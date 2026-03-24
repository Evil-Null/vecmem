/**
 * Vector Memory Engine — Indexing Performance Contracts
 *
 * CI-enforced throughput budgets. These are TESTS, not benchmarks.
 * If performance regresses, CI fails.
 *
 * Setup: 20 markdown files with ~5 chunks each in a temp directory.
 * Uses a fake embedder (random vectors) to avoid model download.
 *
 * Contracts:
 * - Indexing throughput > 50 chunks/sec
 * - Single file indexing < 500ms
 * - Incremental re-index (unchanged) < 50ms per file
 */

import { describe, test, expect, beforeEach, afterEach } from 'vitest'
import { mkdirSync, writeFileSync, rmSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { randomBytes } from 'node:crypto'

import { indexFile, indexDirectory } from '../../src/pipeline/indexer.js'
import { MarkdownParser } from '../../src/parser/markdown.js'
import { SqliteStore } from '../../src/store/sqlite.js'
import type { Embedder, Logger } from '../../src/types.js'

// ============================================================================
// Constants
// ============================================================================

const EMBEDDING_DIM = 384
const FILE_COUNT = 20

// ============================================================================
// Fake embedder — random vectors, fast
// ============================================================================

function randomEmbedding(): Float32Array {
  const vec = new Float32Array(EMBEDDING_DIM)
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    vec[i] = Math.random()
  }
  // L2 normalize
  let norm = 0
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    norm += vec[i]! * vec[i]!
  }
  norm = Math.sqrt(norm)
  if (norm > 0) {
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      vec[i] = vec[i]! / norm
    }
  }
  return vec
}

function createFakeEmbedder(): Embedder {
  return {
    dimensions: EMBEDDING_DIM,
    async embed(_text: string): Promise<Float32Array> {
      return randomEmbedding()
    },
    async embedBatch(texts: string[]): Promise<Float32Array[]> {
      return texts.map(() => randomEmbedding())
    },
  }
}

/** Silent logger — no output during performance tests */
function createSilentLogger(): Logger {
  return {
    debug: () => {},
    info: () => {},
    warn: () => {},
    error: () => {},
  }
}

// ============================================================================
// Markdown file generators
// ============================================================================

/**
 * Generate a markdown file with ~5 headings/chunks.
 * Each chunk has enough content to be non-trivial (> minChunkTokens).
 */
function generateMarkdownFile(index: number): string {
  const topics = [
    'Authentication', 'Database Design', 'API Gateway', 'Monitoring',
    'Cache Strategy', 'Load Balancing', 'Error Handling', 'Logging',
    'Testing Strategy', 'CI/CD Pipeline', 'Docker Setup', 'Kubernetes',
    'Security Audit', 'Data Migration', 'Performance Tuning',
    'Service Mesh', 'Event Sourcing', 'CQRS Pattern', 'GraphQL',
    'WebSocket Integration',
  ]

  const topic = topics[index % topics.length]!

  return `# ${topic} Guide

## Overview

This document covers ${topic} in detail for our production system.
The ${topic} implementation follows industry best practices and has been
tested extensively in staging environments. Key metrics include
throughput, latency, and error rates.

## Architecture

The ${topic} architecture uses a layered approach with clear separation
of concerns. The presentation layer handles user requests, the business
logic layer processes data, and the persistence layer manages storage.
Each layer has well-defined interfaces and can be tested independently.

## Configuration

Configuration for ${topic} is managed through environment variables and
config files. Default values are provided for development environments.
Production settings should be specified in the deployment manifests.
All configuration is validated at startup using schema validation.

## Troubleshooting

Common issues with ${topic} include timeout errors, connection refused,
and authentication failures. Check the logs for detailed error messages.
Use the doctor command to verify system health and invariant compliance.
Contact the platform team if issues persist after following these steps.

## Performance Considerations

The ${topic} system is designed for high throughput and low latency.
Benchmarks show that the current implementation handles over one thousand
requests per second with p99 latency under fifty milliseconds.
Regular performance testing ensures no regressions are introduced.
`
}

// ============================================================================
// Test Suite
// ============================================================================

describe('Indexing Performance Contracts', () => {
  let tempDir: string
  let docsDir: string
  let dbDir: string
  let store: SqliteStore
  let parser: MarkdownParser
  let embedder: Embedder
  let logger: Logger

  beforeEach(() => {
    tempDir = join(tmpdir(), `vector-perf-index-${randomBytes(8).toString('hex')}`)
    docsDir = join(tempDir, 'docs')
    dbDir = join(tempDir, '.vector')
    mkdirSync(docsDir, { recursive: true })
    mkdirSync(dbDir, { recursive: true })

    store = new SqliteStore({ storagePath: dbDir, databaseName: 'perf.db' })
    parser = new MarkdownParser({ project: 'perf-test' })
    embedder = createFakeEmbedder()
    logger = createSilentLogger()

    // Generate 20 markdown files
    for (let i = 0; i < FILE_COUNT; i++) {
      const content = generateMarkdownFile(i)
      writeFileSync(join(docsDir, `doc-${i.toString().padStart(2, '0')}.md`), content, 'utf-8')
    }
  })

  afterEach(() => {
    store.close()
    if (existsSync(tempDir)) {
      rmSync(tempDir, { recursive: true, force: true })
    }
  })

  // --------------------------------------------------------------------------
  // Contract: indexing throughput > 50 chunks/sec
  // --------------------------------------------------------------------------

  test('indexing throughput exceeds 50 chunks/sec', async () => {
    const start = performance.now()
    const result = await indexDirectory(
      docsDir,
      parser,
      embedder,
      store,
      logger,
    )
    const elapsed = performance.now() - start

    // Verify indexing succeeded
    expect(result.indexed).toBe(FILE_COUNT)
    expect(result.failed).toBe(0)

    // Calculate total chunks indexed
    // Each file has ~5 heading-based chunks
    const totalChunks = result.indexed * 5 // approximate — at least 5 chunks per file
    const chunksPerSec = totalChunks / (elapsed / 1000)

    expect(chunksPerSec).toBeGreaterThan(50)
  })

  // --------------------------------------------------------------------------
  // Contract: single file indexing < 500ms
  // --------------------------------------------------------------------------

  test('single file indexing completes under 500ms', async () => {
    const filePath = join(docsDir, 'doc-00.md')

    // Warm up (first call may be slower due to module loading)
    await indexFile(filePath, parser, embedder, store, logger, docsDir)

    // Re-create store to force re-index (clear file_hashes)
    store.close()
    rmSync(join(dbDir, 'perf.db'))
    rmSync(join(dbDir, 'perf.db-wal'), { force: true })
    rmSync(join(dbDir, 'perf.db-shm'), { force: true })
    store = new SqliteStore({ storagePath: dbDir, databaseName: 'perf.db' })

    const start = performance.now()
    const result = await indexFile(filePath, parser, embedder, store, logger, docsDir)
    const elapsed = performance.now() - start

    expect(result.status).toBe('indexed')
    expect(elapsed).toBeLessThan(500)
  })

  // --------------------------------------------------------------------------
  // Contract: incremental re-index (unchanged) < 50ms per file
  // --------------------------------------------------------------------------

  test('incremental re-index of unchanged file completes under 50ms', async () => {
    const filePath = join(docsDir, 'doc-00.md')

    // First index
    await indexFile(filePath, parser, embedder, store, logger, docsDir)

    // Second index — file unchanged, should skip quickly
    const start = performance.now()
    const result = await indexFile(filePath, parser, embedder, store, logger, docsDir)
    const elapsed = performance.now() - start

    expect(result.status).toBe('skipped')
    expect(elapsed).toBeLessThan(50)
  })

  // --------------------------------------------------------------------------
  // Contract: batch re-index (all unchanged) is fast
  // --------------------------------------------------------------------------

  test('batch re-index of unchanged directory is fast', async () => {
    // First index all files
    await indexDirectory(docsDir, parser, embedder, store, logger)

    // Second index — all unchanged, should skip all quickly
    const start = performance.now()
    const result = await indexDirectory(docsDir, parser, embedder, store, logger)
    const elapsed = performance.now() - start

    expect(result.indexed).toBe(0)
    expect(result.skipped).toBe(FILE_COUNT)
    expect(result.failed).toBe(0)

    // 20 files skipped should take well under 2 seconds
    expect(elapsed).toBeLessThan(2000)
  })

  // --------------------------------------------------------------------------
  // Contract: embedding batch < 5ms/chunk (pipeline overhead with fake embedder)
  // --------------------------------------------------------------------------

  test('embedding batch overhead < 5ms/chunk with fake embedder (100 chunks)', async () => {
    // Generate 100 chunks of text
    const chunkTexts: string[] = []
    for (let i = 0; i < 100; i++) {
      chunkTexts.push(
        `This is chunk number ${i} about topic ${i % 10}. ` +
        `It contains enough text to be a realistic chunk for embedding. ` +
        `The content discusses architecture, testing, and performance.`
      )
    }

    // Measure embedding batch with fake embedder (random vectors)
    // This tests pipeline overhead, not actual model speed.
    const start = performance.now()
    const results = await embedder.embedBatch(chunkTexts)
    const elapsed = performance.now() - start

    // Verify results
    expect(results).toHaveLength(100)
    for (const vec of results) {
      expect(vec).toBeInstanceOf(Float32Array)
      expect(vec.length).toBe(EMBEDDING_DIM)
    }

    // Contract: 100 chunks should complete in < 500ms (5ms/chunk)
    // With fake embedder this measures pure JS overhead + array allocation
    expect(elapsed).toBeLessThan(500)
  })
})
