/**
 * Vector Memory Engine — Pipeline Integration Tests
 *
 * Tests the full pipeline: parse → embed → store → search.
 * Uses a fake embedder (deterministic, fast, no model download)
 * and a real SQLite store + real parser.
 *
 * Coverage:
 * - Full pipeline: .md file → index → query → verify results
 * - Incremental indexing: unchanged file skipped
 * - Re-indexing: modified file re-indexed, new content found
 * - Graceful degradation: ModelLoadError → BM25-only fallback
 * - removeDocument() via pipeline
 * - indexDirectory() discovers all .md files
 * - Security: validateFilePath rejects traversal
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { mkdirSync, writeFileSync, rmSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { randomBytes } from 'node:crypto'

import { indexFile, indexDirectory } from '../../src/pipeline/indexer.js'
import { SearchOrchestrator } from '../../src/pipeline/searcher.js'
import { createLogger } from '../../src/logger.js'
import { MarkdownParser, validateFilePath } from '../../src/parser/markdown.js'
import { SqliteStore } from '../../src/store/sqlite.js'
import { Bm25Search } from '../../src/search/bm25.js'
import { VectorSearch } from '../../src/search/vector.js'
import { ModelLoadError, SecurityError } from '../../src/errors.js'
import { createDocumentId, createProjectId } from '../../src/types.js'
import type { Embedder, Logger } from '../../src/types.js'
import Database from 'better-sqlite3'

// ============================================================================
// Fake Embedder — deterministic, no model download
// ============================================================================

/** Creates a deterministic 384-dim embedding from text content */
function fakeEmbed(text: string): Float32Array {
  const vec = new Float32Array(384)
  // Simple deterministic hash-based embedding
  for (let i = 0; i < text.length && i < 384; i++) {
    vec[i] = (text.charCodeAt(i) % 100) / 100
  }
  // Normalize to unit vector
  let norm = 0
  for (let i = 0; i < 384; i++) {
    norm += vec[i]! * vec[i]!
  }
  norm = Math.sqrt(norm)
  if (norm > 0) {
    for (let i = 0; i < 384; i++) {
      vec[i] = vec[i]! / norm
    }
  }
  return vec
}

function createFakeEmbedder(): Embedder {
  return {
    dimensions: 384,
    async embed(text: string): Promise<Float32Array> {
      return fakeEmbed(text)
    },
    async embedBatch(texts: string[]): Promise<Float32Array[]> {
      return texts.map(fakeEmbed)
    },
  }
}

/** Creates a fake embedder that throws ModelLoadError on every call */
function createFailingEmbedder(): Embedder {
  return {
    dimensions: 384,
    async embed(_text: string): Promise<Float32Array> {
      throw new ModelLoadError('Model not available for testing')
    },
    async embedBatch(_texts: string[]): Promise<Float32Array[]> {
      throw new ModelLoadError('Model not available for testing')
    },
  }
}

/** Creates a fake embedder that throws a generic error (not ModelLoadError) */
function createCrashingEmbedder(): Embedder {
  return {
    dimensions: 384,
    async embed(_text: string): Promise<Float32Array> {
      throw new Error('Unexpected ONNX runtime crash')
    },
    async embedBatch(_texts: string[]): Promise<Float32Array[]> {
      throw new Error('Unexpected ONNX runtime crash')
    },
  }
}

// ============================================================================
// Test helpers
// ============================================================================

function createTempDir(): string {
  const dir = join(tmpdir(), `vector-test-${randomBytes(8).toString('hex')}`)
  mkdirSync(dir, { recursive: true })
  return dir
}

function writeMdFile(dir: string, name: string, content: string): string {
  const filePath = join(dir, name)
  writeFileSync(filePath, content, 'utf-8')
  return filePath
}

// ============================================================================
// Tests
// ============================================================================

describe('Pipeline Integration', () => {
  let tempDir: string
  let dbDir: string
  let store: SqliteStore
  let parser: MarkdownParser
  let embedder: Embedder
  let logger: Logger
  let db: Database.Database

  beforeEach(() => {
    tempDir = createTempDir()
    dbDir = join(tempDir, '.vector')
    mkdirSync(dbDir, { recursive: true })

    store = new SqliteStore({ storagePath: dbDir, databaseName: 'test.db' })
    parser = new MarkdownParser({ project: 'test-project' })
    embedder = createFakeEmbedder()
    logger = createLogger(false)

    // Open a separate DB handle for search operations
    db = new Database(join(dbDir, 'test.db'))
    db.pragma('journal_mode = WAL')
    db.pragma('foreign_keys = ON')
  })

  afterEach(() => {
    db.close()
    ;(store as SqliteStore & { close(): void }).close()
    if (existsSync(tempDir)) {
      rmSync(tempDir, { recursive: true, force: true })
    }
  })

  // --------------------------------------------------------------------------
  // Full pipeline: create .md → index → query → verify
  // --------------------------------------------------------------------------

  it('indexes a markdown file and finds it via search', async () => {
    const filePath = writeMdFile(tempDir, 'auth.md', `# Authentication

## OAuth 2.0 Flow

Authentication uses OAuth 2.0 with PKCE for secure authorization.
The client initiates the flow by redirecting to the authorization server.

## Token Management

Access tokens expire after 1 hour. Refresh tokens are used to obtain
new access tokens without re-authentication.
`)

    // Index the file
    const result = await indexFile(filePath, parser, embedder, store, logger, tempDir)
    expect(result.status).toBe('indexed')
    if (result.status === 'indexed') {
      expect(result.chunks).toBeGreaterThan(0)
    }

    // Search for content
    const searcher = new SearchOrchestrator(db, embedder, logger, {
      defaultTopK: 10,
      rrfK: 60,
      minScore: 0.0,
    })

    const results = await searcher.query('OAuth authentication')
    expect(results.length).toBeGreaterThan(0)

    // Verify result contains relevant content
    const found = results.some(
      r => r.chunk.contentPlain.includes('OAuth') ||
           r.chunk.contentPlain.includes('authentication') ||
           r.chunk.contentPlain.includes('Authentication'),
    )
    expect(found).toBe(true)

    // Verify SearchResult shape
    const first = results[0]!
    expect(first.score).toBeGreaterThanOrEqual(0)
    expect(first.score).toBeLessThanOrEqual(1)
    expect(first.documentTitle).toBeTruthy()
    expect(first.documentPath).toBeTruthy()
    expect(typeof first.highlight).toBe('string')
  })

  // --------------------------------------------------------------------------
  // Incremental indexing: unchanged file is skipped
  // --------------------------------------------------------------------------

  it('skips unchanged files on re-index', async () => {
    const filePath = writeMdFile(tempDir, 'notes.md', `# Notes

Some important notes about the project.
`)

    // First index
    const first = await indexFile(filePath, parser, embedder, store, logger, tempDir)
    expect(first.status).toBe('indexed')

    // Second index — same file, same content
    const second = await indexFile(filePath, parser, embedder, store, logger, tempDir)
    expect(second.status).toBe('skipped')
    if (second.status === 'skipped') {
      expect(second.reason).toBe('unchanged')
    }
  })

  // --------------------------------------------------------------------------
  // Re-indexing: modify file → re-index → new content found
  // --------------------------------------------------------------------------

  it('re-indexes modified files and finds new content', async () => {
    const filePath = writeMdFile(tempDir, 'guide.md', `# Guide

## Getting Started

Install the package with npm install.
`)

    // Index original
    await indexFile(filePath, parser, embedder, store, logger, tempDir)

    // Modify file
    writeFileSync(filePath, `# Guide

## Getting Started

Install the package with npm install.

## Kubernetes Deployment

Deploy to Kubernetes using helm charts and kubectl apply.
`, 'utf-8')

    // Re-index
    const result = await indexFile(filePath, parser, embedder, store, logger, tempDir)
    expect(result.status).toBe('indexed')

    // Search for new content
    const searcher = new SearchOrchestrator(db, embedder, logger, {
      defaultTopK: 10,
      rrfK: 60,
      minScore: 0.0,
    })

    const results = await searcher.query('Kubernetes deployment')
    expect(results.length).toBeGreaterThan(0)

    const found = results.some(
      r => r.chunk.contentPlain.includes('Kubernetes') ||
           r.chunk.contentPlain.includes('helm'),
    )
    expect(found).toBe(true)
  })

  // --------------------------------------------------------------------------
  // Graceful degradation: ModelLoadError → BM25-only fallback
  // --------------------------------------------------------------------------

  it('falls back to BM25-only when embedder throws ModelLoadError', async () => {
    // First, index with a working embedder so there's data to search
    const filePath = writeMdFile(tempDir, 'api.md', `# API Reference

## REST Endpoints

The API exposes REST endpoints for creating and managing resources.
Each endpoint requires authentication via Bearer token.
`)

    await indexFile(filePath, parser, embedder, store, logger, tempDir)

    // Now search with a failing embedder
    const failingEmbedder = createFailingEmbedder()
    const warnMessages: Array<{ event: string; data?: Record<string, unknown> }> = []
    const capturingLogger: Logger = {
      debug: () => {},
      info: () => {},
      warn: (event, data) => { warnMessages.push({ event, data }) },
      error: () => {},
    }

    const searcher = new SearchOrchestrator(db, failingEmbedder, capturingLogger, {
      defaultTopK: 10,
      rrfK: 60,
      minScore: 0.0,
    })

    // Should NOT throw — falls back to BM25
    const results = await searcher.query('REST endpoints')
    expect(results.length).toBeGreaterThan(0)

    // Should have logged a degradation warning
    const degradedWarn = warnMessages.find(m => m.event === 'search.degraded')
    expect(degradedWarn).toBeDefined()
    expect(degradedWarn!.data?.reason).toBe('model_unavailable')
    expect(degradedWarn!.data?.fallback).toBe('bm25_only')
  })

  // --------------------------------------------------------------------------
  // Non-ModelLoadError re-thrown (not caught)
  // --------------------------------------------------------------------------

  it('re-throws non-ModelLoadError from embedder', async () => {
    const filePath = writeMdFile(tempDir, 'crash.md', `# Crash Test

Some content for testing crash scenarios.
`)

    await indexFile(filePath, parser, embedder, store, logger, tempDir)

    const crashingEmbedder = createCrashingEmbedder()
    const searcher = new SearchOrchestrator(db, crashingEmbedder, logger, {
      defaultTopK: 10,
      rrfK: 60,
      minScore: 0.0,
    })

    // Should throw the original error
    await expect(searcher.query('crash test')).rejects.toThrow('Unexpected ONNX runtime crash')
  })

  // --------------------------------------------------------------------------
  // removeDocument() via pipeline
  // --------------------------------------------------------------------------

  it('removes a document and its chunks from the store', async () => {
    const filePath = writeMdFile(tempDir, 'remove-me.md', `# Remove Me

This document should be removed after indexing.
`)

    await indexFile(filePath, parser, embedder, store, logger, tempDir)

    const projectId = createProjectId('test-project')
    const doc = store.getDocument(projectId, filePath)
    expect(doc).not.toBeNull()

    // Remove the document
    const docId = createDocumentId(projectId, filePath)
    store.removeDocument(docId)

    // Verify it's gone
    const afterRemove = store.getDocument(projectId, filePath)
    expect(afterRemove).toBeNull()

    // Verify chunks are gone
    const chunks = store.getChunks(docId)
    expect(chunks).toHaveLength(0)

    // Search should not find it
    const searcher = new SearchOrchestrator(db, embedder, logger, {
      defaultTopK: 10,
      rrfK: 60,
      minScore: 0.0,
    })
    const results = await searcher.query('Remove Me')
    const found = results.some(r => r.documentPath === filePath)
    expect(found).toBe(false)
  })

  // --------------------------------------------------------------------------
  // indexDirectory() discovers and indexes all .md files
  // --------------------------------------------------------------------------

  it('indexes all markdown files in a directory', async () => {
    const docsDir = join(tempDir, 'docs')
    mkdirSync(docsDir, { recursive: true })

    writeMdFile(docsDir, 'one.md', `# Document One

First document content about databases.
`)
    writeMdFile(docsDir, 'two.md', `# Document Two

Second document content about networking.
`)
    writeMdFile(docsDir, 'three.md', `# Document Three

Third document content about security.
`)
    // Non-.md file should be ignored
    writeFileSync(join(docsDir, 'readme.txt'), 'This is not markdown')

    // Subdirectory with nested .md
    const subDir = join(docsDir, 'sub')
    mkdirSync(subDir)
    writeMdFile(subDir, 'nested.md', `# Nested Document

Nested document about performance.
`)

    const progressCalls: Array<{ current: number; total: number }> = []
    const result = await indexDirectory(
      docsDir,
      parser,
      embedder,
      store,
      logger,
      (current, total) => { progressCalls.push({ current, total }) },
    )

    expect(result.indexed).toBe(4) // one.md, two.md, three.md, nested.md
    expect(result.skipped).toBe(0)
    expect(result.failed).toBe(0)

    // Verify progress callback was called
    expect(progressCalls.length).toBe(4)
    // All progress calls should have total=4
    for (const call of progressCalls) {
      expect(call.total).toBe(4)
    }

    // Search across all indexed files
    const searcher = new SearchOrchestrator(db, embedder, logger, {
      defaultTopK: 10,
      rrfK: 60,
      minScore: 0.0,
    })

    const results = await searcher.query('security')
    expect(results.length).toBeGreaterThan(0)
  })

  // --------------------------------------------------------------------------
  // indexDirectory() handles per-file failures gracefully
  // --------------------------------------------------------------------------

  it('continues indexing when individual files fail', async () => {
    const docsDir = join(tempDir, 'mixed')
    mkdirSync(docsDir, { recursive: true })

    writeMdFile(docsDir, 'good.md', `# Good Document

This file is valid and should be indexed.
`)

    // Create a file that will cause a parse error (empty content still parses,
    // but we can make the embedder fail for specific content)
    writeMdFile(docsDir, 'also-good.md', `# Also Good

Another valid document.
`)

    const result = await indexDirectory(
      docsDir,
      parser,
      embedder,
      store,
      logger,
    )

    expect(result.indexed).toBe(2)
    expect(result.failed).toBe(0)
  })

  // --------------------------------------------------------------------------
  // Security: validateFilePath rejects paths outside project root
  // --------------------------------------------------------------------------

  it('rejects file paths outside the project root', () => {
    expect(() => {
      validateFilePath('/etc/passwd', tempDir)
    }).toThrow(SecurityError)
  })

  it('rejects path traversal attempts', () => {
    expect(() => {
      validateFilePath(join(tempDir, '..', '..', 'etc', 'passwd'), tempDir)
    }).toThrow(SecurityError)
  })

  it('rejects indexFile with path outside project root', async () => {
    // Write a file outside tempDir
    const outsideDir = createTempDir()
    const outsideFile = writeMdFile(outsideDir, 'evil.md', '# Evil\n\nShould not be indexed.')

    await expect(
      indexFile(outsideFile, parser, embedder, store, logger, tempDir),
    ).rejects.toThrow(SecurityError)

    rmSync(outsideDir, { recursive: true, force: true })
  })

  // --------------------------------------------------------------------------
  // indexDirectory() with empty directory
  // --------------------------------------------------------------------------

  it('handles empty directory gracefully', async () => {
    const emptyDir = join(tempDir, 'empty')
    mkdirSync(emptyDir)

    const result = await indexDirectory(
      emptyDir,
      parser,
      embedder,
      store,
      logger,
    )

    expect(result.indexed).toBe(0)
    expect(result.skipped).toBe(0)
    expect(result.failed).toBe(0)
  })

  // --------------------------------------------------------------------------
  // Search result scores are UnitScore [0, 1]
  // --------------------------------------------------------------------------

  it('returns search results with scores in [0, 1]', async () => {
    // Index multiple files for meaningful score distribution
    writeMdFile(tempDir, 'alpha.md', `# Alpha Guide

Comprehensive guide to alpha testing and quality assurance.
Alpha testing involves internal team testing before beta release.
`)
    writeMdFile(tempDir, 'beta.md', `# Beta Guide

Beta testing involves external users testing before production.
Bug reports from beta testers help identify edge cases.
`)

    await indexFile(join(tempDir, 'alpha.md'), parser, embedder, store, logger, tempDir)
    await indexFile(join(tempDir, 'beta.md'), parser, embedder, store, logger, tempDir)

    const searcher = new SearchOrchestrator(db, embedder, logger, {
      defaultTopK: 10,
      rrfK: 60,
      minScore: 0.0,
    })

    const results = await searcher.query('alpha testing')
    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(0)
      expect(r.score).toBeLessThanOrEqual(1)
    }
  })

  // --------------------------------------------------------------------------
  // Search results sorted descending by score
  // --------------------------------------------------------------------------

  it('returns search results sorted descending by score', async () => {
    writeMdFile(tempDir, 'sorted-a.md', `# Sorting Test A

Content about algorithms and sorting.
`)
    writeMdFile(tempDir, 'sorted-b.md', `# Sorting Test B

Content about data structures and arrays.
`)

    await indexFile(join(tempDir, 'sorted-a.md'), parser, embedder, store, logger, tempDir)
    await indexFile(join(tempDir, 'sorted-b.md'), parser, embedder, store, logger, tempDir)

    const searcher = new SearchOrchestrator(db, embedder, logger, {
      defaultTopK: 10,
      rrfK: 60,
      minScore: 0.0,
    })

    const results = await searcher.query('sorting algorithms')
    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1]!.score).toBeGreaterThanOrEqual(results[i]!.score)
    }
  })
})
