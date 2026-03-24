/**
 * Vector Memory Engine — Search Performance Contracts
 *
 * CI-enforced speed budgets. These are TESTS, not benchmarks.
 * If performance regresses, CI fails.
 *
 * Setup: 1000 chunks with random 384-dim Float32Array embeddings.
 * Uses a fake embedder (random vectors) — we measure search speed,
 * not embedding quality.
 *
 * Contracts:
 * - Hybrid search (BM25 + vector + RRF) < 200ms for 1K chunks
 * - BM25 search < 50ms for 1K chunks
 * - Vector search < 100ms for 1K chunks
 * - RRF fusion < 10ms for 100 results
 */

import { describe, test, expect, beforeAll, afterAll } from 'vitest'
import { mkdirSync, rmSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { randomBytes } from 'node:crypto'
import Database from 'better-sqlite3'

import { SqliteStore } from '../../src/store/sqlite.js'
import { Bm25Search } from '../../src/search/bm25.js'
import { VectorSearch } from '../../src/search/vector.js'
import { rrfFuse } from '../../src/search/fusion.js'
import { SearchOrchestrator } from '../../src/pipeline/searcher.js'
import {
  createProjectId,
  createDocumentId,
  createChunkId,
  type DocumentMeta,
  type ChunkWithEmbedding,
  type RawChunk,
  type Embedder,
  type Logger,
  type ProjectId,
} from '../../src/types.js'

// ============================================================================
// Constants
// ============================================================================

const CHUNK_COUNT = 1000
const EMBEDDING_DIM = 384
const PROJECT: ProjectId = createProjectId('perf-test')

// ============================================================================
// Random data generators
// ============================================================================

/** Generate a random Float32Array with values in [0, 1], then L2-normalize */
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

/** Generate realistic-looking chunk content for FTS5 indexing */
function generateChunkContent(index: number): string {
  const topics = [
    'authentication', 'database', 'deployment', 'monitoring',
    'caching', 'networking', 'security', 'logging', 'testing',
    'configuration', 'migration', 'performance', 'scaling',
    'API', 'microservices', 'containers', 'orchestration',
    'observability', 'debugging', 'profiling',
  ]
  const topic = topics[index % topics.length]!
  return `This section covers ${topic} in detail. ` +
    `The ${topic} system handles request processing and data management. ` +
    `Key considerations for ${topic} include throughput, latency, and reliability. ` +
    `Implementation uses standard patterns for ${topic} with proper error handling. ` +
    `Chunk number ${index} provides additional context about ${topic} operations.`
}

// ============================================================================
// Fake embedder — returns random vectors, measures search not embedding
// ============================================================================

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
// Test Suite
// ============================================================================

describe('Search Performance Contracts', () => {
  let tempDir: string
  let dbDir: string
  let store: SqliteStore
  let db: Database.Database
  let bm25: Bm25Search
  let vectorSearch: VectorSearch
  let queryEmbedding: Float32Array

  // --------------------------------------------------------------------------
  // Setup: create store with 1000 chunks + embeddings
  // --------------------------------------------------------------------------

  beforeAll(() => {
    tempDir = join(tmpdir(), `vector-perf-search-${randomBytes(8).toString('hex')}`)
    dbDir = join(tempDir, '.vector')
    mkdirSync(dbDir, { recursive: true })

    store = new SqliteStore({ storagePath: dbDir, databaseName: 'perf.db' })

    // Insert 1000 chunks across 50 documents (20 chunks each)
    const docsCount = 50
    const chunksPerDoc = CHUNK_COUNT / docsCount

    for (let d = 0; d < docsCount; d++) {
      const filePath = `/perf/doc-${d}.md`
      const doc: DocumentMeta = {
        title: `Performance Test Document ${d}`,
        filePath,
        project: PROJECT,
        contentHash: `hash-${d}-${randomBytes(8).toString('hex')}`,
        tags: ['perf', 'test'],
        frontmatter: {},
        indexedAt: new Date(),
        fileSize: 1000,
      }

      const items: ChunkWithEmbedding[] = []
      for (let c = 0; c < chunksPerDoc; c++) {
        const globalIndex = d * chunksPerDoc + c
        const content = generateChunkContent(globalIndex)
        const chunk: RawChunk = {
          content: `## Section ${c}\n\n${content}`,
          contentPlain: content,
          headingPath: [`Document ${d}`, `Section ${c}`],
          index: c,
          hasCodeBlock: false,
        }
        items.push({ chunk, embedding: randomEmbedding() })
      }

      store.save(doc, items)
    }

    // Open a separate DB handle for search operations
    db = new Database(join(dbDir, 'perf.db'))
    db.pragma('journal_mode = WAL')
    db.pragma('foreign_keys = ON')

    bm25 = new Bm25Search(db)
    vectorSearch = new VectorSearch(db)

    // Pre-generate query embedding
    queryEmbedding = randomEmbedding()
  })

  afterAll(() => {
    db.close()
    store.close()
    if (existsSync(tempDir)) {
      rmSync(tempDir, { recursive: true, force: true })
    }
  })

  // --------------------------------------------------------------------------
  // Contract: hybrid search < 200ms
  // --------------------------------------------------------------------------

  test('hybrid search completes under 200ms for 1K chunks', async () => {
    const embedder = createFakeEmbedder()
    const logger = createSilentLogger()
    const searcher = new SearchOrchestrator(db, embedder, logger, {
      defaultTopK: 10,
      rrfK: 60,
      minScore: 0.0,
    })

    // Warm up (first query may be slower due to JIT/SQLite cache)
    await searcher.query('authentication security')

    // Measure
    const start = performance.now()
    const results = await searcher.query('database deployment monitoring')
    const elapsed = performance.now() - start

    expect(results.length).toBeGreaterThan(0)
    expect(elapsed).toBeLessThan(200)
  })

  // --------------------------------------------------------------------------
  // Contract: BM25 search < 50ms
  // --------------------------------------------------------------------------

  test('BM25 search completes under 50ms for 1K chunks', () => {
    // Warm up — use a single term that exists in chunk content
    bm25.search('throughput')

    const start = performance.now()
    // Use a single term that appears in ALL chunks (generated content includes it)
    const results = bm25.search('throughput', { topK: 50 })
    const elapsed = performance.now() - start

    expect(results.length).toBeGreaterThan(0)
    expect(elapsed).toBeLessThan(50)
  })

  // --------------------------------------------------------------------------
  // Contract: vector search < 100ms
  // --------------------------------------------------------------------------

  test('vector search completes under 100ms for 1K chunks', () => {
    // Warm up
    vectorSearch.search(queryEmbedding, { topK: 10 })

    const start = performance.now()
    const results = vectorSearch.search(queryEmbedding, { topK: 50 })
    const elapsed = performance.now() - start

    expect(results.length).toBeGreaterThan(0)
    expect(elapsed).toBeLessThan(100)
  })

  // --------------------------------------------------------------------------
  // Contract: RRF fusion < 10ms for 100 results
  // --------------------------------------------------------------------------

  test('RRF fusion completes under 10ms for 100 results', () => {
    // Generate 100 BM25 results and 100 vector results with partial overlap
    const bm25Results = Array.from({ length: 100 }, (_, i) => ({
      chunkId: `chunk-${i}`,
      rank: i + 1,
    }))

    const vectorResults = Array.from({ length: 100 }, (_, i) => ({
      chunkId: `chunk-${i + 50}`, // 50 overlap with BM25
      rank: i + 1,
    }))

    // Warm up
    rrfFuse(bm25Results.slice(0, 10), vectorResults.slice(0, 10), 60)

    const start = performance.now()
    const results = rrfFuse(bm25Results, vectorResults, 60)
    const elapsed = performance.now() - start

    expect(results.length).toBeGreaterThan(0)
    expect(elapsed).toBeLessThan(10)
  })

  // --------------------------------------------------------------------------
  // Contract: search results are valid
  // --------------------------------------------------------------------------

  test('all search result scores are in [0, 1]', async () => {
    const embedder = createFakeEmbedder()
    const logger = createSilentLogger()
    const searcher = new SearchOrchestrator(db, embedder, logger, {
      defaultTopK: 50,
      rrfK: 60,
      minScore: 0.0,
    })

    const results = await searcher.query('caching performance scaling')

    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(0)
      expect(r.score).toBeLessThanOrEqual(1)
    }
  })
})
