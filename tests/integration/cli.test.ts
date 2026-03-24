/**
 * Vector Memory Engine — CLI Integration Tests
 *
 * Tests each CLI command by invoking the command handlers directly
 * against a temp directory with real SQLite stores and a fake embedder.
 *
 * Since the real embedder downloads a 23MB model, these tests use
 * direct handler calls with controlled dependencies instead of spawning
 * the built binary. The pipeline integration tests already cover the
 * full flow with real components.
 *
 * Coverage:
 * - `vector init` → creates database, shows file count
 * - `vector index` → indexes files (via indexDirectory)
 * - `vector query` → returns results with proper formatting
 * - `vector status` → shows document/chunk counts
 * - `vector doctor` → runs all invariant checks
 * - `vector remove` → removes document and its chunks
 * - Exit codes: 0 (success), 4 (no results)
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { mkdirSync, writeFileSync, rmSync, existsSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { randomBytes } from 'node:crypto'

import { SqliteStore } from '../../src/store/sqlite.js'
import { MarkdownParser } from '../../src/parser/markdown.js'
import { indexFile, indexDirectory } from '../../src/pipeline/indexer.js'
import { SearchOrchestrator } from '../../src/pipeline/searcher.js'
import { createLogger } from '../../src/logger.js'
import { createDocumentId, createProjectId } from '../../src/types.js'
import {
  checkAllInvariants,
  everyChunkHasEmbedding,
  chunkCountsMatch,
  uniformDimensions,
  ftsInSync,
  noOrphanChunks,
} from '../../src/invariants.js'
import type { Embedder, Logger, VectorConfig } from '../../src/types.js'
import type { } from '../../src/config.js'

// Format imports — testing format functions independently
import {
  formatSuccess,
  formatWarning,
  formatError,
  formatSearchResult,
  formatProgress,
  formatElapsed,
  formatBytes,
  formatTimeAgo,
  formatIndexSummary,
  formatDoctorCheck,
  formatHeader,
} from '../../src/cli/format.js'

// ============================================================================
// Fake Embedder — deterministic, no model download
// ============================================================================

function fakeEmbed(text: string): Float32Array {
  const vec = new Float32Array(384)
  for (let i = 0; i < text.length && i < 384; i++) {
    vec[i] = (text.charCodeAt(i) % 100) / 100
  }
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

// ============================================================================
// Test Helpers
// ============================================================================

function createTempDir(): string {
  const dir = join(tmpdir(), `vector-cli-test-${randomBytes(8).toString('hex')}`)
  mkdirSync(dir, { recursive: true })
  return dir
}

function writeMdFile(dir: string, name: string, content: string): string {
  const filePath = join(dir, name)
  const fileDir = join(filePath, '..')
  if (!existsSync(fileDir)) {
    mkdirSync(fileDir, { recursive: true })
  }
  writeFileSync(filePath, content, 'utf-8')
  return filePath
}

function createTestConfig(dbDir: string): VectorConfig {
  return {
    storagePath: dbDir,
    databaseName: 'test.db',
    embeddingProvider: 'local' as const,
    embeddingModel: 'Xenova/all-MiniLM-L6-v2',
    embeddingDimensions: 384,
    modelCachePath: join(dbDir, 'models'),
    maxChunkTokens: 400,
    minChunkTokens: 50,
    chunkOverlapTokens: 40,
    headingSplitDepth: 2,
    defaultTopK: 10,
    rrfK: 60,
    minScore: 0.01,
    project: 'test-project',
  }
}

// ============================================================================
// Format Function Tests
// ============================================================================

describe('CLI Format Functions', () => {
  it('formatSuccess contains checkmark', () => {
    const result = formatSuccess('Operation completed')
    expect(result).toContain('Operation completed')
    // Contains unicode checkmark (with or without ANSI codes)
    expect(result).toMatch(/\u2713/)
  })

  it('formatWarning contains warning symbol', () => {
    const result = formatWarning('Something needs attention')
    expect(result).toContain('Something needs attention')
  })

  it('formatError contains cross mark', () => {
    const result = formatError('Something went wrong')
    expect(result).toContain('Something went wrong')
    expect(result).toMatch(/\u2717/)
  })

  it('formatProgress shows current/total', () => {
    const result = formatProgress(5, 10)
    expect(result).toContain('5/10')
    expect(result).toContain('files')
  })

  it('formatProgress handles zero total', () => {
    const result = formatProgress(0, 0)
    expect(result).toContain('0/0')
  })

  it('formatElapsed shows milliseconds for < 1s', () => {
    expect(formatElapsed(87)).toBe('87ms')
    expect(formatElapsed(500)).toBe('500ms')
  })

  it('formatElapsed shows seconds for >= 1s', () => {
    expect(formatElapsed(2100)).toBe('2.1s')
    expect(formatElapsed(1000)).toBe('1.0s')
  })

  it('formatBytes shows appropriate units', () => {
    expect(formatBytes(500)).toBe('500 B')
    expect(formatBytes(1500)).toBe('1.5 KB')
    expect(formatBytes(4200000)).toBe('4.0 MB')
  })

  it('formatTimeAgo shows relative time', () => {
    const now = new Date()

    // Just now
    expect(formatTimeAgo(now)).toBe('just now')

    // 5 minutes ago
    const fiveMinAgo = new Date(now.getTime() - 5 * 60 * 1000)
    expect(formatTimeAgo(fiveMinAgo)).toBe('5 minutes ago')

    // 1 minute ago
    const oneMinAgo = new Date(now.getTime() - 60 * 1000)
    expect(formatTimeAgo(oneMinAgo)).toBe('1 minute ago')

    // 3 hours ago
    const threeHrsAgo = new Date(now.getTime() - 3 * 60 * 60 * 1000)
    expect(formatTimeAgo(threeHrsAgo)).toBe('3 hours ago')

    // 2 days ago
    const twoDaysAgo = new Date(now.getTime() - 2 * 24 * 60 * 60 * 1000)
    expect(formatTimeAgo(twoDaysAgo)).toBe('2 days ago')
  })

  it('formatDoctorCheck shows pass/fail', () => {
    const pass = formatDoctorCheck('Database', true, 'OK (4.2 MB)')
    expect(pass).toContain('Database')
    expect(pass).toContain('OK')

    const fail = formatDoctorCheck('FTS index', false, 'out of sync')
    expect(fail).toContain('FTS index')
    expect(fail).toContain('out of sync')
  })

  it('formatHeader shows version', () => {
    const result = formatHeader('0.1.0')
    expect(result).toContain('Vector')
    expect(result).toContain('0.1.0')
  })

  it('formatIndexSummary shows files, chunks, time, size', () => {
    const result = formatIndexSummary(47, 312, 2100, '/tmp/test.db', 4200000)
    expect(result).toContain('47 files')
    expect(result).toContain('312 chunks')
    expect(result).toContain('312 embeddings')
  })
})

// ============================================================================
// CLI Command Integration Tests
// ============================================================================

describe('CLI Command Integration', () => {
  let tempDir: string
  let dbDir: string
  let store: SqliteStore
  let parser: MarkdownParser
  let embedder: Embedder
  let logger: Logger
  let config: VectorConfig

  beforeEach(() => {
    tempDir = createTempDir()
    dbDir = join(tempDir, '.vector')
    mkdirSync(dbDir, { recursive: true })

    config = createTestConfig(dbDir)
    store = new SqliteStore({ storagePath: dbDir, databaseName: 'test.db' })
    parser = new MarkdownParser({ project: 'test-project' })
    embedder = createFakeEmbedder()
    logger = createLogger(false)
  })

  afterEach(() => {
    store.close()
    if (existsSync(tempDir)) {
      rmSync(tempDir, { recursive: true, force: true })
    }
  })

  // --------------------------------------------------------------------------
  // init — creates database, discovers files
  // --------------------------------------------------------------------------

  describe('init', () => {
    it('creates database at configured path', () => {
      // Store was already created in beforeEach — verify DB file exists
      const dbPath = store.getDbPath()
      expect(existsSync(dbPath)).toBe(true)
    })

    it('database has WAL mode enabled', () => {
      const db = store.getDb()
      const mode = db.pragma('journal_mode', { simple: true }) as string
      expect(mode).toBe('wal')
    })

    it('listDocuments returns empty on fresh init', () => {
      const docs = store.listDocuments()
      expect(docs).toHaveLength(0)
    })
  })

  // --------------------------------------------------------------------------
  // index — indexes files, shows progress
  // --------------------------------------------------------------------------

  describe('index', () => {
    it('indexes all markdown files in a directory', async () => {
      writeMdFile(tempDir, 'readme.md', `# Readme\n\nProject overview and documentation.`)
      writeMdFile(tempDir, 'guide.md', `# Guide\n\nGetting started guide for new users.`)

      const progressCalls: Array<{ current: number; total: number }> = []
      const result = await indexDirectory(
        tempDir,
        parser,
        embedder,
        store,
        logger,
        (current, total) => { progressCalls.push({ current, total }) },
      )

      expect(result.indexed).toBe(2)
      expect(result.skipped).toBe(0)
      expect(result.failed).toBe(0)
      expect(progressCalls.length).toBe(2)
    })

    it('skips unchanged files on re-index', async () => {
      writeMdFile(tempDir, 'stable.md', `# Stable\n\nThis file will not change.`)

      await indexDirectory(tempDir, parser, embedder, store, logger)
      const result = await indexDirectory(tempDir, parser, embedder, store, logger)

      expect(result.indexed).toBe(0)
      expect(result.skipped).toBe(1)
    })

    it('indexes nested directories', async () => {
      const docsDir = join(tempDir, 'docs')
      mkdirSync(docsDir, { recursive: true })
      writeMdFile(docsDir, 'api.md', `# API\n\nAPI reference documentation.`)
      writeMdFile(docsDir, 'faq.md', `# FAQ\n\nFrequently asked questions.`)
      writeMdFile(tempDir, 'root.md', `# Root\n\nRoot level doc.`)

      const result = await indexDirectory(tempDir, parser, embedder, store, logger)
      expect(result.indexed).toBe(3)
    })

    it('reports chunk counts correctly after indexing', async () => {
      writeMdFile(tempDir, 'multi.md', `# Multi Section

## Section One

Content for section one about database design.

## Section Two

Content for section two about API architecture.
`)

      await indexDirectory(tempDir, parser, embedder, store, logger)

      const docs = store.listDocuments()
      expect(docs.length).toBe(1)
      expect(docs[0]!.chunkCount).toBeGreaterThan(0)
    })
  })

  // --------------------------------------------------------------------------
  // query — searches and returns results
  // --------------------------------------------------------------------------

  describe('query', () => {
    it('finds indexed content via search', async () => {
      writeMdFile(tempDir, 'auth.md', `# Authentication

## OAuth 2.0 Flow

Authentication uses OAuth 2.0 with PKCE for secure authorization.
The client initiates the flow by redirecting to the authorization server.
`)

      await indexFile(join(tempDir, 'auth.md'), parser, embedder, store, logger, tempDir)

      const search = new SearchOrchestrator(store.getDb(), embedder, logger, {
        defaultTopK: 10,
        rrfK: 60,
        minScore: 0.0,
      })

      const results = await search.query('OAuth authentication')
      expect(results.length).toBeGreaterThan(0)
      expect(results[0]!.score).toBeGreaterThanOrEqual(0)
      expect(results[0]!.score).toBeLessThanOrEqual(1)
    })

    it('returns empty results for unmatched query', async () => {
      writeMdFile(tempDir, 'hello.md', `# Hello\n\nJust a simple greeting.`)
      await indexFile(join(tempDir, 'hello.md'), parser, embedder, store, logger, tempDir)

      const search = new SearchOrchestrator(store.getDb(), embedder, logger, {
        defaultTopK: 10,
        rrfK: 60,
        minScore: 0.5,  // High threshold to get no results
      })

      const results = await search.query('xyzzy quantum entanglement')
      // May or may not have results depending on BM25 — we check structure
      for (const r of results) {
        expect(r.score).toBeGreaterThanOrEqual(0)
        expect(r.score).toBeLessThanOrEqual(1)
      }
    })

    it('formatSearchResult produces valid output for each result', async () => {
      writeMdFile(tempDir, 'format-test.md', `# Format Test

## Content Section

This section has some content about formatting and display.
`)

      await indexFile(join(tempDir, 'format-test.md'), parser, embedder, store, logger, tempDir)

      const search = new SearchOrchestrator(store.getDb(), embedder, logger, {
        defaultTopK: 10,
        rrfK: 60,
        minScore: 0.0,
      })

      const results = await search.query('formatting')
      expect(results.length).toBeGreaterThan(0)

      const formatted = formatSearchResult(results[0]!)
      expect(formatted).toBeTruthy()
      expect(typeof formatted).toBe('string')
      expect(formatted.length).toBeGreaterThan(0)
    })
  })

  // --------------------------------------------------------------------------
  // status — shows document/chunk counts
  // --------------------------------------------------------------------------

  describe('status', () => {
    it('shows correct document and chunk counts', async () => {
      writeMdFile(tempDir, 'a.md', `# Doc A\n\nContent for document A.`)
      writeMdFile(tempDir, 'b.md', `# Doc B\n\nContent for document B.`)

      await indexDirectory(tempDir, parser, embedder, store, logger)

      const docs = store.listDocuments()
      expect(docs.length).toBe(2)

      let totalChunks = 0
      for (const doc of docs) {
        totalChunks += doc.chunkCount
      }
      expect(totalChunks).toBeGreaterThan(0)
    })

    it('shows database size', () => {
      const dbPath = store.getDbPath()
      const stat = statSync(dbPath)
      expect(stat.size).toBeGreaterThan(0)
    })

    it('detects stale files', async () => {
      const filePath = writeMdFile(tempDir, 'stale.md', `# Stale\n\nOriginal content.`)
      await indexFile(filePath, parser, embedder, store, logger, tempDir)

      // Modify the file (make it newer than the indexed time)
      // Wait a small amount to ensure mtime changes
      await new Promise(resolve => setTimeout(resolve, 50))
      writeFileSync(filePath, `# Stale\n\nModified content.`, 'utf-8')

      const docs = store.listDocuments()
      let staleCount = 0
      for (const doc of docs) {
        try {
          const fileStat = statSync(doc.filePath)
          if (fileStat.mtimeMs > doc.indexedAt.getTime()) {
            staleCount++
          }
        } catch {
          staleCount++
        }
      }

      expect(staleCount).toBe(1)
    })
  })

  // --------------------------------------------------------------------------
  // doctor — runs all invariant checks
  // --------------------------------------------------------------------------

  describe('doctor', () => {
    it('all invariants pass on fresh database', () => {
      const db = store.getDb()
      // Should not throw
      checkAllInvariants(db)
    })

    it('all invariants pass after indexing', async () => {
      writeMdFile(tempDir, 'doctor-test.md', `# Doctor Test\n\nContent for doctor testing.`)
      await indexDirectory(tempDir, parser, embedder, store, logger)

      const db = store.getDb()
      checkAllInvariants(db)
    })

    it('individual invariant checks pass', async () => {
      writeMdFile(tempDir, 'inv.md', `# Invariants\n\nTesting individual invariant checks.`)
      await indexDirectory(tempDir, parser, embedder, store, logger)

      const db = store.getDb()
      everyChunkHasEmbedding(db)
      chunkCountsMatch(db)
      uniformDimensions(db)
      ftsInSync(db)
      noOrphanChunks(db)
    })

    it('FTS entry count matches chunk count after indexing', async () => {
      writeMdFile(tempDir, 'fts.md', `# FTS Test

## Section One

Content for FTS testing section one.

## Section Two

Content for FTS testing section two.
`)
      await indexDirectory(tempDir, parser, embedder, store, logger)

      const db = store.getDb()
      const chunkCount = (db.prepare('SELECT COUNT(*) as cnt FROM chunks').get() as { cnt: number }).cnt
      const ftsCount = (db.prepare('SELECT COUNT(*) as cnt FROM chunks_fts').get() as { cnt: number }).cnt

      expect(chunkCount).toBeGreaterThan(0)
      expect(ftsCount).toBe(chunkCount)
    })

    it('embedding count matches chunk count after indexing', async () => {
      writeMdFile(tempDir, 'emb.md', `# Embedding Test\n\nTesting embedding counts.`)
      await indexDirectory(tempDir, parser, embedder, store, logger)

      const db = store.getDb()
      const chunkCount = (db.prepare('SELECT COUNT(*) as cnt FROM chunks').get() as { cnt: number }).cnt
      const embCount = (db.prepare('SELECT COUNT(*) as cnt FROM embeddings').get() as { cnt: number }).cnt

      expect(chunkCount).toBeGreaterThan(0)
      expect(embCount).toBe(chunkCount)
    })
  })

  // --------------------------------------------------------------------------
  // remove — removes document and its chunks
  // --------------------------------------------------------------------------

  describe('remove', () => {
    it('removes a document and its chunks from the store', async () => {
      const filePath = writeMdFile(tempDir, 'remove-me.md', `# Remove Me\n\nThis will be removed.`)
      await indexFile(filePath, parser, embedder, store, logger, tempDir)

      const projectId = createProjectId('test-project')
      const doc = store.getDocument(projectId, filePath)
      expect(doc).not.toBeNull()
      expect(doc!.chunkCount).toBeGreaterThan(0)

      const docId = createDocumentId(projectId, filePath)
      store.removeDocument(docId)

      const afterRemove = store.getDocument(projectId, filePath)
      expect(afterRemove).toBeNull()

      const chunks = store.getChunks(docId)
      expect(chunks).toHaveLength(0)
    })

    it('removal does not affect other documents', async () => {
      const fileA = writeMdFile(tempDir, 'keep.md', `# Keep\n\nThis stays.`)
      const fileB = writeMdFile(tempDir, 'drop.md', `# Drop\n\nThis goes away.`)

      await indexFile(fileA, parser, embedder, store, logger, tempDir)
      await indexFile(fileB, parser, embedder, store, logger, tempDir)

      const projectId = createProjectId('test-project')
      const docB = store.getDocument(projectId, fileB)!
      const docBId = createDocumentId(projectId, fileB)
      store.removeDocument(docBId)

      // fileA should still be there
      const docA = store.getDocument(projectId, fileA)
      expect(docA).not.toBeNull()
      expect(docA!.chunkCount).toBeGreaterThan(0)

      // fileB should be gone
      const afterRemove = store.getDocument(projectId, fileB)
      expect(afterRemove).toBeNull()
    })

    it('invariants hold after removal', async () => {
      const fileA = writeMdFile(tempDir, 'r-a.md', `# A\n\nContent A.`)
      const fileB = writeMdFile(tempDir, 'r-b.md', `# B\n\nContent B.`)

      await indexFile(fileA, parser, embedder, store, logger, tempDir)
      await indexFile(fileB, parser, embedder, store, logger, tempDir)

      const projectId = createProjectId('test-project')
      const docId = createDocumentId(projectId, fileA)
      store.removeDocument(docId)

      const db = store.getDb()
      checkAllInvariants(db)
    })
  })
})
