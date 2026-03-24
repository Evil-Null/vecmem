/**
 * Vector Memory Engine — Invariant System Unit Tests
 *
 * Tests cover:
 * - Each of 5 invariants passes on a valid store (after normal save)
 * - everyChunkHasEmbedding: detects orphan chunk without embedding
 * - chunkCountsMatch: detects mismatch between document.chunk_count and actual chunks
 * - uniformDimensions: detects mixed dimensions in embeddings table
 * - ftsInSync: detects FTS entries missing for existing chunks
 * - noOrphanChunks: detects chunks pointing to non-existent documents
 * - checkAllInvariants() runs all 5 and passes on valid state
 * - checkAllInvariants() throws InvariantViolation when any fails
 * - withInvariantCheck() wrapper runs checks after operation in dev mode
 * - withInvariantCheck() skips checks when NODE_ENV=production
 */

import { describe, test, expect, beforeEach, afterEach } from 'vitest'
import { mkdtempSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import Database from 'better-sqlite3'
import { SqliteStore } from '../../src/store/sqlite.js'
import {
  createDocumentId,
  createChunkId,
  createProjectId,
  type DocumentMeta,
  type ChunkWithEmbedding,
  type ProjectId,
  type RawChunk,
} from '../../src/types.js'
import { InvariantViolation } from '../../src/errors.js'
import {
  everyChunkHasEmbedding,
  chunkCountsMatch,
  uniformDimensions,
  ftsInSync,
  noOrphanChunks,
  checkAllInvariants,
  withInvariantCheck,
} from '../../src/invariants.js'

// ============================================================================
// Test Helpers
// ============================================================================

function makeTmpDir(): string {
  return mkdtempSync(join(tmpdir(), 'vector-invariants-test-'))
}

function makeConfig(storagePath: string) {
  return { storagePath, databaseName: 'test.db' }
}

function makeProjectId(name = 'test-project'): ProjectId {
  return createProjectId(name)
}

function makeDocumentMeta(overrides: Partial<DocumentMeta> = {}): DocumentMeta {
  const project = makeProjectId()
  return {
    title: 'Test Document',
    filePath: '/path/to/test.md',
    project,
    contentHash: 'abc123hash',
    tags: ['test', 'unit'],
    frontmatter: { key: 'value' },
    indexedAt: new Date('2026-03-24T00:00:00Z'),
    fileSize: 1024,
    ...overrides,
  }
}

function makeRawChunk(index: number, overrides: Partial<RawChunk> = {}): RawChunk {
  return {
    content: `## Heading ${index}\n\nChunk content for index ${index}.`,
    contentPlain: `Heading ${index} Chunk content for index ${index}.`,
    headingPath: [`Heading ${index}`],
    index,
    hasCodeBlock: false,
    ...overrides,
  }
}

function makeEmbedding(dims = 384): Float32Array {
  const arr = new Float32Array(dims)
  for (let i = 0; i < dims; i++) {
    arr[i] = Math.random() * 2 - 1
  }
  return arr
}

function makeChunkWithEmbedding(
  index: number,
  dims = 384,
  chunkOverrides: Partial<RawChunk> = {},
): ChunkWithEmbedding {
  return {
    chunk: makeRawChunk(index, chunkOverrides),
    embedding: makeEmbedding(dims),
  }
}

/**
 * Open a raw better-sqlite3 connection to the same database file.
 * This is used to inject invalid states for testing invariant detection.
 */
function openRawDb(tmpDir: string): Database.Database {
  const dbPath = join(tmpDir, 'test.db')
  const db = new Database(dbPath)
  db.pragma('foreign_keys = OFF') // Allow creating invalid states
  return db
}

// ============================================================================
// Tests
// ============================================================================

describe('Invariant System', () => {
  let tmpDir: string
  let store: SqliteStore
  let rawDb: Database.Database

  beforeEach(() => {
    tmpDir = makeTmpDir()
    store = new SqliteStore(makeConfig(tmpDir))
  })

  afterEach(() => {
    if (rawDb) {
      rawDb.close()
    }
    store.close()
    rmSync(tmpDir, { recursive: true, force: true })
  })

  // --------------------------------------------------------------------------
  // Valid state — all invariants pass
  // --------------------------------------------------------------------------

  describe('valid state after normal save', () => {
    test('all 5 invariants pass on a valid store with data', () => {
      const doc = makeDocumentMeta()
      const items = [
        makeChunkWithEmbedding(0),
        makeChunkWithEmbedding(1),
        makeChunkWithEmbedding(2),
      ]

      store.save(doc, items)

      rawDb = openRawDb(tmpDir)
      expect(() => everyChunkHasEmbedding(rawDb)).not.toThrow()
      expect(() => chunkCountsMatch(rawDb)).not.toThrow()
      expect(() => uniformDimensions(rawDb)).not.toThrow()
      expect(() => ftsInSync(rawDb)).not.toThrow()
      expect(() => noOrphanChunks(rawDb)).not.toThrow()
    })

    test('all 5 invariants pass on an empty store', () => {
      rawDb = openRawDb(tmpDir)
      expect(() => everyChunkHasEmbedding(rawDb)).not.toThrow()
      expect(() => chunkCountsMatch(rawDb)).not.toThrow()
      expect(() => uniformDimensions(rawDb)).not.toThrow()
      expect(() => ftsInSync(rawDb)).not.toThrow()
      expect(() => noOrphanChunks(rawDb)).not.toThrow()
    })
  })

  // --------------------------------------------------------------------------
  // everyChunkHasEmbedding
  // --------------------------------------------------------------------------

  describe('everyChunkHasEmbedding', () => {
    test('passes when every chunk has an embedding', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0), makeChunkWithEmbedding(1)])

      rawDb = openRawDb(tmpDir)
      expect(() => everyChunkHasEmbedding(rawDb)).not.toThrow()
    })

    test('detects orphan chunk without embedding', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0)])

      rawDb = openRawDb(tmpDir)

      // Manually insert a chunk without a corresponding embedding
      const docId = createDocumentId(doc.project, doc.filePath)
      rawDb.prepare(`
        INSERT INTO chunks (id, document_id, project, content, content_plain, token_count, heading_path, heading_depth, chunk_index, has_code_block)
        VALUES ('orphan_chunk_id', ?, 'test-project', 'orphan content', 'orphan content', 10, '[]', 0, 99, 0)
      `).run(docId)

      expect(() => everyChunkHasEmbedding(rawDb)).toThrow(InvariantViolation)
    })
  })

  // --------------------------------------------------------------------------
  // chunkCountsMatch
  // --------------------------------------------------------------------------

  describe('chunkCountsMatch', () => {
    test('passes when document.chunk_count matches actual chunk count', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0), makeChunkWithEmbedding(1)])

      rawDb = openRawDb(tmpDir)
      expect(() => chunkCountsMatch(rawDb)).not.toThrow()
    })

    test('detects mismatch between document.chunk_count and actual chunks', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0), makeChunkWithEmbedding(1)])

      rawDb = openRawDb(tmpDir)

      // Manually update chunk_count to wrong value
      const docId = createDocumentId(doc.project, doc.filePath)
      rawDb.prepare('UPDATE documents SET chunk_count = 99 WHERE id = ?').run(docId)

      expect(() => chunkCountsMatch(rawDb)).toThrow(InvariantViolation)
    })
  })

  // --------------------------------------------------------------------------
  // uniformDimensions
  // --------------------------------------------------------------------------

  describe('uniformDimensions', () => {
    test('passes when all embeddings have uniform dimensions', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0, 384), makeChunkWithEmbedding(1, 384)])

      rawDb = openRawDb(tmpDir)
      expect(() => uniformDimensions(rawDb)).not.toThrow()
    })

    test('passes on empty embeddings table', () => {
      rawDb = openRawDb(tmpDir)
      expect(() => uniformDimensions(rawDb)).not.toThrow()
    })

    test('detects mixed dimensions in embeddings table', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0, 384)])

      rawDb = openRawDb(tmpDir)

      // Insert a second document with a chunk that has a different embedding dimension
      const doc2 = makeDocumentMeta({ filePath: '/path/to/other.md' })
      const docId2 = createDocumentId(doc2.project, doc2.filePath)
      const chunkId2 = 'mixed_dim_chunk_id'

      rawDb.prepare(`
        INSERT INTO documents (id, project, file_path, title, content_hash, file_size, tags, frontmatter, chunk_count, indexed_at)
        VALUES (?, ?, ?, 'Other Doc', 'hash2', 512, '[]', '{}', 1, datetime('now'))
      `).run(docId2, doc2.project, doc2.filePath)

      rawDb.prepare(`
        INSERT INTO chunks (id, document_id, project, content, content_plain, token_count, heading_path, heading_depth, chunk_index, has_code_block)
        VALUES (?, ?, ?, 'content', 'content', 5, '[]', 0, 0, 0)
      `).run(chunkId2, docId2, doc2.project)

      // Insert embedding with different dimensions (128 instead of 384)
      const smallEmbedding = new Float32Array(128)
      const embeddingBuffer = Buffer.from(smallEmbedding.buffer, smallEmbedding.byteOffset, smallEmbedding.byteLength)
      rawDb.prepare(`
        INSERT INTO embeddings (chunk_id, model_name, dimensions, vector)
        VALUES (?, 'all-MiniLM-L6-v2', 128, ?)
      `).run(chunkId2, embeddingBuffer)

      expect(() => uniformDimensions(rawDb)).toThrow(InvariantViolation)
    })
  })

  // --------------------------------------------------------------------------
  // ftsInSync
  // --------------------------------------------------------------------------

  describe('ftsInSync', () => {
    test('passes when FTS entries match chunks', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0), makeChunkWithEmbedding(1)])

      rawDb = openRawDb(tmpDir)
      expect(() => ftsInSync(rawDb)).not.toThrow()
    })

    test('detects FTS entries missing for existing chunks', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0)])

      rawDb = openRawDb(tmpDir)

      // Delete the FTS entry directly, leaving the chunk orphaned from FTS
      const docId = createDocumentId(doc.project, doc.filePath)
      const chunkId = createChunkId(docId, 0)
      rawDb.prepare("DELETE FROM chunks_fts WHERE chunk_id = ?").run(chunkId)

      expect(() => ftsInSync(rawDb)).toThrow(InvariantViolation)
    })

    test('detects extra FTS entries without matching chunks', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0)])

      rawDb = openRawDb(tmpDir)

      // Insert an extra FTS entry without a matching chunk
      rawDb.prepare(`
        INSERT INTO chunks_fts(chunk_id, title, content, tags)
        VALUES ('ghost_chunk_id', 'Ghost', 'ghost content', '[]')
      `).run()

      expect(() => ftsInSync(rawDb)).toThrow(InvariantViolation)
    })
  })

  // --------------------------------------------------------------------------
  // noOrphanChunks
  // --------------------------------------------------------------------------

  describe('noOrphanChunks', () => {
    test('passes when all chunks reference existing documents', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0)])

      rawDb = openRawDb(tmpDir)
      expect(() => noOrphanChunks(rawDb)).not.toThrow()
    })

    test('detects chunks pointing to non-existent documents', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0)])

      rawDb = openRawDb(tmpDir)

      // Insert a chunk referencing a non-existent document
      rawDb.prepare(`
        INSERT INTO chunks (id, document_id, project, content, content_plain, token_count, heading_path, heading_depth, chunk_index, has_code_block)
        VALUES ('orphan_ref_chunk', 'nonexistent_doc_id', 'test-project', 'orphan', 'orphan', 5, '[]', 0, 0, 0)
      `).run()

      expect(() => noOrphanChunks(rawDb)).toThrow(InvariantViolation)
    })
  })

  // --------------------------------------------------------------------------
  // checkAllInvariants()
  // --------------------------------------------------------------------------

  describe('checkAllInvariants()', () => {
    test('runs all 5 invariants and passes on valid state', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0), makeChunkWithEmbedding(1)])

      rawDb = openRawDb(tmpDir)
      expect(() => checkAllInvariants(rawDb)).not.toThrow()
    })

    test('passes on empty database', () => {
      rawDb = openRawDb(tmpDir)
      expect(() => checkAllInvariants(rawDb)).not.toThrow()
    })

    test('throws InvariantViolation when any invariant fails', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0)])

      rawDb = openRawDb(tmpDir)

      // Create an orphan chunk (no embedding)
      const docId = createDocumentId(doc.project, doc.filePath)
      rawDb.prepare(`
        INSERT INTO chunks (id, document_id, project, content, content_plain, token_count, heading_path, heading_depth, chunk_index, has_code_block)
        VALUES ('bad_chunk', ?, 'test-project', 'bad', 'bad', 3, '[]', 0, 99, 0)
      `).run(docId)

      expect(() => checkAllInvariants(rawDb)).toThrow(InvariantViolation)
    })
  })

  // --------------------------------------------------------------------------
  // withInvariantCheck()
  // --------------------------------------------------------------------------

  describe('withInvariantCheck()', () => {
    test('runs checks after operation in dev mode and returns result', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0)])

      rawDb = openRawDb(tmpDir)

      const result = withInvariantCheck(rawDb, () => {
        return 42
      })

      expect(result).toBe(42)
    })

    test('throws InvariantViolation if operation leaves invalid state (dev mode)', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0)])

      rawDb = openRawDb(tmpDir)

      const docId = createDocumentId(doc.project, doc.filePath)

      expect(() =>
        withInvariantCheck(rawDb, () => {
          // Create an invalid state inside the operation
          rawDb.prepare(`
            INSERT INTO chunks (id, document_id, project, content, content_plain, token_count, heading_path, heading_depth, chunk_index, has_code_block)
            VALUES ('bad_chunk_2', ?, 'test-project', 'bad', 'bad', 3, '[]', 0, 99, 0)
          `).run(docId)
          return 'result'
        }),
      ).toThrow(InvariantViolation)
    })

    test('skips checks when NODE_ENV=production', () => {
      rawDb = openRawDb(tmpDir)

      const originalEnv = process.env['NODE_ENV']
      try {
        process.env['NODE_ENV'] = 'production'

        // Insert invalid state manually — in production mode, this should NOT throw
        const doc = makeDocumentMeta()
        store.save(doc, [makeChunkWithEmbedding(0)])

        // Close and reopen rawDb to see the new data
        rawDb.close()
        rawDb = openRawDb(tmpDir)

        const docId = createDocumentId(doc.project, doc.filePath)
        rawDb.prepare(`
          INSERT INTO chunks (id, document_id, project, content, content_plain, token_count, heading_path, heading_depth, chunk_index, has_code_block)
          VALUES ('bad_chunk_3', ?, 'test-project', 'bad', 'bad', 3, '[]', 0, 99, 0)
        `).run(docId)

        // Should NOT throw because production mode skips invariant checks
        const result = withInvariantCheck(rawDb, () => 'production-result')
        expect(result).toBe('production-result')
      } finally {
        process.env['NODE_ENV'] = originalEnv
      }
    })

    test('propagates operation errors without running invariant checks', () => {
      rawDb = openRawDb(tmpDir)

      expect(() =>
        withInvariantCheck(rawDb, () => {
          throw new Error('operation failed')
        }),
      ).toThrow('operation failed')
    })
  })
})
