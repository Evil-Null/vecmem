/**
 * Vector Memory Engine — SQLite Store Unit Tests
 *
 * Tests cover:
 * - Atomic save (document + chunks + embeddings)
 * - Save with empty chunks
 * - getDocument returns StoredDocument or null
 * - getChunks returns IndexedChunk[] with Float32Array embeddings
 * - removeDocument cascades: deletes chunks + embeddings + FTS entries
 * - needsReindex: new file, unchanged file, changed file
 * - Idempotent save
 * - FTS5 trigger integration (insert + delete)
 * - listDocuments
 * - Database file permissions (0600)
 */

import { describe, test, expect, beforeEach, afterEach } from 'vitest'
import { mkdtempSync, rmSync, statSync } from 'node:fs'
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

// ============================================================================
// Test Helpers
// ============================================================================

function makeTmpDir(): string {
  return mkdtempSync(join(tmpdir(), 'vector-store-test-'))
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

// ============================================================================
// Tests
// ============================================================================

describe('SqliteStore', () => {
  let tmpDir: string
  let store: SqliteStore

  beforeEach(() => {
    tmpDir = makeTmpDir()
    store = new SqliteStore(makeConfig(tmpDir))
  })

  afterEach(() => {
    store.close()
    rmSync(tmpDir, { recursive: true, force: true })
  })

  // --------------------------------------------------------------------------
  // save() — atomic persistence
  // --------------------------------------------------------------------------

  describe('save()', () => {
    test('persists document + chunks + embeddings atomically', () => {
      const doc = makeDocumentMeta()
      const items = [
        makeChunkWithEmbedding(0),
        makeChunkWithEmbedding(1),
        makeChunkWithEmbedding(2),
      ]

      store.save(doc, items)

      const stored = store.getDocument(doc.project, doc.filePath)
      expect(stored).not.toBeNull()
      expect(stored!.title).toBe('Test Document')
      expect(stored!.chunkCount).toBe(3)

      const docId = createDocumentId(doc.project, doc.filePath)
      const chunks = store.getChunks(docId)
      expect(chunks).toHaveLength(3)

      // Each chunk must have an embedding
      for (const chunk of chunks) {
        expect(chunk.embedding).toBeInstanceOf(Float32Array)
        expect(chunk.embedding.length).toBe(384)
      }
    })

    test('with empty chunks array creates document with 0 chunks', () => {
      const doc = makeDocumentMeta()

      store.save(doc, [])

      const stored = store.getDocument(doc.project, doc.filePath)
      expect(stored).not.toBeNull()
      expect(stored!.chunkCount).toBe(0)

      const docId = createDocumentId(doc.project, doc.filePath)
      const chunks = store.getChunks(docId)
      expect(chunks).toHaveLength(0)
    })

    test('idempotent: saving same data twice produces same state', () => {
      const doc = makeDocumentMeta()
      const items = [makeChunkWithEmbedding(0), makeChunkWithEmbedding(1)]

      store.save(doc, items)
      store.save(doc, items)

      const stored = store.getDocument(doc.project, doc.filePath)
      expect(stored).not.toBeNull()
      expect(stored!.chunkCount).toBe(2)

      const docId = createDocumentId(doc.project, doc.filePath)
      const chunks = store.getChunks(docId)
      expect(chunks).toHaveLength(2)
    })

    test('upsert: saving updated document replaces old data', () => {
      const doc = makeDocumentMeta()
      const items1 = [makeChunkWithEmbedding(0), makeChunkWithEmbedding(1)]

      store.save(doc, items1)

      // Save with different chunk count
      const updatedDoc = makeDocumentMeta({ contentHash: 'newhash' })
      const items2 = [makeChunkWithEmbedding(0)]

      store.save(updatedDoc, items2)

      const stored = store.getDocument(doc.project, doc.filePath)
      expect(stored).not.toBeNull()
      expect(stored!.chunkCount).toBe(1)
      expect(stored!.contentHash).toBe('newhash')
    })
  })

  // --------------------------------------------------------------------------
  // getDocument()
  // --------------------------------------------------------------------------

  describe('getDocument()', () => {
    test('returns StoredDocument with correct fields', () => {
      const doc = makeDocumentMeta()
      const items = [makeChunkWithEmbedding(0)]

      store.save(doc, items)

      const stored = store.getDocument(doc.project, doc.filePath)
      expect(stored).not.toBeNull()
      expect(stored!.id).toBe(createDocumentId(doc.project, doc.filePath))
      expect(stored!.title).toBe('Test Document')
      expect(stored!.filePath).toBe('/path/to/test.md')
      expect(stored!.project).toBe(doc.project)
      expect(stored!.contentHash).toBe('abc123hash')
      expect(stored!.tags).toEqual(['test', 'unit'])
      expect(stored!.frontmatter).toEqual({ key: 'value' })
      expect(stored!.chunkCount).toBe(1)
      expect(stored!.fileSize).toBe(1024)
      expect(stored!.indexedAt).toBeInstanceOf(Date)
    })

    test('returns null for missing document', () => {
      const result = store.getDocument(makeProjectId(), '/nonexistent.md')
      expect(result).toBeNull()
    })
  })

  // --------------------------------------------------------------------------
  // getChunks()
  // --------------------------------------------------------------------------

  describe('getChunks()', () => {
    test('returns IndexedChunk[] with embeddings as Float32Array', () => {
      const doc = makeDocumentMeta()
      const embedding = makeEmbedding(384)
      const items: ChunkWithEmbedding[] = [
        {
          chunk: makeRawChunk(0, { content: '## Auth\n\nOAuth flow', contentPlain: 'Auth OAuth flow' }),
          embedding,
        },
      ]

      store.save(doc, items)

      const docId = createDocumentId(doc.project, doc.filePath)
      const chunks = store.getChunks(docId)

      expect(chunks).toHaveLength(1)
      const chunk = chunks[0]!
      expect(chunk.id).toBe(createChunkId(docId, 0))
      expect(chunk.documentId).toBe(docId)
      expect(chunk.content).toBe('## Auth\n\nOAuth flow')
      expect(chunk.contentPlain).toBe('Auth OAuth flow')
      expect(chunk.headingPath).toEqual(['Heading 0'])
      expect(chunk.index).toBe(0)
      expect(chunk.hasCodeBlock).toBe(false)
      expect(chunk.embedding).toBeInstanceOf(Float32Array)
      expect(chunk.embedding.length).toBe(384)

      // Verify embedding values are preserved through BLOB round-trip
      for (let i = 0; i < embedding.length; i++) {
        expect(chunk.embedding[i]).toBeCloseTo(embedding[i]!, 6)
      }
    })

    test('returns empty array for missing document', () => {
      const docId = createDocumentId(makeProjectId(), '/nonexistent.md')
      const chunks = store.getChunks(docId)
      expect(chunks).toEqual([])
    })

    test('chunks are ordered by chunk_index', () => {
      const doc = makeDocumentMeta()
      const items = [
        makeChunkWithEmbedding(0),
        makeChunkWithEmbedding(1),
        makeChunkWithEmbedding(2),
      ]

      store.save(doc, items)

      const docId = createDocumentId(doc.project, doc.filePath)
      const chunks = store.getChunks(docId)

      expect(chunks).toHaveLength(3)
      expect(chunks[0]!.index).toBe(0)
      expect(chunks[1]!.index).toBe(1)
      expect(chunks[2]!.index).toBe(2)
    })
  })

  // --------------------------------------------------------------------------
  // removeDocument()
  // --------------------------------------------------------------------------

  describe('removeDocument()', () => {
    test('cascades: deletes chunks + embeddings + FTS entries', () => {
      const doc = makeDocumentMeta()
      const items = [makeChunkWithEmbedding(0), makeChunkWithEmbedding(1)]

      store.save(doc, items)

      const docId = createDocumentId(doc.project, doc.filePath)

      // Verify exists
      expect(store.getDocument(doc.project, doc.filePath)).not.toBeNull()
      expect(store.getChunks(docId)).toHaveLength(2)

      // Remove
      store.removeDocument(docId)

      // Document gone
      expect(store.getDocument(doc.project, doc.filePath)).toBeNull()

      // Chunks gone (CASCADE)
      expect(store.getChunks(docId)).toHaveLength(0)

      // FTS entries gone (verified via raw SQL)
      const dbPath = join(tmpDir, 'test.db')
      const db = new Database(dbPath)
      const ftsCount = db.prepare('SELECT count(*) as cnt FROM chunks_fts').get() as { cnt: number }
      expect(ftsCount.cnt).toBe(0)
      db.close()
    })

    test('removing nonexistent document does not throw', () => {
      const docId = createDocumentId(makeProjectId(), '/nonexistent.md')
      expect(() => store.removeDocument(docId)).not.toThrow()
    })
  })

  // --------------------------------------------------------------------------
  // needsReindex()
  // --------------------------------------------------------------------------

  describe('needsReindex()', () => {
    test('returns true for new file', () => {
      const project = makeProjectId()
      expect(store.needsReindex(project, '/new-file.md', 'somehash')).toBe(true)
    })

    test('returns false for unchanged file (same hash)', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0)])

      expect(store.needsReindex(doc.project, doc.filePath, 'abc123hash')).toBe(false)
    })

    test('returns true for changed file (different hash)', () => {
      const doc = makeDocumentMeta()
      store.save(doc, [makeChunkWithEmbedding(0)])

      expect(store.needsReindex(doc.project, doc.filePath, 'differenthash')).toBe(true)
    })
  })

  // --------------------------------------------------------------------------
  // FTS5 triggers
  // --------------------------------------------------------------------------

  describe('FTS5 triggers', () => {
    test('FTS5 entries created via trigger after insert', () => {
      const doc = makeDocumentMeta({ title: 'Authentication Guide' })
      const items: ChunkWithEmbedding[] = [
        {
          chunk: makeRawChunk(0, {
            contentPlain: 'OAuth flow with PKCE authentication',
            headingPath: ['Authentication'],
          }),
          embedding: makeEmbedding(),
        },
      ]

      store.save(doc, items)

      // Query FTS directly
      const dbPath = join(tmpDir, 'test.db')
      const db = new Database(dbPath)
      const ftsRows = db.prepare('SELECT * FROM chunks_fts').all() as Array<{
        chunk_id: string
        title: string
        content: string
        tags: string
      }>
      expect(ftsRows).toHaveLength(1)
      expect(ftsRows[0]!.title).toBe('Authentication Guide')
      expect(ftsRows[0]!.content).toBe('OAuth flow with PKCE authentication')
      db.close()
    })

    test('FTS5 entries deleted via trigger after chunk delete', () => {
      const doc = makeDocumentMeta()
      const items = [makeChunkWithEmbedding(0), makeChunkWithEmbedding(1)]

      store.save(doc, items)

      const docId = createDocumentId(doc.project, doc.filePath)
      store.removeDocument(docId)

      // Verify FTS is empty
      const dbPath = join(tmpDir, 'test.db')
      const db = new Database(dbPath)
      const ftsCount = db.prepare('SELECT count(*) as cnt FROM chunks_fts').get() as { cnt: number }
      expect(ftsCount.cnt).toBe(0)
      db.close()
    })
  })

  // --------------------------------------------------------------------------
  // listDocuments()
  // --------------------------------------------------------------------------

  describe('listDocuments()', () => {
    test('returns all documents for project', () => {
      const project = makeProjectId()
      const doc1 = makeDocumentMeta({ filePath: '/a.md', project })
      const doc2 = makeDocumentMeta({ filePath: '/b.md', project })

      store.save(doc1, [makeChunkWithEmbedding(0)])
      store.save(doc2, [makeChunkWithEmbedding(0), makeChunkWithEmbedding(1)])

      const docs = store.listDocuments(project)
      expect(docs).toHaveLength(2)

      const paths = docs.map(d => d.filePath).sort()
      expect(paths).toEqual(['/a.md', '/b.md'])
    })

    test('returns empty array when no documents exist', () => {
      const docs = store.listDocuments(makeProjectId())
      expect(docs).toEqual([])
    })

    test('filters by project when specified', () => {
      const projectA = createProjectId('project-a')
      const projectB = createProjectId('project-b')

      store.save(
        makeDocumentMeta({ filePath: '/a.md', project: projectA }),
        [makeChunkWithEmbedding(0)],
      )
      store.save(
        makeDocumentMeta({ filePath: '/b.md', project: projectB }),
        [makeChunkWithEmbedding(0)],
      )

      const docsA = store.listDocuments(projectA)
      expect(docsA).toHaveLength(1)
      expect(docsA[0]!.filePath).toBe('/a.md')

      const docsB = store.listDocuments(projectB)
      expect(docsB).toHaveLength(1)
      expect(docsB[0]!.filePath).toBe('/b.md')
    })

    test('returns all documents when no project filter', () => {
      const projectA = createProjectId('project-a')
      const projectB = createProjectId('project-b')

      store.save(
        makeDocumentMeta({ filePath: '/a.md', project: projectA }),
        [makeChunkWithEmbedding(0)],
      )
      store.save(
        makeDocumentMeta({ filePath: '/b.md', project: projectB }),
        [makeChunkWithEmbedding(0)],
      )

      const allDocs = store.listDocuments()
      expect(allDocs).toHaveLength(2)
    })
  })

  // --------------------------------------------------------------------------
  // Database file permissions
  // --------------------------------------------------------------------------

  describe('file permissions', () => {
    test('database file permissions are 0600', () => {
      const dbPath = join(tmpDir, 'test.db')
      const stat = statSync(dbPath)
      // 0600 = owner read/write only = 0o600 = 384 decimal
      // stat.mode includes file type bits, so mask with 0o777
      const permissions = stat.mode & 0o777
      expect(permissions).toBe(0o600)
    })
  })
})
