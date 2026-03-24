/**
 * Vector Memory Engine — Store Property Tests
 *
 * Property-based tests using fast-check to prove store properties
 * hold for all valid inputs.
 *
 * numRuns is set to 500 (not 10K) because each run creates a real SQLite database
 * with transactions, inserts, and deletes. Higher run counts would make the test
 * suite impractically slow without providing proportionally more coverage.
 *
 * Properties tested:
 * 1. Cascade deletion: delete document -> getChunks returns empty
 * 2. Idempotent save: save same data twice -> same document count, same chunk count
 * 3. Every saved chunk is retrievable: save N chunks -> getChunks returns N chunks
 */

import { describe, test, beforeEach, afterEach } from 'vitest'
import fc from 'fast-check'
import { mkdtempSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { SqliteStore } from '../../src/store/sqlite.js'
import {
  createDocumentId,
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
  return mkdtempSync(join(tmpdir(), 'vector-store-prop-'))
}

function makeProjectId(name = 'prop-project'): ProjectId {
  return createProjectId(name)
}

function makeDocumentMeta(filePath: string, project: ProjectId): DocumentMeta {
  return {
    title: `Doc ${filePath}`,
    filePath,
    project,
    contentHash: `hash-${filePath}`,
    tags: ['prop-test'],
    frontmatter: {},
    indexedAt: new Date('2026-03-24T00:00:00Z'),
    fileSize: 256,
  }
}

function makeEmbedding(dims = 384): Float32Array {
  const arr = new Float32Array(dims)
  for (let i = 0; i < dims; i++) {
    arr[i] = Math.random() * 2 - 1
  }
  return arr
}

function makeItems(count: number): ChunkWithEmbedding[] {
  const items: ChunkWithEmbedding[] = []
  for (let i = 0; i < count; i++) {
    const chunk: RawChunk = {
      content: `## Section ${i}\n\nContent for chunk ${i}.`,
      contentPlain: `Section ${i} Content for chunk ${i}.`,
      headingPath: [`Section ${i}`],
      index: i,
      hasCodeBlock: false,
    }
    items.push({ chunk, embedding: makeEmbedding() })
  }
  return items
}

// ============================================================================
// Store Property Tests
// ============================================================================

describe('Store Properties', () => {
  let tmpDir: string
  let store: SqliteStore

  beforeEach(() => {
    tmpDir = makeTmpDir()
    store = new SqliteStore({ storagePath: tmpDir, databaseName: 'prop-test.db' })
  })

  afterEach(() => {
    store.close()
    rmSync(tmpDir, { recursive: true, force: true })
  })

  test('Property 1: Cascade deletion — delete document -> getChunks returns empty', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 20 }),
        fc.integer({ min: 1, max: 1000 }),
        (chunkCount, seed) => {
          const project = makeProjectId()
          const filePath = `/cascade-${seed}.md`
          const doc = makeDocumentMeta(filePath, project)
          const items = makeItems(chunkCount)

          store.save(doc, items)

          const docId = createDocumentId(project, filePath)

          // Delete the document
          store.removeDocument(docId)

          // getChunks must return empty
          const chunks = store.getChunks(docId)
          return chunks.length === 0
        },
      ),
      { numRuns: 500 },
    )
  })

  test('Property 2: Idempotent save — save same data twice -> same document count, same chunk count', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 15 }),
        fc.integer({ min: 1, max: 1000 }),
        (chunkCount, seed) => {
          const project = makeProjectId()
          const filePath = `/idempotent-${seed}.md`
          const doc = makeDocumentMeta(filePath, project)
          const items = makeItems(chunkCount)

          // Save twice
          store.save(doc, items)
          store.save(doc, items)

          // Should still have exactly one document
          const docs = store.listDocuments(project)
          const matchingDocs = docs.filter(d => d.filePath === filePath)
          if (matchingDocs.length !== 1) return false

          // Should have exactly chunkCount chunks
          const docId = createDocumentId(project, filePath)
          const chunks = store.getChunks(docId)
          return chunks.length === chunkCount
        },
      ),
      { numRuns: 500 },
    )
  })

  test('Property 3: Every saved chunk is retrievable — save N chunks -> getChunks returns N chunks', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 25 }),
        fc.integer({ min: 1, max: 1000 }),
        (chunkCount, seed) => {
          const project = makeProjectId()
          const filePath = `/retrievable-${seed}.md`
          const doc = makeDocumentMeta(filePath, project)
          const items = makeItems(chunkCount)

          store.save(doc, items)

          const docId = createDocumentId(project, filePath)
          const chunks = store.getChunks(docId)

          // Must return exactly N chunks
          if (chunks.length !== chunkCount) return false

          // Each chunk must have a valid embedding
          for (const chunk of chunks) {
            if (!(chunk.embedding instanceof Float32Array)) return false
            if (chunk.embedding.length !== 384) return false
          }

          return true
        },
      ),
      { numRuns: 500 },
    )
  })
})
