/**
 * Vector Memory Engine — SQLite Store Implementation
 *
 * Implements the Store interface using better-sqlite3 with:
 * - WAL mode for concurrent reads
 * - Atomic transactions (all-or-nothing saves)
 * - FTS5 full-text search index (synced via triggers)
 * - CASCADE deletes (document -> chunks -> embeddings -> FTS)
 * - Content-based deterministic IDs
 * - File permissions 0600 (owner read/write only)
 *
 * CRITICAL Float32Array/BLOB alignment:
 * - Store: Buffer.from(embedding.buffer, embedding.byteOffset, embedding.byteLength)
 * - Retrieve: new Float32Array(new Uint8Array(blob).buffer)
 */

import Database from 'better-sqlite3'
import { readFileSync, chmodSync, mkdirSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { countTokens } from '../parser/tokens.js'
import { TransactionFailedError, DatabaseCorruptedError } from '../errors.js'
import {
  createDocumentId,
  createChunkId,
  type Store,
  type DocumentMeta,
  type StoredDocument,
  type IndexedChunk,
  type ChunkWithEmbedding,
  type DocumentId,
  type ProjectId,
  type ChunkId,
} from '../types.js'

// ============================================================================
// Schema version
// ============================================================================

const SCHEMA_VERSION = 1

// ============================================================================
// Configuration interface for SqliteStore
// ============================================================================

interface StoreConfig {
  readonly storagePath: string
  readonly databaseName: string
}

// ============================================================================
// SqliteStore Implementation
// ============================================================================

export class SqliteStore implements Store {
  private readonly db: Database.Database

  constructor(config: StoreConfig) {
    const dbDir = config.storagePath
    if (!existsSync(dbDir)) {
      mkdirSync(dbDir, { recursive: true })
    }

    const dbPath = join(dbDir, config.databaseName)
    const isNew = !existsSync(dbPath)

    try {
      this.db = new Database(dbPath)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      throw new DatabaseCorruptedError(
        `Failed to open database at ${dbPath}: ${message}`,
      )
    }

    // Set PRAGMAs before schema
    this.db.pragma('journal_mode = WAL')
    this.db.pragma('foreign_keys = ON')
    this.db.pragma('busy_timeout = 5000')

    // Initialize schema
    this.initSchema()

    // Set file permissions to 0600 (owner read/write only)
    if (isNew) {
      chmodSync(dbPath, 0o600)
    }
  }

  // --------------------------------------------------------------------------
  // Schema initialization
  // --------------------------------------------------------------------------

  private initSchema(): void {
    const currentVersion = this.db.pragma('user_version', {
      simple: true,
    }) as number

    if (currentVersion < SCHEMA_VERSION) {
      const schemaPath = join(
        dirname(fileURLToPath(import.meta.url)),
        'schema.sql',
      )
      const schemaSql = readFileSync(schemaPath, 'utf-8')

      // Execute schema DDL — PRAGMAs in SQL file are re-applied (idempotent)
      this.db.exec(schemaSql)

      // Mark schema version
      this.db.pragma(`user_version = ${SCHEMA_VERSION}`)
    }
  }

  // --------------------------------------------------------------------------
  // save() — atomic upsert of document + chunks + embeddings
  // --------------------------------------------------------------------------

  save(doc: DocumentMeta, items: readonly ChunkWithEmbedding[]): void {
    const docId = createDocumentId(doc.project, doc.filePath)

    const insertDoc = this.db.prepare(`
      INSERT OR REPLACE INTO documents
        (id, project, file_path, title, content_hash, file_size, tags, frontmatter, chunk_count, indexed_at)
      VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    `)

    const deleteChunks = this.db.prepare(
      'DELETE FROM chunks WHERE document_id = ?',
    )

    const insertChunk = this.db.prepare(`
      INSERT INTO chunks
        (id, document_id, project, content, content_plain, token_count, heading_path, heading_depth, chunk_index, has_code_block)
      VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `)

    const insertEmbedding = this.db.prepare(`
      INSERT INTO embeddings (chunk_id, model_name, dimensions, vector)
      VALUES (?, ?, ?, ?)
    `)

    const upsertFileHash = this.db.prepare(`
      INSERT OR REPLACE INTO file_hashes (project, file_path, file_hash, file_size, updated_at)
      VALUES (?, ?, ?, ?, datetime('now'))
    `)

    const runTransaction = this.db.transaction(() => {
      // Delete existing chunks first (triggers will clean FTS)
      deleteChunks.run(docId)

      // Delete existing document to handle upsert cleanly
      this.db.prepare('DELETE FROM documents WHERE id = ?').run(docId)

      // Insert document
      insertDoc.run(
        docId,
        doc.project,
        doc.filePath,
        doc.title,
        doc.contentHash,
        doc.fileSize,
        JSON.stringify(doc.tags),
        JSON.stringify(doc.frontmatter),
        items.length,
      )

      // Insert chunks + embeddings
      for (const item of items) {
        const chunkId = createChunkId(docId, item.chunk.index)

        insertChunk.run(
          chunkId,
          docId,
          doc.project,
          item.chunk.content,
          item.chunk.contentPlain,
          countTokens(item.chunk.contentPlain),
          JSON.stringify(item.chunk.headingPath),
          item.chunk.headingPath.length,
          item.chunk.index,
          item.chunk.hasCodeBlock ? 1 : 0,
        )

        // CRITICAL: Proper Float32Array -> Buffer alignment
        const embeddingBuffer = Buffer.from(
          item.embedding.buffer,
          item.embedding.byteOffset,
          item.embedding.byteLength,
        )

        insertEmbedding.run(
          chunkId,
          'all-MiniLM-L6-v2',
          item.embedding.length,
          embeddingBuffer,
        )
      }

      // Update file hashes for incremental indexing
      upsertFileHash.run(
        doc.project,
        doc.filePath,
        doc.contentHash,
        doc.fileSize,
      )
    })

    try {
      runTransaction()
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      throw new TransactionFailedError(
        `Failed to save document ${doc.filePath}: ${message}`,
      )
    }
  }

  // --------------------------------------------------------------------------
  // getDocument()
  // --------------------------------------------------------------------------

  getDocument(
    project: ProjectId,
    filePath: string,
  ): StoredDocument | null {
    const row = this.db
      .prepare(
        'SELECT * FROM documents WHERE project = ? AND file_path = ?',
      )
      .get(project, filePath) as DocumentRow | undefined

    if (!row) return null

    return rowToStoredDocument(row)
  }

  // --------------------------------------------------------------------------
  // getChunks()
  // --------------------------------------------------------------------------

  getChunks(documentId: DocumentId): IndexedChunk[] {
    const rows = this.db
      .prepare(
        `SELECT c.*, e.vector, e.dimensions
         FROM chunks c
         JOIN embeddings e ON e.chunk_id = c.id
         WHERE c.document_id = ?
         ORDER BY c.chunk_index`,
      )
      .all(documentId) as ChunkRow[]

    return rows.map(rowToIndexedChunk)
  }

  // --------------------------------------------------------------------------
  // removeDocument()
  // --------------------------------------------------------------------------

  removeDocument(documentId: DocumentId): void {
    // CASCADE handles chunks -> embeddings -> FTS (via triggers)
    this.db
      .prepare('DELETE FROM documents WHERE id = ?')
      .run(documentId)
  }

  // --------------------------------------------------------------------------
  // needsReindex()
  // --------------------------------------------------------------------------

  needsReindex(
    project: ProjectId,
    filePath: string,
    contentHash: string,
  ): boolean {
    const row = this.db
      .prepare(
        'SELECT file_hash FROM file_hashes WHERE project = ? AND file_path = ?',
      )
      .get(project, filePath) as { file_hash: string } | undefined

    if (!row) return true
    return row.file_hash !== contentHash
  }

  // --------------------------------------------------------------------------
  // listDocuments()
  // --------------------------------------------------------------------------

  listDocuments(project?: ProjectId): StoredDocument[] {
    let rows: DocumentRow[]

    if (project !== undefined) {
      rows = this.db
        .prepare('SELECT * FROM documents WHERE project = ?')
        .all(project) as DocumentRow[]
    } else {
      rows = this.db
        .prepare('SELECT * FROM documents')
        .all() as DocumentRow[]
    }

    return rows.map(rowToStoredDocument)
  }

  // --------------------------------------------------------------------------
  // getDb() — expose database handle for SearchOrchestrator
  // --------------------------------------------------------------------------

  getDb(): Database.Database {
    return this.db
  }

  // --------------------------------------------------------------------------
  // getDbPath() — full path to the database file
  // --------------------------------------------------------------------------

  getDbPath(): string {
    return this.db.name
  }

  // --------------------------------------------------------------------------
  // close() — release database handle
  // --------------------------------------------------------------------------

  close(): void {
    this.db.close()
  }
}

// ============================================================================
// Row types — internal, not exported
// ============================================================================

interface DocumentRow {
  readonly id: string
  readonly project: string
  readonly file_path: string
  readonly title: string
  readonly content_hash: string
  readonly file_size: number
  readonly tags: string
  readonly frontmatter: string
  readonly chunk_count: number
  readonly indexed_at: string
}

interface ChunkRow {
  readonly id: string
  readonly document_id: string
  readonly project: string
  readonly content: string
  readonly content_plain: string
  readonly token_count: number
  readonly heading_path: string
  readonly heading_depth: number
  readonly chunk_index: number
  readonly has_code_block: number
  readonly vector: Buffer
  readonly dimensions: number
}

// ============================================================================
// Row → Domain type converters
// ============================================================================

function rowToStoredDocument(row: DocumentRow): StoredDocument {
  return {
    id: row.id as DocumentId,
    title: row.title,
    filePath: row.file_path,
    project: row.project as ProjectId,
    contentHash: row.content_hash,
    tags: JSON.parse(row.tags) as readonly string[],
    frontmatter: JSON.parse(row.frontmatter) as Readonly<
      Record<string, unknown>
    >,
    chunkCount: row.chunk_count,
    fileSize: row.file_size,
    indexedAt: new Date(row.indexed_at + 'Z'),
  }
}

function rowToIndexedChunk(row: ChunkRow): IndexedChunk {
  // CRITICAL: Proper BLOB -> Float32Array alignment
  // better-sqlite3 BLOBs may have non-aligned byte offsets
  // Copy through Uint8Array to ensure alignment
  const uint8 = new Uint8Array(row.vector)
  const embedding = new Float32Array(uint8.buffer)

  return {
    id: row.id as ChunkId,
    documentId: row.document_id as DocumentId,
    content: row.content,
    contentPlain: row.content_plain,
    headingPath: JSON.parse(row.heading_path) as readonly string[],
    index: row.chunk_index,
    hasCodeBlock: row.has_code_block === 1,
    embedding,
  }
}
