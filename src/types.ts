/**
 * Vector Memory Engine — Type System
 *
 * "Invalid states are unrepresentable."
 *
 * Branded types prevent ID mixups at compile time.
 * Two-phase chunks enforce embedding presence at type level.
 * All interfaces are readonly — immutable data is data you can trust.
 */

import { createHash } from 'node:crypto'
import { InvariantViolation } from './errors.js'

// ============================================================================
// Branded Types
// ============================================================================

/** Content-addressed document identifier: SHA256(project:filePath)[0:16] */
export type DocumentId = string & { readonly __brand: 'DocumentId' }

/** Content-addressed chunk identifier: SHA256(documentId:index)[0:16] */
export type ChunkId = string & { readonly __brand: 'ChunkId' }

/** Project identifier — branded string, no hash */
export type ProjectId = string & { readonly __brand: 'ProjectId' }

/** Score bounded to [0, 1] — physically enforced at construction */
export type UnitScore = number & { readonly __brand: 'UnitScore' }

// ============================================================================
// Branded Type Factories
// ============================================================================

/**
 * Create a deterministic DocumentId from project name and file path.
 * SHA256 of `${project}:${filePath}`, truncated to 16 hex chars.
 */
export function createDocumentId(project: string, filePath: string): DocumentId {
  return createHash('sha256')
    .update(`${project}:${filePath}`)
    .digest('hex')
    .slice(0, 16) as DocumentId
}

/**
 * Create a deterministic ChunkId from document ID and chunk index.
 * SHA256 of `${docId}:${index}`, truncated to 16 hex chars.
 */
export function createChunkId(docId: DocumentId, index: number): ChunkId {
  return createHash('sha256')
    .update(`${docId}:${index}`)
    .digest('hex')
    .slice(0, 16) as ChunkId
}

/**
 * Create a ProjectId — branded string, no hashing.
 */
export function createProjectId(name: string): ProjectId {
  return name as ProjectId
}

/**
 * Validate and create a UnitScore bounded to [0, 1].
 * Throws InvariantViolation if out of range or NaN.
 */
export function unitScore(n: number): UnitScore {
  if (Number.isNaN(n) || n < 0 || n > 1) {
    throw new InvariantViolation(`Score ${n} outside [0,1]`)
  }
  return n as UnitScore
}

// ============================================================================
// Data Interfaces — ALL readonly
// ============================================================================

/** Metadata for a parsed markdown document */
export interface DocumentMeta {
  readonly title: string
  readonly filePath: string
  readonly project: ProjectId
  readonly contentHash: string
  readonly tags: readonly string[]
  readonly frontmatter: Readonly<Record<string, unknown>>
  readonly indexedAt: Date
  readonly fileSize: number
}

/**
 * Raw chunk from Parser — no embedding, no ID.
 * This is the output of the parsing phase.
 */
export interface RawChunk {
  readonly content: string
  readonly contentPlain: string
  readonly headingPath: readonly string[]
  readonly index: number
  readonly hasCodeBlock: boolean
}

/**
 * Indexed chunk in Store — has ID, has embedding (REQUIRED, not optional).
 * "Chunk without embedding in store" is impossible at type level.
 */
export interface IndexedChunk {
  readonly id: ChunkId
  readonly documentId: DocumentId
  readonly content: string
  readonly contentPlain: string
  readonly headingPath: readonly string[]
  readonly index: number
  readonly hasCodeBlock: boolean
  readonly embedding: Float32Array
}

/**
 * Explicit pairing of chunk and its embedding.
 * Prevents implicit array index coupling (chunks[i] + embeddings[i]).
 */
export interface ChunkWithEmbedding {
  readonly chunk: RawChunk
  readonly embedding: Float32Array
}

/** Document as stored in the database, with computed fields */
export interface StoredDocument extends DocumentMeta {
  readonly id: DocumentId
  readonly chunkCount: number
}

/** Result of parsing a markdown file */
export interface ParseResult {
  readonly document: DocumentMeta
  readonly chunks: readonly RawChunk[]
}

/** Search result with scoring breakdown */
export interface SearchResult {
  readonly chunk: IndexedChunk
  readonly score: UnitScore
  readonly scores: {
    readonly bm25Rank: number
    readonly vectorRank: number
    readonly rrfRaw: number
  }
  readonly documentTitle: string
  readonly documentPath: string
  readonly highlight: string
}

/** Options for search queries */
export interface SearchOptions {
  readonly topK?: number
  readonly minScore?: number
  readonly project?: ProjectId
}

/** Result of indexing a file — discriminated union */
export type IndexResult =
  | { readonly status: 'indexed'; readonly chunks: number }
  | { readonly status: 'skipped'; readonly reason: string }

// ============================================================================
// Interface Contracts
// ============================================================================

/** Markdown parser: file path in, structured data out */
export interface Parser {
  parse(filePath: string): ParseResult
}

/** Embedding model: text in, Float32Array out */
export interface Embedder {
  embed(text: string): Promise<Float32Array>
  embedBatch(texts: string[]): Promise<Float32Array[]>
  readonly dimensions: number
}

/** Persistent storage for documents and chunks */
export interface Store {
  save(doc: DocumentMeta, items: ChunkWithEmbedding[]): void
  getDocument(project: ProjectId, filePath: string): StoredDocument | null
  getChunks(documentId: DocumentId): IndexedChunk[]
  removeDocument(documentId: DocumentId): void
  needsReindex(project: ProjectId, filePath: string, contentHash: string): boolean
  listDocuments(project?: ProjectId): StoredDocument[]
}

/** Search engine: query text in, ranked results out */
export interface Search {
  query(text: string, options?: SearchOptions): Promise<SearchResult[]>
}

/** Structured logger with event-based API */
export interface Logger {
  debug(event: string, data?: Record<string, unknown>): void
  info(event: string, data?: Record<string, unknown>): void
  warn(event: string, data?: Record<string, unknown>): void
  error(event: string, error: import('./errors.js').VectorError): void
}
