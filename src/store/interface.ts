/**
 * Vector Memory Engine — Store Interface
 *
 * Re-exports the Store interface and related types from the central type system.
 * All type definitions live in src/types.ts — single source of truth.
 */

export type {
  Store,
  DocumentMeta,
  StoredDocument,
  RawChunk,
  IndexedChunk,
  ChunkWithEmbedding,
  DocumentId,
  ChunkId,
  ProjectId,
} from '../types.js'
