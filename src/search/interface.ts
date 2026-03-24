/**
 * Vector Memory Engine — Search Interface
 *
 * Re-exports the Search interface and related types from the central type system.
 * All type definitions live in src/types.ts — single source of truth.
 */

export type {
  Search,
  SearchResult,
  SearchOptions,
  UnitScore,
  ChunkId,
  ProjectId,
  IndexedChunk,
} from '../types.js'
