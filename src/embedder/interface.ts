/**
 * Vector Memory Engine — Embedder Interface
 *
 * Re-exports the Embedder interface from the central type system.
 * Other modules import from here, not directly from types.ts.
 * This provides a clean module boundary for the embedding subsystem.
 */

export type { Embedder } from '../types.js'
