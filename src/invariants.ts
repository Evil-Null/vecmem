/**
 * Vector Memory Engine — System Invariants
 *
 * 5 runtime invariant checks that verify database consistency:
 * 1. everyChunkHasEmbedding — no chunk without an embedding row
 * 2. chunkCountsMatch — document.chunk_count == actual COUNT(chunks)
 * 3. uniformDimensions — exactly one DISTINCT dimensions value in embeddings table
 * 4. ftsInSync — chunks table IDs == chunks_fts table IDs
 * 5. noOrphanChunks — no chunks pointing to deleted documents
 *
 * Dev/test mode: checked after every Store.save() and Store.removeDocument()
 * Production: disabled for performance.
 * `vector doctor`: runs all 5 on demand regardless of mode.
 *
 * If an invariant is violated, it is a BUG — throw InvariantViolation immediately.
 */

import type Database from 'better-sqlite3'
import { InvariantViolation } from './errors.js'

// ============================================================================
// Individual Invariant Checks
// ============================================================================

/**
 * Invariant 1: Every chunk has a corresponding embedding row.
 * A chunk without an embedding is a corrupted state — the type system
 * (IndexedChunk requires embedding) prevents this at compile time,
 * but data corruption or raw SQL could bypass it.
 */
export function everyChunkHasEmbedding(db: Database.Database): void {
  const row = db.prepare(`
    SELECT COUNT(*) as orphan_count
    FROM chunks c
    LEFT JOIN embeddings e ON e.chunk_id = c.id
    WHERE e.chunk_id IS NULL
  `).get() as { orphan_count: number }

  if (row.orphan_count > 0) {
    throw new InvariantViolation(
      `everyChunkHasEmbedding: ${row.orphan_count} chunk(s) found without embeddings`,
    )
  }
}

/**
 * Invariant 2: document.chunk_count matches the actual number of chunks.
 * A mismatch means a partial write or data corruption occurred.
 */
export function chunkCountsMatch(db: Database.Database): void {
  const rows = db.prepare(`
    SELECT d.id, d.chunk_count AS expected, COUNT(c.id) AS actual
    FROM documents d
    LEFT JOIN chunks c ON c.document_id = d.id
    GROUP BY d.id
    HAVING d.chunk_count != COUNT(c.id)
  `).all() as Array<{ id: string; expected: number; actual: number }>

  if (rows.length > 0) {
    const details = rows
      .map(r => `doc ${r.id}: expected ${r.expected}, actual ${r.actual}`)
      .join('; ')
    throw new InvariantViolation(
      `chunkCountsMatch: ${rows.length} document(s) with mismatched chunk counts — ${details}`,
    )
  }
}

/**
 * Invariant 3: All embeddings have uniform dimensions.
 * Mixed dimensions would break cosine similarity computations.
 * An empty embeddings table is valid (no data yet).
 */
export function uniformDimensions(db: Database.Database): void {
  const row = db.prepare(`
    SELECT COUNT(DISTINCT dimensions) AS dim_count
    FROM embeddings
  `).get() as { dim_count: number }

  if (row.dim_count > 1) {
    const dims = db.prepare(`
      SELECT DISTINCT dimensions FROM embeddings ORDER BY dimensions
    `).all() as Array<{ dimensions: number }>
    const dimValues = dims.map(d => d.dimensions).join(', ')
    throw new InvariantViolation(
      `uniformDimensions: found ${row.dim_count} distinct dimension values: [${dimValues}]`,
    )
  }
}

/**
 * Invariant 4: FTS5 index is in sync with chunks table.
 * Every chunk must have a corresponding FTS entry and vice versa.
 * Missing FTS entries means search won't find valid chunks.
 * Extra FTS entries means search returns ghost results.
 */
export function ftsInSync(db: Database.Database): void {
  // Chunks without FTS entries
  const missingFts = db.prepare(`
    SELECT COUNT(*) as cnt
    FROM chunks c
    WHERE c.id NOT IN (SELECT chunk_id FROM chunks_fts)
  `).get() as { cnt: number }

  if (missingFts.cnt > 0) {
    throw new InvariantViolation(
      `ftsInSync: ${missingFts.cnt} chunk(s) missing from FTS index`,
    )
  }

  // FTS entries without chunks
  const extraFts = db.prepare(`
    SELECT COUNT(*) as cnt
    FROM chunks_fts
    WHERE chunk_id NOT IN (SELECT id FROM chunks)
  `).get() as { cnt: number }

  if (extraFts.cnt > 0) {
    throw new InvariantViolation(
      `ftsInSync: ${extraFts.cnt} FTS entry/entries without matching chunk(s)`,
    )
  }
}

/**
 * Invariant 5: No orphan chunks — every chunk references an existing document.
 * Orphan chunks can result from failed CASCADE deletes or raw SQL manipulation.
 */
export function noOrphanChunks(db: Database.Database): void {
  const row = db.prepare(`
    SELECT COUNT(*) as orphan_count
    FROM chunks c
    WHERE c.document_id NOT IN (SELECT id FROM documents)
  `).get() as { orphan_count: number }

  if (row.orphan_count > 0) {
    throw new InvariantViolation(
      `noOrphanChunks: ${row.orphan_count} chunk(s) reference non-existent documents`,
    )
  }
}

// ============================================================================
// Aggregate check
// ============================================================================

/**
 * Run all 5 system invariants. If any fails, throws InvariantViolation.
 * Used by `vector doctor` and by withInvariantCheck() in dev mode.
 */
export function checkAllInvariants(db: Database.Database): void {
  everyChunkHasEmbedding(db)
  chunkCountsMatch(db)
  uniformDimensions(db)
  ftsInSync(db)
  noOrphanChunks(db)
}

// ============================================================================
// Dev-mode wrapper
// ============================================================================

/**
 * Wrapper that runs invariant checks after an operation — but only in dev/test mode.
 * In production (NODE_ENV=production), invariant checks are skipped for performance.
 *
 * If the operation itself throws, the error propagates without running checks.
 */
export function withInvariantCheck<T>(db: Database.Database, operation: () => T): T {
  const result = operation()

  if (process.env['NODE_ENV'] !== 'production') {
    checkAllInvariants(db)
  }

  return result
}
