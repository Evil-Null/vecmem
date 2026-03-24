/**
 * Vector Memory Engine — RRF Fusion + Highlight Generation
 *
 * Reciprocal Rank Fusion combines BM25 and vector search results
 * using rank positions (not raw scores, which are on incomparable scales).
 *
 * Formula: rrfRaw(chunk) = 1/(k + bm25_rank) + 1/(k + vector_rank)
 *
 * Scores are normalized to UnitScore [0, 1] via min-max normalization.
 * For chunks in only one list, only that component contributes.
 */

import type Database from 'better-sqlite3'
import { unitScore, type UnitScore } from '../types.js'

// ============================================================================
// Fusion Result — internal type before full SearchResult assembly
// ============================================================================

export interface FusionResult {
  readonly chunkId: string
  readonly score: UnitScore
  readonly bm25Rank: number
  readonly vectorRank: number
  readonly rrfRaw: number
}

// ============================================================================
// Ranked Input — what BM25 and Vector searches produce
// ============================================================================

interface RankedInput {
  readonly chunkId: string
  readonly rank: number
}

// ============================================================================
// RRF Fusion
// ============================================================================

/**
 * Fuse BM25 and vector search results using Reciprocal Rank Fusion.
 *
 * @param bm25Results - Ranked results from BM25 (1-based ranks)
 * @param vectorResults - Ranked results from vector search (1-based ranks)
 * @param k - RRF smoothing parameter (default: 60). Higher = less rank sensitivity.
 * @returns Fused results sorted descending by score, normalized to [0, 1]
 */
export function rrfFuse(
  bm25Results: ReadonlyArray<RankedInput>,
  vectorResults: ReadonlyArray<RankedInput>,
  k: number,
): FusionResult[] {
  // Collect all unique chunk IDs with their ranks
  const chunkMap = new Map<string, { bm25Rank: number; vectorRank: number }>()

  for (const r of bm25Results) {
    const existing = chunkMap.get(r.chunkId)
    if (existing !== undefined) {
      existing.bm25Rank = r.rank
    } else {
      chunkMap.set(r.chunkId, { bm25Rank: r.rank, vectorRank: 0 })
    }
  }

  for (const r of vectorResults) {
    const existing = chunkMap.get(r.chunkId)
    if (existing !== undefined) {
      existing.vectorRank = r.rank
    } else {
      chunkMap.set(r.chunkId, { bm25Rank: 0, vectorRank: r.rank })
    }
  }

  // Empty inputs -> empty results
  if (chunkMap.size === 0) {
    return []
  }

  // Compute RRF raw scores
  const rawResults: Array<{
    chunkId: string
    bm25Rank: number
    vectorRank: number
    rrfRaw: number
  }> = []

  for (const [chunkId, ranks] of chunkMap) {
    let rrfRaw = 0

    if (ranks.bm25Rank > 0) {
      rrfRaw += 1 / (k + ranks.bm25Rank)
    }

    if (ranks.vectorRank > 0) {
      rrfRaw += 1 / (k + ranks.vectorRank)
    }

    rawResults.push({
      chunkId,
      bm25Rank: ranks.bm25Rank,
      vectorRank: ranks.vectorRank,
      rrfRaw,
    })
  }

  // Sort by rrfRaw descending
  rawResults.sort((a, b) => b.rrfRaw - a.rrfRaw)

  // Min-max normalization to [0, 1]
  return normalizeToUnitScore(rawResults)
}

/**
 * Normalize raw RRF scores to UnitScore [0, 1] via min-max normalization.
 *
 * Single result gets score 1.0.
 * All identical scores get score 1.0.
 */
function normalizeToUnitScore(
  results: ReadonlyArray<{
    readonly chunkId: string
    readonly bm25Rank: number
    readonly vectorRank: number
    readonly rrfRaw: number
  }>,
): FusionResult[] {
  if (results.length === 0) {
    return []
  }

  // Find min and max
  let min = Infinity
  let max = -Infinity

  for (const r of results) {
    if (r.rrfRaw < min) min = r.rrfRaw
    if (r.rrfRaw > max) max = r.rrfRaw
  }

  const range = max - min

  return results.map(r => ({
    chunkId: r.chunkId,
    score: range === 0
      ? unitScore(1.0)  // single result or all identical -> 1.0
      : unitScore((r.rrfRaw - min) / range),
    bm25Rank: r.bm25Rank,
    vectorRank: r.vectorRank,
    rrfRaw: r.rrfRaw,
  }))
}

// ============================================================================
// Highlight Generation
// ============================================================================

/**
 * Generate a text highlight/snippet for a search result.
 *
 * For BM25-matchable content: uses FTS5 snippet() function.
 * For vector-only matches or when FTS5 snippet fails: first 150 chars of content_plain.
 *
 * @param db - Database handle (better-sqlite3)
 * @param chunkId - The chunk to generate a highlight for
 * @param query - The original search query
 * @returns Highlighted text snippet, or empty string if chunk not found
 */
export function generateHighlight(
  db: Database.Database,
  chunkId: string,
  query: string,
): string {
  // Try FTS5 snippet first
  const sanitized = query.replace(/[()]/g, ' ').trim()

  if (sanitized.length > 0) {
    try {
      const ftsRow = db.prepare(`
        SELECT snippet(chunks_fts, 2, '', '', '...', 32) as snip
        FROM chunks_fts
        WHERE chunk_id = ?
          AND chunks_fts MATCH ?
      `).get(chunkId, sanitized) as { snip: string } | undefined

      if (ftsRow !== undefined && ftsRow.snip.length > 0) {
        return ftsRow.snip
      }
    } catch {
      // FTS5 query failed — fall through to plain text fallback
    }
  }

  // Fallback: first 150 chars of content_plain
  const chunkRow = db.prepare(
    'SELECT content_plain FROM chunks WHERE id = ?',
  ).get(chunkId) as { content_plain: string } | undefined

  if (chunkRow === undefined) {
    return ''
  }

  const plain = chunkRow.content_plain
  if (plain.length <= 150) {
    return plain
  }

  return plain.slice(0, 150) + '...'
}
