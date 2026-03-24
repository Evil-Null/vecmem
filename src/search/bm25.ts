/**
 * Vector Memory Engine — BM25 Search via FTS5
 *
 * Wraps SQLite FTS5 with bm25() ranking function.
 * Queries the chunks_fts virtual table using MATCH.
 *
 * FTS5 query sanitization:
 * - Preserves * (prefix search) and "" (phrase matching)
 * - Escapes () (grouping/NEAR operators)
 * - Balances unmatched quotes
 */

import type Database from 'better-sqlite3'

// ============================================================================
// BM25 Result — internal ranked result
// ============================================================================

export interface Bm25Result {
  readonly chunkId: string
  readonly rank: number
}

// ============================================================================
// FTS5 Query Sanitization
// ============================================================================

/**
 * Sanitize user input for FTS5 MATCH queries.
 *
 * Preserves:
 * - `*` for prefix search (e.g., "auth*")
 * - `""` for phrase matching (e.g., "OAuth flow")
 *
 * Escapes/removes:
 * - `()` — used for grouping/NEAR in FTS5 syntax
 * - Unbalanced quotes — auto-balanced
 */
export function sanitizeFtsQuery(query: string): string {
  // Remove parentheses (FTS5 grouping/NEAR operators)
  let sanitized = query.replace(/[()]/g, ' ')

  // Balance unmatched quotes
  sanitized = balanceQuotes(sanitized)

  // Collapse whitespace and trim
  return sanitized.replace(/\s+/g, ' ').trim()
}

/**
 * Balance double quotes in a string.
 * If the number of `"` is odd, remove the last one.
 */
function balanceQuotes(input: string): string {
  const quoteCount = (input.match(/"/g) ?? []).length
  if (quoteCount % 2 !== 0) {
    // Remove the last unmatched quote
    const lastIdx = input.lastIndexOf('"')
    return input.slice(0, lastIdx) + input.slice(lastIdx + 1)
  }
  return input
}

// ============================================================================
// Bm25Search Class
// ============================================================================

export class Bm25Search {
  private readonly db: Database.Database

  constructor(db: Database.Database) {
    this.db = db
  }

  /**
   * Search the FTS5 index using BM25 ranking.
   *
   * @param query - User search query (will be sanitized)
   * @param options - Optional topK and project filter
   * @returns Ranked results with chunk IDs and 1-based ranks
   */
  search(
    query: string,
    options?: { readonly topK?: number; readonly project?: string },
  ): Bm25Result[] {
    const sanitized = sanitizeFtsQuery(query)
    if (sanitized.length === 0) {
      return []
    }

    const topK = options?.topK ?? 50
    const project = options?.project

    // Build the SQL query
    // bm25() returns negative values (lower = better match), so ORDER BY rank
    let sql: string
    const params: (string | number)[] = []

    if (project !== undefined) {
      sql = `
        SELECT f.chunk_id, rank
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.chunk_id
        WHERE chunks_fts MATCH ?
          AND c.project = ?
        ORDER BY rank
        LIMIT ?
      `
      params.push(sanitized, project, topK)
    } else {
      sql = `
        SELECT chunk_id, rank
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        LIMIT ?
      `
      params.push(sanitized, topK)
    }

    const rows = this.db.prepare(sql).all(...params) as ReadonlyArray<{
      readonly chunk_id: string
      readonly rank: number
    }>

    // Convert to 1-based ranking
    return rows.map((row, idx) => ({
      chunkId: row.chunk_id,
      rank: idx + 1,
    }))
  }
}
