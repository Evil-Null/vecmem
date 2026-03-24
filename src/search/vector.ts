/**
 * Vector Memory Engine — Vector Search (Cosine Similarity)
 *
 * Brute-force vector search for v1 (<50K chunks, ~15ms).
 * Loads all embeddings from DB, computes cosine similarity with query,
 * returns top-K results.
 *
 * INTENTIONAL: brute-force is simpler, correct, and fast enough for v1.
 * If profiling shows it's a bottleneck at scale, add approximate NN later.
 */

import type Database from 'better-sqlite3'

// ============================================================================
// Vector Result — internal ranked result
// ============================================================================

export interface VectorResult {
  readonly chunkId: string
  readonly rank: number
  readonly similarity: number
}

// ============================================================================
// Cosine Similarity — pure function
// ============================================================================

/**
 * Compute cosine similarity between two Float32Array vectors.
 *
 * cosine(a, b) = (a . b) / (||a|| * ||b||)
 *
 * Returns:
 * - 1.0 for identical direction
 * - 0.0 for orthogonal vectors
 * - -1.0 for opposite direction
 * - 0.0 if either vector is zero (graceful handling)
 *
 * @param a - First vector
 * @param b - Second vector (must be same length)
 * @returns Similarity in [-1, 1]
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0
  let normA = 0
  let normB = 0

  for (let i = 0; i < a.length; i++) {
    const ai = a[i]!
    const bi = b[i]!
    dot += ai * bi
    normA += ai * ai
    normB += bi * bi
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB)

  // Graceful handling of zero vectors
  if (denominator === 0) {
    return 0
  }

  return dot / denominator
}

// ============================================================================
// VectorSearch Class
// ============================================================================

export class VectorSearch {
  private readonly db: Database.Database

  constructor(db: Database.Database) {
    this.db = db
  }

  /**
   * Search for chunks most similar to the query embedding.
   *
   * Brute-force: loads all embeddings, computes cosine similarity, returns top-K.
   * Intentionally simple for v1 — correct and fast enough (<50K chunks).
   *
   * @param queryEmbedding - Query vector (Float32Array from embedder)
   * @param options - Optional topK and project filter
   * @returns Ranked results with chunk IDs, 1-based ranks, and similarity scores
   */
  search(
    queryEmbedding: Float32Array,
    options?: { readonly topK?: number; readonly project?: string },
  ): VectorResult[] {
    const topK = options?.topK ?? 50
    const project = options?.project

    // Load all embeddings from DB
    let sql: string
    const params: string[] = []

    if (project !== undefined) {
      sql = `
        SELECT e.chunk_id, e.vector
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        WHERE c.project = ?
      `
      params.push(project)
    } else {
      sql = 'SELECT chunk_id, vector FROM embeddings'
    }

    const rows = this.db.prepare(sql).all(...params) as ReadonlyArray<{
      readonly chunk_id: string
      readonly vector: Buffer
    }>

    // Compute cosine similarity for each embedding
    const scored: Array<{ readonly chunkId: string; readonly similarity: number }> = []

    for (const row of rows) {
      // CRITICAL: Proper BLOB -> Float32Array alignment
      const uint8 = new Uint8Array(row.vector)
      const embedding = new Float32Array(uint8.buffer)

      const similarity = cosineSimilarity(queryEmbedding, embedding)
      scored.push({ chunkId: row.chunk_id, similarity })
    }

    // Sort by similarity descending
    scored.sort((a, b) => b.similarity - a.similarity)

    // Take top-K and assign 1-based ranks
    return scored.slice(0, topK).map((item, idx) => ({
      chunkId: item.chunkId,
      rank: idx + 1,
      similarity: item.similarity,
    }))
  }
}
