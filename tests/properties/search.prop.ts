/**
 * Vector Memory Engine — Search Property Tests
 *
 * Property-based tests using fast-check to prove search properties
 * hold for all valid inputs.
 *
 * Properties tested:
 * 1. Results always sorted descending by score
 * 2. All scores are valid UnitScore [0, 1]
 * 3. RRF raw score is always positive for any ranks and k > 0
 * 4. Cosine similarity bounded in [-1, 1] for any two vectors of same length
 */

import { describe, test, expect } from 'vitest'
import fc from 'fast-check'
import { cosineSimilarity } from '../../src/search/vector.js'
import { rrfFuse } from '../../src/search/fusion.js'

// ============================================================================
// Arbitraries
// ============================================================================

/** Generate a non-zero Float32Array of given length */
const float32Array = (length: number) =>
  fc.array(fc.float({ min: -10, max: 10, noNaN: true, noDefaultInfinity: true }), {
    minLength: length,
    maxLength: length,
  }).map(arr => new Float32Array(arr))

/** Generate a ranked result list with unique chunkIds and 1-based ranks */
const rankedResults = (maxLen: number) =>
  fc.integer({ min: 0, max: maxLen }).chain(len =>
    fc.array(
      fc.string({ minLength: 1, maxLength: 16 }),
      { minLength: len, maxLength: len },
    ).map(ids => {
      // Deduplicate
      const unique = [...new Set(ids)]
      return unique.map((id, i) => ({ chunkId: id, rank: i + 1 }))
    }),
  )

// ============================================================================
// Property 1: Results always sorted descending by score
// ============================================================================

describe('Search Properties', () => {
  test('Property 1: RRF fusion results always sorted descending by score', () => {
    fc.assert(
      fc.property(
        rankedResults(20),
        rankedResults(20),
        fc.integer({ min: 1, max: 200 }),
        (bm25, vector, k) => {
          const results = rrfFuse(bm25, vector, k)

          for (let i = 1; i < results.length; i++) {
            if (results[i - 1]!.score < results[i]!.score) {
              return false
            }
          }
          return true
        },
      ),
      { numRuns: 10_000 },
    )
  })

  // ==========================================================================
  // Property 2: All scores are valid UnitScore [0, 1]
  // ==========================================================================

  test('Property 2: All fused scores are in [0, 1]', () => {
    fc.assert(
      fc.property(
        rankedResults(20),
        rankedResults(20),
        fc.integer({ min: 1, max: 200 }),
        (bm25, vector, k) => {
          const results = rrfFuse(bm25, vector, k)

          for (const r of results) {
            if (r.score < 0 || r.score > 1) {
              return false
            }
            if (Number.isNaN(r.score)) {
              return false
            }
          }
          return true
        },
      ),
      { numRuns: 10_000 },
    )
  })

  // ==========================================================================
  // Property 3: RRF raw score is always positive
  // ==========================================================================

  test('Property 3: RRF raw score is always positive for any ranks and k > 0', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 10_000 }), // rank
        fc.integer({ min: 1, max: 10_000 }), // k
        (rank, k) => {
          const rrfComponent = 1 / (k + rank)
          return rrfComponent > 0
        },
      ),
      { numRuns: 10_000 },
    )
  })

  test('Property 3b: RRF fused raw scores are always positive', () => {
    fc.assert(
      fc.property(
        rankedResults(15),
        rankedResults(15),
        fc.integer({ min: 1, max: 200 }),
        (bm25, vector, k) => {
          const results = rrfFuse(bm25, vector, k)

          for (const r of results) {
            if (r.rrfRaw <= 0) return false
          }
          return true
        },
      ),
      { numRuns: 10_000 },
    )
  })

  // ==========================================================================
  // Property 4: Cosine similarity bounded in [-1, 1]
  // ==========================================================================

  test('Property 4: Cosine similarity bounded in [-1, 1] for any two vectors', () => {
    fc.assert(
      fc.property(
        float32Array(384),
        float32Array(384),
        (a, b) => {
          const sim = cosineSimilarity(a, b)

          // Zero vectors produce 0, which is in bounds
          if (Number.isNaN(sim)) return false
          return sim >= -1.0001 && sim <= 1.0001 // small epsilon for float32 rounding
        },
      ),
      { numRuns: 10_000 },
    )
  })

  test('Property 4b: Cosine similarity of vector with itself is approximately 1.0', () => {
    fc.assert(
      fc.property(
        float32Array(384),
        (v) => {
          const sim = cosineSimilarity(v, v)

          // Zero vectors produce 0
          const hasNonZero = v.some(x => x !== 0)
          if (!hasNonZero) {
            return sim === 0
          }

          // Non-zero vectors should produce ~1.0
          return Math.abs(sim - 1.0) < 0.001
        },
      ),
      { numRuns: 5_000 },
    )
  })
})
