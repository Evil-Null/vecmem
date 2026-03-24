/**
 * Vector Memory Engine — Embedder Property Tests
 *
 * Property-based tests using fast-check to prove embedding determinism
 * holds for arbitrary text inputs.
 *
 * Uses the REAL LocalEmbedder model (all-MiniLM-L6-v2) — not a fake.
 * numRuns is set to 10 because each run invokes the ONNX model, which
 * takes ~50-200ms per call. 10 runs with random text inputs is sufficient
 * to verify the determinism property without making the test suite slow.
 *
 * A comprehensive determinism unit test already exists in
 * tests/unit/embedder.test.ts — this property test adds randomized
 * input coverage to strengthen the invariant.
 *
 * Property tested:
 * 1. Same text -> same embedding (determinism, invariant 7)
 */

import { describe, test, beforeAll } from 'vitest'
import fc from 'fast-check'
import { LocalEmbedder } from '../../src/embedder/local.js'
import { DEFAULT_CONFIG } from '../../src/config.js'

describe('Embedder Properties', () => {
  let embedder: LocalEmbedder

  beforeAll(() => {
    embedder = new LocalEmbedder({
      model: DEFAULT_CONFIG.embeddingModel,
      cachePath: DEFAULT_CONFIG.modelCachePath,
    })
  })

  // Arbitrary for generating non-empty text inputs
  const textArb = fc.stringOf(
    fc.constantFrom(
      ...'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789.,!? '.split(''),
    ),
    { minLength: 5, maxLength: 200 },
  ).map(s => s.trim()).filter(s => s.length >= 3)

  test('Property 1: Determinism — same text produces identical embedding (exact Float32Array equality)', async () => {
    await fc.assert(
      fc.asyncProperty(textArb, async (text) => {
        const first = await embedder.embed(text)
        const second = await embedder.embed(text)

        if (first.length !== second.length) return false

        // Exact equality — same process, same model, same input
        for (let i = 0; i < first.length; i++) {
          if (first[i] !== second[i]) return false
        }

        return true
      }),
      // 10 runs with real model — each invocation is ~50-200ms.
      // This provides randomized coverage without making CI slow.
      { numRuns: 10 },
    )
  }, 120_000) // 2 minute timeout for model download on first run
})
