import { describe, test, expect, beforeAll } from 'vitest'
import { LocalEmbedder } from '../../src/embedder/local.js'
import { EmptyInputError } from '../../src/errors.js'
import { DEFAULT_CONFIG } from '../../src/config.js'

/**
 * LocalEmbedder unit tests.
 *
 * Model: Xenova/all-MiniLM-L6-v2 (384-dimensional, L2-normalized)
 * First run downloads ~23MB model — timeout set accordingly.
 */

describe('LocalEmbedder', () => {
  let embedder: LocalEmbedder

  beforeAll(() => {
    embedder = new LocalEmbedder({
      model: DEFAULT_CONFIG.embeddingModel,
      cachePath: DEFAULT_CONFIG.modelCachePath,
    })
  })

  // =========================================================================
  // dimensions property
  // =========================================================================

  test('dimensions property returns 384', () => {
    expect(embedder.dimensions).toBe(384)
  })

  // =========================================================================
  // embed() — single text
  // =========================================================================

  test('embed() returns Float32Array', async () => {
    const result = await embedder.embed('hello world')
    expect(result).toBeInstanceOf(Float32Array)
  }, 120_000)

  test('embed() returns correct dimensions (384)', async () => {
    const result = await embedder.embed('the quick brown fox jumps over the lazy dog')
    expect(result.length).toBe(384)
  }, 120_000)

  test('embed() produces L2-normalized vectors (unit length)', async () => {
    const result = await embedder.embed('vector normalization test')
    const magnitude = Math.sqrt(
      result.reduce((sum, val) => sum + val * val, 0),
    )
    expect(magnitude).toBeCloseTo(1.0, 4)
  }, 120_000)

  // =========================================================================
  // embedBatch() — multiple texts
  // =========================================================================

  test('embedBatch() returns array of Float32Array with correct length', async () => {
    const texts = ['first text', 'second text', 'third text']
    const results = await embedder.embedBatch(texts)

    expect(results).toHaveLength(3)
    for (const vec of results) {
      expect(vec).toBeInstanceOf(Float32Array)
      expect(vec.length).toBe(384)
    }
  }, 120_000)

  test('embedBatch() results match individual embed() results', async () => {
    const texts = ['alpha beta gamma', 'delta epsilon zeta']

    const batchResults = await embedder.embedBatch(texts)
    const individualResults = await Promise.all(
      texts.map((t) => embedder.embed(t)),
    )

    expect(batchResults).toHaveLength(individualResults.length)
    for (let i = 0; i < batchResults.length; i++) {
      const batch = batchResults[i]!
      const individual = individualResults[i]!
      // Exact Float32Array equality — same model, same input, same process
      expect(batch.length).toBe(individual.length)
      for (let j = 0; j < batch.length; j++) {
        expect(batch[j]).toBe(individual[j])
      }
    }
  }, 120_000)

  test('embedBatch() with empty array returns empty array', async () => {
    const results = await embedder.embedBatch([])
    expect(results).toEqual([])
  }, 120_000)

  // =========================================================================
  // Error handling — EmptyInputError
  // =========================================================================

  test('embed() throws EmptyInputError for empty string', async () => {
    await expect(embedder.embed('')).rejects.toThrow(EmptyInputError)
  }, 120_000)

  test('embed() throws EmptyInputError for whitespace-only string', async () => {
    await expect(embedder.embed('   \t\n  ')).rejects.toThrow(EmptyInputError)
  }, 120_000)

  // =========================================================================
  // Determinism — invariant 7
  // =========================================================================

  test('determinism: same text produces same embedding (exact Float32Array equality)', async () => {
    const text = 'determinism is a critical invariant of the embedding pipeline'

    const first = await embedder.embed(text)
    const second = await embedder.embed(text)

    expect(first).toBeInstanceOf(Float32Array)
    expect(second).toBeInstanceOf(Float32Array)
    expect(first.length).toBe(second.length)

    // Exact equality — not approximate. Same process, same model, same input.
    for (let i = 0; i < first.length; i++) {
      expect(first[i]).toBe(second[i])
    }
  }, 120_000)

  // =========================================================================
  // Semantic sanity — different inputs produce different vectors
  // =========================================================================

  test('different texts produce different embeddings', async () => {
    const vec1 = await embedder.embed('the sun is shining brightly')
    const vec2 = await embedder.embed('quantum mechanics explains particle behavior')

    // At least some elements must differ for semantically different texts
    let diffCount = 0
    for (let i = 0; i < vec1.length; i++) {
      if (vec1[i] !== vec2[i]) diffCount++
    }
    expect(diffCount).toBeGreaterThan(0)
  }, 120_000)
})
