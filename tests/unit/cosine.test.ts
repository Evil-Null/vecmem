/**
 * Vector Memory Engine — Cosine Similarity Unit Tests
 *
 * Tests cover:
 * - Identical vectors return 1.0
 * - Orthogonal vectors return 0.0
 * - Opposite vectors return -1.0
 * - Result always in [-1, 1]
 * - Zero vector handling
 */

import { describe, test, expect } from 'vitest'
import { cosineSimilarity } from '../../src/search/vector.js'

describe('cosineSimilarity', () => {
  test('identical vectors return 1.0', () => {
    const v = new Float32Array([1, 0, 0])
    expect(cosineSimilarity(v, v)).toBeCloseTo(1.0)
  })

  test('orthogonal vectors return 0.0', () => {
    const a = new Float32Array([1, 0, 0])
    const b = new Float32Array([0, 1, 0])
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.0)
  })

  test('opposite vectors return -1.0', () => {
    const a = new Float32Array([1, 0, 0])
    const b = new Float32Array([-1, 0, 0])
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1.0)
  })

  test('result always in [-1, 1]', () => {
    const a = new Float32Array([0.5, 0.3, 0.8, 0.1])
    const b = new Float32Array([0.2, 0.9, 0.1, 0.7])
    const sim = cosineSimilarity(a, b)
    expect(sim).toBeGreaterThanOrEqual(-1)
    expect(sim).toBeLessThanOrEqual(1)
  })

  test('normalized vectors: identical direction returns 1.0', () => {
    const a = new Float32Array([3, 4, 0])
    const b = new Float32Array([6, 8, 0]) // same direction, different magnitude
    expect(cosineSimilarity(a, b)).toBeCloseTo(1.0)
  })

  test('zero vector returns 0.0', () => {
    const a = new Float32Array([1, 2, 3])
    const zero = new Float32Array([0, 0, 0])
    expect(cosineSimilarity(a, zero)).toBe(0)
  })

  test('both zero vectors return 0.0', () => {
    const zero = new Float32Array([0, 0, 0])
    expect(cosineSimilarity(zero, zero)).toBe(0)
  })

  test('high-dimensional vectors produce valid result', () => {
    const dims = 384
    const a = new Float32Array(dims)
    const b = new Float32Array(dims)
    for (let i = 0; i < dims; i++) {
      a[i] = Math.random() * 2 - 1
      b[i] = Math.random() * 2 - 1
    }
    const sim = cosineSimilarity(a, b)
    expect(sim).toBeGreaterThanOrEqual(-1)
    expect(sim).toBeLessThanOrEqual(1)
  })
})
