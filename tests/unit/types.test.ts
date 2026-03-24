import { describe, test, expect } from 'vitest'
import {
  createDocumentId, createChunkId, createProjectId, unitScore,
  type DocumentId, type ChunkId, type ProjectId, type UnitScore,
} from '../../src/types.js'

describe('Branded types', () => {
  test('createDocumentId returns DocumentId', () => {
    const id = createDocumentId('test-project', '/path/to/file.md')
    expect(typeof id).toBe('string')
    expect(id.length).toBe(16)
  })

  test('same input produces same DocumentId', () => {
    const id1 = createDocumentId('proj', '/a.md')
    const id2 = createDocumentId('proj', '/a.md')
    expect(id1).toBe(id2)
  })

  test('different input produces different DocumentId', () => {
    const id1 = createDocumentId('proj', '/a.md')
    const id2 = createDocumentId('proj', '/b.md')
    expect(id1).not.toBe(id2)
  })

  test('createChunkId is deterministic', () => {
    const docId = createDocumentId('proj', '/a.md')
    const c1 = createChunkId(docId, 0)
    const c2 = createChunkId(docId, 0)
    expect(c1).toBe(c2)
  })

  test('createChunkId returns 16-char hex', () => {
    const docId = createDocumentId('proj', '/a.md')
    const chunkId = createChunkId(docId, 0)
    expect(typeof chunkId).toBe('string')
    expect(chunkId.length).toBe(16)
    expect(chunkId).toMatch(/^[0-9a-f]{16}$/)
  })

  test('different chunk indices produce different ChunkIds', () => {
    const docId = createDocumentId('proj', '/a.md')
    const c1 = createChunkId(docId, 0)
    const c2 = createChunkId(docId, 1)
    expect(c1).not.toBe(c2)
  })

  test('createProjectId returns branded string', () => {
    const pid = createProjectId('my-project')
    expect(pid).toBe('my-project')
  })

  test('unitScore accepts valid scores', () => {
    expect(unitScore(0)).toBe(0)
    expect(unitScore(0.5)).toBe(0.5)
    expect(unitScore(1)).toBe(1)
  })

  test('unitScore rejects invalid scores', () => {
    expect(() => unitScore(-0.1)).toThrow()
    expect(() => unitScore(1.1)).toThrow()
  })

  test('unitScore rejects NaN', () => {
    expect(() => unitScore(NaN)).toThrow()
  })

  test('unitScore boundary values', () => {
    expect(unitScore(0)).toBe(0)
    expect(unitScore(1)).toBe(1)
    expect(unitScore(Number.MIN_VALUE)).toBe(Number.MIN_VALUE)
  })

  test('DocumentId format is hex string', () => {
    const id = createDocumentId('proj', '/file.md')
    expect(id).toMatch(/^[0-9a-f]{16}$/)
  })
})
