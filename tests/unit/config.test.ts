import { describe, test, expect, beforeEach, afterEach } from 'vitest'
import { resolveConfig, DEFAULT_CONFIG, configSchema } from '../../src/config.js'
import { writeFileSync, mkdirSync, rmSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

describe('Config defaults', () => {
  test('defaults are applied when no overrides given', () => {
    const config = resolveConfig({})
    expect(config.embeddingProvider).toBe('local')
    expect(config.embeddingModel).toBe('Xenova/all-MiniLM-L6-v2')
    expect(config.embeddingDimensions).toBe(384)
    expect(config.maxChunkTokens).toBe(400)
    expect(config.minChunkTokens).toBe(50)
    expect(config.chunkOverlapTokens).toBe(40)
    expect(config.headingSplitDepth).toBe(2)
    expect(config.defaultTopK).toBe(10)
    expect(config.rrfK).toBe(60)
    expect(config.minScore).toBe(0.01)
    expect(config.databaseName).toBe('vector.db')
  })

  test('DEFAULT_CONFIG contains all required fields', () => {
    expect(DEFAULT_CONFIG.storagePath).toBeDefined()
    expect(DEFAULT_CONFIG.databaseName).toBe('vector.db')
    expect(DEFAULT_CONFIG.embeddingProvider).toBe('local')
    expect(DEFAULT_CONFIG.embeddingModel).toBe('Xenova/all-MiniLM-L6-v2')
    expect(DEFAULT_CONFIG.embeddingDimensions).toBe(384)
    expect(DEFAULT_CONFIG.modelCachePath).toBeDefined()
    expect(DEFAULT_CONFIG.maxChunkTokens).toBe(400)
    expect(DEFAULT_CONFIG.minChunkTokens).toBe(50)
    expect(DEFAULT_CONFIG.chunkOverlapTokens).toBe(40)
    expect(DEFAULT_CONFIG.headingSplitDepth).toBe(2)
    expect(DEFAULT_CONFIG.defaultTopK).toBe(10)
    expect(DEFAULT_CONFIG.rrfK).toBe(60)
    expect(DEFAULT_CONFIG.minScore).toBe(0.01)
    expect(DEFAULT_CONFIG.project).toBeDefined()
  })
})

describe('Config overrides', () => {
  test('CLI overrides apply on top of defaults', () => {
    const config = resolveConfig({ defaultTopK: 20, rrfK: 80 })
    expect(config.defaultTopK).toBe(20)
    expect(config.rrfK).toBe(80)
    // Other defaults unchanged
    expect(config.embeddingDimensions).toBe(384)
  })

  test('partial overrides merge with defaults', () => {
    const config = resolveConfig({ maxChunkTokens: 500 })
    expect(config.maxChunkTokens).toBe(500)
    expect(config.minChunkTokens).toBe(50) // default preserved
  })
})

describe('Config validation', () => {
  test('invalid embeddingDimensions throws', () => {
    expect(() => resolveConfig({ embeddingDimensions: -1 })).toThrow()
  })

  test('invalid maxChunkTokens throws', () => {
    expect(() => resolveConfig({ maxChunkTokens: 0 })).toThrow()
  })

  test('invalid minChunkTokens throws', () => {
    expect(() => resolveConfig({ minChunkTokens: -5 })).toThrow()
  })

  test('invalid defaultTopK throws', () => {
    expect(() => resolveConfig({ defaultTopK: 0 })).toThrow()
  })

  test('invalid rrfK throws', () => {
    expect(() => resolveConfig({ rrfK: 0 })).toThrow()
  })

  test('invalid minScore throws', () => {
    expect(() => resolveConfig({ minScore: -1 })).toThrow()
  })

  test('configSchema validates correct config', () => {
    const result = configSchema.safeParse({
      storagePath: '/tmp/test',
      databaseName: 'test.db',
      embeddingProvider: 'local',
      embeddingModel: 'Xenova/all-MiniLM-L6-v2',
      embeddingDimensions: 384,
      modelCachePath: '/tmp/models',
      maxChunkTokens: 400,
      minChunkTokens: 50,
      chunkOverlapTokens: 40,
      headingSplitDepth: 2,
      defaultTopK: 10,
      rrfK: 60,
      minScore: 0.01,
      project: 'test',
    })
    expect(result.success).toBe(true)
  })

  test('configSchema rejects invalid embeddingProvider', () => {
    const result = configSchema.safeParse({
      embeddingProvider: 'openai',
    })
    expect(result.success).toBe(false)
  })
})

describe('Config resolution chain', () => {
  const tmpBase = join(tmpdir(), `vector-config-test-${Date.now()}`)
  const globalDir = join(tmpBase, '.vector')
  const projectDir = join(tmpBase, 'project', '.vector')

  beforeEach(() => {
    mkdirSync(globalDir, { recursive: true })
    mkdirSync(projectDir, { recursive: true })
  })

  afterEach(() => {
    rmSync(tmpBase, { recursive: true, force: true })
  })

  test('global config overrides defaults', () => {
    writeFileSync(
      join(globalDir, 'config.json'),
      JSON.stringify({ defaultTopK: 25 })
    )
    const config = resolveConfig(
      {},
      join(globalDir, 'config.json'),
      undefined
    )
    expect(config.defaultTopK).toBe(25)
  })

  test('project config overrides global', () => {
    writeFileSync(
      join(globalDir, 'config.json'),
      JSON.stringify({ defaultTopK: 25, rrfK: 80 })
    )
    writeFileSync(
      join(projectDir, 'config.json'),
      JSON.stringify({ defaultTopK: 30 })
    )
    const config = resolveConfig(
      {},
      join(globalDir, 'config.json'),
      join(projectDir, 'config.json')
    )
    expect(config.defaultTopK).toBe(30)
    // global still applies for rrfK
    expect(config.rrfK).toBe(80)
  })

  test('CLI overrides everything', () => {
    writeFileSync(
      join(globalDir, 'config.json'),
      JSON.stringify({ defaultTopK: 25 })
    )
    writeFileSync(
      join(projectDir, 'config.json'),
      JSON.stringify({ defaultTopK: 30 })
    )
    const config = resolveConfig(
      { defaultTopK: 5 },
      join(globalDir, 'config.json'),
      join(projectDir, 'config.json')
    )
    expect(config.defaultTopK).toBe(5)
  })

  test('missing config files are ignored gracefully', () => {
    const config = resolveConfig(
      {},
      '/nonexistent/path/config.json',
      '/also/nonexistent/config.json'
    )
    // Should fall back to defaults
    expect(config.defaultTopK).toBe(10)
  })
})

describe('Config file loading', () => {
  const tmpBase = join(tmpdir(), `vector-config-load-${Date.now()}`)
  const configDir = join(tmpBase, '.vector')

  beforeEach(() => {
    mkdirSync(configDir, { recursive: true })
  })

  afterEach(() => {
    rmSync(tmpBase, { recursive: true, force: true })
  })

  test('malformed JSON in config file throws', () => {
    writeFileSync(join(configDir, 'config.json'), '{ invalid json }')
    expect(() =>
      resolveConfig({}, join(configDir, 'config.json'), undefined)
    ).toThrow()
  })
})
