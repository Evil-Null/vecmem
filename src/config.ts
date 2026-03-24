/**
 * Vector Memory Engine — Configuration
 *
 * Zero-config defaults with layered overrides:
 * 1. Built-in defaults
 * 2. ~/.vector/config.json (global)
 * 3. .vector/config.json (project-local)
 * 4. CLI flags
 *
 * Validated at startup via Zod schema.
 */

import { z } from 'zod'
import { join, basename } from 'node:path'
import { homedir } from 'node:os'
import { readFileSync, existsSync } from 'node:fs'

// ============================================================================
// Zod Schema — single source of truth for config shape and validation
// ============================================================================

export const configSchema = z.object({
  // Storage
  storagePath: z.string().min(1),
  databaseName: z.string().min(1),

  // Embedding
  embeddingProvider: z.enum(['local']),
  embeddingModel: z.string().min(1),
  embeddingDimensions: z.number().int().positive(),
  modelCachePath: z.string().min(1),

  // Chunking
  maxChunkTokens: z.number().int().positive(),
  minChunkTokens: z.number().int().nonnegative(),
  chunkOverlapTokens: z.number().int().nonnegative(),
  headingSplitDepth: z.number().int().positive(),

  // Retrieval
  defaultTopK: z.number().int().positive(),
  rrfK: z.number().int().positive(),
  minScore: z.number().nonnegative(),

  // Project
  project: z.string().min(1),
})

export type VectorConfig = z.infer<typeof configSchema>

// ============================================================================
// Defaults — from spec section 13
// ============================================================================

const defaultStoragePath = join(homedir(), '.vector')

export const DEFAULT_CONFIG: VectorConfig = {
  storagePath: defaultStoragePath,
  databaseName: 'vector.db',
  embeddingProvider: 'local',
  embeddingModel: 'Xenova/all-MiniLM-L6-v2',
  embeddingDimensions: 384,
  modelCachePath: join(defaultStoragePath, 'models'),
  maxChunkTokens: 400,
  minChunkTokens: 50,
  chunkOverlapTokens: 40,
  headingSplitDepth: 2,
  defaultTopK: 10,
  rrfK: 60,
  minScore: 0.01,
  project: basename(process.cwd()),
}

// ============================================================================
// Config File Loading
// ============================================================================

/** Read and parse a JSON config file. Returns empty object if file doesn't exist. */
function readConfigFile(filePath: string): Record<string, unknown> {
  if (!existsSync(filePath)) {
    return {}
  }
  const raw = readFileSync(filePath, 'utf-8')
  return JSON.parse(raw) as Record<string, unknown>
}

// ============================================================================
// Config Resolution
// ============================================================================

/**
 * Resolve configuration by merging layers:
 * defaults -> global config -> project config -> CLI overrides
 *
 * @param overrides - CLI flag overrides (highest priority)
 * @param globalConfigPath - Path to global config file (~/.vector/config.json)
 * @param projectConfigPath - Path to project config file (.vector/config.json)
 * @returns Fully resolved and validated VectorConfig
 * @throws ZodError if the merged config is invalid
 */
export function resolveConfig(
  overrides: Partial<VectorConfig>,
  globalConfigPath?: string,
  projectConfigPath?: string,
): VectorConfig {
  const globalPath = globalConfigPath ?? join(homedir(), '.vector', 'config.json')
  const projectPath = projectConfigPath ?? join(process.cwd(), '.vector', 'config.json')

  const globalConfig = readConfigFile(globalPath)
  const projectConfig = readConfigFile(projectPath)

  const merged = {
    ...DEFAULT_CONFIG,
    ...globalConfig,
    ...projectConfig,
    ...overrides,
  }

  return configSchema.parse(merged)
}

/**
 * Load the full config for the current environment.
 * Convenience wrapper around resolveConfig with default file paths.
 *
 * @param overrides - CLI flag overrides
 * @returns Fully resolved and validated VectorConfig
 */
export function loadConfig(overrides: Partial<VectorConfig> = {}): VectorConfig {
  return resolveConfig(overrides)
}
