/**
 * Vector Memory Engine — CLI `query` Command
 *
 * `vector query "search text"` — hybrid search with formatted results.
 *
 * Shows: result count, timing, box-drawn results with path/score/snippet.
 * Exit code 4 if no results found (distinct from errors).
 */

import { SearchOrchestrator } from '../pipeline/searcher.js'
import { SqliteStore } from '../store/sqlite.js'
import { LocalEmbedder } from '../embedder/local.js'
import { createLogger } from '../logger.js'
import {
  formatSearchResult,
  formatElapsed,
  formatError,
  formatWarning,
} from './format.js'
import type { VectorConfig } from '../config.js'
import { createProjectId } from '../types.js'

// ============================================================================
// query Command Handler
// ============================================================================

/**
 * Execute the `vector query` command.
 *
 * @param queryText - The search query string
 * @param options - Command options (topK, project, all)
 * @param config - Resolved VectorConfig
 * @param verbose - Whether to show debug logs
 */
export async function runQuery(
  queryText: string,
  options: { readonly topK?: number; readonly project?: string; readonly all?: boolean },
  config: VectorConfig,
  verbose: boolean,
): Promise<void> {
  const logger = createLogger(verbose)

  if (!queryText || queryText.trim().length === 0) {
    process.stderr.write(formatError('Query text is required.') + '\n')
    process.stderr.write('  Usage: vector query "your search text"\n')
    process.exitCode = 2
    return
  }

  const store = new SqliteStore({
    storagePath: config.storagePath,
    databaseName: config.databaseName,
  })

  const embedder = new LocalEmbedder({
    model: config.embeddingModel,
    cachePath: config.modelCachePath,
  })

  try {
    const search = new SearchOrchestrator(store.getDb(), embedder, logger, {
      defaultTopK: config.defaultTopK,
      rrfK: config.rrfK,
      minScore: config.minScore,
    })

    const startMs = performance.now()

    const topK = options.topK ?? config.defaultTopK
    const projectFilter = options.project !== undefined ? createProjectId(options.project) : undefined

    const results = await search.query(queryText, {
      topK,
      project: projectFilter,
    })

    const elapsedMs = Math.round(performance.now() - startMs)

    if (results.length === 0) {
      process.stdout.write(`  No results found (${formatElapsed(elapsedMs)})\n`)
      process.stdout.write('  Try a different query or run `vector index` to add more files.\n')
      process.exitCode = 4
      return
    }

    // Show result count and timing
    process.stdout.write(`  Found ${results.length} result${results.length === 1 ? '' : 's'} (${formatElapsed(elapsedMs)})\n`)

    // Determine how many to show (default: 5, --all shows all)
    const displayCount = options.all === true ? results.length : Math.min(5, results.length)

    for (let i = 0; i < displayCount; i++) {
      const result = results[i]!
      process.stdout.write(formatSearchResult(result) + '\n')
    }

    // Show "N more results" hint
    const remaining = results.length - displayCount
    if (remaining > 0) {
      process.stdout.write(`  ${remaining} more result${remaining === 1 ? '' : 's'}. Use --all to see all.\n`)
    }
  } catch (err: unknown) {
    // Check for degraded mode — ModelLoadError is handled inside SearchOrchestrator
    // Other errors bubble up here
    const message = err instanceof Error ? err.message : String(err)
    if (message.includes('model') || message.includes('Model')) {
      process.stderr.write(formatWarning('Vector search unavailable (embedding model not loaded)') + '\n')
      process.stderr.write('  Run `vector doctor` to diagnose.\n')
    } else {
      process.stderr.write(formatError(message) + '\n')
    }
    process.exitCode = 1
  } finally {
    store.close()
  }
}
