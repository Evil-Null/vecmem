/**
 * Vector Memory Engine — CLI `index` Command
 *
 * `vector index [path]` — index .md files with progress bar.
 *
 * Calls indexDirectory() from the pipeline with a progress callback,
 * shows a progress bar during indexing, and prints a summary with
 * files, chunks, elapsed time, throughput, and database size.
 */

import { resolve } from 'node:path'
import { statSync } from 'node:fs'
import { indexDirectory } from '../pipeline/indexer.js'
import { SqliteStore } from '../store/sqlite.js'
import { MarkdownParser } from '../parser/markdown.js'
import { LocalEmbedder } from '../embedder/local.js'
import { createLogger } from '../logger.js'
import {
  formatProgress,
  formatIndexSummary,
  formatError,
} from './format.js'
import type { VectorConfig } from '../config.js'

// ============================================================================
// index Command Handler
// ============================================================================

/**
 * Execute the `vector index` command.
 *
 * @param pathArg - Optional path to index (default: cwd)
 * @param config - Resolved VectorConfig
 * @param verbose - Whether to show debug logs
 */
export async function runIndex(
  pathArg: string | undefined,
  config: VectorConfig,
  verbose: boolean,
): Promise<void> {
  const targetPath = resolve(pathArg ?? process.cwd())
  const logger = createLogger(verbose)

  // Validate target path exists
  try {
    const stat = statSync(targetPath)
    if (!stat.isDirectory()) {
      process.stderr.write(formatError(`Not a directory: ${targetPath}`) + '\n')
      process.exitCode = 2
      return
    }
  } catch {
    process.stderr.write(formatError(`Path not found: ${targetPath}`) + '\n')
    process.exitCode = 2
    return
  }

  // Create app components
  const store = new SqliteStore({
    storagePath: config.storagePath,
    databaseName: config.databaseName,
  })

  const parser = new MarkdownParser({
    maxChunkTokens: config.maxChunkTokens,
    minChunkTokens: config.minChunkTokens,
    chunkOverlapTokens: config.chunkOverlapTokens,
    headingSplitDepth: config.headingSplitDepth,
    project: config.project,
  })

  const embedder = new LocalEmbedder({
    model: config.embeddingModel,
    cachePath: config.modelCachePath,
  })

  const startMs = performance.now()
  let totalChunks = 0

  try {
    // Track chunk counts by subscribing to store saves
    // We do this by wrapping the progress callback
    const result = await indexDirectory(
      targetPath,
      parser,
      embedder,
      store,
      logger,
      (current, total) => {
        // Clear line and rewrite progress bar
        if (process.stdout.isTTY) {
          process.stdout.write('\r\x1b[K')
        }
        process.stdout.write(formatProgress(current, total))
        if (!process.stdout.isTTY) {
          process.stdout.write('\n')
        }
      },
    )

    // Clear progress bar line
    if (process.stdout.isTTY) {
      process.stdout.write('\r\x1b[K')
    }

    const elapsedMs = Math.round(performance.now() - startMs)

    // Count total chunks from indexed documents
    const docs = store.listDocuments()
    for (const doc of docs) {
      totalChunks += doc.chunkCount
    }

    // Get database file size
    let dbSizeBytes = 0
    try {
      const dbPath = store.getDbPath()
      const dbStat = statSync(dbPath)
      dbSizeBytes = dbStat.size
    } catch {
      // Cannot stat db — not critical
    }

    // Show summary
    const indexed = result.indexed
    const skipped = result.skipped
    const failed = result.failed
    const total = indexed + skipped + failed

    process.stdout.write(
      formatIndexSummary(
        total,
        totalChunks,
        elapsedMs,
        store.getDbPath(),
        dbSizeBytes,
      ) + '\n',
    )

    if (skipped > 0) {
      process.stdout.write(`  Skipped: ${skipped} unchanged file${skipped === 1 ? '' : 's'}\n`)
    }
    if (failed > 0) {
      process.stderr.write(formatError(`Failed: ${failed} file${failed === 1 ? '' : 's'}. Run with --verbose for details.`) + '\n')
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err)
    process.stderr.write(formatError(message) + '\n')
    process.exitCode = 1
  } finally {
    store.close()
  }
}
