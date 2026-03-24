/**
 * Vector Memory Engine — CLI `status` Command
 *
 * `vector status` — show document count, chunk count, database size,
 * stale files, and last indexed time.
 *
 * Always answers: "What is the current state of my index?"
 */

import { statSync } from 'node:fs'
import { SqliteStore } from '../store/sqlite.js'
import {
  formatBytes,
  formatTimeAgo,
  formatWarning,
  formatError,
} from './format.js'
import type { VectorConfig } from '../config.js'

// ============================================================================
// status Command Handler
// ============================================================================

/**
 * Execute the `vector status` command.
 *
 * Shows: document count, chunk count, database size, last indexed time,
 * and stale file count.
 */
export function runStatus(config: VectorConfig): void {
  let store: SqliteStore

  try {
    store = new SqliteStore({
      storagePath: config.storagePath,
      databaseName: config.databaseName,
    })
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err)
    process.stderr.write(formatError(`Cannot open database: ${message}`) + '\n')
    process.stderr.write('  Run `vector init` to create a new database.\n')
    process.exitCode = 3
    return
  }

  try {
    const docs = store.listDocuments()
    const docCount = docs.length
    let chunkCount = 0
    let lastIndexed: Date | null = null

    for (const doc of docs) {
      chunkCount += doc.chunkCount
      if (lastIndexed === null || doc.indexedAt > lastIndexed) {
        lastIndexed = doc.indexedAt
      }
    }

    // Database file size
    let dbSizeBytes = 0
    try {
      const dbPath = store.getDbPath()
      const dbStat = statSync(dbPath)
      dbSizeBytes = dbStat.size
    } catch {
      // Cannot stat — not critical
    }

    // Count stale files (files that have changed since last index)
    let staleCount = 0
    for (const doc of docs) {
      try {
        const fileStat = statSync(doc.filePath)
        if (fileStat.mtimeMs > doc.indexedAt.getTime()) {
          staleCount++
        }
      } catch {
        // File no longer exists — that's stale
        staleCount++
      }
    }

    // Output
    process.stdout.write(`  Documents: ${docCount} (${chunkCount} chunks)\n`)
    process.stdout.write(`  Database: ${formatBytes(dbSizeBytes)}\n`)

    if (lastIndexed !== null) {
      process.stdout.write(`  Last indexed: ${formatTimeAgo(lastIndexed)}\n`)
    } else {
      process.stdout.write('  Last indexed: never\n')
    }

    if (staleCount > 0) {
      process.stdout.write(formatWarning(`Stale files: ${staleCount}`) + '\n')
      process.stdout.write('  Run `vector index` to update.\n')
    } else {
      process.stdout.write(`  Stale files: 0\n`)
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err)
    process.stderr.write(formatError(message) + '\n')
    process.exitCode = 1
  } finally {
    store.close()
  }
}
