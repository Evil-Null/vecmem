/**
 * Vector Memory Engine — CLI `doctor` Command
 *
 * `vector doctor` — run all 5 system invariants, check model availability,
 * check FTS sync, show each check with pass/fail status.
 *
 * Doctor runs ALL checks regardless of NODE_ENV. It is the user's
 * diagnostic tool for when something seems wrong.
 */

import { statSync } from 'node:fs'
import Database from 'better-sqlite3'
import { join } from 'node:path'
import {
  everyChunkHasEmbedding,
  chunkCountsMatch,
  uniformDimensions,
  ftsInSync,
  noOrphanChunks,
} from '../invariants.js'
import { InvariantViolation } from '../errors.js'
import { SqliteStore } from '../store/sqlite.js'
import {
  formatDoctorCheck,
  formatBytes,
  formatHeader,
  formatError,
} from './format.js'
import type { VectorConfig } from '../config.js'

// ============================================================================
// doctor Command Handler
// ============================================================================

/**
 * Execute the `vector doctor` command.
 *
 * Runs all system health checks and displays results.
 */
export function runDoctor(config: VectorConfig, version: string): void {
  process.stdout.write('\n' + formatHeader(version) + ' \u2014 System Health Check\n\n')

  let store: SqliteStore

  try {
    store = new SqliteStore({
      storagePath: config.storagePath,
      databaseName: config.databaseName,
    })
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err)
    process.stdout.write(formatDoctorCheck('Database', false, `FAILED: ${message}`) + '\n')
    process.exitCode = 1
    return
  }

  const db = store.getDb()
  let allPassed = true

  try {
    // Check 1: Database health
    let dbSizeBytes = 0
    try {
      const dbPath = store.getDbPath()
      const dbStat = statSync(dbPath)
      dbSizeBytes = dbStat.size
    } catch {
      // Not critical
    }

    const walMode = db.pragma('journal_mode', { simple: true }) as string
    process.stdout.write(
      formatDoctorCheck(
        'Database',
        true,
        `OK (${formatBytes(dbSizeBytes)}, ${walMode.toUpperCase()} mode)`,
      ) + '\n',
    )

    // Check 2: FTS index sync
    try {
      ftsInSync(db)
      const ftsCount = (db.prepare('SELECT COUNT(*) as cnt FROM chunks_fts').get() as { cnt: number }).cnt
      process.stdout.write(
        formatDoctorCheck('FTS index', true, `in sync (${ftsCount} entries)`) + '\n',
      )
    } catch (err: unknown) {
      allPassed = false
      const msg = err instanceof InvariantViolation ? err.message : 'check failed'
      process.stdout.write(formatDoctorCheck('FTS index', false, msg) + '\n')
    }

    // Check 3: Embeddings
    try {
      everyChunkHasEmbedding(db)
      const embCount = (db.prepare('SELECT COUNT(*) as cnt FROM embeddings').get() as { cnt: number }).cnt
      const chunkCount = (db.prepare('SELECT COUNT(*) as cnt FROM chunks').get() as { cnt: number }).cnt
      process.stdout.write(
        formatDoctorCheck('Embeddings', true, `all present (${embCount}/${chunkCount})`) + '\n',
      )
    } catch (err: unknown) {
      allPassed = false
      const msg = err instanceof InvariantViolation ? err.message : 'check failed'
      process.stdout.write(formatDoctorCheck('Embeddings', false, msg) + '\n')
    }

    // Check 4: Model availability
    const modelPath = join(config.modelCachePath, 'Xenova')
    let modelAvailable = false
    try {
      const modelStat = statSync(modelPath)
      modelAvailable = modelStat.isDirectory()
    } catch {
      // Model directory doesn't exist
    }

    if (modelAvailable) {
      process.stdout.write(
        formatDoctorCheck('Model', true, `cached (${config.embeddingModel})`) + '\n',
      )
    } else {
      process.stdout.write(
        formatDoctorCheck('Model', true, `will download on first use (${config.embeddingModel})`) + '\n',
      )
    }

    // Check 5: All 5 invariants
    let invariantsPassing = 0
    const invariantChecks: Array<{ name: string; fn: (db: Database.Database) => void }> = [
      { name: 'everyChunkHasEmbedding', fn: everyChunkHasEmbedding },
      { name: 'chunkCountsMatch', fn: chunkCountsMatch },
      { name: 'uniformDimensions', fn: uniformDimensions },
      { name: 'ftsInSync', fn: ftsInSync },
      { name: 'noOrphanChunks', fn: noOrphanChunks },
    ]

    for (const check of invariantChecks) {
      try {
        check.fn(db)
        invariantsPassing++
      } catch {
        // Counted as failing
      }
    }

    if (invariantsPassing === 5) {
      process.stdout.write(
        formatDoctorCheck('Invariants', true, `all passing (${invariantsPassing}/5)`) + '\n',
      )
    } else {
      allPassed = false
      process.stdout.write(
        formatDoctorCheck('Invariants', false, `${invariantsPassing}/5 passing`) + '\n',
      )
    }

    // Check 6: Stale files
    const docs = store.listDocuments()
    let staleFiles: string[] = []
    for (const doc of docs) {
      try {
        const fileStat = statSync(doc.filePath)
        if (fileStat.mtimeMs > doc.indexedAt.getTime()) {
          staleFiles.push(doc.filePath)
        }
      } catch {
        staleFiles.push(doc.filePath)
      }
    }

    if (staleFiles.length > 0) {
      process.stdout.write(
        formatDoctorCheck('Stale files', false, `${staleFiles.length} file${staleFiles.length === 1 ? '' : 's'} changed since last index`) + '\n',
      )
      for (const f of staleFiles.slice(0, 5)) {
        process.stdout.write(`    \u2192 ${f}\n`)
      }
      if (staleFiles.length > 5) {
        process.stdout.write(`    ... and ${staleFiles.length - 5} more\n`)
      }
      process.stdout.write('\n  Run `vector index` to update stale files.\n')
    } else if (docs.length > 0) {
      process.stdout.write(
        formatDoctorCheck('Stale files', true, 'none') + '\n',
      )
    }

    process.stdout.write('\n')

    if (!allPassed) {
      process.exitCode = 1
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err)
    process.stderr.write(formatError(message) + '\n')
    process.exitCode = 1
  } finally {
    store.close()
  }
}
