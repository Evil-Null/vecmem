/**
 * Vector Memory Engine — CLI `remove` Command
 *
 * `vector remove <path>` — remove a document and its chunks from the index.
 *
 * Finds the document by path, removes it (CASCADE handles chunks/embeddings/FTS),
 * and shows what was removed.
 */

import { resolve } from 'node:path'
import { SqliteStore } from '../store/sqlite.js'
import { createDocumentId, createProjectId } from '../types.js'
import {
  formatSuccess,
  formatError,
  formatWarning,
} from './format.js'
import type { VectorConfig } from '../config.js'

// ============================================================================
// remove Command Handler
// ============================================================================

/**
 * Execute the `vector remove` command.
 *
 * @param filePath - Path to the file to remove from index
 * @param config - Resolved VectorConfig
 */
export function runRemove(filePath: string, config: VectorConfig): void {
  if (!filePath || filePath.trim().length === 0) {
    process.stderr.write(formatError('File path is required.') + '\n')
    process.stderr.write('  Usage: vector remove <path>\n')
    process.exitCode = 2
    return
  }

  const resolvedPath = resolve(filePath)

  let store: SqliteStore
  try {
    store = new SqliteStore({
      storagePath: config.storagePath,
      databaseName: config.databaseName,
    })
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err)
    process.stderr.write(formatError(`Cannot open database: ${message}`) + '\n')
    process.exitCode = 3
    return
  }

  try {
    const projectId = createProjectId(config.project)

    // Try to find the document by resolved path
    let doc = store.getDocument(projectId, resolvedPath)

    // If not found by resolved path, try the literal path
    if (doc === null && filePath !== resolvedPath) {
      doc = store.getDocument(projectId, filePath)
    }

    if (doc === null) {
      process.stderr.write(formatWarning(`Not found in index: ${filePath}`) + '\n')
      process.stderr.write('  Run `vector status` to see indexed files.\n')
      process.exitCode = 4
      return
    }

    const chunkCount = doc.chunkCount
    const docId = createDocumentId(projectId, doc.filePath)
    store.removeDocument(docId)

    process.stdout.write(
      formatSuccess(`Removed: ${doc.filePath} (${chunkCount} chunk${chunkCount === 1 ? '' : 's'})`) + '\n',
    )
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err)
    process.stderr.write(formatError(message) + '\n')
    process.exitCode = 1
  } finally {
    store.close()
  }
}
