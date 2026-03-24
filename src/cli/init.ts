/**
 * Vector Memory Engine — CLI `init` Command
 *
 * `vector init` — auto-discover .md files, create database directory,
 * initialize SQLite store, show summary.
 *
 * Zero-config: discovers files in cwd, creates DB at default path.
 * The user never needs a config file for basic usage.
 */

import { readdirSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { SqliteStore } from '../store/sqlite.js'
import { formatSuccess, formatHeader } from './format.js'
import type { VectorConfig } from '../config.js'

// ============================================================================
// init Command Handler
// ============================================================================

/**
 * Execute the `vector init` command.
 *
 * 1. Auto-discover .md files in cwd (recursively)
 * 2. Create database directory if needed
 * 3. Initialize SqliteStore (creates DB + schema)
 * 4. Show summary: file count, database path
 */
export function runInit(config: VectorConfig, version: string): void {
  const cwd = process.cwd()

  // Show header
  process.stdout.write(formatHeader(version) + '\n')

  // Discover .md files
  const mdFiles = discoverMarkdownFilesForInit(cwd)
  const relDir = relative(cwd, cwd) || '.'
  process.stdout.write(`  Found ${mdFiles.length} markdown file${mdFiles.length === 1 ? '' : 's'} in ${relDir}\n`)

  // Initialize store (creates DB directory + DB file + schema)
  const store = new SqliteStore({
    storagePath: config.storagePath,
    databaseName: config.databaseName,
  })

  const dbPath = store.getDbPath()
  const isNew = store.listDocuments().length === 0
  process.stdout.write(`  Database: ${dbPath} (${isNew ? 'new' : 'existing'})\n`)

  store.close()

  // Show next step
  if (mdFiles.length > 0) {
    process.stdout.write(formatSuccess('Ready. Run `vector index` to start.') + '\n')
  } else {
    process.stdout.write('  No markdown files found. Add some .md files and run `vector index`.\n')
  }
}

// ============================================================================
// File Discovery (init-specific — lightweight, no security validation needed)
// ============================================================================

/**
 * Recursively discover .md files for the init summary.
 * Unlike the indexer's discovery, this is purely informational.
 */
function discoverMarkdownFilesForInit(dirPath: string): string[] {
  const results: string[] = []
  collectMdFiles(dirPath, results)
  return results
}

function collectMdFiles(dirPath: string, results: string[]): void {
  let entries: string[]
  try {
    entries = readdirSync(dirPath)
  } catch {
    return
  }

  for (const entry of entries) {
    // Skip hidden directories and node_modules
    if (entry.startsWith('.') || entry === 'node_modules') {
      continue
    }

    const fullPath = join(dirPath, entry)
    let stat
    try {
      stat = statSync(fullPath)
    } catch {
      continue
    }

    if (stat.isDirectory()) {
      collectMdFiles(fullPath, results)
    } else if (stat.isFile() && entry.endsWith('.md')) {
      results.push(fullPath)
    }
  }
}
