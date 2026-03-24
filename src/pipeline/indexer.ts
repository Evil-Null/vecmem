/**
 * Vector Memory Engine — Index Pipeline Orchestrator
 *
 * Orchestrates: parse → embed → store (one direction, no circular deps).
 *
 * indexFile():
 * 1. validateFilePath() — security check
 * 2. parser.parse() → DocumentMeta + RawChunk[]
 * 3. store.needsReindex() — skip if unchanged (idempotent)
 * 4. embedder.embedBatch() → Float32Array[]
 * 5. Pair chunks + embeddings → ChunkWithEmbedding[]
 * 6. store.save() — atomic transaction
 *
 * indexDirectory():
 * - Recursively discovers all .md files
 * - Calls indexFile() for each
 * - Reports progress via callback
 * - Catches errors per file (report and continue)
 */

import { readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { validateFilePath } from '../parser/markdown.js'
import type {
  Parser,
  Embedder,
  Store,
  Logger,
  IndexResult,
  ChunkWithEmbedding,
} from '../types.js'

// ============================================================================
// indexFile — single file pipeline
// ============================================================================

/**
 * Index a single markdown file through the full pipeline.
 *
 * @param filePath - Path to the .md file
 * @param parser - Markdown parser
 * @param embedder - Embedding model
 * @param store - Persistent store
 * @param logger - Structured logger
 * @param projectRoot - Root directory for security validation
 * @returns IndexResult — 'indexed' with chunk count, or 'skipped' with reason
 * @throws SecurityError if filePath escapes projectRoot
 * @throws FileNotFoundError if file does not exist
 */
export async function indexFile(
  filePath: string,
  parser: Parser,
  embedder: Embedder,
  store: Store,
  logger: Logger,
  projectRoot: string,
): Promise<IndexResult> {
  const startMs = performance.now()

  // 1. Security check — validates path and existence
  const validPath = validateFilePath(filePath, projectRoot)

  // 2. Parse → DocumentMeta + RawChunk[]
  const { document, chunks } = parser.parse(validPath)

  // 3. Check if re-indexing is needed (idempotent)
  if (!store.needsReindex(document.project, document.filePath, document.contentHash)) {
    const elapsedMs = Math.round(performance.now() - startMs)
    logger.debug('file.skipped', {
      path: validPath,
      reason: 'unchanged',
      elapsed_ms: elapsedMs,
    })
    return { status: 'skipped', reason: 'unchanged' }
  }

  // 4. Embed all chunks
  const texts = chunks.map(c => c.contentPlain)
  const embeddings = await embedder.embedBatch(texts)

  // 5. Pair chunks + embeddings → ChunkWithEmbedding[]
  const items: ChunkWithEmbedding[] = chunks.map((chunk, i) => ({
    chunk,
    embedding: embeddings[i]!,
  }))

  // 6. Atomic save
  store.save(document, items)

  const elapsedMs = Math.round(performance.now() - startMs)
  logger.info('file.indexed', {
    path: validPath,
    chunks: chunks.length,
    elapsed_ms: elapsedMs,
  })

  return { status: 'indexed', chunks: chunks.length }
}

// ============================================================================
// indexDirectory — batch pipeline
// ============================================================================

/**
 * Discover and index all .md files in a directory (recursively).
 *
 * @param dirPath - Root directory to scan
 * @param parser - Markdown parser
 * @param embedder - Embedding model
 * @param store - Persistent store
 * @param logger - Structured logger
 * @param onProgress - Optional progress callback (current, total)
 * @returns Summary: { indexed, skipped, failed }
 */
export async function indexDirectory(
  dirPath: string,
  parser: Parser,
  embedder: Embedder,
  store: Store,
  logger: Logger,
  onProgress?: (current: number, total: number) => void,
): Promise<{ indexed: number; skipped: number; failed: number }> {
  const startMs = performance.now()

  // Discover all .md files
  const files = discoverMarkdownFiles(dirPath)

  let indexed = 0
  let skipped = 0
  let failed = 0

  for (let i = 0; i < files.length; i++) {
    const file = files[i]!

    try {
      const result = await indexFile(file, parser, embedder, store, logger, dirPath)

      if (result.status === 'indexed') {
        indexed++
      } else {
        skipped++
      }
    } catch (err: unknown) {
      failed++
      const message = err instanceof Error ? err.message : String(err)
      logger.warn('file.failed', {
        path: file,
        error: message,
      })
    }

    if (onProgress !== undefined) {
      onProgress(i + 1, files.length)
    }
  }

  const elapsedMs = Math.round(performance.now() - startMs)
  logger.info('dir.indexed', {
    path: dirPath,
    total: files.length,
    indexed,
    skipped,
    failed,
    elapsed_ms: elapsedMs,
  })

  return { indexed, skipped, failed }
}

// ============================================================================
// File Discovery — recursive .md search
// ============================================================================

/**
 * Recursively discover all .md files in a directory.
 * Returns sorted array of absolute file paths.
 */
function discoverMarkdownFiles(dirPath: string): string[] {
  const results: string[] = []
  collectMarkdownFiles(dirPath, results)
  results.sort()
  return results
}

/** Directories to always skip during discovery */
const IGNORED_DIRS = new Set([
  'node_modules',
  '.git',
  '.hg',
  '.svn',
  'dist',
  'build',
  'out',
  '.next',
  '.nuxt',
  '.output',
  '.cache',
  '.vector',
  '__pycache__',
  '.claude',
])

/** Recursive helper for markdown file discovery */
function collectMarkdownFiles(dirPath: string, results: string[]): void {
  let entries: string[]
  try {
    entries = readdirSync(dirPath)
  } catch {
    // Directory not readable — skip silently
    return
  }

  for (const entry of entries) {
    // Skip hidden directories (starting with .) and known ignore patterns
    if (entry.startsWith('.') && entry !== '.') {
      continue
    }
    if (IGNORED_DIRS.has(entry)) {
      continue
    }

    const fullPath = join(dirPath, entry)

    let stat
    try {
      stat = statSync(fullPath)
    } catch {
      // File not accessible — skip
      continue
    }

    if (stat.isDirectory()) {
      collectMarkdownFiles(fullPath, results)
    } else if (stat.isFile() && entry.endsWith('.md')) {
      results.push(fullPath)
    }
  }
}
