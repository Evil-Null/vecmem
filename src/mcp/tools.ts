/**
 * Vector Memory Engine — MCP Tool Definitions
 *
 * 7 tools exposing Vector's core functionality to AI clients via MCP:
 * - search_memory: hybrid BM25 + vector search
 * - index_files: index markdown files into memory
 * - get_document: retrieve a specific document
 * - get_chunks: retrieve all chunks for a document
 * - list_documents: list all indexed documents
 * - remove_document: remove a document from the index
 * - status: aggregate index statistics
 *
 * Tool schemas defined in Zod — auto-converted to JSON Schema by the MCP SDK.
 * Each handler maps directly to pipeline/store functions.
 * Errors are structured with VectorError.code for machine-readable responses.
 *
 * Project isolation: all tools with optional `project` params default to
 * ctx.config.project (the server's configured project) when client omits it.
 */

import { z } from 'zod'
import { statSync } from 'node:fs'
import { indexFile } from '../pipeline/indexer.js'
import { SearchOrchestrator } from '../pipeline/searcher.js'
import { VectorError } from '../errors.js'
import {
  createDocumentId,
  createProjectId,
  type Store,
  type Parser,
  type Embedder,
  type Logger,
} from '../types.js'
import type Database from 'better-sqlite3'
import type { CallToolResult } from '@modelcontextprotocol/sdk/types.js'

// ============================================================================
// Tool Context — dependencies injected from server.ts
// ============================================================================

export interface ToolContext {
  readonly store: Store
  readonly parser: Parser
  readonly parserFactory: (project: string) => Parser
  readonly embedder: Embedder
  readonly logger: Logger
  readonly db: Database.Database
  readonly dbPath: string
  readonly projectRoot: string
  readonly config: {
    readonly defaultTopK: number
    readonly rrfK: number
    readonly minScore: number
    readonly project: string
  }
}

// ============================================================================
// Input Schemas — Zod definitions (auto-converted to JSON Schema by MCP SDK)
// ============================================================================

export const searchMemorySchema = {
  query: z.string().describe('Search query text'),
  topK: z.number().optional().describe('Maximum number of results to return'),
  project: z.string().optional().describe('Project to search (defaults to server project)'),
} as const

export const indexFilesSchema = {
  paths: z.array(z.string()).describe('Array of file paths to index'),
  project: z.string().optional().describe('Project to index into (defaults to server project)'),
} as const

export const getDocumentSchema = {
  project: z.string().describe('Project name'),
  filePath: z.string().describe('File path of the document'),
} as const

export const listDocumentsSchema = {
  project: z.string().optional().describe('Project to list (defaults to server project)'),
} as const

export const removeDocumentSchema = {
  project: z.string().describe('Project name'),
  filePath: z.string().describe('File path of the document to remove'),
} as const

export const getChunksSchema = {
  project: z.string().describe('Project name'),
  filePath: z.string().describe('File path of the document'),
} as const

export const statusSchema = {
  project: z.string().optional().describe('Project to report on (defaults to server project)'),
} as const

// ============================================================================
// Tool Handlers
// ============================================================================

/**
 * search_memory — Hybrid BM25 + vector search over indexed memory.
 */
export async function handleSearchMemory(
  input: { readonly query: string; readonly topK?: number; readonly project?: string },
  ctx: ToolContext,
): Promise<CallToolResult> {
  try {
    const searcher = new SearchOrchestrator(ctx.db, ctx.embedder, ctx.logger, {
      defaultTopK: ctx.config.defaultTopK,
      rrfK: ctx.config.rrfK,
      minScore: ctx.config.minScore,
    })

    const projectId = createProjectId(input.project ?? ctx.config.project)

    const results = await searcher.query(input.query, {
      topK: input.topK,
      project: projectId,
    })

    const serialized = results.map(r => ({
      documentTitle: r.documentTitle,
      documentPath: r.documentPath,
      score: r.score,
      scores: r.scores,
      content: r.chunk.contentPlain,
      headingPath: r.chunk.headingPath,
      highlight: r.highlight,
    }))

    return {
      content: [{ type: 'text' as const, text: JSON.stringify(serialized, null, 2) }],
    }
  } catch (err: unknown) {
    return toErrorResult(err)
  }
}

/**
 * index_files — Index markdown files into memory.
 */
export async function handleIndexFiles(
  input: { readonly paths: readonly string[]; readonly project?: string },
  ctx: ToolContext,
): Promise<CallToolResult> {
  try {
    // Use project-specific parser if client overrides project
    const activeParser = input.project !== undefined
      ? ctx.parserFactory(input.project)
      : ctx.parser

    const results: Array<{ path: string; status: string; chunks?: number; reason?: string }> = []

    for (const filePath of input.paths) {
      try {
        const result = await indexFile(
          filePath,
          activeParser,
          ctx.embedder,
          ctx.store,
          ctx.logger,
          ctx.projectRoot,
        )

        if (result.status === 'indexed') {
          results.push({ path: filePath, status: 'indexed', chunks: result.chunks })
        } else {
          results.push({ path: filePath, status: 'skipped', reason: result.reason })
        }
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err)
        results.push({ path: filePath, status: 'failed', reason: message })
      }
    }

    return {
      content: [{ type: 'text' as const, text: JSON.stringify(results, null, 2) }],
    }
  } catch (err: unknown) {
    return toErrorResult(err)
  }
}

/**
 * get_document — Retrieve a specific document with its full content (all chunks).
 */
export function handleGetDocument(
  input: { readonly project: string; readonly filePath: string },
  ctx: ToolContext,
): CallToolResult {
  try {
    const projectId = createProjectId(input.project)
    const doc = ctx.store.getDocument(projectId, input.filePath)

    if (doc === null) {
      return {
        content: [{ type: 'text' as const, text: JSON.stringify(null) }],
      }
    }

    // Include chunk content so AI clients can actually READ the document
    const chunks = ctx.store.getChunks(doc.id)
    const serialized = {
      id: doc.id,
      title: doc.title,
      filePath: doc.filePath,
      project: doc.project,
      contentHash: doc.contentHash,
      tags: doc.tags,
      frontmatter: doc.frontmatter,
      chunkCount: doc.chunkCount,
      fileSize: doc.fileSize,
      indexedAt: doc.indexedAt.toISOString(),
      chunks: chunks.map(c => ({
        id: c.id,
        content: c.content,
        contentPlain: c.contentPlain,
        headingPath: c.headingPath,
        index: c.index,
        hasCodeBlock: c.hasCodeBlock,
      })),
    }

    return {
      content: [{ type: 'text' as const, text: JSON.stringify(serialized, null, 2) }],
    }
  } catch (err: unknown) {
    return toErrorResult(err)
  }
}

/**
 * get_chunks — Retrieve all chunks (with full text) for a document.
 * This is the primary tool for reading document content from the index.
 */
export function handleGetChunks(
  input: { readonly project: string; readonly filePath: string },
  ctx: ToolContext,
): CallToolResult {
  try {
    const projectId = createProjectId(input.project)
    const doc = ctx.store.getDocument(projectId, input.filePath)

    if (doc === null) {
      return {
        content: [{ type: 'text' as const, text: JSON.stringify({ error: 'Document not found', filePath: input.filePath }) }],
        isError: true,
      }
    }

    const chunks = ctx.store.getChunks(doc.id)
    const serialized = chunks.map(c => ({
      id: c.id,
      content: c.content,
      contentPlain: c.contentPlain,
      headingPath: c.headingPath,
      index: c.index,
      hasCodeBlock: c.hasCodeBlock,
    }))

    return {
      content: [{ type: 'text' as const, text: JSON.stringify(serialized, null, 2) }],
    }
  } catch (err: unknown) {
    return toErrorResult(err)
  }
}

/**
 * list_documents — List all indexed documents, optionally filtered by project.
 */
export function handleListDocuments(
  input: { readonly project?: string },
  ctx: ToolContext,
): CallToolResult {
  try {
    const projectId = createProjectId(input.project ?? ctx.config.project)

    const docs = ctx.store.listDocuments(projectId)

    const serialized = docs.map(d => ({
      id: d.id,
      title: d.title,
      filePath: d.filePath,
      project: d.project,
      chunkCount: d.chunkCount,
      fileSize: d.fileSize,
      indexedAt: d.indexedAt.toISOString(),
    }))

    return {
      content: [{ type: 'text' as const, text: JSON.stringify(serialized, null, 2) }],
    }
  } catch (err: unknown) {
    return toErrorResult(err)
  }
}

/**
 * remove_document — Remove a document and all its chunks from the index.
 */
export function handleRemoveDocument(
  input: { readonly project: string; readonly filePath: string },
  ctx: ToolContext,
): CallToolResult {
  try {
    const projectId = createProjectId(input.project)
    const doc = ctx.store.getDocument(projectId, input.filePath)

    if (doc === null) {
      return {
        content: [{ type: 'text' as const, text: JSON.stringify({ removed: false }) }],
      }
    }

    const docId = createDocumentId(projectId, input.filePath)
    ctx.store.removeDocument(docId)

    return {
      content: [{ type: 'text' as const, text: JSON.stringify({ removed: true }) }],
    }
  } catch (err: unknown) {
    return toErrorResult(err)
  }
}

/**
 * status — Aggregate index statistics.
 *
 * Output fields match spec section 16: { documents, chunks, dbSize, staleFiles }
 */
export function handleStatus(
  input: { readonly project?: string },
  ctx: ToolContext,
): CallToolResult {
  try {
    const projectId = createProjectId(input.project ?? ctx.config.project)

    const docs = ctx.store.listDocuments(projectId)

    const documents = docs.length
    const chunks = docs.reduce((sum, d) => sum + d.chunkCount, 0)

    // Database file size in bytes
    let dbSize = 0
    try {
      const dbStat = statSync(ctx.dbPath)
      dbSize = dbStat.size
    } catch {
      // Database file not accessible — report 0
    }

    // Count files modified since last index (stale files)
    let staleFiles = 0
    for (const doc of docs) {
      try {
        const fileStat = statSync(doc.filePath)
        if (fileStat.mtimeMs > doc.indexedAt.getTime()) {
          staleFiles++
        }
      } catch {
        // File not accessible (deleted or moved) — counts as stale
        staleFiles++
      }
    }

    const stats = {
      documents,
      chunks,
      dbSize,
      staleFiles,
      projects: [...new Set(docs.map(d => d.project as string))],
    }

    return {
      content: [{ type: 'text' as const, text: JSON.stringify(stats, null, 2) }],
    }
  } catch (err: unknown) {
    return toErrorResult(err)
  }
}

// ============================================================================
// Error Handling — VectorError → MCP error response
// ============================================================================

function toErrorResult(err: unknown): CallToolResult {
  if (err instanceof VectorError) {
    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({
          error: err.message,
          code: err.code,
          recoverable: err.recoverable,
        }),
      }],
      isError: true,
    }
  }

  const message = err instanceof Error ? err.message : String(err)
  return {
    content: [{ type: 'text' as const, text: JSON.stringify({ error: message }) }],
    isError: true,
  }
}
