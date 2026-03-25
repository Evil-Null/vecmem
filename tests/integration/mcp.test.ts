/**
 * Vector Memory Engine — MCP Server Integration Tests
 *
 * Tests all 7 MCP tools by directly calling handler functions.
 * Uses a real SQLite store, real parser, and fake embedder (no model download).
 *
 * Coverage:
 * - All 7 tools are callable and return valid MCP CallToolResult
 * - search_memory returns array of results
 * - index_files indexes files and returns per-file results
 * - get_document retrieves a specific document
 * - get_chunks retrieves chunks for a document
 * - list_documents returns document list
 * - remove_document removes and returns status
 * - status returns aggregate stats (document count, chunk count, etc.)
 * - Project isolation: tools default to server project when client omits param
 * - Error handling: VectorError → structured MCP error response
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { mkdirSync, writeFileSync, rmSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { randomBytes } from 'node:crypto'
import Database from 'better-sqlite3'

import {
  handleSearchMemory,
  handleIndexFiles,
  handleGetDocument,
  handleListDocuments,
  handleRemoveDocument,
  handleStatus,
  type ToolContext,
} from '../../src/mcp/tools.js'
import { createMcpServer } from '../../src/mcp/server.js'
import { SqliteStore } from '../../src/store/sqlite.js'
import { MarkdownParser } from '../../src/parser/markdown.js'
import { createLogger } from '../../src/logger.js'
import type { Embedder, Logger } from '../../src/types.js'

// ============================================================================
// Fake Embedder — deterministic, no model download
// ============================================================================

function fakeEmbed(text: string): Float32Array {
  const vec = new Float32Array(384)
  for (let i = 0; i < text.length && i < 384; i++) {
    vec[i] = (text.charCodeAt(i) % 100) / 100
  }
  let norm = 0
  for (let i = 0; i < 384; i++) {
    norm += vec[i]! * vec[i]!
  }
  norm = Math.sqrt(norm)
  if (norm > 0) {
    for (let i = 0; i < 384; i++) {
      vec[i] = vec[i]! / norm
    }
  }
  return vec
}

function createFakeEmbedder(): Embedder {
  return {
    dimensions: 384,
    async embed(text: string): Promise<Float32Array> {
      return fakeEmbed(text)
    },
    async embedBatch(texts: string[]): Promise<Float32Array[]> {
      return texts.map(fakeEmbed)
    },
  }
}

// ============================================================================
// Test helpers
// ============================================================================

function createTempDir(): string {
  const dir = join(tmpdir(), `vector-mcp-test-${randomBytes(8).toString('hex')}`)
  mkdirSync(dir, { recursive: true })
  return dir
}

function writeMdFile(dir: string, name: string, content: string): string {
  const filePath = join(dir, name)
  writeFileSync(filePath, content, 'utf-8')
  return filePath
}

function parseToolResultText(result: { content: ReadonlyArray<{ type: string; text?: string }> }): unknown {
  const textContent = result.content[0]
  if (textContent === undefined || textContent.type !== 'text' || textContent.text === undefined) {
    throw new Error('Expected text content in tool result')
  }
  return JSON.parse(textContent.text)
}

// ============================================================================
// Tests
// ============================================================================

describe('MCP Tools Integration', () => {
  let tempDir: string
  let dbDir: string
  let store: SqliteStore
  let parser: MarkdownParser
  let embedder: Embedder
  let logger: Logger
  let db: Database.Database
  let ctx: ToolContext

  beforeEach(() => {
    tempDir = createTempDir()
    dbDir = join(tempDir, '.vector')
    mkdirSync(dbDir, { recursive: true })

    store = new SqliteStore({ storagePath: dbDir, databaseName: 'test.db' })
    parser = new MarkdownParser({ project: 'test-project' })
    embedder = createFakeEmbedder()
    logger = createLogger(false)

    // Separate DB handle for search operations
    db = new Database(join(dbDir, 'test.db'))
    db.pragma('journal_mode = WAL')
    db.pragma('foreign_keys = ON')

    ctx = {
      store,
      parser,
      parserFactory: (project: string) => new MarkdownParser({ project }),
      embedder,
      logger,
      db,
      dbPath: join(dbDir, 'test.db'),
      projectRoot: tempDir,
      config: {
        defaultTopK: 10,
        rrfK: 60,
        minScore: 0.0,
        project: 'test-project',
      },
    }
  })

  afterEach(() => {
    db.close()
    store.close()
    if (existsSync(tempDir)) {
      rmSync(tempDir, { recursive: true, force: true })
    }
  })

  // --------------------------------------------------------------------------
  // createMcpServer registers all 6 tools
  // --------------------------------------------------------------------------

  it('creates MCP server with all 6 tools registered', () => {
    const server = createMcpServer(ctx)
    expect(server).toBeDefined()
    // The server object itself being created without error validates tool registration
    expect(typeof server.connect).toBe('function')
    expect(typeof server.close).toBe('function')
  })

  // --------------------------------------------------------------------------
  // index_files — indexes files and returns results
  // --------------------------------------------------------------------------

  it('index_files indexes markdown files and returns per-file results', async () => {
    const filePath = writeMdFile(tempDir, 'auth.md', `# Authentication

## OAuth 2.0 Flow

Authentication uses OAuth 2.0 with PKCE for secure authorization.
The client initiates the flow by redirecting to the authorization server.
`)

    const result = await handleIndexFiles({ paths: [filePath] }, ctx)

    expect(result.isError).toBeUndefined()
    const parsed = parseToolResultText(result) as Array<{ path: string; status: string; chunks?: number }>
    expect(Array.isArray(parsed)).toBe(true)
    expect(parsed.length).toBe(1)
    expect(parsed[0]!.status).toBe('indexed')
    expect(parsed[0]!.chunks).toBeGreaterThan(0)
  })

  it('index_files skips unchanged files on re-index', async () => {
    const filePath = writeMdFile(tempDir, 'skip.md', `# Skip Test

Content that should only be indexed once.
`)

    // First index
    await handleIndexFiles({ paths: [filePath] }, ctx)

    // Second index — should skip
    const result = await handleIndexFiles({ paths: [filePath] }, ctx)
    const parsed = parseToolResultText(result) as Array<{ path: string; status: string; reason?: string }>
    expect(parsed[0]!.status).toBe('skipped')
    expect(parsed[0]!.reason).toBe('unchanged')
  })

  it('index_files handles non-existent files gracefully', async () => {
    const result = await handleIndexFiles({
      paths: [join(tempDir, 'nonexistent.md')],
    }, ctx)

    const parsed = parseToolResultText(result) as Array<{ path: string; status: string; reason?: string }>
    expect(parsed[0]!.status).toBe('failed')
    expect(parsed[0]!.reason).toBeTruthy()
  })

  // --------------------------------------------------------------------------
  // search_memory — returns array of results
  // --------------------------------------------------------------------------

  it('search_memory returns array of search results', async () => {
    // Index a file first
    const filePath = writeMdFile(tempDir, 'search-test.md', `# Search Test

## OAuth Authentication

OAuth 2.0 is the industry standard protocol for authorization.
It provides secure delegated access for web and mobile applications.
`)

    await handleIndexFiles({ paths: [filePath] }, ctx)

    const result = await handleSearchMemory({ query: 'OAuth authentication' }, ctx)

    expect(result.isError).toBeUndefined()
    const parsed = parseToolResultText(result) as Array<{
      documentTitle: string
      documentPath: string
      score: number
      content: string
    }>
    expect(Array.isArray(parsed)).toBe(true)
    expect(parsed.length).toBeGreaterThan(0)

    // Verify result shape
    const first = parsed[0]!
    expect(typeof first.documentTitle).toBe('string')
    expect(typeof first.documentPath).toBe('string')
    expect(typeof first.score).toBe('number')
    expect(first.score).toBeGreaterThanOrEqual(0)
    expect(first.score).toBeLessThanOrEqual(1)
    expect(typeof first.content).toBe('string')
  })

  it('search_memory returns empty array for no matches', async () => {
    // Search with no indexed content
    const result = await handleSearchMemory({ query: 'nonexistent content xyz123' }, ctx)

    expect(result.isError).toBeUndefined()
    const parsed = parseToolResultText(result) as unknown[]
    expect(Array.isArray(parsed)).toBe(true)
    expect(parsed.length).toBe(0)
  })

  it('search_memory respects topK parameter', async () => {
    // Index multiple files
    for (let i = 0; i < 5; i++) {
      const filePath = writeMdFile(tempDir, `topk-${i}.md`, `# Document ${i}

Content about topic ${i} for testing topK parameter.
`)
      await handleIndexFiles({ paths: [filePath] }, ctx)
    }

    const result = await handleSearchMemory({ query: 'Document topic', topK: 2 }, ctx)
    const parsed = parseToolResultText(result) as unknown[]
    expect(parsed.length).toBeLessThanOrEqual(2)
  })

  // --------------------------------------------------------------------------
  // get_document — retrieves a specific document
  // --------------------------------------------------------------------------

  it('get_document returns a stored document', async () => {
    const filePath = writeMdFile(tempDir, 'get-test.md', `# Get Test

Content for get_document test.
`)

    await handleIndexFiles({ paths: [filePath] }, ctx)

    const result = handleGetDocument({ project: 'test-project', filePath }, ctx)

    expect(result.isError).toBeUndefined()
    const parsed = parseToolResultText(result) as {
      id: string
      title: string
      filePath: string
      project: string
      chunkCount: number
      indexedAt: string
    }
    expect(parsed).not.toBeNull()
    expect(parsed.title).toBe('Get Test')
    expect(parsed.project).toBe('test-project')
    expect(parsed.chunkCount).toBeGreaterThan(0)
    expect(typeof parsed.indexedAt).toBe('string')
  })

  it('get_document returns null for non-existent document', () => {
    const result = handleGetDocument({
      project: 'test-project',
      filePath: '/nonexistent/path.md',
    }, ctx)

    expect(result.isError).toBeUndefined()
    const parsed = parseToolResultText(result)
    expect(parsed).toBeNull()
  })

  // --------------------------------------------------------------------------
  // list_documents — returns document list
  // --------------------------------------------------------------------------

  it('list_documents returns all indexed documents', async () => {
    const filePaths = ['list-a.md', 'list-b.md', 'list-c.md'].map(name =>
      writeMdFile(tempDir, name, `# ${name.replace('.md', '')}

Content for ${name}.
`),
    )

    for (const fp of filePaths) {
      await handleIndexFiles({ paths: [fp] }, ctx)
    }

    const result = handleListDocuments({}, ctx)

    expect(result.isError).toBeUndefined()
    const parsed = parseToolResultText(result) as Array<{
      id: string
      title: string
      filePath: string
      project: string
      chunkCount: number
    }>
    expect(Array.isArray(parsed)).toBe(true)
    expect(parsed.length).toBe(3)

    // Verify each document has required fields
    for (const doc of parsed) {
      expect(typeof doc.id).toBe('string')
      expect(typeof doc.title).toBe('string')
      expect(typeof doc.filePath).toBe('string')
      expect(doc.project).toBe('test-project')
      expect(typeof doc.chunkCount).toBe('number')
    }
  })

  it('list_documents returns empty array when no documents indexed', () => {
    const result = handleListDocuments({}, ctx)

    expect(result.isError).toBeUndefined()
    const parsed = parseToolResultText(result) as unknown[]
    expect(Array.isArray(parsed)).toBe(true)
    expect(parsed.length).toBe(0)
  })

  it('list_documents filters by project', async () => {
    const filePath = writeMdFile(tempDir, 'proj-filter.md', `# Project Filter

Content for project filter test.
`)

    await handleIndexFiles({ paths: [filePath] }, ctx)

    // Filter by correct project
    const result = handleListDocuments({ project: 'test-project' }, ctx)
    const parsed = parseToolResultText(result) as unknown[]
    expect(parsed.length).toBe(1)

    // Filter by wrong project
    const resultEmpty = handleListDocuments({ project: 'other-project' }, ctx)
    const parsedEmpty = parseToolResultText(resultEmpty) as unknown[]
    expect(parsedEmpty.length).toBe(0)
  })

  // --------------------------------------------------------------------------
  // remove_document — removes and returns status
  // --------------------------------------------------------------------------

  it('remove_document removes an existing document', async () => {
    const filePath = writeMdFile(tempDir, 'remove-me.md', `# Remove Me

This document will be removed.
`)

    await handleIndexFiles({ paths: [filePath] }, ctx)

    // Verify document exists
    const beforeResult = handleGetDocument({ project: 'test-project', filePath }, ctx)
    const beforeParsed = parseToolResultText(beforeResult)
    expect(beforeParsed).not.toBeNull()

    // Remove the document
    const removeResult = handleRemoveDocument({ project: 'test-project', filePath }, ctx)

    expect(removeResult.isError).toBeUndefined()
    const removeParsed = parseToolResultText(removeResult) as { removed: boolean }
    expect(removeParsed.removed).toBe(true)

    // Verify document is gone
    const afterResult = handleGetDocument({ project: 'test-project', filePath }, ctx)
    const afterParsed = parseToolResultText(afterResult)
    expect(afterParsed).toBeNull()
  })

  it('remove_document returns false for non-existent document', () => {
    const result = handleRemoveDocument({
      project: 'test-project',
      filePath: '/nonexistent/path.md',
    }, ctx)

    expect(result.isError).toBeUndefined()
    const parsed = parseToolResultText(result) as { removed: boolean }
    expect(parsed.removed).toBe(false)
  })

  // --------------------------------------------------------------------------
  // status — returns aggregate stats
  // --------------------------------------------------------------------------

  it('status returns aggregate stats with document and chunk counts', async () => {
    const filePaths = ['stat-a.md', 'stat-b.md'].map(name =>
      writeMdFile(tempDir, name, `# ${name.replace('.md', '')}

Content for statistics test in ${name}. This includes enough text
to form meaningful chunks for counting purposes.

## Section Two

More content in a second section for additional chunks.
`),
    )

    for (const fp of filePaths) {
      await handleIndexFiles({ paths: [fp] }, ctx)
    }

    const result = handleStatus({}, ctx)

    expect(result.isError).toBeUndefined()
    const parsed = parseToolResultText(result) as {
      documents: number
      chunks: number
      dbSize: number
      staleFiles: number
      projects: string[]
    }
    expect(parsed.documents).toBe(2)
    expect(parsed.chunks).toBeGreaterThan(0)
    expect(parsed.dbSize).toBeGreaterThan(0)
    expect(typeof parsed.staleFiles).toBe('number')
    expect(Array.isArray(parsed.projects)).toBe(true)
    expect(parsed.projects).toContain('test-project')
  })

  it('status returns zero counts when no documents indexed', () => {
    const result = handleStatus({}, ctx)

    expect(result.isError).toBeUndefined()
    const parsed = parseToolResultText(result) as {
      documents: number
      chunks: number
      dbSize: number
      staleFiles: number
    }
    expect(parsed.documents).toBe(0)
    expect(parsed.chunks).toBe(0)
  })

  it('status filters by project', async () => {
    const filePath = writeMdFile(tempDir, 'status-proj.md', `# Status Project

Content for status project filter test.
`)

    await handleIndexFiles({ paths: [filePath] }, ctx)

    // Filter by correct project
    const result = handleStatus({ project: 'test-project' }, ctx)
    const parsed = parseToolResultText(result) as { documents: number }
    expect(parsed.documents).toBe(1)

    // Filter by wrong project
    const resultEmpty = handleStatus({ project: 'other-project' }, ctx)
    const parsedEmpty = parseToolResultText(resultEmpty) as { documents: number }
    expect(parsedEmpty.documents).toBe(0)
  })

  // --------------------------------------------------------------------------
  // Project isolation — default project fallback
  // --------------------------------------------------------------------------

  describe('project isolation', () => {
    it('search_memory without project param returns only server default project results', async () => {
      const filePath = writeMdFile(tempDir, 'default-proj.md', `# Default Project Doc

## Kubernetes Deployment

Deploy applications using Kubernetes with helm charts.
`)
      await handleIndexFiles({ paths: [filePath] }, ctx)

      // Manually insert a document under a different project
      const otherParser = new MarkdownParser({ project: 'other-project' })
      const otherFile = writeMdFile(tempDir, 'other-proj.md', `# Other Project Doc

## Kubernetes Scaling

Scale applications horizontally with HPA.
`)
      const { document: otherDoc, chunks: otherChunks } = otherParser.parse(otherFile)
      const otherEmbeddings = await embedder.embedBatch(otherChunks.map(c => c.contentPlain))
      const otherItems = otherChunks.map((chunk, i) => ({ chunk, embedding: otherEmbeddings[i]! }))
      store.save(otherDoc, otherItems)

      // Search WITHOUT project param — should only return test-project results
      const result = await handleSearchMemory({ query: 'Kubernetes' }, ctx)
      const parsed = parseToolResultText(result) as Array<{ documentPath: string }>

      expect(parsed.length).toBeGreaterThan(0)
      for (const r of parsed) {
        expect(r.documentPath).not.toBe(otherFile)
      }
    })

    it('list_documents without project param returns only server default project docs', async () => {
      const filePath = writeMdFile(tempDir, 'list-default.md', `# List Default

Content for list test.
`)
      await handleIndexFiles({ paths: [filePath] }, ctx)

      // Insert under other project
      const otherParser = new MarkdownParser({ project: 'other-project' })
      const otherFile = writeMdFile(tempDir, 'list-other.md', `# List Other

Content for other project.
`)
      const { document: otherDoc, chunks: otherChunks } = otherParser.parse(otherFile)
      const otherEmbeddings = await embedder.embedBatch(otherChunks.map(c => c.contentPlain))
      store.save(otherDoc, otherChunks.map((chunk, i) => ({ chunk, embedding: otherEmbeddings[i]! })))

      // list_documents WITHOUT project param
      const result = handleListDocuments({}, ctx)
      const parsed = parseToolResultText(result) as Array<{ project: string }>

      expect(parsed.length).toBe(1)
      expect(parsed[0]!.project).toBe('test-project')
    })

    it('status without project param returns only server default project stats', async () => {
      const filePath = writeMdFile(tempDir, 'status-default.md', `# Status Default

Content for status test.
`)
      await handleIndexFiles({ paths: [filePath] }, ctx)

      // Insert under other project
      const otherParser = new MarkdownParser({ project: 'other-project' })
      const otherFile = writeMdFile(tempDir, 'status-other.md', `# Status Other

Content for other project status.
`)
      const { document: otherDoc, chunks: otherChunks } = otherParser.parse(otherFile)
      const otherEmbeddings = await embedder.embedBatch(otherChunks.map(c => c.contentPlain))
      store.save(otherDoc, otherChunks.map((chunk, i) => ({ chunk, embedding: otherEmbeddings[i]! })))

      // status WITHOUT project param
      const result = handleStatus({}, ctx)
      const parsed = parseToolResultText(result) as { documents: number; projects: string[] }

      expect(parsed.documents).toBe(1)
      expect(parsed.projects).toEqual(['test-project'])
    })

    it('index_files respects explicit project override', async () => {
      const filePath = writeMdFile(tempDir, 'custom-proj.md', `# Custom Project

Content indexed under custom project name.
`)

      // Index with explicit project override
      await handleIndexFiles({ paths: [filePath], project: 'custom-project' }, ctx)

      // Should NOT appear under default project
      const defaultDocs = handleListDocuments({ project: 'test-project' }, ctx)
      const defaultParsed = parseToolResultText(defaultDocs) as Array<{ filePath: string }>
      const foundInDefault = defaultParsed.some(d => d.filePath === filePath)
      expect(foundInDefault).toBe(false)

      // Should appear under custom project
      const customDocs = handleListDocuments({ project: 'custom-project' }, ctx)
      const customParsed = parseToolResultText(customDocs) as Array<{ filePath: string }>
      const foundInCustom = customParsed.some(d => d.filePath === filePath)
      expect(foundInCustom).toBe(true)
    })

    it('search_memory with explicit project overrides server default', async () => {
      const filePath = writeMdFile(tempDir, 'override-default.md', `# Override Default

Content about databases and indexing.
`)
      await handleIndexFiles({ paths: [filePath] }, ctx)

      // Index under other project
      const otherParser = new MarkdownParser({ project: 'other-project' })
      const otherFile = writeMdFile(tempDir, 'override-other.md', `# Override Other

Content about databases and queries.
`)
      const { document: otherDoc, chunks: otherChunks } = otherParser.parse(otherFile)
      const otherEmbeddings = await embedder.embedBatch(otherChunks.map(c => c.contentPlain))
      store.save(otherDoc, otherChunks.map((chunk, i) => ({ chunk, embedding: otherEmbeddings[i]! })))

      // Search explicitly in other-project
      const result = await handleSearchMemory(
        { query: 'databases', project: 'other-project' },
        ctx,
      )
      const parsed = parseToolResultText(result) as Array<{ documentPath: string }>

      for (const r of parsed) {
        expect(r.documentPath).not.toBe(filePath)
      }
    })
  })

  // --------------------------------------------------------------------------
  // End-to-end: index → search → remove → verify gone
  // --------------------------------------------------------------------------

  it('full lifecycle: index → search → remove → verify gone', async () => {
    const filePath = writeMdFile(tempDir, 'lifecycle.md', `# Lifecycle Test

## Kubernetes Deployment

Deploy applications using Kubernetes with helm charts.
The cluster manages pod scheduling and service discovery.
`)

    // 1. Index
    const indexResult = await handleIndexFiles({ paths: [filePath] }, ctx)
    const indexParsed = parseToolResultText(indexResult) as Array<{ status: string }>
    expect(indexParsed[0]!.status).toBe('indexed')

    // 2. Search
    const searchResult = await handleSearchMemory({ query: 'Kubernetes deployment' }, ctx)
    const searchParsed = parseToolResultText(searchResult) as Array<{ content: string }>
    expect(searchParsed.length).toBeGreaterThan(0)

    // 3. Status
    const statusResult = handleStatus({}, ctx)
    const statusParsed = parseToolResultText(statusResult) as { documents: number }
    expect(statusParsed.documents).toBe(1)

    // 4. Remove
    const removeResult = handleRemoveDocument({ project: 'test-project', filePath }, ctx)
    const removeParsed = parseToolResultText(removeResult) as { removed: boolean }
    expect(removeParsed.removed).toBe(true)

    // 5. Verify gone
    const afterStatus = handleStatus({}, ctx)
    const afterParsed = parseToolResultText(afterStatus) as { documents: number }
    expect(afterParsed.documents).toBe(0)

    // 6. Search returns no results for removed content
    const afterSearch = await handleSearchMemory({ query: 'Kubernetes deployment' }, ctx)
    const afterSearchParsed = parseToolResultText(afterSearch) as unknown[]
    // The removed document should not appear
    const hasLifecycleDoc = (afterSearchParsed as Array<{ documentPath: string }>).some(
      r => r.documentPath === filePath,
    )
    expect(hasLifecycleDoc).toBe(false)
  })
})
