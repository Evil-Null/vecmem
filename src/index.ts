#!/usr/bin/env node
/**
 * Vector Memory Engine — Entry Point
 *
 * Dispatches to CLI (Commander) or MCP server (stdio) based on --mcp flag.
 */

const isMcpMode = process.argv.includes('--mcp')

if (isMcpMode) {
  // MCP server mode — stdio transport for AI clients
  const { startMcpServer } = await import('./mcp/server.js')
  const { SqliteStore } = await import('./store/sqlite.js')
  const { MarkdownParser } = await import('./parser/markdown.js')
  const { LocalEmbedder } = await import('./embedder/local.js')
  const { createLogger } = await import('./logger.js')
  const { resolveConfig } = await import('./config.js')

  const config = resolveConfig({})
  const store = new SqliteStore({
    storagePath: config.storagePath,
    databaseName: config.databaseName,
    modelName: config.embeddingModel,
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
  const logger = createLogger(false)

  await startMcpServer({
    store,
    parser,
    parserFactory: (project: string) => new MarkdownParser({
      maxChunkTokens: config.maxChunkTokens,
      minChunkTokens: config.minChunkTokens,
      chunkOverlapTokens: config.chunkOverlapTokens,
      headingSplitDepth: config.headingSplitDepth,
      project,
    }),
    embedder,
    logger,
    db: store.getDb(),
    dbPath: store.getDbPath(),
    projectRoot: process.cwd(),
    config: {
      defaultTopK: config.defaultTopK,
      rrfK: config.rrfK,
      minScore: config.minScore,
      project: config.project,
    },
  })
} else {
  // CLI mode — Commander for terminal usage
  const { program } = await import('./cli/program.js')
  program.parse()
}
