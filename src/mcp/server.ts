/**
 * Vector Memory Engine — MCP Server
 *
 * Exposes Vector's core functionality to AI clients (Claude CLI, Cursor, Copilot)
 * via stdio transport using the Model Context Protocol.
 *
 * 6 tools registered:
 * - search_memory: hybrid BM25 + vector search
 * - index_files: index markdown files into memory
 * - get_document: retrieve a specific document
 * - list_documents: list all indexed documents
 * - remove_document: remove a document from the index
 * - status: aggregate index statistics
 *
 * stdio transport only (v1). Tool schemas auto-generated from Zod definitions.
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import {
  searchMemorySchema,
  indexFilesSchema,
  getDocumentSchema,
  listDocumentsSchema,
  removeDocumentSchema,
  statusSchema,
  handleSearchMemory,
  handleIndexFiles,
  handleGetDocument,
  handleListDocuments,
  handleRemoveDocument,
  handleStatus,
  type ToolContext,
} from './tools.js'

// ============================================================================
// MCP Server Factory
// ============================================================================

/**
 * Create an MCP server with all 6 tools registered.
 *
 * @param context - Dependencies (store, parser, embedder, logger, db, config)
 * @returns McpServer instance ready to connect to a transport
 */
export function createMcpServer(context: ToolContext): McpServer {
  const server = new McpServer(
    {
      name: 'vector-memory',
      version: '0.1.0',
    },
    {
      capabilities: {
        tools: {},
      },
    },
  )

  // --------------------------------------------------------------------------
  // Register tools
  // --------------------------------------------------------------------------

  server.registerTool(
    'search_memory',
    {
      description: 'Search memory using hybrid BM25 + vector search',
      inputSchema: searchMemorySchema,
    },
    async (args) => handleSearchMemory(args, context),
  )

  server.registerTool(
    'index_files',
    {
      description: 'Index markdown files into memory',
      inputSchema: indexFilesSchema,
    },
    async (args) => handleIndexFiles(args, context),
  )

  server.registerTool(
    'get_document',
    {
      description: 'Get a specific document by project and path',
      inputSchema: getDocumentSchema,
    },
    (args) => handleGetDocument(args, context),
  )

  server.registerTool(
    'list_documents',
    {
      description: 'List all indexed documents',
      inputSchema: listDocumentsSchema,
    },
    (args) => handleListDocuments(args, context),
  )

  server.registerTool(
    'remove_document',
    {
      description: 'Remove a document from the index',
      inputSchema: removeDocumentSchema,
    },
    (args) => handleRemoveDocument(args, context),
  )

  server.registerTool(
    'status',
    {
      description: 'Get index status and statistics',
      inputSchema: statusSchema,
    },
    (args) => handleStatus(args, context),
  )

  return server
}

// ============================================================================
// Start MCP Server via stdio transport
// ============================================================================

/**
 * Start the MCP server with stdio transport.
 * This connects stdin/stdout for communication with AI clients.
 *
 * @param context - Dependencies (store, parser, embedder, logger, db, config)
 */
export async function startMcpServer(context: ToolContext): Promise<void> {
  const server = createMcpServer(context)
  const transport = new StdioServerTransport()
  await server.connect(transport)
}
