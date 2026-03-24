# vecmem

> Your notes become AI's memory

A CLI tool and MCP server that transforms `.md` files into a hybrid AI memory system with semantic vector search, full-text search, and instant retrieval.

## Install

```bash
npm install -g vecmem
```

## Quick Start

```bash
vecmem init                              # Auto-discover .md files
vecmem index                             # Index → chunks → embeddings
vecmem query "how does auth work?"       # Hybrid search (BM25 + vector)
vecmem status                            # Stats and health
vecmem doctor                            # System health check
```

## What It Does

You write notes in `.md` files. vecmem automatically:

- **Parses** markdown into semantic chunks (heading-aware, code-block-safe)
- **Embeds** chunks locally using transformer models (no API keys, works offline)
- **Stores** everything in a single SQLite database (metadata + FTS5 + vector embeddings)
- **Searches** using hybrid BM25 + vector similarity + Reciprocal Rank Fusion
- **Serves** results via MCP protocol to any AI client (Claude CLI, Cursor, Copilot, Windsurf)

## CLI Commands

| Command | Description |
|---------|-------------|
| `vecmem init` | Initialize project, auto-discover .md files |
| `vecmem index [path]` | Index markdown files (incremental, skips unchanged) |
| `vecmem query "..."` | Hybrid search — BM25 + vector + RRF |
| `vecmem status` | Document count, chunks, DB size, stale files |
| `vecmem doctor` | Health check — DB, FTS sync, invariants, model |
| `vecmem remove <path>` | Remove a document from the index |

## MCP Server

Works with any MCP-compatible AI client:

```bash
vecmem --mcp    # Start stdio MCP server
```

7 tools: `search_memory`, `index_files`, `get_document`, `get_chunks`, `list_documents`, `remove_document`, `status`

### Claude CLI Setup

Add to `.mcp.json` in your project:

```json
{
  "mcpServers": {
    "vecmem": {
      "command": "vecmem",
      "args": ["--mcp"]
    }
  }
}
```

## Architecture

```
.md files → Parser → Embedder → SQLite Store → Hybrid Search
                                    │
                              CLI ←─┤─→ MCP Server
```

- **Parser** — remark AST, heading-aware chunking, frontmatter extraction
- **Embedder** — local HuggingFace transformers (all-MiniLM-L6-v2, 384-dim)
- **Store** — SQLite with WAL mode, FTS5 triggers, atomic transactions
- **Search** — BM25 + cosine similarity + RRF fusion, normalized to [0,1]

## Engineering

Built at Distinguished/Fellow level (Level 3):

- **Branded types** — `DocumentId`, `ChunkId`, `UnitScore` prevent ID mixups at compile time
- **System invariants** — 5 invariant checks verified after every write in dev mode
- **Property-based testing** — fast-check with 1000+ random inputs per property
- **Performance contracts** — CI fails if search > 200ms or indexing < 50 chunks/sec
- **Graceful degradation** — BM25 fallback when embedding model unavailable
- **255 tests** across 19 test files

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | TypeScript 5.7+ (strict mode) |
| Runtime | Node.js 22+ |
| Storage | SQLite via better-sqlite3 (WAL, FTS5) |
| Embeddings | @huggingface/transformers (all-MiniLM-L6-v2) |
| Markdown | remark + remark-frontmatter |
| CLI | commander |
| MCP | @modelcontextprotocol/sdk |
| Testing | vitest + fast-check |

## License

[MIT](LICENSE)
