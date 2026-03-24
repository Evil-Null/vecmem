# Vector Ideafix

A CLI tool and MCP server that automatically converts `.md` files into a hybrid AI memory system with vector search, full-text search, trust scoring, relation graphs, and token-optimized retrieval.

> **Status:** In development

## What It Does

You write plans and notes in `.md` files. Vector Ideafix automatically:

- **Parses** markdown into semantically meaningful chunks (heading-aware, code-block-safe)
- **Embeds** chunks locally using transformer models (no API keys required)
- **Stores** everything in a single SQLite database (structured metadata + FTS5 full-text index + vector embeddings)
- **Retrieves** the most relevant context using hybrid search (BM25 + vector similarity + Reciprocal Rank Fusion)
- **Scores trust** on chunks based on usage feedback, with automatic decay for stale content
- **Serves** results via MCP protocol to any compatible AI client (Claude CLI, VS Code, Cursor, etc.)

## Architecture

```
.md files
    |
    v
[ Parser Engine ]  -- remark AST, frontmatter extraction, heading-aware chunking
    |
    v
[ Embedding Engine ]  -- local transformers (all-MiniLM-L6-v2, 384-dim)
    |
    v
[ SQLite Storage ]  -- structured tables + FTS5 + vector BLOBs (single file, zero services)
    |
    v
[ Hybrid Retrieval ]  -- BM25 + cosine similarity + RRF fusion + trust weighting
    |
    v
[ CLI / MCP Server ]  -- commander CLI + MCP protocol for AI client integration
```

## Key Design Decisions

- **Single SQLite database** — atomic transactions across all data types, single-file backup, zero external dependencies
- **Local embeddings** — no API costs, works offline, reproducible results
- **Reciprocal Rank Fusion** — proven method to combine keyword and vector search without score normalization
- **Incremental indexing** — SHA256 content hashing, only re-process changed files
- **Heading-aware chunking** — respects document structure, preserves code blocks, inherits parent heading context

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | TypeScript 5.7+, Node.js 22+ |
| Storage | SQLite via better-sqlite3 (FTS5 + BLOB vectors) |
| Embeddings | @huggingface/transformers (all-MiniLM-L6-v2) |
| Markdown | remark + remark-frontmatter |
| CLI | commander |
| MCP | @modelcontextprotocol/sdk |
| Validation | zod |

## Planned CLI

```bash
vector init                          # Initialize project
vector index ./memory                # Parse, embed, and store .md files
vector query "fix login bug"         # Hybrid search
vector status                        # Stats and health
```

## Planned MCP Tools

- `search_memory` — hybrid search with filtering
- `index_memory` — index new or updated files
- `get_chunk` — retrieve specific chunk by ID
- `list_documents` — list indexed documents
- `check_freshness` — detect stale content
- `report_usage` — feedback for trust scoring
- `stats` — database statistics

## Roadmap

- [ ] **Phase 1** — Foundation (project scaffolding, types, config, database schema)
- [ ] **Phase 2** — Parser Engine (markdown parsing, chunking, token counting)
- [ ] **Phase 3** — Embedding Engine (local transformer integration)
- [ ] **Phase 4** — Storage & Search (document/chunk CRUD, FTS5, vector storage)
- [ ] **Phase 5** — Hybrid Retrieval (BM25, vector search, RRF fusion)
- [ ] **Phase 6** — Trust, Lifecycle & Relations (scoring, decay, relation graphs)
- [ ] **Phase 7** — CLI Interface (commander-based commands)
- [ ] **Phase 8** — MCP Server (protocol integration)
- [ ] **Phase 9** — LLM Context Injection (token-optimized prompt compilation)

## License

[MIT](LICENSE)
