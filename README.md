# vecmem

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](package.json)
[![Platform](https://img.shields.io/badge/platform-Node.js%2022%2B-brightgreen.svg)](https://nodejs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7%2B-3178C6.svg)](https://www.typescriptlang.org)
[![Tests](https://img.shields.io/badge/tests-255%20passing-success.svg)](tests/)
[![MCP](https://img.shields.io/badge/MCP-7%20tools-purple.svg)](src/mcp/)

> Your notes become AI's memory

You have markdown notes scattered across folders. You know the answer is *somewhere* in there, but where?

**vecmem** indexes your `.md` files and lets you (or your AI assistant) search them instantly — not just by keywords, but by meaning.

## The Problem

```
You: "How did we set up authentication?"
AI:  "I don't have access to your notes."

You: *manually searches 50 files*
You: *finds it 10 minutes later in docs/auth/oauth.md*
```

## The Solution

```bash
vecmem index ./docs                        # One-time: index your notes
vecmem query "how did we set up auth?"     # Instant: finds it in 100ms
```

```
Found 5 results (87ms)

┌─ docs/auth/oauth.md ─── score: 0.94 ────────────┐
│ ## OAuth 2.0 Flow                                 │
│ Authentication uses OAuth 2.0 with PKCE.          │
│ The flow starts with a redirect to /auth/login... │
└───────────────────────────────────────────────────┘
```

It finds `oauth.md` even though you searched "auth" — because it understands meaning, not just words.

## Install

```bash
npm install -g vecmem
```

## Quick Start

```bash
vecmem init                              # Find all .md files in your project
vecmem index                             # Index them (takes a few seconds)
vecmem query "database schema"           # Search by meaning
vecmem status                            # See what's indexed
```

## Works With AI Assistants

vecmem is an [MCP server](https://modelcontextprotocol.io/) — AI assistants can use it directly:

**Claude CLI** — add to `.mcp.json` in your project:

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

Also works with **Cursor**, **VS Code Copilot**, **Windsurf**, and any MCP-compatible client.

Once connected, your AI can search your notes automatically:

```
You: "What do we know about the deployment process?"
AI:  *searches vecmem* → finds 3 relevant notes → gives you the answer
```

## How It Works

```
Your .md files
      │
      ▼
   vecmem index
      │
      ├── Reads and splits into sections (heading-aware)
      ├── Converts each section into a vector (meaning)
      └── Stores in local SQLite database

   vecmem query "..."
      │
      ├── Keyword search (exact matches)
      ├── Meaning search (similar concepts)
      └── Combines both → best results first
```

**Everything runs locally.** No API keys, no cloud, no data leaves your machine.

## All Commands

| Command | What it does |
|---------|-------------|
| `vecmem init` | Find .md files and set up database |
| `vecmem index [path]` | Index files (only re-indexes changed ones) |
| `vecmem query "..."` | Search your notes |
| `vecmem status` | How many files/sections are indexed |
| `vecmem doctor` | Check everything is healthy |
| `vecmem remove <path>` | Remove a file from the index |

## FAQ

**How is this different from grep?**
grep finds exact words. vecmem finds related concepts. Search "auth" and it finds documents about "login", "OAuth", "session management".

**Does it need an internet connection?**
Only once — to download the search model (23 MB). After that, everything works offline.

**How fast is it?**
Search takes ~100ms across hundreds of files. Indexing processes ~30 files/sec.

**What files does it support?**
Markdown (`.md`) files only. It understands headings, code blocks, and frontmatter.

**Where is my data stored?**
In `~/.vector/vector.db` — a single file on your machine. Delete it anytime.

**Does it work with languages other than English?**
The search model works best with English. Other languages may have reduced accuracy for meaning-based search, but keyword search works for any language.

## Tech Details

<details>
<summary>For developers who want to know what's under the hood</summary>

### Architecture

```
.md files → Parser → Embedder → SQLite Store → Hybrid Search
                                    │
                              CLI ←─┤─→ MCP Server
```

- **Parser** — remark AST, heading-aware chunking, frontmatter extraction
- **Embedder** — HuggingFace transformers (all-MiniLM-L6-v2, 384-dim, local ONNX)
- **Store** — SQLite with WAL mode, FTS5 full-text index, atomic transactions
- **Search** — BM25 + cosine similarity + Reciprocal Rank Fusion, scores normalized to [0,1]

### Engineering

- Branded types (`DocumentId`, `ChunkId`, `UnitScore`) — compile-time safety
- 5 system invariants checked after every write in dev mode
- Property-based testing with fast-check (1000+ random inputs per property)
- Performance contracts enforced in CI
- Graceful degradation — falls back to keyword search if model unavailable
- 255 tests across 19 test files

### Stack

TypeScript 5.7+ | Node.js 22+ | SQLite (better-sqlite3) | HuggingFace transformers | remark | commander | MCP SDK | zod | vitest + fast-check

</details>

## License

[MIT](LICENSE)
