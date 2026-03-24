# Vector Memory Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool + MCP server that automatically converts `.md` files into a hybrid (vector + structured + full-text) AI memory system with trust scoring, relation graphs, and token-optimized retrieval.

**Architecture:** Extends ColdContext's proven patterns (trust scoring, relation graph, lifecycle management, MCP protocol) with three new engines: (1) automatic markdown parsing + intelligent chunking, (2) local embedding generation via transformer models, (3) hybrid retrieval combining BM25 full-text search, vector similarity, and structured metadata filtering with Reciprocal Rank Fusion. All storage is unified in SQLite (structured tables + FTS5 virtual tables + vector BLOBs) — zero external services required.

**Tech Stack:** TypeScript 5.7+, Node.js 22+ (LTS), SQLite via better-sqlite3 (FTS5 + BLOB vectors), @huggingface/transformers (local embeddings, all-MiniLM-L6-v2, 384-dim), remark + remark-frontmatter (markdown AST parsing), commander (CLI), @modelcontextprotocol/sdk (MCP server), zod (validation).

---

## Table of Contents

1. [Architecture Decision Records](#i-architecture-decision-records)
2. [Data Model](#ii-data-model)
3. [File Structure](#iii-file-structure)
4. [Phase 1 — Foundation](#phase-1--foundation-tasks-1-4)
5. [Phase 2 — Parser Engine](#phase-2--parser-engine-tasks-5-7)
6. [Phase 3 — Embedding Engine](#phase-3--embedding-engine-tasks-8-9)
7. [Phase 4 — Storage & Search](#phase-4--storage--search-tasks-10-13)
8. [Phase 5 — Hybrid Retrieval](#phase-5--hybrid-retrieval-tasks-14-16)
9. [Phase 6 — Trust, Lifecycle & Relations](#phase-6--trust-lifecycle--relations-tasks-17-19)
10. [Phase 7 — CLI Interface](#phase-7--cli-interface-tasks-20-22)
11. [Phase 8 — MCP Server](#phase-8--mcp-server-tasks-23-24)
12. [Phase 9 — LLM Context Injection](#phase-9--llm-context-injection-task-25)
13. [Testing Strategy](#testing-strategy)
14. [Security Model](#security-model)
15. [Performance Considerations](#performance-considerations)

---

## I. Architecture Decision Records

### ADR-001: TypeScript over Python

**Decision:** TypeScript (ES2022, Node16 module resolution)

**Why:**
- Consistent with ColdContext ecosystem — shared patterns, types, tooling
- better-sqlite3 is synchronous (no async overhead for DB ops)
- @modelcontextprotocol/sdk is TypeScript-native
- Type safety catches contract violations at compile time
- Single `npm install` — no virtualenv/pip complexity

**Trade-off:** Python has richer ML ecosystem, but @huggingface/transformers (JS) now runs ONNX models locally with identical quality. For embedding generation (our only ML need), JS parity is complete.

---

### ADR-002: SQLite as Unified Storage (Structured + FTS5 + Vector BLOBs)

**Decision:** Single SQLite database for ALL data — structured metadata, full-text search index (FTS5), and vector embeddings (Float32Array BLOBs).

**Why:**
- **Atomic transactions** across all three data types — chunk + embedding + FTS index update in one transaction
- **Zero external services** — no Chroma/FAISS/Qdrant process to manage
- **Single file backup** — `cp vector.db vector.db.bak`
- **FTS5 built-in BM25** — `rank` function returns BM25 scores natively
- **WAL mode** — concurrent reads + single writer, no lock contention for CLI use
- better-sqlite3 is the fastest SQLite driver for Node.js (synchronous C++ bindings)

**Vector search approach:** Store embeddings as BLOB columns (Float32Array). Compute cosine similarity in JavaScript. For our expected scale (<50K chunks from .md files), brute-force scan of 384-dim vectors takes <15ms. This eliminates native dependency issues (sqlite-vss, faiss-node) while maintaining sub-second response times.

**When to upgrade:** If chunk count exceeds 100K, add hnswlib-node as an ANN index alongside SQLite. The storage layer is designed with an interface that makes this swap transparent.

---

### ADR-003: Local Embeddings First (@huggingface/transformers)

**Decision:** Default to local embedding via `@huggingface/transformers` with `Xenova/all-MiniLM-L6-v2` (384 dimensions, ~23MB ONNX model).

**Why:**
- **Zero API cost** — no OpenAI billing for hobby/open-source projects
- **Zero network dependency** — works offline, in CI, behind firewalls
- **Reproducible** — same model version = identical vectors = deterministic tests
- **Fast** — ~5ms per sentence on CPU (no GPU needed for short .md chunks)
- **Good quality** — MiniLM-L6-v2 scores 68.06 on STS benchmark, sufficient for .md memory retrieval

**Context window limitation:** all-MiniLM-L6-v2 has a 256-token context window. Input beyond that length is silently truncated by the HuggingFace transformers pipeline. Since our target chunk size is 200-500 tokens, chunks near the upper boundary will lose semantic information in their embeddings. Mitigation: the chunker's `maxTokens` default (400) should be considered alongside this limit. For embedding purposes, `contentPlain` is used (stripped markdown = fewer tokens than raw). If higher fidelity is needed, a model with larger context (e.g., `nomic-ai/nomic-embed-text-v1.5` with 8192 context) can be swapped via the `EmbeddingEngine` interface.

**Abstraction:** `EmbeddingEngine` interface with `embed(text): Promise<Float32Array>` and `embedBatch(texts): Promise<Float32Array[]>`. Factory pattern selects provider from config. OpenAI adapter is a future Phase 5 addition (same interface, different provider).

**Model cache:** `~/.vector/models/` — downloaded once, reused across projects.

---

### ADR-004: Reciprocal Rank Fusion (RRF) for Hybrid Search

**Decision:** Combine BM25 and vector search results using RRF with configurable k parameter.

**Why:**
- **Simple, effective, proven** — used by Elasticsearch, Pinecone, Weaviate in production
- **No score normalization needed** — RRF uses ranks, not raw scores (BM25 and cosine similarity have incompatible scales)
- **Tunable** — single `k` parameter (default 60) controls fusion behavior
- **Outperforms** linear interpolation of scores in benchmarks (Cormack et al., 2009)

**Formula:**
```
RRF_score(doc) = sum( 1 / (k + rank_i(doc)) )
```
Where `rank_i` is the document's position in result list `i`, and `k=60` (standard).

**Enhancement:** Trust score acts as a multiplier: `final_score = RRF_score * trust_score`. This naturally surfaces trusted chunks and demotes stale ones.

---

### ADR-005: MCP Protocol for AI Integration

**Decision:** Expose all functionality as MCP tools via stdio transport (same as ColdContext).

**Why:**
- **Universal AI client support** — Claude CLI, VS Code Copilot, Cursor, Windsurf, any MCP-compatible client
- **Standardized tool schemas** — Zod schemas to JSON Schema to client auto-discovery
- **Composable** — Vector + ColdContext + Context7 = multi-MCP workflow
- **No custom integration code** — clients discover and call tools automatically

---

### ADR-006: Heading-Aware Chunking with Metadata Inheritance

**Decision:** Parse markdown into AST (remark), split by headings (configurable depth), preserve code blocks as atomic units, and propagate parent heading context to child chunks.

**Why:**
- **Heading-aware** — respects document structure instead of arbitrary token boundaries
- **Metadata inheritance** — chunk "### Fix login bug" under "## Auth System" carries parent context `["Auth System", "Fix login bug"]` for better retrieval
- **Code block preservation** — never splits a code block mid-way (common failure in naive chunkers)
- **Frontmatter extraction** — YAML frontmatter becomes structured metadata (tags, priority, dates)

**Chunk size control:** Target 200-500 tokens per chunk. If a section exceeds max, split at paragraph boundaries within the section. If a single paragraph exceeds max, split at sentence boundaries.

---

### ADR-007: Incremental Indexing via Content Hashing

**Decision:** SHA256 hash each .md file. On re-index, only process files whose hash changed. Adapted from ColdContext's `file_hashes` pattern.

**Why:**
- **O(delta) not O(n)** — re-indexing 1 changed file out of 500 takes milliseconds, not seconds
- **Embedding cost** — even local embeddings have CPU cost; skipping unchanged files matters
- **Deterministic** — same content = same hash = no unnecessary work
- **Audit trail** — hash history enables "what changed since last index" reporting

---

## II. Data Model

### Core TypeScript Interfaces

```typescript
// -- Document: represents one .md file ----------------------------------------
interface MemoryDocument {
  id: string;                    // "doc_" + sha256(project + filePath)[:12]
  project: string;               // project name
  filePath: string;              // relative path to .md file
  fileName: string;              // basename
  contentHash: string;           // "sha256:<hex>" of file content
  fileSize: number;              // bytes

  // Extracted metadata
  frontmatter: Record<string, unknown>;  // YAML frontmatter
  title: string;                 // first H1, or filename
  tags: string[];                // from frontmatter or inline #tags
  headingTree: string[];         // top-level heading structure

  chunkCount: number;
  totalTokens: number;
  trustScore: number;            // 0.0-1.0 (aggregate of chunk trusts)

  createdAt: string;             // ISO 8601
  updatedAt: string;
  indexedAt: string;             // last successful indexing time
}

// -- Chunk: a semantic unit within a document ---------------------------------
interface MemoryChunk {
  id: string;                    // "chk_" + sha256(docId + headingPath + index)[:12]
  documentId: string;            // FK to MemoryDocument.id
  project: string;               // denormalized for query efficiency

  // Content
  content: string;               // raw markdown text of this chunk
  contentPlain: string;          // stripped markdown (for FTS5 indexing)
  tokenCount: number;            // estimated tokens

  // Structure
  headingPath: string[];         // ["Auth System", "Login", "Fix bug"]
  headingDepth: number;          // depth of the heading that starts this chunk
  chunkIndex: number;            // position within document (0-based)
  hasCodeBlock: boolean;         // contains fenced code block(s)

  // Metadata (inherited from document + section)
  tags: string[];                // document tags + section-specific tags
  priority: string | null;       // "high" | "medium" | "low" | null
  metadata: Record<string, unknown>;  // arbitrary key-value from frontmatter

  // Trust and usage
  trustScore: number;            // 0.0-1.0
  accessCount: number;
  successCount: number;
  failureCount: number;

  createdAt: string;
  updatedAt: string;
}

// -- Embedding: vector representation of a chunk ------------------------------
interface ChunkEmbedding {
  chunkId: string;               // FK to MemoryChunk.id
  modelName: string;             // "all-MiniLM-L6-v2"
  dimensions: number;            // 384
  vector: Float32Array;          // the embedding vector
  createdAt: string;
}

// -- Relation: edge between documents or chunks -------------------------------
interface MemoryRelation {
  sourceId: string;              // doc or chunk ID
  targetId: string;              // doc or chunk ID
  relationType: RelationType;
  strength: number;              // 0.0-1.0
  description?: string;
  createdAt: string;
}

type RelationType =
  | 'references'      // [[wiki-link]] or [text](file.md)
  | 'parent-of'       // document to chunk
  | 'sibling-of'      // chunks under same heading
  | 'related-to'      // semantic similarity > threshold
  | 'depends-on'      // explicit dependency
  | 'supersedes'      // newer version of same topic
  | 'tagged-with';    // shared tag connection

// -- Search Result ------------------------------------------------------------
interface SearchResult {
  chunk: MemoryChunk;
  scores: {
    bm25: number | null;         // FTS5 BM25 score (null if not used)
    vector: number | null;       // cosine similarity (null if not used)
    trust: number;               // trust score component
    fused: number;               // final RRF + trust weighted score
  };
  highlights?: string[];         // FTS5 snippet highlights
  documentTitle: string;
  documentPath: string;
}

// -- Configuration ------------------------------------------------------------
interface VectorConfig {
  storage: {
    path: string;                // default: "~/.vector/vector.db"
    maxSizeMb: number;           // default: 500
  };
  embedding: {
    provider: 'local' | 'openai';
    model: string;               // default: "Xenova/all-MiniLM-L6-v2"
    dimensions: number;          // default: 384
    modelCachePath: string;      // default: "~/.vector/models"
    batchSize: number;           // default: 32
  };
  chunking: {
    maxTokens: number;           // default: 400
    minTokens: number;           // default: 50
    overlapTokens: number;       // default: 40
    splitDepth: number;          // heading depth to split at (default: 2 = ##)
    preserveCodeBlocks: boolean; // default: true
  };
  retrieval: {
    defaultTopK: number;         // default: 10
    rrfK: number;                // RRF k parameter (default: 60)
    minScore: number;            // minimum fused score (default: 0.1)
    trustWeight: number;         // trust multiplier strength (default: 0.3)
  };
  trust: {
    autoInjectThreshold: number; // default: 0.8
    availableThreshold: number;  // default: 0.4
    decayRate: number;           // daily unused decay (default: 0.02)
    decayStartDays: number;      // days before decay starts (default: 7)
  };
  gc: {
    retentionDays: number;       // default: 90
    autoGcEnabled: boolean;      // default: true
  };
}
```

### SQLite Schema

```sql
-- DOCUMENTS: one row per .md file
CREATE TABLE IF NOT EXISTS documents (
  id              TEXT PRIMARY KEY,
  project         TEXT NOT NULL,
  file_path       TEXT NOT NULL,
  file_name       TEXT NOT NULL,
  content_hash    TEXT NOT NULL,
  file_size       INTEGER NOT NULL,
  frontmatter     TEXT NOT NULL DEFAULT '{}',
  title           TEXT NOT NULL,
  tags            TEXT NOT NULL DEFAULT '[]',
  heading_tree    TEXT NOT NULL DEFAULT '[]',
  chunk_count     INTEGER DEFAULT 0,
  total_tokens    INTEGER DEFAULT 0,
  trust_score     REAL DEFAULT 1.0,
  created_at      TEXT DEFAULT (datetime('now')),
  updated_at      TEXT DEFAULT (datetime('now')),
  indexed_at      TEXT DEFAULT (datetime('now')),
  UNIQUE(project, file_path)
);

-- CHUNKS: semantic units within documents
CREATE TABLE IF NOT EXISTS chunks (
  id              TEXT PRIMARY KEY,
  document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  project         TEXT NOT NULL,
  content         TEXT NOT NULL,
  content_plain   TEXT NOT NULL,
  token_count     INTEGER NOT NULL,
  heading_path    TEXT NOT NULL DEFAULT '[]',
  heading_depth   INTEGER DEFAULT 0,
  chunk_index     INTEGER NOT NULL,
  has_code_block  INTEGER DEFAULT 0,
  tags            TEXT NOT NULL DEFAULT '[]',
  priority        TEXT,
  metadata        TEXT NOT NULL DEFAULT '{}',
  trust_score     REAL DEFAULT 1.0,
  access_count    INTEGER DEFAULT 0,
  success_count   INTEGER DEFAULT 0,
  failure_count   INTEGER DEFAULT 0,
  created_at      TEXT DEFAULT (datetime('now')),
  updated_at      TEXT DEFAULT (datetime('now'))
);

-- EMBEDDINGS: vector representations (Float32Array as BLOB)
CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id        TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  model_name      TEXT NOT NULL,
  dimensions      INTEGER NOT NULL,
  vector          BLOB NOT NULL,
  created_at      TEXT DEFAULT (datetime('now'))
);

-- FTS5: standalone full-text search index (BM25)
-- Note: NOT using content= sync mode. The title column is synthetic
-- (extracted from heading_path JSON), so content-sync would break
-- FTS5 rebuild. Instead, triggers manage insert/delete manually.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED,
  title,
  content,
  tags,
  tokenize='porter unicode61'
);

-- Triggers to keep FTS5 in sync with chunks table
CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(chunk_id, title, content, tags)
  VALUES (
    new.id,
    COALESCE(json_extract(new.heading_path, '$[#-1]'), ''),
    new.content_plain,
    new.tags
  );
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
  DELETE FROM chunks_fts WHERE chunk_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
  DELETE FROM chunks_fts WHERE chunk_id = old.id;
  INSERT INTO chunks_fts(chunk_id, title, content, tags)
  VALUES (
    new.id,
    COALESCE(json_extract(new.heading_path, '$[#-1]'), ''),
    new.content_plain,
    new.tags
  );
END;

-- RELATIONS: graph edges between documents/chunks
CREATE TABLE IF NOT EXISTS relations (
  source_id       TEXT NOT NULL,
  source_type     TEXT NOT NULL DEFAULT 'document',  -- 'document' | 'chunk'
  target_id       TEXT NOT NULL,
  target_type     TEXT NOT NULL DEFAULT 'document',  -- 'document' | 'chunk'
  relation_type   TEXT NOT NULL,
  strength        REAL DEFAULT 1.0,
  description     TEXT,
  created_at      TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (source_id, target_id, relation_type)
);

-- USAGE: trust scoring audit trail
CREATE TABLE IF NOT EXISTS usage_log (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id        TEXT REFERENCES chunks(id) ON DELETE CASCADE,
  task_result     TEXT NOT NULL,
  task_description TEXT,
  tokens_used     INTEGER,
  failure_reason  TEXT,
  created_at      TEXT DEFAULT (datetime('now'))
);

-- FILE HASHES: change detection for incremental indexing
CREATE TABLE IF NOT EXISTS file_hashes (
  project         TEXT NOT NULL,
  file_path       TEXT NOT NULL,
  file_hash       TEXT NOT NULL,
  file_size       INTEGER,
  updated_at      TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (project, file_path)
);

-- SCHEMA VERSION: for future migrations
CREATE TABLE IF NOT EXISTS schema_version (
  version         INTEGER NOT NULL,
  applied_at      TEXT DEFAULT (datetime('now'))
);
-- Insert v1 if table is empty (first initialization)
INSERT OR IGNORE INTO schema_version(version) VALUES (1);

-- INDEXES
CREATE INDEX IF NOT EXISTS idx_doc_project ON documents(project);
CREATE INDEX IF NOT EXISTS idx_doc_trust ON documents(trust_score);
CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(content_hash);

CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunk_project ON chunks(project);
CREATE INDEX IF NOT EXISTS idx_chunk_trust ON chunks(trust_score);
CREATE INDEX IF NOT EXISTS idx_chunk_heading ON chunks(heading_depth);

CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model_name);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relations(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relations(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relations(relation_type);

CREATE INDEX IF NOT EXISTS idx_usage_chunk ON usage_log(chunk_id);
CREATE INDEX IF NOT EXISTS idx_hash_project ON file_hashes(project);
```

---

## III. File Structure

```
vector/
|-- package.json
|-- tsconfig.json
|-- .gitignore
|-- src/
|   |-- index.ts                         # Entry point: CLI dispatch + MCP server start
|   |-- config.ts                        # Zod config schema, defaults, file loading
|   |-- types.ts                         # All shared interfaces (from Data Model above)
|   |
|   |-- parser/
|   |   |-- markdown.ts                  # remark AST parsing, frontmatter extraction
|   |   |-- chunker.ts                   # Heading-aware chunking with overlap
|   |   +-- tokens.ts                    # Token counting (simple word-based estimator)
|   |
|   |-- embedding/
|   |   |-- engine.ts                    # EmbeddingEngine interface + factory
|   |   +-- local.ts                     # @huggingface/transformers adapter
|   |
|   |-- storage/
|   |   |-- database.ts                  # Schema init, getDb(), closeDb(), migrations
|   |   |-- documents.ts                 # Document CRUD operations
|   |   |-- chunks.ts                    # Chunk CRUD + FTS5 sync
|   |   +-- vectors.ts                   # Embedding CRUD + cosine similarity
|   |
|   |-- retrieval/
|   |   |-- bm25.ts                      # FTS5 BM25 search wrapper
|   |   |-- vector-search.ts             # Vector similarity search
|   |   |-- fusion.ts                    # RRF score fusion + trust weighting
|   |   +-- engine.ts                    # Hybrid retrieval orchestrator
|   |
|   |-- engine/
|   |   |-- trust.ts                     # Trust scoring (adapted from ColdContext)
|   |   |-- hash.ts                      # SHA256 hashing (adapted from ColdContext)
|   |   |-- lifecycle.ts                 # GC, freshness detection, decay
|   |   +-- prompt.ts                    # Token-optimized prompt compilation
|   |
|   |-- graph/
|   |   +-- relations.ts                 # Relation CRUD, link extraction, BFS cascade
|   |
|   |-- cli/
|   |   |-- program.ts                   # Commander setup + global options
|   |   |-- init.ts                      # `vector init` -- initialize project
|   |   |-- index-cmd.ts                 # `vector index` -- parse + embed + store
|   |   |-- query.ts                     # `vector query` -- hybrid search
|   |   +-- status.ts                    # `vector status` -- stats + health
|   |
|   +-- mcp/
|       |-- server.ts                    # MCP server setup + tool registration
|       +-- tools/
|           |-- store.ts                 # index_memory, update_memory
|           |-- query.ts                 # search_memory, get_chunk, list_documents
|           |-- validate.ts              # check_freshness, check_cascade
|           +-- admin.ts                 # stats, gc, report_usage
|
|-- tests/
|   |-- fixtures/
|   |   |-- simple.md                    # Basic markdown for testing
|   |   |-- frontmatter.md               # With YAML frontmatter
|   |   |-- complex.md                   # Nested headings, code blocks, links
|   |   |-- large.md                     # >2000 tokens (tests chunking boundaries)
|   |   +-- linked-a.md / linked-b.md   # Cross-referencing files
|   |-- parser/
|   |   |-- markdown.test.ts
|   |   +-- chunker.test.ts
|   |-- embedding/
|   |   +-- engine.test.ts
|   |-- storage/
|   |   |-- database.test.ts
|   |   |-- documents.test.ts
|   |   |-- chunks.test.ts
|   |   +-- vectors.test.ts
|   |-- retrieval/
|   |   |-- bm25.test.ts
|   |   |-- vector-search.test.ts
|   |   +-- fusion.test.ts
|   |-- engine/
|   |   |-- trust.test.ts
|   |   +-- hash.test.ts
|   +-- integration/
|       +-- full-pipeline.test.ts
|
+-- docs/
    +-- plans/
        +-- 2026-03-23-vector-memory-engine.md   # this file
```

**Responsibility boundaries:**
- `parser/` — pure functions: text in, structured chunks out. No DB, no I/O beyond file reading.
- `embedding/` — pure functions: text in, Float32Array out. No DB.
- `storage/` — all SQLite operations. Only module that imports `better-sqlite3`.
- `retrieval/` — orchestrates search across storage. Stateless query logic.
- `engine/` — cross-cutting utilities (trust, hash, lifecycle, prompt compilation).
- `graph/` — relation management. Reads/writes via storage layer.
- `cli/` — commander handlers. Thin wrappers that compose parser + embedding + storage + retrieval.
- `mcp/` — MCP protocol wrappers. Thin wrappers same as CLI.

---

## Phase 1 — Foundation (Tasks 1-4)

### Task 1: Project Scaffolding

**Files:**
- Create: `vector/package.json`
- Create: `vector/tsconfig.json`
- Create: `vector/.gitignore`
- Create: `vector/src/index.ts`

- [ ] **Step 1: Create project directory**

```bash
mkdir -p /home/makho/Documents/Vector/vector
cd /home/makho/Documents/Vector/vector
```

- [ ] **Step 2: Create package.json**

```json
{
  "name": "vector-memory",
  "version": "0.1.0",
  "description": "Developer Memory Engine -- .md to hybrid AI memory system",
  "type": "module",
  "main": "dist/index.js",
  "bin": {
    "vector": "dist/index.js"
  },
  "scripts": {
    "build": "tsc",
    "dev": "tsc --watch",
    "start": "node dist/index.js",
    "test": "node --test dist/tests/**/*.test.js",
    "clean": "rm -rf dist",
    "prepublishOnly": "npm run build"
  },
  "engines": {
    "node": ">=22.0.0"
  },
  "keywords": ["mcp", "memory", "vector", "markdown", "rag", "ai-agent"],
  "license": "MIT",
  "dependencies": {
    "@huggingface/transformers": "^3.4.0",
    "@modelcontextprotocol/sdk": "^1.26.0",
    "better-sqlite3": "^11.0.0",
    "commander": "^13.0.0",
    "remark-frontmatter": "^5.0.0",
    "remark-parse": "^11.0.0",
    "unified": "^11.0.0",
    "zod": "^3.25.0"
  },
  "devDependencies": {
    "@types/better-sqlite3": "^7.6.0",
    "@types/node": "^22.0.0",
    "typescript": "^5.7.0"
  }
}
```

- [ ] **Step 3: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "node16",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

Note: Tests are compiled separately. Use `npx tsc -p tsconfig.test.json` for tests, with `rootDir: "."` and `include: ["src/**/*", "tests/**/*"]`. Or use Node.js test runner with `tsx` for direct TypeScript execution in tests.

- [ ] **Step 4: Create .gitignore**

```
node_modules/
dist/
*.db
*.db-wal
*.db-shm
.env
.vector/
```

- [ ] **Step 5: Create minimal entry point**

```typescript
// src/index.ts
#!/usr/bin/env node
console.log('vector-memory v0.1.0');
```

- [ ] **Step 6: Install dependencies and verify build**

Run: `cd /home/makho/Documents/Vector/vector && npm install && npx tsc`
Expected: Clean compilation, no errors.

- [ ] **Step 7: Commit**

```bash
git init
git add package.json tsconfig.json .gitignore src/index.ts
git commit -m "feat: project scaffolding with TypeScript + dependencies"
```

---

### Task 2: Configuration System

**Files:**
- Create: `src/config.ts`
- Test: `tests/config.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// tests/config.test.ts
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { getDefaultConfig, loadConfig, configSchema } from '../src/config.js';

describe('config', () => {
  it('returns valid default config', () => {
    const config = getDefaultConfig();
    assert.equal(config.embedding.provider, 'local');
    assert.equal(config.embedding.dimensions, 384);
    assert.equal(config.chunking.maxTokens, 400);
    assert.equal(config.retrieval.rrfK, 60);
    assert.equal(config.trust.autoInjectThreshold, 0.8);
  });

  it('validates config with zod schema', () => {
    const config = getDefaultConfig();
    const result = configSchema.safeParse(config);
    assert.equal(result.success, true);
  });

  it('rejects invalid config', () => {
    const bad = { storage: { path: '', maxSizeMb: -1 } };
    const result = configSchema.safeParse(bad);
    assert.equal(result.success, false);
  });

  it('merges partial overrides with defaults', () => {
    const config = loadConfig({ chunking: { maxTokens: 800 } });
    assert.equal(config.chunking.maxTokens, 800);
    assert.equal(config.chunking.minTokens, 50); // default preserved
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsc && node --test dist/tests/config.test.js`
Expected: FAIL — module `../src/config.js` not found

- [ ] **Step 3: Write implementation**

`src/config.ts`: Zod schema matching VectorConfig interface. `getDefaultConfig()` returns all defaults. `loadConfig(overrides?)` deep-merges defaults, config file (`~/.vector/config.json` if exists), and explicit overrides. Deep merge function for nested objects.

- [ ] **Step 4: Run test to verify it passes**

Run: `npx tsc && node --test dist/tests/config.test.js`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.ts tests/config.test.ts
git commit -m "feat: configuration system with Zod validation and deep merge"
```

---

### Task 3: Core Types

**Files:**
- Create: `src/types.ts`

- [ ] **Step 1: Create types file**

Write the complete TypeScript interfaces from the Data Model section above into `src/types.ts`. These are pure type definitions — no runtime code, no tests needed. Include all interfaces: `MemoryDocument`, `MemoryChunk`, `ChunkEmbedding`, `MemoryRelation`, `SearchResult`, `RelationType`, plus SQLite row types (`DocumentRow`, `ChunkRow`, `EmbeddingRow`).

- [ ] **Step 2: Verify compilation**

Run: `npx tsc`
Expected: Clean compilation

- [ ] **Step 3: Commit**

```bash
git add src/types.ts
git commit -m "feat: core type definitions for documents, chunks, embeddings, relations"
```

---

### Task 4: Database Schema and Initialization

**Files:**
- Create: `src/storage/database.ts`
- Test: `tests/storage/database.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// tests/storage/database.test.ts
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { getDb, closeDb } from '../../src/storage/database.js';
import { unlinkSync, existsSync } from 'fs';

const TEST_DB = '/tmp/vector-test.db';

describe('database', () => {
  afterEach(() => {
    closeDb();
    for (const f of [TEST_DB, `${TEST_DB}-wal`, `${TEST_DB}-shm`]) {
      if (existsSync(f)) unlinkSync(f);
    }
  });

  it('creates database with all tables', () => {
    const db = getDb(TEST_DB);
    const tables = db.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).all().map((r: any) => r.name);

    assert.ok(tables.includes('documents'));
    assert.ok(tables.includes('chunks'));
    assert.ok(tables.includes('embeddings'));
    assert.ok(tables.includes('relations'));
    assert.ok(tables.includes('usage_log'));
    assert.ok(tables.includes('file_hashes'));
  });

  it('creates FTS5 virtual table', () => {
    const db = getDb(TEST_DB);
    const vtables = db.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%fts5%'"
    ).all().map((r: any) => r.name);
    assert.ok(vtables.includes('chunks_fts'));
  });

  it('enables WAL mode', () => {
    const db = getDb(TEST_DB);
    const mode = db.pragma('journal_mode', { simple: true });
    assert.equal(mode, 'wal');
  });

  it('enables foreign keys', () => {
    const db = getDb(TEST_DB);
    const fk = db.pragma('foreign_keys', { simple: true });
    assert.equal(fk, 1);
  });

  it('returns same instance on second call', () => {
    const db1 = getDb(TEST_DB);
    const db2 = getDb(TEST_DB);
    assert.equal(db1, db2);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx tsc && node --test dist/tests/storage/database.test.js`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

`src/storage/database.ts` with the full SQL schema from the Data Model section. Follow ColdContext's pattern: module-level `db` variable, `getDb(path?)` factory, `closeDb()` cleanup. Include all CREATE TABLE, CREATE INDEX, CREATE TRIGGER statements. Use `db.exec(SCHEMA)` for idempotent initialization.

- [ ] **Step 4: Run test to verify it passes**

Run: `npx tsc && node --test dist/tests/storage/database.test.js`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/storage/database.ts tests/storage/database.test.ts
git commit -m "feat: SQLite schema with FTS5, triggers, and WAL mode"
```

---

## Phase 2 — Parser Engine (Tasks 5-7)

### Task 5: Markdown Parser

**Files:**
- Create: `src/parser/markdown.ts`
- Create: `tests/fixtures/simple.md`
- Create: `tests/fixtures/frontmatter.md`
- Test: `tests/parser/markdown.test.ts`

- [ ] **Step 1: Create test fixtures**

```markdown
<!-- tests/fixtures/simple.md -->
# Project Plan

## Auth System

Fix the login bug. Users report timeout errors.

### Steps

1. Check session middleware
2. Add retry logic

## Database

Migrate to PostgreSQL.
```

```markdown
<!-- tests/fixtures/frontmatter.md -->
---
title: Sprint 12 Notes
tags: [auth, backend, urgent]
priority: high
assignee: makho
---

# Sprint 12

## Goals

- Fix login timeout
- Deploy new auth service

## Notes

This is a critical sprint.
```

- [ ] **Step 2: Write the failing test**

```typescript
// tests/parser/markdown.test.ts
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { parseMarkdown } from '../../src/parser/markdown.js';
import { readFileSync } from 'fs';
import { resolve } from 'path';

const fixture = (name: string) =>
  readFileSync(resolve(import.meta.dirname, '../fixtures', name), 'utf-8');

describe('parseMarkdown', () => {
  it('extracts title from first H1', () => {
    const result = parseMarkdown(fixture('simple.md'));
    assert.equal(result.title, 'Project Plan');
  });

  it('extracts heading tree', () => {
    const result = parseMarkdown(fixture('simple.md'));
    assert.deepEqual(result.headingTree, [
      'Project Plan', 'Auth System', 'Steps', 'Database'
    ]);
  });

  it('extracts sections with heading paths', () => {
    const result = parseMarkdown(fixture('simple.md'));
    assert.ok(result.sections.length >= 3);
    const authSection = result.sections.find(s =>
      s.headingPath.includes('Auth System')
    );
    assert.ok(authSection);
    assert.ok(authSection.content.includes('login bug'));
  });

  it('extracts YAML frontmatter', () => {
    const result = parseMarkdown(fixture('frontmatter.md'));
    assert.equal(result.frontmatter.title, 'Sprint 12 Notes');
    assert.deepEqual(result.frontmatter.tags, ['auth', 'backend', 'urgent']);
    assert.equal(result.frontmatter.priority, 'high');
  });

  it('extracts tags from frontmatter', () => {
    const result = parseMarkdown(fixture('frontmatter.md'));
    assert.deepEqual(result.tags, ['auth', 'backend', 'urgent']);
  });

  it('marks sections containing code blocks', () => {
    const md = '# Title\n\n## Code\n\n```js\nconsole.log("hi")\n```\n\n## Text\n\nNo code here.';
    const result = parseMarkdown(md);
    const codeSection = result.sections.find(s =>
      s.headingPath.includes('Code'));
    assert.equal(codeSection?.hasCodeBlock, true);
    const textSection = result.sections.find(s =>
      s.headingPath.includes('Text'));
    assert.equal(textSection?.hasCodeBlock, false);
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `npx tsc && node --test dist/tests/parser/markdown.test.js`
Expected: FAIL

- [ ] **Step 4: Write implementation**

`src/parser/markdown.ts` using `unified` + `remark-parse` + `remark-frontmatter`:
- Parse markdown to AST (mdast)
- Walk AST to extract headings, sections, code blocks
- Extract YAML frontmatter via `remark-frontmatter` + manual YAML parsing
- Return structured `ParseResult` with title, headingTree, sections[], frontmatter, tags

Key implementation: Walk the AST depth-first. Each heading node starts a new section. Collect all content nodes between headings into the current section. Track heading path as a stack.

- [ ] **Step 5: Run test to verify it passes**

Run: `npx tsc && node --test dist/tests/parser/markdown.test.js`
Expected: 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/parser/markdown.ts tests/parser/markdown.test.ts tests/fixtures/
git commit -m "feat: markdown parser with remark AST, frontmatter, heading extraction"
```

---

### Task 6: Token Counter

**Files:**
- Create: `src/parser/tokens.ts`
- Test: `tests/parser/tokens.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// tests/parser/tokens.test.ts
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { estimateTokens } from '../../src/parser/tokens.js';

describe('estimateTokens', () => {
  it('estimates English text (approx 1.3 tokens per word)', () => {
    const text = 'The quick brown fox jumps over the lazy dog';
    const tokens = estimateTokens(text);
    // 9 words * ~1.3 tokens/word = ~12
    assert.ok(tokens >= 9 && tokens <= 15);
  });

  it('estimates code tokens (higher ratio)', () => {
    const code = 'const result = await fetch("/api/login", { method: "POST" });';
    const tokens = estimateTokens(code);
    assert.ok(tokens > 10);
  });

  it('returns 0 for empty string', () => {
    assert.equal(estimateTokens(''), 0);
  });
});
```

- [ ] **Step 2: Run test, verify failure, implement, verify pass**

Implementation: Simple word-based estimator. Split on whitespace + punctuation, apply 1.3x multiplier (average for English text in BPE tokenizers).

```typescript
// src/parser/tokens.ts
export function estimateTokens(text: string): number {
  if (!text) return 0;
  const words = text.split(/\s+/).filter(w => w.length > 0);
  return Math.ceil(words.length * 1.3);
}
```

- [ ] **Step 3: Commit**

```bash
git add src/parser/tokens.ts tests/parser/tokens.test.ts
git commit -m "feat: token estimator for chunk size control"
```

---

### Task 7: Intelligent Chunker

**Files:**
- Create: `src/parser/chunker.ts`
- Create: `tests/fixtures/complex.md`
- Create: `tests/fixtures/large.md`
- Test: `tests/parser/chunker.test.ts`

- [ ] **Step 1: Create test fixtures**

`tests/fixtures/complex.md` — nested headings (H1 to H2 to H3), code blocks, lists, >1000 tokens total.
`tests/fixtures/large.md` — single section with >2000 tokens (forces paragraph-level splitting).

- [ ] **Step 2: Write the failing test**

```typescript
// tests/parser/chunker.test.ts
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { chunkDocument } from '../../src/parser/chunker.js';
import { parseMarkdown } from '../../src/parser/markdown.js';
import { readFileSync } from 'fs';
import { resolve } from 'path';

const fixture = (name: string) =>
  readFileSync(resolve(import.meta.dirname, '../fixtures', name), 'utf-8');

describe('chunkDocument', () => {
  const defaultOpts = {
    maxTokens: 400, minTokens: 50, overlapTokens: 40,
    splitDepth: 2, preserveCodeBlocks: true,
  };

  it('splits by headings at configured depth', () => {
    const parsed = parseMarkdown(fixture('simple.md'));
    const chunks = chunkDocument(parsed, defaultOpts);
    assert.ok(chunks.length >= 2);
    for (const chunk of chunks) {
      assert.ok(chunk.headingPath.length > 0);
    }
  });

  it('preserves code blocks as atomic units', () => {
    const md = '# Title\n\n## Section\n\n```python\ndef foo():\n    return 42\n```\n\nAfter code.';
    const parsed = parseMarkdown(md);
    const chunks = chunkDocument(parsed, defaultOpts);
    const codeChunk = chunks.find(c => c.content.includes('def foo'));
    assert.ok(codeChunk);
    assert.ok(codeChunk.content.includes('return 42'));
    assert.equal(codeChunk.hasCodeBlock, true);
  });

  it('inherits parent heading context', () => {
    const parsed = parseMarkdown(fixture('simple.md'));
    const chunks = chunkDocument(parsed, defaultOpts);
    const stepsChunk = chunks.find(c => c.headingPath.includes('Steps'));
    assert.ok(stepsChunk?.headingPath.includes('Auth System'));
  });

  it('splits oversized sections at paragraph boundaries', () => {
    const parsed = parseMarkdown(fixture('large.md'));
    const chunks = chunkDocument(parsed, { ...defaultOpts, maxTokens: 200 });
    assert.ok(chunks.length >= 3);
    for (const chunk of chunks) {
      assert.ok(chunk.tokenCount <= 250); // allow some slack
    }
  });

  it('filters out chunks below minTokens', () => {
    const md = '# T\n\n## A\n\nHello\n\n## B\n\nThis is a longer section with enough content to pass the minimum token threshold for chunking purposes.';
    const parsed = parseMarkdown(md);
    const chunks = chunkDocument(parsed, { ...defaultOpts, minTokens: 10 });
    const tinyChunk = chunks.find(c => c.content.trim() === 'Hello');
    assert.equal(tinyChunk, undefined);
  });

  it('assigns sequential chunk indices', () => {
    const parsed = parseMarkdown(fixture('simple.md'));
    const chunks = chunkDocument(parsed, defaultOpts);
    for (let i = 0; i < chunks.length; i++) {
      assert.equal(chunks[i].chunkIndex, i);
    }
  });

  it('strips markdown for contentPlain', () => {
    const md = '# Title\n\n## Section\n\n**Bold** and [link](http://example.com)';
    const parsed = parseMarkdown(md);
    const chunks = chunkDocument(parsed, defaultOpts);
    const chunk = chunks.find(c => c.content.includes('Bold'));
    assert.ok(chunk);
    assert.ok(!chunk.contentPlain.includes('**'));
    assert.ok(chunk.contentPlain.includes('Bold'));
  });
});
```

- [ ] **Step 3: Run test, verify failure**

- [ ] **Step 4: Write implementation**

`src/parser/chunker.ts` — Algorithm:
1. Take `ParseResult.sections` from the markdown parser
2. For each section at depth <= `splitDepth`: create a chunk boundary
3. For sections deeper than `splitDepth`: merge into parent chunk
4. For each candidate chunk:
   a. If tokenCount <= maxTokens: emit as-is
   b. If tokenCount > maxTokens: split at paragraph boundaries (`\n\n`)
   c. If a single paragraph > maxTokens: split at sentence boundaries
   d. If `preserveCodeBlocks` and a code block would be split: keep it atomic
5. For chunks below `minTokens`: merge with previous chunk
6. Generate `contentPlain` by stripping markdown syntax
7. Assign sequential `chunkIndex`

- [ ] **Step 5: Run test, verify pass**

- [ ] **Step 6: Commit**

```bash
git add src/parser/chunker.ts tests/parser/chunker.test.ts tests/fixtures/
git commit -m "feat: heading-aware chunker with code block preservation and paragraph splitting"
```

---

## Phase 3 — Embedding Engine (Tasks 8-9)

### Task 8: Embedding Engine Interface and Factory

**Files:**
- Create: `src/embedding/engine.ts`
- Test: `tests/embedding/engine.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// tests/embedding/engine.test.ts
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createEmbeddingEngine, cosineSimilarity } from '../../src/embedding/engine.js';

describe('EmbeddingEngine', () => {
  it('creates local engine from config', async () => {
    const engine = await createEmbeddingEngine({
      provider: 'local',
      model: 'Xenova/all-MiniLM-L6-v2',
      dimensions: 384,
      modelCachePath: '/tmp/vector-test-models',
      batchSize: 32,
    });
    assert.equal(engine.dimensions, 384);
    assert.equal(engine.modelName, 'Xenova/all-MiniLM-L6-v2');
  });

  it('generates embedding with correct dimensions', async () => {
    const engine = await createEmbeddingEngine({
      provider: 'local',
      model: 'Xenova/all-MiniLM-L6-v2',
      dimensions: 384,
      modelCachePath: '/tmp/vector-test-models',
      batchSize: 32,
    });
    const vector = await engine.embed('Fix the login bug in auth system');
    assert.ok(vector instanceof Float32Array);
    assert.equal(vector.length, 384);
  });

  it('generates batch embeddings', async () => {
    const engine = await createEmbeddingEngine({
      provider: 'local',
      model: 'Xenova/all-MiniLM-L6-v2',
      dimensions: 384,
      modelCachePath: '/tmp/vector-test-models',
      batchSize: 32,
    });
    const vectors = await engine.embedBatch([
      'Login authentication',
      'Database migration',
      'API endpoint design',
    ]);
    assert.equal(vectors.length, 3);
    for (const v of vectors) {
      assert.equal(v.length, 384);
    }
  });

  it('produces similar vectors for similar texts', async () => {
    const engine = await createEmbeddingEngine({
      provider: 'local',
      model: 'Xenova/all-MiniLM-L6-v2',
      dimensions: 384,
      modelCachePath: '/tmp/vector-test-models',
      batchSize: 32,
    });
    const [a, b, c] = await engine.embedBatch([
      'Fix the login authentication bug',
      'Repair the sign-in auth issue',
      'Database schema migration to PostgreSQL',
    ]);
    const simAB = cosineSimilarity(a, b);
    const simAC = cosineSimilarity(a, c);
    assert.ok(simAB > simAC,
      `similar texts (${simAB}) should score higher than different (${simAC})`);
  });
});
```

- [ ] **Step 2: Run test, verify failure**

- [ ] **Step 3: Write implementation**

`src/embedding/engine.ts`:
```typescript
export interface EmbeddingEngine {
  readonly modelName: string;
  readonly dimensions: number;
  embed(text: string): Promise<Float32Array>;
  embedBatch(texts: string[]): Promise<Float32Array[]>;
}

export async function createEmbeddingEngine(
  config: VectorConfig['embedding']
): Promise<EmbeddingEngine> {
  if (config.provider === 'local') {
    const { LocalEmbeddingEngine } = await import('./local.js');
    return LocalEmbeddingEngine.create(config);
  }
  throw new Error(`Unknown embedding provider: ${config.provider}`);
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
```

- [ ] **Step 4: Commit after tests pass**

---

### Task 9: Local Embedding Adapter

**Files:**
- Create: `src/embedding/local.ts`

- [ ] **Step 1: Write implementation**

```typescript
// src/embedding/local.ts
import type { EmbeddingEngine } from './engine.js';
import type { VectorConfig } from '../config.js';

export class LocalEmbeddingEngine implements EmbeddingEngine {
  readonly modelName: string;
  readonly dimensions: number;
  private pipeline: any;

  private constructor(modelName: string, dimensions: number, pipeline: any) {
    this.modelName = modelName;
    this.dimensions = dimensions;
    this.pipeline = pipeline;
  }

  static async create(config: VectorConfig['embedding']): Promise<LocalEmbeddingEngine> {
    const { pipeline, env } = await import('@huggingface/transformers');
    env.cacheDir = config.modelCachePath;
    const pipe = await pipeline('feature-extraction', config.model, {
      dtype: 'fp32',
    });
    return new LocalEmbeddingEngine(config.model, config.dimensions, pipe);
  }

  async embed(text: string): Promise<Float32Array> {
    const output = await this.pipeline(text, {
      pooling: 'mean',
      normalize: true,
    });
    return new Float32Array(output.data);
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    const results: Float32Array[] = [];
    for (const text of texts) {
      results.push(await this.embed(text));
    }
    return results;
  }
}
```

- [ ] **Step 2: Run all embedding tests**

Run: `npx tsc && node --test dist/tests/embedding/engine.test.js`
Expected: 4 tests PASS (note: first run downloads model ~23MB)

- [ ] **Step 3: Commit**

```bash
git add src/embedding/engine.ts src/embedding/local.ts tests/embedding/
git commit -m "feat: local embedding engine with HuggingFace transformers"
```

---

## Phase 4 — Storage and Search (Tasks 10-13)

### Task 10: Document Storage

**Files:**
- Create: `src/storage/documents.ts`
- Test: `tests/storage/documents.test.ts`

- [ ] **Step 1: Write the failing test**

Test CRUD operations: `insertDocument`, `getDocument`, `getDocumentByPath`, `updateDocument`, `deleteDocument`, `listDocuments`. Each test uses a fresh temp DB.

- [ ] **Step 2: Implement**

`src/storage/documents.ts`:
- `insertDocument(db, doc)` — INSERT OR REPLACE into documents table
- `getDocument(db, id)` — SELECT by id, parse JSON columns (frontmatter, tags, heading_tree)
- `getDocumentByPath(db, project, filePath)` — SELECT by UNIQUE(project, file_path)
- `updateDocument(db, id, updates)` — UPDATE specific columns
- `deleteDocument(db, id)` — DELETE (cascades to chunks, embeddings via FK)
- `listDocuments(db, project)` — SELECT all for project

All functions accept `db: Database.Database` as first parameter (dependency injection for testing).

- [ ] **Step 3: Run tests, commit**

---

### Task 11: Chunk Storage with FTS5

**Files:**
- Create: `src/storage/chunks.ts`
- Test: `tests/storage/chunks.test.ts`

- [ ] **Step 1: Write the failing test**

Key test cases:
1. insertChunk stores data and FTS5 trigger fires
2. FTS5 search returns matching chunks with BM25 scores
3. deleteChunk removes from both chunks and FTS5
4. getChunksByDocument returns all chunks for a document
5. updateChunkTrust modifies trust_score

- [ ] **Step 2: Implement**

`src/storage/chunks.ts`:
- `insertChunk(db, chunk)` — INSERT into chunks (triggers auto-insert into chunks_fts)
- `insertChunks(db, chunks)` — Batch insert within a transaction
- `getChunk(db, id)` — SELECT by id
- `getChunksByDocument(db, docId)` — SELECT all chunks for document
- `deleteChunksByDocument(db, docId)` — DELETE all chunks for document
- `searchFTS(db, query, opts)` — FTS5 query with BM25 ranking

FTS5 search query:
```sql
SELECT
  fts.chunk_id,
  rank as bm25_score,
  snippet(chunks_fts, 2, '<b>', '</b>', '...', 32) as snippet
FROM chunks_fts fts
JOIN chunks c ON c.id = fts.chunk_id
WHERE chunks_fts MATCH ?
  AND c.trust_score >= ?
ORDER BY rank
LIMIT ?
```

**Important:** FTS5 MATCH syntax can throw errors on special characters (`"`, `*`, `NEAR`, `OR`, `AND`, `NOT`, unbalanced parentheses). User input must be sanitized before MATCH — either wrap each term in double quotes or strip FTS5 operators. Implement `sanitizeFtsQuery(input: string): string` that handles this.

- [ ] **Step 3: Run tests, commit**

---

### Task 12: Vector Storage

**Files:**
- Create: `src/storage/vectors.ts`
- Test: `tests/storage/vectors.test.ts`

- [ ] **Step 1: Write the failing test**

Key test cases:
1. storeEmbedding saves Float32Array as BLOB
2. getEmbedding retrieves and reconstructs Float32Array
3. getAllEmbeddings returns all vectors for a project
4. deleteEmbedding removes entry
5. BLOB round-trip preserves exact float values

- [ ] **Step 2: Implement**

`src/storage/vectors.ts`:
- `storeEmbedding(db, embedding)` — INSERT with `Buffer.from(vector.buffer)`
- `getEmbedding(db, chunkId)` — SELECT, reconstruct `new Float32Array(blob.buffer)`
- `getProjectEmbeddings(db, project, opts)` — SELECT all embeddings joined with chunks table, filtered by trust
- `deleteEmbedding(db, chunkId)` — DELETE

Critical: SQLite BLOB to Float32Array conversion:
```typescript
// Store: Float32Array to Buffer
const buf = Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength);

// Retrieve: Buffer to Float32Array
const vector = new Float32Array(
  blob.buffer, blob.byteOffset, blob.byteLength / 4
);
```

- [ ] **Step 3: Run tests, commit**

---

### Task 13: Vector Similarity Search

**Files:**
- Modify: `src/storage/vectors.ts` (add `searchSimilar`)
- Test: `tests/storage/vectors.test.ts` (add search tests)

- [ ] **Step 1: Write the failing test**

```typescript
it('finds most similar vectors', () => {
  // Store 5 embeddings with known vectors
  // Query with a vector similar to one of them
  // Verify top result is the expected match
  // Verify results are sorted by similarity descending
});
```

- [ ] **Step 2: Implement**

```typescript
export function searchSimilar(
  db: Database.Database,
  queryVector: Float32Array,
  opts: { project: string; topK?: number; minTrust?: number }
): Array<{ chunkId: string; similarity: number }> {
  const embeddings = getProjectEmbeddings(db, opts.project, {
    minTrust: opts.minTrust ?? 0.4,
  });

  const scored = embeddings.map(({ chunkId, vector }) => ({
    chunkId,
    similarity: cosineSimilarity(queryVector, vector),
  }));

  scored.sort((a, b) => b.similarity - a.similarity);
  return scored.slice(0, opts.topK ?? 10);
}
```

- [ ] **Step 3: Run tests, commit**

---

## Phase 5 — Hybrid Retrieval (Tasks 14-16)

### Task 14: BM25 Search Wrapper

**Files:**
- Create: `src/retrieval/bm25.ts`
- Test: `tests/retrieval/bm25.test.ts`

- [ ] **Step 1: Write the failing test**

Test that `searchBM25(db, query, opts)` returns ranked results with BM25 scores, respects project filter, respects trust threshold, and returns empty for non-matching queries.

- [ ] **Step 2: Implement**

Thin wrapper around `searchFTS` from `storage/chunks.ts`. Normalizes output to `Array<{ chunkId, score, rank }>` format needed by fusion.

Must include `sanitizeFtsQuery(input)` function that:
- Wraps each whitespace-separated term in double quotes
- Strips FTS5 operators (OR, AND, NOT, NEAR) from user input
- Escapes unbalanced parentheses and asterisks
- Example: `fix "login" bug OR error` becomes `"fix" "login" "bug" "error"`

- [ ] **Step 3: Run tests, commit**

---

### Task 15: Vector Search Wrapper

**Files:**
- Create: `src/retrieval/vector-search.ts`
- Test: `tests/retrieval/vector-search.test.ts`

- [ ] **Step 1: Write the failing test**

Test that `searchVector(db, engine, query, opts)`:
1. Embeds the query text
2. Calls `searchSimilar` with the query vector
3. Returns results as `Array<{ chunkId, score, rank }>`

- [ ] **Step 2: Implement**

```typescript
export async function searchVector(
  db: Database.Database,
  engine: EmbeddingEngine,
  query: string,
  opts: { project: string; topK?: number; minTrust?: number }
): Promise<Array<{ chunkId: string; score: number; rank: number }>> {
  const queryVector = await engine.embed(query);
  const results = searchSimilar(db, queryVector, opts);
  return results.map((r, i) => ({
    chunkId: r.chunkId,
    score: r.similarity,
    rank: i + 1,
  }));
}
```

- [ ] **Step 3: Run tests, commit**

---

### Task 16: Reciprocal Rank Fusion and Hybrid Engine

**Files:**
- Create: `src/retrieval/fusion.ts`
- Create: `src/retrieval/engine.ts`
- Test: `tests/retrieval/fusion.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// tests/retrieval/fusion.test.ts
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { fuseResults } from '../../src/retrieval/fusion.js';

describe('fuseResults (RRF)', () => {
  it('combines two result lists with RRF', () => {
    const bm25 = [
      { chunkId: 'a', score: 5.0, rank: 1 },
      { chunkId: 'b', score: 3.0, rank: 2 },
      { chunkId: 'c', score: 1.0, rank: 3 },
    ];
    const vector = [
      { chunkId: 'b', score: 0.95, rank: 1 },
      { chunkId: 'a', score: 0.80, rank: 2 },
      { chunkId: 'd', score: 0.70, rank: 3 },
    ];
    const fused = fuseResults([bm25, vector], { k: 60 });

    // Both a and b appear in both lists with mirrored ranks
    const topTwo = fused.slice(0, 2).map(r => r.chunkId).sort();
    assert.deepEqual(topTwo, ['a', 'b']);

    // 'd' appears only in vector, lower score
    const dResult = fused.find(r => r.chunkId === 'd');
    assert.ok(dResult);
    assert.ok(dResult.score < fused[0].score);
  });

  it('applies trust weighting', () => {
    const list = [
      { chunkId: 'high-trust', score: 1.0, rank: 1 },
      { chunkId: 'low-trust', score: 0.9, rank: 2 },
    ];
    const trustScores = { 'high-trust': 0.95, 'low-trust': 0.3 };
    const fused = fuseResults([list], {
      k: 60, trustScores, trustWeight: 0.3
    });
    assert.equal(fused[0].chunkId, 'high-trust');
    assert.ok(fused[0].score > fused[1].score);
  });
});
```

- [ ] **Step 2: Implement fusion**

```typescript
// src/retrieval/fusion.ts
interface RankedResult {
  chunkId: string;
  score: number;
  rank: number;
}

interface FusionOpts {
  k?: number;              // RRF k parameter (default: 60)
  trustScores?: Record<string, number>;
  trustWeight?: number;    // 0-1 (default: 0.3)
}

export function fuseResults(
  resultLists: RankedResult[][],
  opts: FusionOpts = {}
): Array<{ chunkId: string; score: number }> {
  const k = opts.k ?? 60;
  const trustWeight = opts.trustWeight ?? 0.3;
  const scores = new Map<string, number>();

  for (const list of resultLists) {
    for (const result of list) {
      const rrfScore = 1 / (k + result.rank);
      scores.set(
        result.chunkId,
        (scores.get(result.chunkId) ?? 0) + rrfScore
      );
    }
  }

  if (opts.trustScores) {
    for (const [chunkId, rrfScore] of scores) {
      const trust = opts.trustScores[chunkId] ?? 0.5;
      const weighted = rrfScore * (1 - trustWeight + trustWeight * trust);
      scores.set(chunkId, weighted);
    }
  }

  return [...scores.entries()]
    .map(([chunkId, score]) => ({ chunkId, score }))
    .sort((a, b) => b.score - a.score);
}
```

- [ ] **Step 3: Implement hybrid engine**

`src/retrieval/engine.ts` — orchestrates the full search pipeline:

```typescript
export async function hybridSearch(
  db: Database.Database,
  embeddingEngine: EmbeddingEngine,
  query: string,
  opts: SearchOptions
): Promise<SearchResult[]> {
  // 1. Run BM25 and vector search in parallel
  const [bm25Results, vectorResults] = await Promise.all([
    searchBM25(db, query, opts),
    searchVector(db, embeddingEngine, query, opts),
  ]);

  // 2. Collect trust scores for all result chunks
  const allChunkIds = new Set([
    ...bm25Results.map(r => r.chunkId),
    ...vectorResults.map(r => r.chunkId),
  ]);
  const trustScores = getTrustScores(db, [...allChunkIds]);

  // 3. Fuse with RRF + trust weighting
  const fused = fuseResults([bm25Results, vectorResults], {
    k: opts.rrfK,
    trustScores,
    trustWeight: opts.trustWeight,
  });

  // 4. Fetch full chunk data for top results
  const topResults = fused.slice(0, opts.topK);
  return enrichResults(db, topResults, bm25Results, vectorResults);
}
```

- [ ] **Step 4: Run tests, commit**

---

## Phase 6 — Trust, Lifecycle and Relations (Tasks 17-19)

### Task 17: Trust Scoring Engine

**Files:**
- Create: `src/engine/trust.ts`
- Test: `tests/engine/trust.test.ts`

- [ ] **Step 1: Write the failing test**

Adapt ColdContext's trust tests. Test all events: TASK_SUCCESS (+0.05), TASK_FAILURE (-0.15), FILE_CHANGE, DECAY. Test clamping to [0.0, 1.0]. Test labels: fresh/trusted/suspect/deprecated.

- [ ] **Step 2: Implement**

Direct adaptation of ColdContext's `src/engine/trust.ts`:

```typescript
const TRUST_EVENTS = {
  TASK_SUCCESS:           0.05,
  TASK_PARTIAL_SUCCESS:   0.02,
  FRESHNESS_PASS:         0.01,
  TASK_FAILURE:          -0.15,
  FILE_SMALL_CHANGE:     -0.05,
  FILE_MEDIUM_CHANGE:    -0.10,
  FILE_DELETED:          -0.10,
  DOCUMENT_REINDEXED:    -0.08,  // chunk structure changed on re-index
  EMBEDDING_REGENERATED: -0.03,  // vectors regenerated (model change etc.)
  DAILY_UNUSED_DECAY:    -0.02,
} as const;

const TRUST_ABSOLUTE: Record<string, number> = {
  FILE_LARGE_CHANGE:      0.30,
  USER_DEPRECATE:         0.00,
  USER_RESTORE:           0.50,
  MANUAL_REFRESH:         1.00,
};

export function updateTrust(current: number, event: TrustEvent): number {
  if (event in TRUST_ABSOLUTE) return TRUST_ABSOLUTE[event]!;
  const delta = (TRUST_EVENTS as Record<string, number>)[event];
  if (delta === undefined) return current;
  return Math.max(0.0, Math.min(1.0, current + delta));
}

export function getTrustLabel(score: number): string {
  if (score >= 0.8) return 'fresh';
  if (score >= 0.4) return 'trusted';
  if (score >= 0.2) return 'suspect';
  return 'deprecated';
}
```

- [ ] **Step 3: Run tests, commit**

---

### Task 18: Hash Engine and Freshness Detection

**Files:**
- Create: `src/engine/hash.ts`
- Create: `src/engine/lifecycle.ts`
- Test: `tests/engine/hash.test.ts`

- [ ] **Step 1: Implement hash functions**

Adapt ColdContext's `hashFile`, `hashString`, `computeCompositeHash`, `generateId`. Change ID prefix from `cu_` to `doc_` / `chk_`.

- [ ] **Step 2: Implement freshness detection**

`src/engine/lifecycle.ts`:
- `checkFreshness(db, project, filePath)` — compare stored hash vs current file hash
- `detectChangedFiles(db, project, dirPath)` — scan directory, compare all hashes, return changed/added/deleted lists
- `applyDecay(db, project)` — find chunks unused for > decayStartDays, apply daily decay

- [ ] **Step 3: Run tests, commit**

---

### Task 19: Relation Graph

**Files:**
- Create: `src/graph/relations.ts`
- Test: `tests/graph/relations.test.ts`

- [ ] **Step 1: Write the failing test**

Test: `addRelation`, `removeRelation`, `getRelations`, `findCascade` (BFS), link extraction from markdown (`[[wiki-links]]` and `[text](file.md)` patterns).

- [ ] **Step 2: Implement**

`src/graph/relations.ts`:
- `addRelation(db, rel)` — INSERT OR REPLACE into relations table
- `removeRelation(db, sourceId, targetId, relationType?)` — DELETE by composite key. If relationType omitted, removes all relations between the pair.
- `getRelations(db, id)` — SELECT all relations where source or target = id
- `extractLinks(content, currentFile)` — regex to find `[[...]]` and `[...](*.md)` patterns, resolve relative paths
- `findCascade(db, sourceId, maxDepth)` — BFS traversal of relation graph, returns all affected IDs with distance

- [ ] **Step 3: Run tests, commit**

---

## Phase 7 — CLI Interface (Tasks 20-22)

### Task 20: CLI Framework + `vector init`

**Files:**
- Create: `src/cli/program.ts`
- Create: `src/cli/init.ts`
- Modify: `src/index.ts`

- [ ] **Step 1: Write failing test**

Test that `vector init --project=test-project --path=/tmp/test` creates `~/.vector/` directory and initializes DB.

- [ ] **Step 2: Implement**

`src/cli/program.ts` — Commander setup:
```typescript
import { Command } from 'commander';

export function createProgram(): Command {
  const program = new Command();
  program
    .name('vector')
    .description('Developer Memory Engine -- .md to hybrid AI memory')
    .version('0.1.0');
  return program;
}
```

`src/cli/init.ts` — Initialize project (create dirs, init DB, store project metadata).

Update `src/index.ts` to dispatch CLI commands vs MCP server (same pattern as ColdContext).

- [ ] **Step 3: Run tests, commit**

---

### Task 21: `vector index` — Full Indexing Pipeline

**Files:**
- Create: `src/cli/index-cmd.ts`

- [ ] **Step 1: Write failing test**

Test end-to-end: given a directory with .md files, `vector index` should:
1. Discover all .md files
2. Hash each file, skip unchanged
3. Parse markdown into sections
4. Chunk sections
5. Generate embeddings
6. Store documents, chunks, embeddings, FTS5
7. Extract and store relations (links between files)
8. Print summary (files indexed, chunks created, time elapsed)

- [ ] **Step 2: Implement**

`src/cli/index-cmd.ts`:
```typescript
export function registerIndex(program: Command): void {
  program
    .command('index')
    .description('Index .md files into Vector memory')
    .requiredOption('--project <name>', 'Project name')
    .option('--path <dir>', 'Directory to index', '.')
    .option('--pattern <glob>', 'File pattern', '**/*.md')
    .option('--exclude <dirs>', 'Dirs to exclude', 'node_modules,.git,dist')
    .option('--force', 'Re-index all files (ignore hashes)', false)
    .action(async (opts) => {
      // Full pipeline: discover -> hash -> parse -> chunk -> embed -> store
    });
}
```

- [ ] **Step 3: Run tests, commit**

---

### Task 22: `vector query` and `vector status`

**Files:**
- Create: `src/cli/query.ts`
- Create: `src/cli/status.ts`

- [ ] **Step 1: Write failing test**

Test that `vector query "fix login bug" --project=test` returns formatted results.

- [ ] **Step 2: Implement**

`vector query`:
```
vector query "fix login bug" --project=myapp --top=5 --mode=hybrid --format=text
```
Output:
```
1. [0.0325] Sprint 12 Notes > Auth System > Login
   Trust: 0.92 | BM25: 4.21 | Vector: 0.87
   Fix the login bug. Users report timeout errors...

2. [0.0298] Project Plan > Auth System > Steps
   Trust: 0.85 | BM25: 2.10 | Vector: 0.82
   Check session middleware. Add retry logic...
```

`vector status`:
```
vector status --project=myapp
```
Output: document count, chunk count, index freshness, trust distribution, total tokens.

- [ ] **Step 3: Run tests, commit**

---

## Phase 8 — MCP Server (Tasks 23-24)

### Task 23: MCP Server Setup

**Files:**
- Create: `src/mcp/server.ts`

- [ ] **Step 1: Implement MCP server**

Follow ColdContext's pattern. Create `McpServer` with tool registration.

```typescript
const INSTRUCTIONS = `Vector Memory Engine -- hybrid search over .md memory files.

WORKFLOW:
1. search_memory(project, query) -- find relevant memory chunks
2. get_chunk(project, chunkId) -- retrieve specific chunk
3. list_documents(project) -- see all indexed .md files
4. index_memory(project, path) -- index new or changed .md files
5. check_freshness(project) -- verify index is current
6. report_usage(chunkId, taskResult) -- feed trust scoring

SEARCH MODES: hybrid (BM25 + vector), bm25 (keyword only), vector (semantic only)
TRUST: >=0.8 fresh | >=0.4 trusted | <0.4 suspect`;
```

- [ ] **Step 2: Update entry point**

Update `src/index.ts` to start MCP server when no CLI command is given.

- [ ] **Step 3: Commit**

---

### Task 24: MCP Tool Registration

**Files:**
- Create: `src/mcp/tools/store.ts`
- Create: `src/mcp/tools/query.ts`
- Create: `src/mcp/tools/validate.ts`
- Create: `src/mcp/tools/admin.ts`

- [ ] **Step 1: Implement MCP tools**

10 core tools:

| Tool | Description |
|------|-------------|
| `search_memory` | Hybrid search (BM25 + vector + trust). Core tool. |
| `get_chunk` | Retrieve specific chunk by ID. Returns json/prompt/dual. |
| `get_document` | Retrieve document metadata + chunk list. |
| `list_documents` | List all indexed .md files for a project. |
| `index_memory` | Index/re-index .md files from a directory. |
| `check_freshness` | Compare file hashes to detect changes. |
| `report_usage` | Record task result, update trust scores. |
| `add_relation` | Create edge between documents/chunks. |
| `stats` | Memory statistics: counts, sizes, trust distribution. |
| `gc` | Garbage collect low-trust/old chunks. |

Each tool follows MCP return format: `{ content: [{ type: 'text', text: JSON.stringify(result) }] }`.

- [ ] **Step 2: Verify MCP server starts and lists tools**

Run: `echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | node dist/index.js`
Expected: JSON response listing all 10 tools with schemas.

- [ ] **Step 3: Commit**

---

## Phase 9 — LLM Context Injection (Task 25)

### Task 25: Token-Optimized Prompt Compilation

**Files:**
- Create: `src/engine/prompt.ts`
- Test: `tests/engine/prompt.test.ts`

- [ ] **Step 1: Write the failing test**

Test that `compileChunkPrompt(searchResult)` produces compact output:
```
[MEM:auth-system/login-fix] Fix login timeout bug | priority:high [trust:0.92]
PATH: notes/sprint-12.md > Auth System > Login > Fix bug
TAGS: auth,backend,urgent
---
Check session middleware for timeout. Add retry logic with exponential backoff.
```

- [ ] **Step 2: Implement**

Adapt ColdContext's `compilePrompt` for chunk-based output:
- `[MEM:...]` prefix (vs ColdContext's `[CTX:...]`)
- `PATH:` shows heading breadcrumb
- `TAGS:` from chunk metadata
- Raw content follows separator

Also implement `compileSearchResults(results)` — batches multiple chunks into a single prompt with deduplication and token budget awareness.

- [ ] **Step 3: Run tests, commit**

---

## Testing Strategy

### Unit Tests (per-module)
- **Parser:** Diverse .md fixtures (frontmatter, nested headings, code blocks, edge cases)
- **Chunker:** Boundary conditions (max/min tokens, empty sections, single-line sections)
- **Embedding:** Dimensionality, similarity properties, batch consistency
- **Storage:** CRUD, FTS5 sync, BLOB round-trip, transaction isolation
- **Retrieval:** BM25 ranking, vector ranking, RRF fusion math, trust weighting
- **Trust:** All events, clamping, labels

### Integration Test
`tests/integration/full-pipeline.test.ts`:
1. Create temp directory with 5 .md files (various complexity)
2. Run full indexing pipeline
3. Query and verify relevant results rank higher
4. Modify one file, re-index, verify incremental
5. Report usage, verify trust updates
6. Run GC, verify cleanup

### Test Fixtures
- `simple.md` — basic headings and text
- `frontmatter.md` — YAML frontmatter with tags/priority
- `complex.md` — nested H1 to H2 to H3 to H4, code blocks, lists, tables
- `large.md` — >2000 tokens in single section (tests splitting)
- `linked-a.md` / `linked-b.md` — cross-referencing files (tests relation extraction)

### Edge Case Tests (Critical)
- **Chunker:** Empty markdown file, file with only frontmatter, file with no headings (just paragraphs), single massive code block exceeding maxTokens, markdown with only H1 headings when splitDepth=2, Unicode/CJK content
- **Embedding:** Empty string input, very long string (>256 tokens, exceeds MiniLM context window — verify graceful truncation), batch with mix of empty and valid strings
- **FTS5:** Verify triggers fire on chunk insert/update/delete, query with FTS5 special characters, empty query string
- **Database:** FTS5 trigger verification (insert chunk then query chunks_fts to confirm data), concurrent read during write (SQLITE_BUSY handling)
- **Relations:** Dangling relation IDs (target deleted), self-referential relations, duplicate relation types between same pair

### Quality Metrics
- **Retrieval quality:** For each test fixture, define "ground truth" relevant chunks for known queries. Assert that precision@5 >= 0.6 and that the expected chunk appears in top 3.
- **Chunking quality:** Assert no chunk exceeds maxTokens by >20%. Assert no chunk below minTokens (except final chunk).

---

## Security Model

1. **API keys** — If OpenAI provider is configured, key is read from `OPENAI_API_KEY` env var. Never stored in config file. Never logged.
2. **File access** — Only reads files matching `--pattern` in specified `--path`. No traversal above project root.
3. **SQLite** — Database file permissions: user-only (0600). Created in `~/.vector/` by default.
4. **No network** — Local provider makes zero network calls after model download. All data stays local.
5. **MCP isolation** — MCP tools only access Vector's own database. No filesystem write access exposed via MCP.
6. **Input sanitization** — All user inputs pass through Zod validation before reaching SQLite. FTS5 queries are parameterized (no SQL injection).

---

## Performance Considerations

| Operation | Expected Latency | Bottleneck | Mitigation |
|-----------|-----------------|------------|------------|
| Index 100 .md files | ~30s | Embedding generation | Batch embedding, skip unchanged files |
| Index 1 changed file | ~500ms | Single embed batch | Incremental by design |
| Hybrid search | <200ms | Vector scan (brute-force) | Acceptable for <50K chunks |
| BM25 search alone | <10ms | FTS5 is very fast | Built-in optimization |
| Vector search alone | <50ms | Cosine sim over all chunks | Consider HNSW at >100K |
| CLI startup | ~300ms | Model loading | Lazy load: only init embedding engine when needed |
| MCP tool call | <100ms (search) | Same as search | Cached engine instance |

### Memory Budget
- 50K chunks x 384 dims x 4 bytes = ~74MB for all vectors
- SQLite WAL mode keeps memory stable
- Embedding model: ~100MB in memory when loaded

### Optimization Opportunities (Post-MVP)
- Vector quantization (Float32 to Int8) reduces memory 4x
- HNSW index (hnswlib-node) reduces search from O(n) to O(log n)
- Embedding cache (LRU) for repeated queries
- SQLite FTS5 `optimize` command for index compaction

---

## Summary: What Makes This Professional-Grade

1. **ADR-documented decisions** — every tech choice has a "why" and "trade-off"
2. **Concrete data model** — exact TypeScript interfaces and SQL schema, not hand-waving
3. **Interface-driven design** — `EmbeddingEngine` interface enables provider swap without code changes
4. **Proven patterns** — trust scoring, relation graphs, lifecycle management from ColdContext
5. **Hybrid retrieval with RRF** — industry-standard approach (Elasticsearch, Pinecone, Weaviate)
6. **Incremental by design** — SHA256 hashing avoids redundant work
7. **Zero external services** — SQLite unified storage, local embeddings
8. **TDD throughout** — every module has tests before implementation
9. **Security-conscious** — parameterized queries, file permissions, no credential storage
10. **Performance-aware** — latency targets, memory budgets, upgrade path documented
