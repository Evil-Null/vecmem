# Vector Memory Engine — Elite Design Spec (Level 3)

> "Your notes become AI's memory"

**Date:** 2026-03-24
**Status:** Approved
**Approach:** Distinguished/Fellow-level engineering — branded types, system invariants, property-based testing

---

## 1. Product Definition

**What:** CLI tool + MCP server that transforms `.md` files into a hybrid AI memory system with vector search, full-text search, and semantic chunking.

**For:** Broad audience — developers, semi-technical users, anyone with markdown notes and AI tools.

**Interface:** CLI only. No GUI, no web UI.

**Value proposition:** "Your notes become AI's memory" — write markdown as you always do, Vector turns it into AI-accessible memory automatically.

**Distribution:** Open-source (MIT), npm package. User installs locally, everything runs on their machine. No cloud server required.

---

## 2. Core Abstractions

### Entities

```
Document  →  one .md file
    │
    ├── Chunk  →  semantic unit (heading-split)
    │     │
    │     └── Embedding  →  vector representation (384 floats)
    │
    └── Metadata  →  frontmatter, tags, title
```

**4 entities. Nothing more in v1.** No trust, no relations, no usage_log.

### Invariants — always true

1. Every Chunk belongs to exactly one Document. Document deleted → its chunks deleted.
2. Every Chunk has exactly one Embedding. Chunk cannot exist without embedding.
3. Document is identified by `(project, filePath)`. Two files with same path in same project — impossible.
4. Embedding dimension is fixed within a project. If 384-dim, all are 384-dim.
5. Search score is in [0, 1]. Always. No exceptions. Raw RRF scores are normalized to [0,1] via min-max normalization per result set.
6. Indexing is idempotent. Run twice → identical state.
7. Same input → same output. Deterministic IDs (content-based SHA256). Embedding determinism is a tested property — verified but not absolutely guaranteed across ONNX runtime versions.

---

## 3. Type System — "Invalid states are unrepresentable"

### Branded Types

```typescript
type DocumentId = string & { readonly __brand: 'DocumentId' }
type ChunkId = string & { readonly __brand: 'ChunkId' }
type ProjectId = string & { readonly __brand: 'ProjectId' }
type UnitScore = number & { readonly __brand: 'UnitScore' }
```

DocumentId cannot be passed where ChunkId is expected. Compiler stops you.

### Score — physically bounded to [0,1]

```typescript
function unitScore(n: number): UnitScore {
  if (n < 0 || n > 1) {
    throw new InvariantViolation(`Score ${n} outside [0,1]`)
  }
  return n as UnitScore
}
```

### Two-phase Chunk types

```typescript
// From Parser — no embedding yet
interface RawChunk {
  readonly content: string
  readonly contentPlain: string
  readonly headingPath: readonly string[]
  readonly index: number
  readonly hasCodeBlock: boolean
}

// In Store — embedding required
interface IndexedChunk {
  readonly id: ChunkId
  readonly documentId: DocumentId
  readonly content: string
  readonly contentPlain: string
  readonly headingPath: readonly string[]
  readonly index: number
  readonly hasCodeBlock: boolean
  readonly embedding: Float32Array  // required, not optional
}
```

Pipeline: `Parser → RawChunk[] → Embedder → IndexedChunk[] → Store`

"Chunk without embedding in store" — impossible at type level.

### Immutable everywhere

```typescript
interface DocumentMeta {
  readonly title: string
  readonly filePath: string
  readonly project: ProjectId
  readonly contentHash: string
  readonly tags: readonly string[]
  readonly frontmatter: Readonly<Record<string, unknown>>
  readonly indexedAt: Date
  readonly fileSize: number
}
```

---

## 4. Interface Contracts

### Parser

```typescript
interface Parser {
  parse(filePath: string): ParseResult
}

interface ParseResult {
  document: DocumentMeta
  chunks: RawChunk[]
}
```

Knows: markdown syntax, heading hierarchy, code block boundaries. Computes `contentHash` (SHA256 of raw file bytes) and sets it on `DocumentMeta` — this is the single source of truth for content hashing. Path validation (`validateFilePath()`) happens in the orchestrator before calling `parser.parse()`.

Does not know: embeddings, database, search.

### Embedder

```typescript
interface Embedder {
  embed(text: string): Promise<Float32Array>
  embedBatch(texts: string[]): Promise<Float32Array[]>
  readonly dimensions: number
}
```

3 members. Text in, Float32Array out. Embedder owns the conversion to Float32Array — consumers never deal with raw number[].

### Store

```typescript
// Explicit pairing — chunks[i] and embeddings[i] are always matched
interface ChunkWithEmbedding {
  readonly chunk: RawChunk
  readonly embedding: Float32Array
}

interface Store {
  save(doc: DocumentMeta, items: ChunkWithEmbedding[]): void
  getDocument(project: ProjectId, filePath: string): StoredDocument | null
  getChunks(documentId: DocumentId): IndexedChunk[]
  removeDocument(documentId: DocumentId): void
  needsReindex(project: ProjectId, filePath: string, contentHash: string): boolean
}
```

`save` is atomic. Document + chunks + embeddings in one transaction. Either all are written, or none. The `ChunkWithEmbedding` type makes chunk-embedding pairing explicit — no implicit array index coupling.

### Search

```typescript
interface Search {
  query(text: string, options?: SearchOptions): Promise<SearchResult[]>
}

interface SearchOptions {
  topK?: number        // default: 10
  minScore?: number    // default: 0.1
  project?: ProjectId  // filter by project
}

interface SearchResult {
  chunk: IndexedChunk
  score: UnitScore           // normalized fused score ∈ [0,1] — THE ranking score
  scores: {
    bm25Rank: number         // rank position in BM25 results (1-based)
    vectorRank: number       // rank position in vector results (1-based)
    rrfRaw: number           // raw RRF score before normalization
  }
  documentTitle: string
  documentPath: string
  highlight: string
}
```

One method: `query`. Text in, results out. BM25 + vector + RRF fusion happens inside.

### Properties of all interfaces

- Each interface is testable in isolation
- Each interface is replaceable (LocalEmbedder / OpenAIEmbedder)
- Error boundaries are clear per interface
- Data flows one direction: Parser → Embedder → Store → Search
- No circular dependencies

---

## 5. Data Flow

### Pipeline 1: Index (Write)

```
.md file
    │
    ▼
  Parser.parse(filePath)
    │
    ├── DocumentMeta
    └── RawChunk[]
            │
            ▼
      Embedder.embedBatch(chunks.map(c => c.contentPlain))
            │
            └── number[][]
                    │
                    ▼
              Store.save(doc, chunks, embeddings)  ← one transaction
                    │
                    ✓ done
```

Orchestrator:

```typescript
async function indexFile(
  filePath: string,
  parser: Parser,
  embedder: Embedder,
  store: Store
): Promise<IndexResult> {
  const validPath = validateFilePath(filePath)
  const { document, chunks } = parser.parse(validPath)

  // contentHash is computed by parser — single source of truth
  if (!store.needsReindex(document.project, filePath, document.contentHash)) {
    return { status: 'skipped', reason: 'unchanged' }
  }

  const texts = chunks.map(c => c.contentPlain)
  const embeddings = await embedder.embedBatch(texts)

  const items = chunks.map((chunk, i) => ({ chunk, embedding: embeddings[i] }))
  store.save(document, items)

  return { status: 'indexed', chunks: chunks.length }
}
```

### Pipeline 2: Search (Read)

```
query string
        │
        ▼
  Embedder.embed(query)
        │
        ▼
  ┌─────┴─────┐
  │            │
  BM25         Vector
  (FTS5)       (cosine)
  │            │
  rank[]       rank[]
  └─────┬─────┘
        │
        ▼
    RRF Fusion: score(chunk) = 1/(k + bm25_rank) + 1/(k + vector_rank)
        │
        ▼
  SearchResult[] (sorted by fused score)
```

Two paths in parallel:
- **BM25** — keyword search via FTS5 index. Fast, exact terms.
- **Vector** — semantic search via cosine similarity. "auth" finds "authentication".
- **RRF** — Reciprocal Rank Fusion. Uses ranks not scores (scales are incomparable).

### Graceful Degradation

If embedding model fails to load, search falls back to BM25-only mode:

```typescript
class SearchOrchestrator implements Search {
  async query(text: string, options?: SearchOptions): Promise<SearchResult[]> {
    const bm25Results = this.bm25.search(text)

    let vectorResults: RankedResult[] = []
    try {
      const embedding = await this.embedder.embed(text)
      vectorResults = this.vectorSearch.search(embedding)
    } catch (e) {
      if (e instanceof ModelLoadError) {
        logger.warn('search.degraded', {
          reason: 'model_unavailable',
          fallback: 'bm25_only'
        })
        return this.fuseResults(bm25Results, [], options)
      }
      throw e
    }

    return this.fuseResults(bm25Results, vectorResults, options)
  }
}
```

---

## 6. Error Architecture

### Error Hierarchy

```typescript
abstract class VectorError extends Error {
  abstract readonly code: string
  abstract readonly recoverable: boolean
}

// Parser errors
class FileNotFoundError extends VectorError {
  code = 'PARSE_FILE_NOT_FOUND' as const
  recoverable = false
}
class InvalidMarkdownError extends VectorError {
  code = 'PARSE_INVALID_MARKDOWN' as const
  recoverable = false
}

// Embedder errors
class ModelLoadError extends VectorError {
  code = 'EMBED_MODEL_LOAD' as const
  recoverable = true
}
class InputTooLongError extends VectorError {
  code = 'EMBED_INPUT_TOO_LONG' as const
  recoverable = false
}

// Store errors
class DatabaseCorruptedError extends VectorError {
  code = 'STORE_CORRUPTED' as const
  recoverable = false
}
class TransactionFailedError extends VectorError {
  code = 'STORE_TRANSACTION_FAILED' as const
  recoverable = true
}
```

`recoverable` field: CLI automatically decides — retry or report and continue.

No error is silently swallowed. No error crashes the entire system.

---

## 7. Invariant System

### Three levels

1. **Compile-time** — TypeScript branded types, readonly, required fields
2. **Construction-time** — unitScore(), chunkId() validate on creation
3. **System-time** — after every write operation (dev mode only)

### System Invariants

```typescript
const SYSTEM_INVARIANTS = {
  everyChunkHasEmbedding: (store) => {
    // chunks LEFT JOIN embeddings WHERE embedding IS NULL → must be 0
  },
  chunkCountsMatch: (store) => {
    // document.chunk_count == COUNT(actual chunks) for every document
  },
  uniformDimensions: (store) => {
    // SELECT DISTINCT dimensions FROM embeddings → must be exactly 1
  },
  ftsInSync: (store) => {
    // chunks table IDs == chunks_fts table IDs
  },
  noOrphanChunks: (store) => {
    // no chunks whose document_id points to a deleted document
  }
}
```

Dev/test mode: checked after every write. Production: disabled for performance.

---

## 8. Testing Strategy — Level 3

### Testing Pyramid

```
         ╱╲
        ╱  ╲         Property tests (proves properties for ALL inputs)
       ╱    ╲         fast-check, 10K+ inputs per property
      ╱──────╲
     ╱        ╲       Invariant checks (every write in dev mode)
    ╱          ╲       system self-verification
   ╱────────────╲
  ╱              ╲    Unit tests (specific cases, edge cases)
 ╱                ╲    regression tests
╱──────────────────╲
     Integration       End-to-end pipeline test
```

### Core Properties (fast-check)

1. **Indexing is idempotent** — index twice → identical state
2. **Search results always sorted descending** — for any query
3. **Document deletion cascades completely** — no orphan chunks, embeddings, FTS entries
4. **Cosine similarity in [-1, 1]** — for any two vectors
5. **RRF score > 0** — always positive
6. **Parsed chunks reconstruct original content** — for any markdown
7. **Same input → same embedding** — determinism
8. **Same file → same chunk IDs** — content-based, not random

### Performance Contracts (CI-enforced)

```typescript
test('search < 100ms for 10K chunks', ...)
test('indexing throughput > 100 chunks/sec', ...)
test('embedding batch < 5ms/chunk', ...)
```

CI fails if performance regresses. Speed is a contract, not a hope.

---

## 9. Observability

### Structured Logging

```typescript
interface Logger {
  debug(event: string, data?: Record<string, unknown>): void
  info(event: string, data?: Record<string, unknown>): void
  warn(event: string, data?: Record<string, unknown>): void
  error(event: string, error: VectorError): void
}
```

Events: file.indexed, file.skipped, search.completed, search.degraded, invariant.violated

`vector --verbose` shows all. Normal mode — silent.

### Doctor Command

```bash
$ vector doctor
  ✓ Database: OK (4.2 MB, 312 chunks)
  ✓ FTS index: in sync
  ✓ Embeddings: all present (312/312)
  ✓ Model: loaded (all-MiniLM-L6-v2, 23 MB)
  ✓ Invariants: all passing (5/5)
  ⚠ Stale files: 2 changed since last index
```

---

## 10. Security Boundaries

```typescript
// Path traversal protection
function validateFilePath(path: string): string {
  const resolved = resolve(path)
  if (!resolved.startsWith(projectRoot)) {
    throw new SecurityError(`Path traversal: ${path}`)
  }
  return resolved
}

// FTS5 query sanitization — preserve useful operators (* for prefix, " for phrase)
// Only escape characters that cause FTS5 syntax errors or enable injection
function sanitizeFtsQuery(query: string): string {
  // Remove unbalanced quotes, escape parens (used for grouping/NEAR)
  const balanced = balanceQuotes(query)
  return balanced.replace(/[()]/g, ' ').trim()
}

// Chunk size limit
function validateChunk(chunk: RawChunk): void {
  if (chunk.content.length > 50_000) {
    throw new InputTooLongError(`Chunk too large: ${chunk.content.length} chars`)
  }
}
```

---

## 11. Determinism

### Content-based IDs

```typescript
function chunkId(documentId: DocumentId, index: number): ChunkId {
  return createHash('sha256')
    .update(`${documentId}:${index}`)
    .digest('hex')
    .slice(0, 16) as ChunkId
}
```

Same document + same chunk index = same ID. Always. Idempotent indexing depends on this.

### Embedding determinism

Tested via property: embed(text) deep-equals embed(text) for any text within the same process and ONNX runtime version. If model is non-deterministic, test fails and we know. Note: small floating-point variations may occur across CPU architectures or ONNX runtime updates. Content-based IDs depend on documentId + chunk index (not embedding values), so this does not break idempotent indexing.

---

## 12. SQLite Schema (v1) and Migration

### v1 Schema DDL

```sql
-- Database settings
PRAGMA journal_mode = WAL;          -- concurrent reads + single writer
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

-- Documents: one row per .md file
CREATE TABLE documents (
  id           TEXT PRIMARY KEY,     -- DocumentId (SHA256 of project:filePath)
  project      TEXT NOT NULL,
  file_path    TEXT NOT NULL,
  title        TEXT NOT NULL DEFAULT '',
  content_hash TEXT NOT NULL,         -- SHA256 of file content
  file_size    INTEGER NOT NULL DEFAULT 0,
  tags         TEXT NOT NULL DEFAULT '[]',  -- JSON array
  frontmatter  TEXT NOT NULL DEFAULT '{}',  -- JSON object
  chunk_count  INTEGER NOT NULL DEFAULT 0,
  indexed_at   TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(project, file_path)
);

-- Chunks: semantic units within documents
CREATE TABLE chunks (
  id            TEXT PRIMARY KEY,     -- ChunkId (SHA256 of documentId:index)
  document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  project       TEXT NOT NULL,
  content       TEXT NOT NULL,         -- original markdown
  content_plain TEXT NOT NULL,         -- stripped text (for embedding)
  token_count   INTEGER NOT NULL DEFAULT 0,
  heading_path  TEXT NOT NULL DEFAULT '[]',  -- JSON array
  heading_depth INTEGER NOT NULL DEFAULT 0,
  chunk_index   INTEGER NOT NULL,
  has_code_block INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_project ON chunks(project);

-- Embeddings: vector representations
CREATE TABLE embeddings (
  chunk_id   TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  model_name TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  vector     BLOB NOT NULL            -- Float32Array as BLOB
);

-- FTS5 full-text search index
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  chunk_id UNINDEXED,
  title,
  content,
  tags,
  content='',
  contentless_delete=1
);

-- FTS5 sync triggers
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(chunk_id, title, content, tags)
  SELECT NEW.id,
         COALESCE((SELECT title FROM documents WHERE id = NEW.document_id), ''),
         NEW.content_plain,
         NEW.heading_path;
END;

CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
  DELETE FROM chunks_fts WHERE chunk_id = OLD.id;
END;

-- File hashes for incremental indexing
CREATE TABLE file_hashes (
  project    TEXT NOT NULL,
  file_path  TEXT NOT NULL,
  file_hash  TEXT NOT NULL,
  file_size  INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (project, file_path)
);
```

Database file permissions: 0600 (owner read/write only).

### Schema Migration

```typescript
const MIGRATIONS: Migration[] = [
  { version: 1, up: (db) => db.exec(SCHEMA_V1) },
  { version: 2, up: (db) => {
    // v2: trust scoring
    db.exec(`ALTER TABLE chunks ADD COLUMN trust_score REAL DEFAULT 1.0`)
  }},
]

function migrate(db: Database): void {
  const current = db.pragma('user_version') as number
  for (const m of MIGRATIONS) {
    if (m.version > current) {
      db.transaction(() => {
        m.up(db)
        db.pragma(`user_version = ${m.version}`)
      })()
    }
  }
}
```

v1 user upgrades to v2 → database auto-migrates. No data loss.

---

## 13. Configuration

### Zero-config defaults with overrides

```typescript
interface VectorConfig {
  // Storage
  readonly storagePath: string          // default: ~/.vector
  readonly databaseName: string         // default: vector.db

  // Embedding
  readonly embeddingProvider: 'local'    // v1: only local
  readonly embeddingModel: string       // default: Xenova/all-MiniLM-L6-v2
  readonly embeddingDimensions: number  // default: 384
  readonly modelCachePath: string       // default: ~/.vector/models

  // Chunking
  readonly maxChunkTokens: number       // default: 400
  readonly minChunkTokens: number       // default: 50
  readonly chunkOverlapTokens: number   // default: 40
  readonly headingSplitDepth: number    // default: 2 (split on ## and above)

  // Retrieval
  readonly defaultTopK: number          // default: 10
  readonly rrfK: number                 // default: 60
  readonly minScore: number             // default: 0.01

  // Project
  readonly project: string              // default: basename of cwd
}
```

Config resolution order (later overrides earlier):
1. Built-in defaults
2. `~/.vector/config.json` (global)
3. `.vector/config.json` (project-local)
4. CLI flags (`--top-k 20`, `--model ...`)

Validated at startup via Zod schema. Invalid config → clear error with what's wrong and what's expected.

### Chunk overlap

`chunkOverlapTokens: 40` — consecutive chunks share ~40 tokens of context to avoid losing information at heading boundaries. This is a standard RAG technique. When heading-based splitting produces a natural break, overlap is not added (headings are self-contained boundaries).

---

## 14. CLI UX

```bash
$ vector init
  Vector v1.0.0
  Found 47 markdown files in ./docs
  Database: ~/.vector/vector.db (new)
  Ready. Run `vector index` to start.

$ vector index
  Indexing ████████████████████████░░░░  38/47 files
  ✓ 47 files → 312 chunks → 312 embeddings
  ⏱ 2.1s (149 chunks/sec)

$ vector query "how does authentication work?"
  Found 5 results (87ms)
  ┌─ docs/auth/oauth.md ─── score: 0.94 ──────────┐
  │ ## OAuth 2.0 Flow                               │
  │ Authentication uses OAuth 2.0 with PKCE...      │
  └─────────────────────────────────────────────────┘

$ vector status
  Documents: 47 (312 chunks)
  Database: 4.2 MB
  Last indexed: 2 minutes ago
  Stale files: 0

$ vector doctor
  ✓ Database: OK
  ✓ FTS index: in sync
  ✓ Embeddings: 312/312
  ✓ Model: loaded
  ✓ Invariants: 5/5 passing

$ vector remove docs/old-notes.md
  Removed: docs/old-notes.md (8 chunks)
```

### CLI Commands Summary

| Command | Description |
|---------|-------------|
| `vector init` | Initialize project, auto-discover .md files |
| `vector index [path]` | Index .md files (incremental, skips unchanged) |
| `vector query "..."` | Hybrid search (BM25 + vector + RRF) |
| `vector status` | Show index stats, stale files |
| `vector doctor` | Health check: DB, FTS sync, invariants |
| `vector remove <path>` | Remove a document and its chunks from index |

### UX Principles

- **Zero-config** — vector init auto-discovers .md files
- **Progress feedback** — progress bar during indexing
- **Speed visible** — always show elapsed time
- **Result preview** — text snippet, not just file name
- **Human errors** — "File not found: ./notes.md — did you mean ./notes/?"
- **Degraded mode message** — warn, don't crash

---

## 15. Project Structure

```
vector/
├── src/
│   ├── types.ts           # Branded types, all interfaces
│   ├── config.ts          # Zod config schema, defaults, resolution
│   ├── invariants.ts      # System invariant checks
│   ├── errors.ts          # Error hierarchy
│   │
│   ├── parser/
│   │   ├── markdown.ts    # remark → AST → DocumentMeta
│   │   ├── chunker.ts     # AST → RawChunk[]
│   │   └── tokens.ts      # token counting
│   │
│   ├── embedder/
│   │   ├── interface.ts   # Embedder interface only
│   │   └── local.ts       # HuggingFace implementation
│   │
│   ├── store/
│   │   ├── interface.ts   # Store interface only
│   │   ├── sqlite.ts      # SQLite implementation
│   │   └── schema.sql     # DDL in separate file
│   │
│   ├── search/
│   │   ├── interface.ts   # Search interface only
│   │   ├── bm25.ts        # FTS5 wrapper
│   │   ├── vector.ts      # cosine similarity
│   │   └── fusion.ts      # RRF
│   │
│   ├── pipeline/
│   │   ├── indexer.ts     # indexFile orchestrator
│   │   └── searcher.ts   # search orchestrator
│   │
│   ├── cli/
│   │   ├── program.ts    # commander setup
│   │   ├── init.ts
│   │   ├── index.ts
│   │   ├── query.ts
│   │   ├── status.ts
│   │   ├── doctor.ts
│   │   ├── remove.ts
│   │   └── format.ts     # output formatting, colors
│   │
│   ├── mcp/
│   │   ├── server.ts     # MCP server setup, stdio transport
│   │   └── tools.ts      # tool definitions (see MCP Tools section)
│   │
│   └── index.ts          # entry point
│
├── tests/
│   ├── properties/        # Property-based tests (fast-check)
│   │   ├── parser.prop.ts
│   │   ├── search.prop.ts
│   │   └── store.prop.ts
│   │
│   ├── unit/              # Unit tests
│   │   ├── chunker.test.ts
│   │   ├── fusion.test.ts
│   │   ├── tokens.test.ts
│   │   └── cosine.test.ts
│   │
│   ├── integration/       # Pipeline tests
│   │   └── pipeline.test.ts
│   │
│   ├── performance/       # Performance contracts
│   │   ├── search.perf.ts
│   │   └── index.perf.ts
│   │
│   └── fixtures/          # Test .md files
│       ├── simple.md
│       ├── frontmatter.md
│       ├── code-blocks.md
│       ├── deep-headings.md
│       └── large.md
│
├── package.json
├── tsconfig.json
└── vitest.config.ts
```

---

## 16. MCP Server Tools

The MCP server exposes Vector's core functionality to AI clients (Claude CLI, Cursor, Copilot, Windsurf) via stdio transport.

### v1 Tools

| Tool | Input | Output | Maps to |
|------|-------|--------|---------|
| `search_memory` | `{ query: string, topK?: number, project?: string }` | `SearchResult[]` | `Search.query()` |
| `index_files` | `{ paths: string[], project?: string }` | `IndexResult[]` | `pipeline/indexer` |
| `get_document` | `{ project: string, filePath: string }` | `StoredDocument \| null` | `Store.getDocument()` |
| `list_documents` | `{ project?: string }` | `DocumentMeta[]` | `Store.listDocuments()` |
| `remove_document` | `{ project: string, filePath: string }` | `{ removed: boolean }` | `Store.removeDocument()` |
| `status` | `{ project?: string }` | `{ documents, chunks, dbSize, staleFiles }` | Aggregation query |

Tool schemas auto-generated from Zod definitions → JSON Schema for MCP tool discovery.

---

## 17. Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | TypeScript 5.7+ | Type safety, branded types |
| Runtime | Node.js 22+ | LTS, native ESM |
| Storage | better-sqlite3 11.x | Fastest SQLite for Node, sync C++ bindings |
| Embeddings | @huggingface/transformers 3.x | Local, zero API cost, offline |
| Model | all-MiniLM-L6-v2 | 384-dim, 23MB ONNX, 256-token context |
| Markdown | remark + remark-frontmatter | AST parsing, extensible |
| CLI | commander 13.x | Standard, well-tested |
| MCP | @modelcontextprotocol/sdk 1.x | AI client integration |
| Validation | zod 3.x | Runtime type checking |
| Testing | vitest + fast-check | Fast, ESM-native, property testing |

---

## 18. v1 vs v2 Boundary

### v1 — "Core Perfect"

- Parser (markdown → chunks)
- Embedder (local, HuggingFace)
- Store (SQLite, atomic transactions)
- Search (BM25 + vector + RRF)
- CLI (init, index, query, status, doctor, remove)
- MCP server (search, index)
- Branded types (compile-time safety)
- System invariants (self-checking)
- Property-based tests (proves correctness)
- Performance contracts (speed tested in CI)
- Graceful degradation (BM25 fallback)
- Structured logging (observability)
- Determinism guarantee (reproducible)
- Schema migrations (future-proof)
- Security boundaries (input validation)
- Error hierarchy (typed, recoverable flag)

### v2 — "Extend"

- Trust scoring
- Relation graph
- Garbage collection
- OpenAI embedder provider
- `vector watch` (auto-reindex on file change)
- Multi-project support
- Plugin system
- Performance benchmarks dashboard
- LLM prompt compilation (token-optimized context injection)

---

## 19. Elite Engineering Summary

| Level | What it means | Vector implements |
|-------|--------------|-------------------|
| Level 1: Clean Code | Interfaces, SOLID, tests | All interfaces isolated and replaceable |
| Level 2: Invalid states impossible | Type system prevents bugs | Branded types, two-phase chunks, immutable data |
| Level 3: System proves itself | Property tests, invariants | fast-check properties, system invariants, perf contracts |

> Level 1: "I found the bug with a test."
> Level 2: "Writing the bug is impossible."
> Level 3: "The system proves no bug exists."

---

## 20. Design Decisions Log

| Decision | Chosen | Rejected | Why |
|----------|--------|----------|-----|
| Trust scoring in v1 | No (v2) | Yes | Not part of core loop |
| Relation graph in v1 | No (v2) | Yes | Not part of core loop |
| Branded types | Yes | Plain string/number | Prevents ID mixup at compile time |
| Property-based testing | Yes | Unit-only | Proves properties for all inputs |
| Graceful degradation | BM25 fallback | Crash on model fail | Partial results better than nothing |
| Score type | UnitScore [0,1] | Raw number | Invalid scores impossible |
| Chunk types | RawChunk + IndexedChunk | Single Chunk with optional embedding | Embedding-less chunk in store impossible |
| ID generation | Content-based SHA256 | Random UUID | Enables idempotent indexing |
| Schema migration | Version-based up() | Manual ALTER TABLE | Automatic, safe upgrade path |
| Test runner | vitest | jest | ESM-native, faster, TypeScript built-in |
| Embedder return type | Float32Array | number[] | Direct BLOB storage, no conversion at store boundary |
| Store.save() param | ChunkWithEmbedding[] | Separate arrays | Explicit chunk-embedding pairing, no index coupling |
| contentHash source | Parser computes it | Orchestrator computes it | Single source of truth, no divergence risk |
| RRF score normalization | Min-max to [0,1] | Raw RRF values | Satisfies UnitScore invariant |
| FTS5 query operators | Preserve * and "" | Strip everything | Users need prefix search and phrase matching |
| Chunk overlap | 40 tokens default | No overlap | Standard RAG technique, prevents context loss at boundaries |
| WAL mode | Enabled | Default journal | Allows CLI + MCP concurrent reads |
