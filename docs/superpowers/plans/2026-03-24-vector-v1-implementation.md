# Vector Memory Engine v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **MANDATORY:** Before starting ANY task, read `CLAUDE.md`, `AGENTS.md`, the design spec at `docs/superpowers/specs/2026-03-24-vector-elite-design.md`, and the role file matching your task (see `docs/roles/`).

**Goal:** Build a complete CLI tool + MCP server that transforms `.md` files into hybrid AI memory with vector search, BM25, and RRF fusion — at Distinguished/Fellow Level 3 engineering quality.

**Architecture:** Four independent components (Parser, Embedder, Store, Search) connected by a Pipeline orchestrator. Data flows one direction: Parser → Embedder → Store → Search. CLI and MCP are thin presentation layers over the pipeline. All types are branded, all data is immutable, all invariants are checked in dev mode.

**Tech Stack:** TypeScript 5.7+, Node.js 22+, better-sqlite3, @huggingface/transformers, remark, commander, @modelcontextprotocol/sdk, zod, vitest, fast-check

**Parallelization note:** Tasks 5 (Embedder) and 6 (Store) are independent and can be executed in parallel by subagents.

---

## Task 1: Project Scaffold

**Role:** Core Engineer (`docs/roles/core-engineer.md`)

**Files:**
- Create: `package.json`
- Create: `tsconfig.json`
- Create: `vitest.config.ts`
- Create: `src/index.ts` (empty entry point)

- [ ] **Step 1: Create package.json**

```json
{
  "name": "vector-ideafix",
  "version": "0.1.0",
  "description": "Your notes become AI's memory — CLI + MCP hybrid search for markdown",
  "type": "module",
  "engines": { "node": ">=22.0.0" },
  "bin": { "vector": "./dist/index.js" },
  "main": "./dist/index.js",
  "scripts": {
    "build": "tsc",
    "dev": "tsc --watch",
    "test": "vitest run",
    "test:watch": "vitest",
    "test:props": "vitest run tests/properties",
    "test:perf": "vitest run tests/performance",
    "typecheck": "tsc --noEmit",
    "lint": "tsc --noEmit",
    "prepublishOnly": "npm run build"
  },
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
    "fast-check": "^3.15.0",
    "typescript": "^5.7.0",
    "vitest": "^2.0.0"
  },
  "license": "MIT"
}
```

- [ ] **Step 2: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": false,
    "noUnusedLocals": true,
    "noUnusedParameters": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

- [ ] **Step 3: Create vitest.config.ts**

```typescript
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    include: ['tests/**/*.{test,prop,perf}.ts'],
    testTimeout: 30000,
    pool: 'forks',
  },
})
```

- [ ] **Step 4: Create src/index.ts placeholder**

```typescript
#!/usr/bin/env node
// Vector Memory Engine — entry point
// Will dispatch to CLI or MCP server based on args
export {}
```

- [ ] **Step 5: Install dependencies**

Run: `npm install`
Expected: `node_modules/` created, `package-lock.json` generated

- [ ] **Step 6: Verify TypeScript compiles**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 7: Verify vitest runs**

Run: `npx vitest run`
Expected: "No test files found" (no tests yet, but vitest works)

- [ ] **Step 8: Commit**

```bash
git add package.json package-lock.json tsconfig.json vitest.config.ts src/index.ts
git commit -m "feat: project scaffold with TypeScript, vitest, and dependencies"
```

---

## Task 2: Types, Errors, and Config

**Role:** Core Engineer (`docs/roles/core-engineer.md`)

**Files:**
- Create: `src/types.ts`
- Create: `src/errors.ts`
- Create: `src/config.ts`
- Create: `tests/unit/types.test.ts`

- [ ] **Step 1: Write types tests**

```typescript
// tests/unit/types.test.ts
import { describe, test, expect } from 'vitest'
import {
  createDocumentId, createChunkId, createProjectId, unitScore,
  type DocumentId, type ChunkId, type ProjectId, type UnitScore
} from '../../src/types.js'

describe('Branded types', () => {
  test('createDocumentId returns DocumentId', () => {
    const id = createDocumentId('test-project', '/path/to/file.md')
    expect(typeof id).toBe('string')
    expect(id.length).toBe(16)
  })

  test('same input produces same DocumentId', () => {
    const id1 = createDocumentId('proj', '/a.md')
    const id2 = createDocumentId('proj', '/a.md')
    expect(id1).toBe(id2)
  })

  test('different input produces different DocumentId', () => {
    const id1 = createDocumentId('proj', '/a.md')
    const id2 = createDocumentId('proj', '/b.md')
    expect(id1).not.toBe(id2)
  })

  test('createChunkId is deterministic', () => {
    const docId = createDocumentId('proj', '/a.md')
    const c1 = createChunkId(docId, 0)
    const c2 = createChunkId(docId, 0)
    expect(c1).toBe(c2)
  })

  test('unitScore accepts valid scores', () => {
    expect(unitScore(0)).toBe(0)
    expect(unitScore(0.5)).toBe(0.5)
    expect(unitScore(1)).toBe(1)
  })

  test('unitScore rejects invalid scores', () => {
    expect(() => unitScore(-0.1)).toThrow()
    expect(() => unitScore(1.1)).toThrow()
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run tests/unit/types.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Implement src/types.ts**

All branded types, factory functions, interfaces, and interface contracts — exactly as specified in the design spec sections 3-4. Specifically:

**Branded types + factories:** `DocumentId` + `createDocumentId(project, filePath)`, `ChunkId` + `createChunkId(docId, index)`, `ProjectId` + `createProjectId(name)`, `UnitScore` + `unitScore(n)`. All use SHA256 truncated to 16 hex chars.

**Data interfaces:**
- `DocumentMeta` — title, filePath, project, contentHash, tags, frontmatter, indexedAt, fileSize
- `RawChunk` — content, contentPlain, headingPath, index, hasCodeBlock (all readonly)
- `IndexedChunk` — id, documentId + all RawChunk fields + embedding: Float32Array (required)
- `ChunkWithEmbedding` — { chunk: RawChunk, embedding: Float32Array }
- `StoredDocument` — extends DocumentMeta with { id: DocumentId, chunkCount: number }
- `ParseResult` — { document: DocumentMeta, chunks: RawChunk[] }
- `SearchResult` — { chunk, score: UnitScore, scores: { bm25Rank, vectorRank, rrfRaw }, documentTitle, documentPath, highlight }
- `SearchOptions` — { topK?, minScore?, project? }
- `IndexResult` — discriminated union: `{ status: 'indexed', chunks: number }` | `{ status: 'skipped', reason: string }`

**Interface contracts:** `Parser`, `Embedder`, `Store` (including `listDocuments(project?: ProjectId): StoredDocument[]`), `Search`

**Note:** `Store.listDocuments()` is needed by the MCP `list_documents` tool — add it to the Store interface.

- [ ] **Step 4: Implement src/errors.ts**

VectorError base class, all error subclasses (FileNotFoundError, InvalidMarkdownError, ModelLoadError, InputTooLongError, DatabaseCorruptedError, TransactionFailedError, InvariantViolation, SecurityError, EmptyInputError) — exactly as specified in design spec section 6.

- [ ] **Step 5: Implement src/config.ts**

VectorConfig Zod schema with all defaults from design spec section 13. Config resolution: built-in defaults → `~/.vector/config.json` (global) → `.vector/config.json` (project-local) → CLI overrides. `loadConfig()` reads and merges, `resolveConfig(overrides?)` returns frozen config. Invalid config → clear Zod validation error.

- [ ] **Step 6: Write config resolution tests**

Tests for: default values applied when no config file exists, global config overrides defaults, project config overrides global, CLI flags override all, invalid config throws with helpful message.

- [ ] **Step 7: Run tests to verify they pass**

Run: `npx vitest run tests/unit/types.test.ts`
Expected: All PASS

- [ ] **Step 8: Typecheck**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 9: Commit**

```bash
git add src/types.ts src/errors.ts src/config.ts tests/unit/types.test.ts
git commit -m "feat: branded types, error hierarchy, and config system"
```

---

## Task 3: Token Counter and Test Fixtures

**Role:** Quality Engineer (`docs/roles/quality-engineer.md`)

**Files:**
- Create: `src/parser/tokens.ts`
- Create: `tests/unit/tokens.test.ts`
- Create: `tests/fixtures/simple.md`
- Create: `tests/fixtures/frontmatter.md`
- Create: `tests/fixtures/code-blocks.md`
- Create: `tests/fixtures/deep-headings.md`
- Create: `tests/fixtures/large.md`

- [ ] **Step 1: Write token counter tests**

```typescript
// tests/unit/tokens.test.ts
import { describe, test, expect } from 'vitest'
import { countTokens } from '../../src/parser/tokens.js'

describe('countTokens', () => {
  test('empty string returns 0', () => {
    expect(countTokens('')).toBe(0)
  })

  test('simple text uses ~4 chars per token estimate', () => {
    const text = 'Hello world this is a test'  // 26 chars
    const tokens = countTokens(text)
    expect(tokens).toBeGreaterThan(4)
    expect(tokens).toBeLessThan(10)
  })

  test('code blocks count tokens', () => {
    const code = 'function hello() { return "world" }'
    expect(countTokens(code)).toBeGreaterThan(0)
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run tests/unit/tokens.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement src/parser/tokens.ts**

Simple word-based estimator: ~4 chars per token. `countTokens(text: string): number`.

- [ ] **Step 4: Create all 5 test fixture markdown files**

Each fixture tests a different aspect: basic markdown, YAML frontmatter, fenced code blocks, deeply nested headings (h1→h4), and a large file (>2000 tokens).

- [ ] **Step 5: Run tests**

Run: `npx vitest run tests/unit/tokens.test.ts`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/parser/tokens.ts tests/unit/tokens.test.ts tests/fixtures/
git commit -m "feat: token counter and test fixtures"
```

---

## Task 4: Markdown Parser and Chunker

**Role:** Core Engineer (`docs/roles/core-engineer.md`)

**Files:**
- Create: `src/parser/markdown.ts`
- Create: `src/parser/chunker.ts`
- Create: `tests/unit/chunker.test.ts`
- Create: `tests/properties/parser.prop.ts`

- [ ] **Step 1: Write chunker unit tests**

Tests for: heading-based splitting, headingPath extraction, code block preservation (no split mid-block), frontmatter extraction, empty files, files with only frontmatter, contentHash computation, chunk index ordering.

- [ ] **Step 2: Write parser property tests**

```typescript
// tests/properties/parser.prop.ts
import { describe, test } from 'vitest'
import fc from 'fast-check'
import { MarkdownParser } from '../../src/parser/markdown.js'

describe('Parser properties', () => {
  test('parsed chunks reconstruct original content', () => {
    fc.assert(fc.property(
      fc.string({ minLength: 1, maxLength: 2000 }),
      (markdown) => {
        // Write to temp file, parse, verify content coverage
      }
    ), { numRuns: 1000 })
  })

  test('chunk indices are sequential starting from 0', () => {
    fc.assert(fc.property(
      fc.string({ minLength: 1, maxLength: 2000 }),
      (markdown) => {
        // Parse and verify indices
      }
    ), { numRuns: 1000 })
  })
})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `npx vitest run tests/unit/chunker.test.ts tests/properties/parser.prop.ts`
Expected: FAIL

- [ ] **Step 4: Implement src/parser/markdown.ts**

`MarkdownParser` implements `Parser` interface. Uses `unified` + `remark-parse` + `remark-frontmatter` to parse markdown to AST. Extracts `DocumentMeta` (title from first h1, tags from frontmatter, contentHash via SHA256 of file bytes). Delegates chunking to `chunker.ts`.

- [ ] **Step 5: Implement src/parser/chunker.ts**

`chunkify(ast, config): RawChunk[]`. Algorithm:

1. Walk remark AST depth-first. Track current `headingPath` (e.g., `["Auth", "Login"]`).
2. When a heading at depth <= `headingSplitDepth` (default: 2, meaning h1 and h2) is encountered, start a new chunk. This is a **clean break** — no overlap added.
3. Between headings, if accumulated content exceeds `maxChunkTokens`, split at paragraph boundary. For these non-heading splits, **append `chunkOverlapTokens` (40)** from the end of the previous chunk to the start of the new one.
4. Code blocks (fenced ``` or indented) are **never split**. If a code block pushes a chunk over `maxChunkTokens`, keep the whole block in one chunk.
5. `contentPlain` = strip markdown syntax (headings markers, bold/italic markers, link syntax, code fences) but keep the text content. This is what gets embedded.
6. `validateChunk()` — reject any chunk > 50,000 chars.
7. Set `hasCodeBlock = true` if chunk contains any fenced code block.
8. Chunks are indexed sequentially: 0, 1, 2, ...

**Security:** `validateChunk()` lives in this file — it's a parser concern.

- [ ] **Step 5b: Implement security utilities**

Add `validateFilePath(path, projectRoot)` to `src/parser/markdown.ts` (exported separately). This is a security boundary — resolves path and verifies it's within `projectRoot`. Throws `SecurityError` on traversal attempt. The pipeline orchestrator calls this before `parser.parse()`.

**CRITICAL:** `contentHash` is computed by the Parser inside `markdown.ts` using `createHash('sha256').update(rawFileBytes).digest('hex')` and stored on `DocumentMeta`. This is the **single source of truth** — no other code computes content hashes.

- [ ] **Step 6: Run tests**

Run: `npx vitest run tests/unit/chunker.test.ts tests/properties/parser.prop.ts`
Expected: All PASS

- [ ] **Step 7: Run against all fixtures**

Run: `npx vitest run` (all tests)
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/parser/ tests/unit/chunker.test.ts tests/properties/parser.prop.ts
git commit -m "feat: markdown parser with heading-aware chunking"
```

---

## Task 5: Embedder Interface and Local Implementation

**Role:** Quality Engineer (`docs/roles/quality-engineer.md`)

**Files:**
- Create: `src/embedder/interface.ts`
- Create: `src/embedder/local.ts`
- Create: `tests/unit/embedder.test.ts`

- [ ] **Step 1: Write embedder tests**

Tests for: `embed()` returns Float32Array of correct dimensions (384), `embedBatch()` returns array of correct length, empty string throws `EmptyInputError`, determinism (same text → same embedding), batch results match individual results.

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run tests/unit/embedder.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement src/embedder/interface.ts**

Re-export the `Embedder` interface from `types.ts`. This file exists for clear module boundaries.

- [ ] **Step 4: Implement src/embedder/local.ts**

`LocalEmbedder` implements `Embedder`. Uses `@huggingface/transformers` with `Xenova/all-MiniLM-L6-v2`. Model cached at `config.modelCachePath`. `embed()` returns `Float32Array`. `embedBatch()` processes all texts in one pass. Handles model loading lazily on first call.

- [ ] **Step 5: Run tests**

Run: `npx vitest run tests/unit/embedder.test.ts`
Expected: All PASS (note: first run downloads ~23MB model)

- [ ] **Step 6: Commit**

```bash
git add src/embedder/ tests/unit/embedder.test.ts
git commit -m "feat: local embedder with HuggingFace transformers"
```

---

## Task 6: SQLite Store

**Role:** Core Engineer (`docs/roles/core-engineer.md`)

**Files:**
- Create: `src/store/interface.ts`
- Create: `src/store/schema.sql`
- Create: `src/store/sqlite.ts`
- Create: `tests/unit/store.test.ts`
- Create: `tests/properties/store.prop.ts`

- [ ] **Step 1: Write store unit tests**

Tests for: `save()` persists document + chunks + embeddings atomically, `getDocument()` returns null for missing, `getChunks()` returns IndexedChunks with embeddings, `removeDocument()` cascades (chunks + embeddings + FTS entries deleted), `needsReindex()` detects changed content hash, idempotent save (same data twice → same state), FTS5 entries created via trigger.

- [ ] **Step 2: Write store property tests**

```typescript
// tests/properties/store.prop.ts — cascade deletion property, idempotent save property
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `npx vitest run tests/unit/store.test.ts tests/properties/store.prop.ts`
Expected: FAIL

- [ ] **Step 4: Create src/store/schema.sql**

Exact DDL from design spec section 12. Copy verbatim — documents, chunks, embeddings, chunks_fts, triggers, file_hashes.

- [ ] **Step 5: Implement src/store/sqlite.ts**

`SqliteStore` implements `Store` (including `listDocuments()`). Opens database at `config.storagePath/config.databaseName`. Sets WAL mode, foreign_keys, busy_timeout. Runs migration on open. Sets file permissions to **0600** (owner read/write only) after creation.

`save()` uses `db.transaction()` — inserts/upserts document, chunks, embeddings, updates file_hashes, all in one transaction.

**CRITICAL — Float32Array/BLOB alignment:** Embeddings stored as `Buffer.from(embedding.buffer, embedding.byteOffset, embedding.byteLength)`. Retrieved with safe alignment: `new Float32Array(new Uint8Array(blob).buffer)` — do NOT use `new Float32Array(blob.buffer)` directly, as better-sqlite3 BLOBs may have non-aligned byte offsets which causes runtime crashes.

`removeDocument()` cascades via ON DELETE CASCADE. `needsReindex()` compares content hash from file_hashes table.

- [ ] **Step 6: Run tests**

Run: `npx vitest run tests/unit/store.test.ts tests/properties/store.prop.ts`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/store/ tests/unit/store.test.ts tests/properties/store.prop.ts
git commit -m "feat: SQLite store with atomic save, FTS5, and cascade delete"
```

---

## Task 7: Invariant System

**Role:** Quality Engineer (`docs/roles/quality-engineer.md`)

**Files:**
- Create: `src/invariants.ts`
- Create: `tests/unit/invariants.test.ts`

- [ ] **Step 1: Write invariant tests**

Tests for: each of 5 invariants passes on valid store, each invariant detects its specific violation (create invalid state manually in test, verify invariant throws `InvariantViolation`), `checkAllInvariants()` runs all 5.

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run tests/unit/invariants.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement src/invariants.ts**

5 invariant check functions + `checkAllInvariants(db)` + `withInvariantCheck(db, operation)` wrapper that runs checks after write operations when `NODE_ENV !== 'production'`.

- [ ] **Step 4: Run tests**

Run: `npx vitest run tests/unit/invariants.test.ts`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/invariants.ts tests/unit/invariants.test.ts
git commit -m "feat: 5 system invariants with dev-mode auto-checking"
```

---

## Task 8: Search Engine (BM25 + Vector + RRF)

**Role:** Core Engineer (`docs/roles/core-engineer.md`)

**Files:**
- Create: `src/search/interface.ts`
- Create: `src/search/bm25.ts`
- Create: `src/search/vector.ts`
- Create: `src/search/fusion.ts`
- Create: `tests/unit/cosine.test.ts`
- Create: `tests/unit/fusion.test.ts`
- Create: `tests/properties/search.prop.ts`

- [ ] **Step 1: Write cosine similarity tests**

Tests for: identical vectors → 1.0, orthogonal vectors → 0.0, opposite vectors → -1.0, result always in [-1, 1].

- [ ] **Step 2: Write RRF fusion tests**

Tests for: single-source fusion (BM25 only, vector only), dual-source fusion, result normalization to [0, 1], correct rank ordering, k parameter effect.

- [ ] **Step 3: Write search property tests**

Properties: results always sorted descending, all scores are UnitScore [0,1], RRF raw score always positive, cosine similarity bounded.

- [ ] **Step 4: Run tests to verify they fail**

Run: `npx vitest run tests/unit/cosine.test.ts tests/unit/fusion.test.ts tests/properties/search.prop.ts`
Expected: FAIL

- [ ] **Step 5: Implement src/search/bm25.ts**

`Bm25Search` class. Queries FTS5 `chunks_fts` table using `MATCH` with `bm25()` ranking function. Returns ranked results with chunk_ids.

`sanitizeFtsQuery(query)` — preserves `*` (prefix search: `auth*` matches "authentication") and `""` (phrase search: `"login bug"`). Strips unbalanced quotes via `balanceQuotes()`. Escapes `()` (FTS5 grouping/NEAR operators) by replacing with spaces. This function lives in this file with its own tests.

- [ ] **Step 6: Implement src/search/vector.ts**

`VectorSearch` class. Brute-force scan: loads all embeddings from store, computes cosine similarity with query embedding, returns top-K ranked results. This is intentional for v1 — for <50K chunks, brute-force takes ~15ms. ANN index (hnswlib) is a v2 optimization.

`cosineSimilarity(a: Float32Array, b: Float32Array): number` as a pure exported function. Result always in [-1, 1] (for arbitrary vectors) or [0, 1] (for L2-normalized MiniLM embeddings).

- [ ] **Step 7: Implement src/search/fusion.ts**

`rrfFuse(bm25Ranks, vectorRanks, k): FusedResult[]`. Reciprocal Rank Fusion formula: `rrfRaw = 1/(k + rank_bm25) + 1/(k + rank_vector)`. Min-max normalization of `rrfRaw` values to [0,1] via `unitScore()` → becomes `SearchResult.score`. Each result also carries `scores: { bm25Rank, vectorRank, rrfRaw }` for transparency. Returns sorted descending by `score`.

**Highlight generation:** `highlight` is the FTS5 `snippet()` function output for BM25 matches. For vector-only matches (no BM25 hit), use first 150 chars of `chunk.contentPlain`.

- [ ] **Step 8: Run tests**

Run: `npx vitest run tests/unit/cosine.test.ts tests/unit/fusion.test.ts tests/properties/search.prop.ts`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/search/ tests/unit/cosine.test.ts tests/unit/fusion.test.ts tests/properties/search.prop.ts
git commit -m "feat: hybrid search with BM25, vector similarity, and RRF fusion"
```

---

## Task 9: Pipeline Orchestrators

**Role:** Core Engineer (`docs/roles/core-engineer.md`)

**Files:**
- Create: `src/pipeline/indexer.ts`
- Create: `src/pipeline/searcher.ts`
- Create: `src/logger.ts`
- Create: `tests/integration/pipeline.test.ts`

- [ ] **Step 1: Write integration tests**

Tests for: full pipeline (create .md file → index → query → get results), incremental indexing (unchanged file skipped), re-indexing (changed file updated), graceful degradation (embedder fails → BM25-only results with warning), `removeDocument()` via pipeline.

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run tests/integration/pipeline.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement src/logger.ts**

Structured `Logger` interface implementation. Writes to stderr. `--verbose` shows debug level. Normal mode shows info+ only. Events: `file.indexed`, `file.skipped`, `search.completed`, `search.degraded`, `invariant.violated`. Each event includes structured data (path, elapsed_ms, count, etc.).

- [ ] **Step 3b: Write logger tests**

Tests for: log levels filter correctly (debug hidden in normal mode, shown in verbose), structured data appears in output, error events include VectorError code.

- [ ] **Step 4: Implement src/pipeline/indexer.ts**

`indexFile()` function exactly as in spec section 5. `indexDirectory()` function that discovers all `.md` files, filters by `needsReindex()`, indexes in batch. Reports progress via callback. Security: `validateFilePath()` before parse.

- [ ] **Step 5: Implement src/pipeline/searcher.ts**

`SearchOrchestrator` implements `Search`. Wraps BM25 + vector + RRF. **Graceful degradation:** specifically catches `ModelLoadError` (and only `ModelLoadError`) — falls back to BM25-only results. All other errors are re-thrown. Logs `search.degraded` warning via logger with `{ reason: 'model_unavailable', fallback: 'bm25_only' }`.

- [ ] **Step 6: Run tests**

Run: `npx vitest run tests/integration/pipeline.test.ts`
Expected: All PASS

- [ ] **Step 7: Run ALL tests**

Run: `npx vitest run`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/pipeline/ src/logger.ts tests/integration/pipeline.test.ts
git commit -m "feat: pipeline orchestrators with graceful degradation"
```

---

## Task 10: CLI Commands

**Role:** DX Engineer (`docs/roles/dx-engineer.md`)

**Files:**
- Create: `src/cli/program.ts`
- Create: `src/cli/init.ts`
- Create: `src/cli/index.ts`
- Create: `src/cli/query.ts`
- Create: `src/cli/status.ts`
- Create: `src/cli/doctor.ts`
- Create: `src/cli/remove.ts`
- Create: `src/cli/format.ts`
- Modify: `src/index.ts`

**Dependency injection pattern:** Create a `createApp(config: VectorConfig)` factory in `program.ts` that instantiates Parser, Embedder, Store, Search once. Each CLI command receives the app context — no command creates its own instances.

- [ ] **Step 1: Write CLI integration tests**

Create `tests/integration/cli.test.ts`. Tests for: `vector init` creates database, `vector index` indexes fixtures, `vector query` returns results with scores, `vector status` shows correct counts, `vector doctor` reports all passing, `vector remove` removes document and chunks. Use `execFile` to run the built CLI binary against a temp directory with fixture files. Assert exit codes and output patterns.

- [ ] **Step 2: Implement src/cli/format.ts**

Output formatting utilities: `formatSearchResult()`, `formatProgress()`, `formatSuccess()`, `formatWarning()`, `formatError()`, `formatBox()`. Color-aware (detect `process.stdout.isTTY`). No color when piped.

- [ ] **Step 3: Implement src/cli/program.ts**

Commander program setup. Global flags: `--verbose`, `--project <name>`, `--storage-path <path>`. Version from package.json. Registers all subcommands.

- [ ] **Step 4: Implement each CLI command**

`init.ts` — auto-discover .md files, create database, show summary.
`index.ts` — index files with progress bar, show timing and counts.
`query.ts` — search with formatted result boxes, scores, highlights.
`status.ts` — show document count, chunk count, DB size, stale files.
`doctor.ts` — run all 5 invariants + model check + FTS sync check.
`remove.ts` — remove document by path, show what was removed.

- [ ] **Step 5: Update src/index.ts**

Entry point dispatches to CLI program. Add shebang `#!/usr/bin/env node`.

- [ ] **Step 6: Build and test CLI manually**

Run: `npm run build && node dist/index.js --help`
Expected: Help output with all commands listed

Run: `node dist/index.js --version`
Expected: `0.1.0`

- [ ] **Step 7: Run CLI integration tests**

Run: `npx vitest run tests/integration/cli.test.ts`
Expected: All PASS

- [ ] **Step 8: Test each command manually with test fixtures**

```bash
mkdir /tmp/vector-test && cp tests/fixtures/*.md /tmp/vector-test/
node dist/index.js init --project test
node dist/index.js index /tmp/vector-test
node dist/index.js query "test heading"
node dist/index.js status
node dist/index.js doctor
node dist/index.js remove /tmp/vector-test/simple.md
```

- [ ] **Step 9: Commit**

```bash
git add src/cli/ src/index.ts tests/integration/cli.test.ts
git commit -m "feat: CLI with init, index, query, status, doctor, remove commands"
```

---

## Task 11: MCP Server

**Role:** DX Engineer (`docs/roles/dx-engineer.md`)

**Files:**
- Create: `src/mcp/server.ts`
- Create: `src/mcp/tools.ts`

- [ ] **Step 1: Write MCP integration tests**

Create `tests/integration/mcp.test.ts`. Tests for: all 6 tools are registered, `search_memory` returns SearchResult array, `index_files` indexes and returns results, `list_documents` returns document list, `remove_document` removes and returns status, `status` returns aggregate stats. Use MCP SDK's test client or spawn the server in-process.

- [ ] **Step 2: Implement src/mcp/tools.ts**

Define all 6 MCP tools with Zod input schemas: `search_memory`, `index_files`, `get_document`, `list_documents`, `remove_document`, `status`. Each tool maps to pipeline/store functions. Auto-generate JSON Schema from Zod for MCP discovery using `zodToJsonSchema` or MCP SDK's built-in schema support.

- [ ] **Step 3: Implement src/mcp/server.ts**

MCP server with stdio transport via `@modelcontextprotocol/sdk`. Registers all tools from `tools.ts`. Error responses include `VectorError.code` for structured error handling by AI clients. Start server when `src/index.ts` receives `--mcp` flag.

- [ ] **Step 4: Update src/index.ts for MCP mode**

Add MCP server launch: `vector --mcp` starts stdio server instead of CLI.

- [ ] **Step 5: Run MCP tests**

Run: `npx vitest run tests/integration/mcp.test.ts`
Expected: All PASS

- [ ] **Step 6: Test MCP server manually**

Run: `echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | node dist/index.js --mcp`
Expected: JSON response listing all 6 tools

- [ ] **Step 7: Commit**

```bash
git add src/mcp/ src/index.ts
git commit -m "feat: MCP server with 6 tools via stdio transport"
```

---

## Task 12: Performance Contracts

**Role:** Quality Engineer (`docs/roles/quality-engineer.md`)

**Files:**
- Create: `tests/performance/search.perf.ts`
- Create: `tests/performance/index.perf.ts`

- [ ] **Step 1: Write search performance test**

Generate 10K chunks in SQLite with random embeddings. Run search query. Assert elapsed < 100ms.

- [ ] **Step 2: Write indexing throughput test**

Generate 50 markdown files. Index all. Assert throughput > 100 chunks/sec.

- [ ] **Step 3: Run performance tests**

Run: `npx vitest run tests/performance/`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/performance/
git commit -m "test: performance contracts — search <100ms, indexing >100 chunks/sec"
```

---

## Task 13: Final Integration and Polish

**Role:** All roles — cross-cutting

**Files:**
- Modify: various files for edge cases
- Update: `README.md`
- Update: `.gitignore`

- [ ] **Step 1: Run complete test suite**

Run: `npx vitest run`
Expected: All tests PASS (unit, property, integration, performance)

- [ ] **Step 2: Run typecheck**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Test end-to-end with real markdown files**

```bash
npm run build
node dist/index.js init
node dist/index.js index ./docs
node dist/index.js query "branded types"
node dist/index.js status
node dist/index.js doctor
```

- [ ] **Step 4: Update .gitignore**

Add: `dist/`, `*.db`, `.vector/`

- [ ] **Step 5: Update README.md with installation and usage**

Quick start: `npm install -g vector-ideafix`, `vector init`, `vector index`, `vector query`.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat: Vector Memory Engine v1.0.0 — complete implementation"
```

- [ ] **Step 7: Push to GitHub**

```bash
git push origin main
```
