# Vector Memory Engine — Project Instructions

## CRITICAL: Read Before ANY Work

**STOP. Before writing any code, running any command, or making any decision, you MUST:**

1. **Read the design spec:** `docs/superpowers/specs/2026-03-24-vector-elite-design.md`
2. **Read the role that matches your task** (see Role Activation below)
3. **Understand what you are building and why**

This is not optional. This is not a suggestion. Code written without reading the spec and role will be **wrong** — not because it doesn't compile, but because it will violate invariants, miss type safety requirements, or break the architecture.

**If you are a subagent:** Read `AGENTS.md` first. It has specific instructions for you.

---

## Project Overview

Vector is a CLI tool + MCP server that transforms `.md` files into a hybrid AI memory system with vector search (cosine similarity), full-text search (BM25 via FTS5), and Reciprocal Rank Fusion.

**Value proposition:** "Your notes become AI's memory."

**Engineering level:** Distinguished/Fellow (Level 3) — branded types, system invariants, property-based testing, performance contracts.

---

## Role Activation

Every task maps to exactly one role. Read the role file before starting work.

| Working on... | Role | File |
|---------------|------|------|
| types, interfaces, config, errors | **Core Engineer** | `docs/roles/core-engineer.md` |
| parser, chunker, tokens | **Core Engineer** | `docs/roles/core-engineer.md` |
| store, sqlite, schema, FTS5 | **Core Engineer** | `docs/roles/core-engineer.md` |
| search, bm25, vector, fusion, pipeline | **Core Engineer** | `docs/roles/core-engineer.md` |
| embedder, model integration | **Quality Engineer** | `docs/roles/quality-engineer.md` |
| tests (any kind), invariants | **Quality Engineer** | `docs/roles/quality-engineer.md` |
| performance contracts, benchmarks | **Quality Engineer** | `docs/roles/quality-engineer.md` |
| CLI commands, output, formatting | **DX Engineer** | `docs/roles/dx-engineer.md` |
| MCP server, tools | **DX Engineer** | `docs/roles/dx-engineer.md` |
| error messages, user experience | **DX Engineer** | `docs/roles/dx-engineer.md` |

**Cross-cutting tasks** (e.g., adding a new feature end-to-end): read ALL three roles.

---

## Architecture — Non-Negotiable Rules

### Invariants (must ALWAYS be true)

1. Every Chunk belongs to exactly one Document (CASCADE delete)
2. Every IndexedChunk has an embedding (type-level enforcement)
3. Document identified by `(project, filePath)` — UNIQUE
4. Embedding dimensions uniform within project
5. Search score is `UnitScore` in [0, 1]
6. Indexing is idempotent
7. Same input → same output (deterministic IDs)

### Type Safety

- **Branded types:** `DocumentId`, `ChunkId`, `ProjectId`, `UnitScore` — never raw `string`/`number`
- **Two-phase chunks:** `RawChunk` (from parser) → `IndexedChunk` (in store). No optional embeddings.
- **Explicit pairing:** `ChunkWithEmbedding { chunk: RawChunk, embedding: Float32Array }`
- **Immutable:** `readonly` on every field, every interface
- **No `any`**

### Data Flow

```
Parser → Embedder → Store → Search
  (one direction only, no circular dependencies)
```

### Error Handling

- All errors extend `VectorError` with `code` and `recoverable`
- Recoverable → retry. Non-recoverable → report and continue.
- No silent swallowing. No bare `catch {}`.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | TypeScript 5.7+ (strict mode) |
| Runtime | Node.js 22+ |
| Storage | better-sqlite3 11.x (WAL mode) |
| Embeddings | @huggingface/transformers 3.x (all-MiniLM-L6-v2) |
| Markdown | remark + remark-frontmatter |
| CLI | commander 13.x |
| MCP | @modelcontextprotocol/sdk 1.x |
| Validation | zod 3.x |
| Testing | vitest + fast-check |

---

## Testing Requirements

- **Property-based tests** (fast-check) for all core properties — 10,000+ random inputs
- **Performance contracts** enforced in CI — search < 100ms, indexing > 100 chunks/sec
- **System invariants** checked after every write in dev mode
- **Test first, implement second** — always

---

## File Structure

```
src/
  types.ts, config.ts, errors.ts, invariants.ts
  parser/    — markdown.ts, chunker.ts, tokens.ts
  embedder/  — interface.ts, local.ts
  store/     — interface.ts, sqlite.ts, schema.sql
  search/    — interface.ts, bm25.ts, vector.ts, fusion.ts
  pipeline/  — indexer.ts, searcher.ts
  cli/       — program.ts, init.ts, index.ts, query.ts, status.ts, doctor.ts, remove.ts, format.ts
  mcp/       — server.ts, tools.ts
  index.ts
tests/
  properties/ — *.prop.ts
  unit/       — *.test.ts
  integration/ — pipeline.test.ts
  performance/ — *.perf.ts
  fixtures/   — *.md
```

---

## Commit Standards

- Conventional commits: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`
- Every commit must pass: `tsc --noEmit && vitest run`
- No commits that break invariants or type safety
