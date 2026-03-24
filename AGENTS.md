# Vector Memory Engine — Subagent Instructions

## MANDATORY: Read Before Executing ANY Task

You are a subagent working on the Vector Memory Engine. You do NOT have the full conversation context. You MUST orient yourself before taking any action.

**HARD REQUIREMENT — complete these steps IN ORDER before writing any code:**

### Step 1: Read the design spec
```
docs/superpowers/specs/2026-03-24-vector-elite-design.md
```
This is the single source of truth for the system architecture, type definitions, interfaces, invariants, data flow, error handling, and testing strategy. If your code contradicts this spec, your code is wrong.

### Step 2: Read the role that matches your task
```
docs/roles/core-engineer.md    — types, interfaces, parser, store, search, pipeline, config, errors
docs/roles/quality-engineer.md — embedder, tests, invariants, performance contracts
docs/roles/dx-engineer.md      — CLI commands, MCP server, output formatting, error messages
```
Each role defines: what files you own, what invariants you enforce, what code standards apply, and what you must check before writing code.

### Step 3: Read CLAUDE.md
```
CLAUDE.md
```
Project-wide rules: non-negotiable invariants, type safety requirements, data flow direction, technology stack.

### Step 4: Now you may begin work

Only after completing steps 1-3.

---

## Why This Matters

Vector is engineered at **Level 3 (Distinguished/Fellow)** — the highest standard of software engineering:

- **Branded types** prevent ID mixups at compile time
- **System invariants** are verified after every write in dev mode
- **Property-based tests** prove correctness for 10,000+ random inputs
- **Performance contracts** fail CI if speed regresses

**One line of code that ignores these principles can break the entire system's guarantees.** A `string` where a `DocumentId` is required. An optional embedding where it must be required. A mutable object where it must be readonly. These are not style preferences — they are safety invariants.

---

## Quick Reference: What You Must NOT Do

| Forbidden | Why | Instead |
|-----------|-----|---------|
| `any` type | Breaks all type safety | Find the correct type |
| `embedding?: Float32Array` | Violates invariant 2 | Use `RawChunk` or `IndexedChunk` (two separate types) |
| `function f(id: string)` | IDs are branded | Use `DocumentId`, `ChunkId`, `ProjectId` |
| Mutable interface fields | Immutability guarantee | `readonly` on every field |
| `chunks[i]` + `embeddings[i]` | Implicit coupling | Use `ChunkWithEmbedding` type |
| `catch {}` (empty) | Silent error swallowing | Handle or re-throw `VectorError` |
| `console.log` for output | Bypasses formatting | Use `Logger` or `format.ts` |
| `score > 1.0` or `score < 0.0` | Violates UnitScore invariant | Use `unitScore()` constructor |
| Test with 1 example | Insufficient proof | Use property-based tests (fast-check) |
| Skip reading the spec | Will produce wrong code | Read it. Every time. |

---

## Quick Reference: Data Flow

```
.md file → Parser.parse() → RawChunk[]
  → Embedder.embedBatch() → Float32Array[]
    → ChunkWithEmbedding[] → Store.save() [atomic transaction]
      → Search.query() → SearchResult[] [UnitScore in [0,1]]
```

One direction. No circular dependencies. Parser does not know Store exists.

---

## Quick Reference: File Ownership

| Directory | Owner Role | Key Responsibility |
|-----------|-----------|-------------------|
| `src/types.ts` | Core Engineer | ALL type definitions live here |
| `src/parser/` | Core Engineer | Markdown → RawChunk[], contentHash |
| `src/store/` | Core Engineer | SQLite, atomic save, FTS5 sync |
| `src/search/` | Core Engineer | BM25, cosine, RRF fusion |
| `src/pipeline/` | Core Engineer | Orchestrators (indexer, searcher) |
| `src/config.ts` | Core Engineer | Zod schema, defaults |
| `src/errors.ts` | Core Engineer | VectorError hierarchy |
| `src/embedder/` | Quality Engineer | HuggingFace, Float32Array output |
| `src/invariants.ts` | Quality Engineer | 5 system invariant checks |
| `tests/` | Quality Engineer | All test files |
| `src/cli/` | DX Engineer | Commander, formatting, UX |
| `src/mcp/` | DX Engineer | MCP server, tool schemas |

---

## Reporting Back

When your task is complete, your response must include:

1. **What you did** — specific files created/modified
2. **Which role you followed** — Core/Quality/DX Engineer
3. **Which invariants you verified** — by name
4. **What tests you wrote or ran** — property tests, unit tests, or performance tests
5. **Any concerns** — anything that felt wrong, unclear, or potentially violating the spec
