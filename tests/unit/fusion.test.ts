/**
 * Vector Memory Engine — RRF Fusion Unit Tests
 *
 * Tests cover:
 * - Single-source fusion (BM25 only, vector only)
 * - Dual-source fusion: chunk ranked high in both lists gets highest score
 * - Result normalization: all scores are UnitScore [0, 1]
 * - Correct rank ordering: results sorted descending by score
 * - k parameter effect: higher k reduces difference between ranks
 * - Empty inputs: both empty -> empty results
 * - Highlight generation
 */

import { describe, test, expect, beforeEach, afterEach } from 'vitest'
import { mkdtempSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import Database from 'better-sqlite3'
import { rrfFuse, generateHighlight } from '../../src/search/fusion.js'
import { type UnitScore } from '../../src/types.js'

// ============================================================================
// Helper: validate UnitScore
// ============================================================================

function isValidUnitScore(score: UnitScore): boolean {
  return typeof score === 'number' && score >= 0 && score <= 1
}

// ============================================================================
// rrfFuse tests
// ============================================================================

describe('rrfFuse', () => {
  const k = 60

  // --------------------------------------------------------------------------
  // Empty inputs
  // --------------------------------------------------------------------------

  test('both empty -> empty results', () => {
    const results = rrfFuse([], [], k)
    expect(results).toEqual([])
  })

  // --------------------------------------------------------------------------
  // Single-source fusion
  // --------------------------------------------------------------------------

  test('BM25 only: returns results with vector ranks as 0', () => {
    const bm25 = [
      { chunkId: 'a', rank: 1 },
      { chunkId: 'b', rank: 2 },
      { chunkId: 'c', rank: 3 },
    ]
    const results = rrfFuse(bm25, [], k)

    expect(results).toHaveLength(3)
    // First result should be chunk 'a' (rank 1)
    expect(results[0]!.chunkId).toBe('a')
    // All vector ranks should be 0 (absent)
    for (const r of results) {
      expect(r.vectorRank).toBe(0)
      expect(r.bm25Rank).toBeGreaterThan(0)
    }
  })

  test('vector only: returns results with bm25 ranks as 0', () => {
    const vector = [
      { chunkId: 'x', rank: 1 },
      { chunkId: 'y', rank: 2 },
    ]
    const results = rrfFuse([], vector, k)

    expect(results).toHaveLength(2)
    expect(results[0]!.chunkId).toBe('x')
    for (const r of results) {
      expect(r.bm25Rank).toBe(0)
      expect(r.vectorRank).toBeGreaterThan(0)
    }
  })

  // --------------------------------------------------------------------------
  // Dual-source fusion
  // --------------------------------------------------------------------------

  test('chunk ranked high in both lists gets highest score', () => {
    const bm25 = [
      { chunkId: 'winner', rank: 1 },
      { chunkId: 'bm25-only', rank: 2 },
    ]
    const vector = [
      { chunkId: 'winner', rank: 1 },
      { chunkId: 'vec-only', rank: 2 },
    ]

    const results = rrfFuse(bm25, vector, k)

    // 'winner' appears in both at rank 1 -> highest fused score
    expect(results[0]!.chunkId).toBe('winner')
    expect(results[0]!.bm25Rank).toBe(1)
    expect(results[0]!.vectorRank).toBe(1)

    // Its score should be highest
    for (let i = 1; i < results.length; i++) {
      expect(results[0]!.score).toBeGreaterThanOrEqual(results[i]!.score)
    }
  })

  test('dual-source produces 3 unique results from 2+2 with 1 overlap', () => {
    const bm25 = [
      { chunkId: 'overlap', rank: 1 },
      { chunkId: 'bm25-unique', rank: 2 },
    ]
    const vector = [
      { chunkId: 'overlap', rank: 1 },
      { chunkId: 'vec-unique', rank: 2 },
    ]

    const results = rrfFuse(bm25, vector, k)
    expect(results).toHaveLength(3) // overlap + bm25-unique + vec-unique

    const ids = results.map(r => r.chunkId)
    expect(ids).toContain('overlap')
    expect(ids).toContain('bm25-unique')
    expect(ids).toContain('vec-unique')
  })

  // --------------------------------------------------------------------------
  // Result normalization
  // --------------------------------------------------------------------------

  test('all scores are valid UnitScore [0, 1]', () => {
    const bm25 = [
      { chunkId: 'a', rank: 1 },
      { chunkId: 'b', rank: 5 },
      { chunkId: 'c', rank: 10 },
    ]
    const vector = [
      { chunkId: 'b', rank: 1 },
      { chunkId: 'd', rank: 3 },
      { chunkId: 'a', rank: 7 },
    ]

    const results = rrfFuse(bm25, vector, k)

    for (const r of results) {
      expect(isValidUnitScore(r.score)).toBe(true)
    }
  })

  test('single result gets score 1.0', () => {
    const bm25 = [{ chunkId: 'only', rank: 1 }]
    const results = rrfFuse(bm25, [], k)

    expect(results).toHaveLength(1)
    expect(results[0]!.score).toBeCloseTo(1.0)
  })

  // --------------------------------------------------------------------------
  // Correct rank ordering
  // --------------------------------------------------------------------------

  test('results sorted descending by score', () => {
    const bm25 = [
      { chunkId: 'a', rank: 1 },
      { chunkId: 'b', rank: 3 },
      { chunkId: 'c', rank: 5 },
    ]
    const vector = [
      { chunkId: 'c', rank: 1 },
      { chunkId: 'b', rank: 2 },
      { chunkId: 'a', rank: 4 },
    ]

    const results = rrfFuse(bm25, vector, k)

    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1]!.score).toBeGreaterThanOrEqual(results[i]!.score)
    }
  })

  // --------------------------------------------------------------------------
  // k parameter effect
  // --------------------------------------------------------------------------

  test('higher k reduces difference between rank positions', () => {
    const bm25 = [
      { chunkId: 'top', rank: 1 },
      { chunkId: 'bottom', rank: 100 },
    ]

    // With low k, difference between rank 1 and rank 100 is large
    const resultsLowK = rrfFuse(bm25, [], 1)
    const lowKDiff = resultsLowK[0]!.rrfRaw - resultsLowK[1]!.rrfRaw

    // With high k, difference is smaller
    const resultsHighK = rrfFuse(bm25, [], 1000)
    const highKDiff = resultsHighK[0]!.rrfRaw - resultsHighK[1]!.rrfRaw

    expect(lowKDiff).toBeGreaterThan(highKDiff)
  })

  // --------------------------------------------------------------------------
  // RRF raw score correctness
  // --------------------------------------------------------------------------

  test('rrfRaw matches expected formula', () => {
    const bm25 = [{ chunkId: 'x', rank: 3 }]
    const vector = [{ chunkId: 'x', rank: 7 }]
    const kVal = 60

    const results = rrfFuse(bm25, vector, kVal)

    // rrfRaw = 1/(k + bm25_rank) + 1/(k + vector_rank)
    const expectedRaw = 1 / (kVal + 3) + 1 / (kVal + 7)
    expect(results[0]!.rrfRaw).toBeCloseTo(expectedRaw)
  })

  test('single-source rrfRaw has only one component', () => {
    const bm25 = [{ chunkId: 'x', rank: 5 }]
    const kVal = 60

    const results = rrfFuse(bm25, [], kVal)

    // Only BM25 contributes
    const expectedRaw = 1 / (kVal + 5)
    expect(results[0]!.rrfRaw).toBeCloseTo(expectedRaw)
  })
})

// ============================================================================
// generateHighlight tests
// ============================================================================

describe('generateHighlight', () => {
  let tmpDir: string
  let db: Database.Database

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), 'vector-highlight-'))
    const dbPath = join(tmpDir, 'test.db')
    db = new Database(dbPath)
    db.pragma('foreign_keys = ON')

    // Minimal schema for highlight tests
    db.prepare(`CREATE TABLE documents (
      id TEXT PRIMARY KEY,
      project TEXT NOT NULL,
      file_path TEXT NOT NULL,
      title TEXT NOT NULL DEFAULT '',
      content_hash TEXT NOT NULL,
      file_size INTEGER NOT NULL DEFAULT 0,
      tags TEXT NOT NULL DEFAULT '[]',
      frontmatter TEXT NOT NULL DEFAULT '{}',
      chunk_count INTEGER NOT NULL DEFAULT 0,
      indexed_at TEXT NOT NULL DEFAULT (datetime('now')),
      UNIQUE(project, file_path)
    )`).run()

    db.prepare(`CREATE TABLE chunks (
      id TEXT PRIMARY KEY,
      document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      project TEXT NOT NULL,
      content TEXT NOT NULL,
      content_plain TEXT NOT NULL,
      token_count INTEGER NOT NULL DEFAULT 0,
      heading_path TEXT NOT NULL DEFAULT '[]',
      heading_depth INTEGER NOT NULL DEFAULT 0,
      chunk_index INTEGER NOT NULL,
      has_code_block INTEGER NOT NULL DEFAULT 0
    )`).run()

    db.prepare(`CREATE VIRTUAL TABLE chunks_fts USING fts5(
      chunk_id UNINDEXED,
      title,
      content,
      tags
    )`).run()

    db.prepare(`CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO chunks_fts(chunk_id, title, content, tags)
      SELECT NEW.id,
             COALESCE((SELECT title FROM documents WHERE id = NEW.document_id), ''),
             NEW.content_plain,
             NEW.heading_path;
    END`).run()

    // Insert test data
    db.prepare(`INSERT INTO documents (id, project, file_path, title, content_hash)
      VALUES ('doc1', 'test', '/test.md', 'Test Doc', 'hash1')`).run()

    db.prepare(`INSERT INTO chunks (id, document_id, project, content, content_plain, chunk_index)
      VALUES ('chunk1', 'doc1', 'test', '## Auth\n\nOAuth authentication flow with PKCE',
              'Auth OAuth authentication flow with PKCE', 0)`).run()

    db.prepare(`INSERT INTO chunks (id, document_id, project, content, content_plain, chunk_index)
      VALUES ('chunk2', 'doc1', 'test', '## Setup\n\nConfiguration and setup guide for the project. This section covers all the details needed to get started including installation steps and environment configuration.',
              'Setup Configuration and setup guide for the project. This section covers all the details needed to get started including installation steps and environment configuration.', 1)`).run()
  })

  afterEach(() => {
    db.close()
    rmSync(tmpDir, { recursive: true, force: true })
  })

  test('returns snippet for BM25-matchable content', () => {
    const highlight = generateHighlight(db, 'chunk1', 'authentication')
    expect(highlight.length).toBeGreaterThan(0)
  })

  test('returns first 150 chars for non-matching query', () => {
    const highlight = generateHighlight(db, 'chunk2', 'zzzznonexistenttermzzzz')
    expect(highlight.length).toBeLessThanOrEqual(153) // 150 + "..."
    expect(highlight.length).toBeGreaterThan(0)
  })

  test('returns empty string for nonexistent chunk', () => {
    const highlight = generateHighlight(db, 'nonexistent', 'test')
    expect(highlight).toBe('')
  })
})
