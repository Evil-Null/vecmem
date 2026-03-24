/**
 * Vector Memory Engine — Search Pipeline Orchestrator
 *
 * Orchestrates: embed query → BM25 + vector → RRF fuse → SearchResult[]
 *
 * Graceful degradation:
 * - If embedder throws ModelLoadError → fall back to BM25-only
 * - All other errors re-throw (not swallowed)
 * - Logs search.degraded warning on fallback
 *
 * Score guarantee: all returned scores are UnitScore in [0, 1].
 */

import type Database from 'better-sqlite3'
import { Bm25Search } from '../search/bm25.js'
import { VectorSearch } from '../search/vector.js'
import { rrfFuse, generateHighlight, type FusionResult } from '../search/fusion.js'
import { ModelLoadError } from '../errors.js'
import type {
  Search,
  SearchResult,
  SearchOptions,
  Embedder,
  Logger,
  IndexedChunk,
  DocumentId,
  ChunkId,
} from '../types.js'

// ============================================================================
// Search Configuration
// ============================================================================

interface SearchConfig {
  readonly defaultTopK: number
  readonly rrfK: number
  readonly minScore: number
}

// ============================================================================
// SearchOrchestrator
// ============================================================================

export class SearchOrchestrator implements Search {
  private readonly db: Database.Database
  private readonly embedder: Embedder
  private readonly logger: Logger
  private readonly config: SearchConfig
  private readonly bm25: Bm25Search
  private readonly vectorSearch: VectorSearch

  constructor(
    db: Database.Database,
    embedder: Embedder,
    logger: Logger,
    config: SearchConfig,
  ) {
    this.db = db
    this.embedder = embedder
    this.logger = logger
    this.config = config
    this.bm25 = new Bm25Search(db)
    this.vectorSearch = new VectorSearch(db)
  }

  /**
   * Execute a hybrid search: BM25 + vector + RRF fusion.
   *
   * Falls back to BM25-only if the embedder throws ModelLoadError.
   * All other errors are re-thrown.
   *
   * @param text - Query string
   * @param options - Optional topK, minScore, project filter
   * @returns SearchResult[] sorted descending by score, bounded to [0, 1]
   */
  async query(text: string, options?: SearchOptions): Promise<SearchResult[]> {
    const startMs = performance.now()
    const topK = options?.topK ?? this.config.defaultTopK
    const minScore = options?.minScore ?? this.config.minScore
    const project = options?.project

    // Fetch a larger set for fusion, then trim to topK
    const fetchK = topK * 5

    // 1. BM25 search — always runs
    const bm25Results = this.bm25.search(text, {
      topK: fetchK,
      project: project as string | undefined,
    })

    // 2. Vector search — may degrade gracefully
    let vectorResults: Array<{ readonly chunkId: string; readonly rank: number }> = []
    let degraded = false

    try {
      const queryEmbedding = await this.embedder.embed(text)
      vectorResults = this.vectorSearch.search(queryEmbedding, {
        topK: fetchK,
        project: project as string | undefined,
      })
    } catch (err: unknown) {
      if (err instanceof ModelLoadError) {
        degraded = true
        this.logger.warn('search.degraded', {
          reason: 'model_unavailable',
          fallback: 'bm25_only',
        })
      } else {
        throw err
      }
    }

    // 3. RRF fusion
    const fused = rrfFuse(bm25Results, vectorResults, this.config.rrfK)

    // 4. Build full SearchResult objects
    const results = this.buildResults(fused, text, topK, minScore)

    const elapsedMs = Math.round(performance.now() - startMs)
    this.logger.debug('search.completed', {
      query: text,
      results: results.length,
      degraded,
      elapsed_ms: elapsedMs,
    })

    return results
  }

  // --------------------------------------------------------------------------
  // Private — build SearchResult from FusionResult
  // --------------------------------------------------------------------------

  private buildResults(
    fused: FusionResult[],
    query: string,
    topK: number,
    minScore: number,
  ): SearchResult[] {
    const results: SearchResult[] = []

    for (const f of fused) {
      // Filter by minScore
      if (f.score < minScore) {
        continue
      }

      // Load the chunk from DB
      const chunk = this.loadChunk(f.chunkId)
      if (chunk === null) {
        continue
      }

      // Load document metadata
      const doc = this.loadDocumentMeta(chunk.documentId)

      // Generate highlight
      const highlight = generateHighlight(this.db, f.chunkId, query)

      results.push({
        chunk,
        score: f.score,
        scores: {
          bm25Rank: f.bm25Rank,
          vectorRank: f.vectorRank,
          rrfRaw: f.rrfRaw,
        },
        documentTitle: doc.title,
        documentPath: doc.filePath,
        highlight,
      })

      // Stop at topK
      if (results.length >= topK) {
        break
      }
    }

    return results
  }

  /**
   * Load a single IndexedChunk by its ID, including embedding.
   */
  private loadChunk(chunkId: string): IndexedChunk | null {
    const row = this.db.prepare(`
      SELECT c.*, e.vector, e.dimensions
      FROM chunks c
      JOIN embeddings e ON e.chunk_id = c.id
      WHERE c.id = ?
    `).get(chunkId) as ChunkRow | undefined

    if (row === undefined) {
      return null
    }

    // CRITICAL: Proper BLOB -> Float32Array alignment
    const uint8 = new Uint8Array(row.vector)
    const embedding = new Float32Array(uint8.buffer)

    return {
      id: row.id as ChunkId,
      documentId: row.document_id as DocumentId,
      content: row.content,
      contentPlain: row.content_plain,
      headingPath: JSON.parse(row.heading_path) as readonly string[],
      index: row.chunk_index,
      hasCodeBlock: row.has_code_block === 1,
      embedding,
    }
  }

  /**
   * Load document title and file path for a given document ID.
   */
  private loadDocumentMeta(documentId: DocumentId): { title: string; filePath: string } {
    const row = this.db.prepare(
      'SELECT title, file_path FROM documents WHERE id = ?',
    ).get(documentId) as { title: string; file_path: string } | undefined

    if (row === undefined) {
      return { title: '', filePath: '' }
    }

    return { title: row.title, filePath: row.file_path }
  }
}

// ============================================================================
// Internal row type
// ============================================================================

interface ChunkRow {
  readonly id: string
  readonly document_id: string
  readonly content: string
  readonly content_plain: string
  readonly heading_path: string
  readonly chunk_index: number
  readonly has_code_block: number
  readonly vector: Buffer
  readonly dimensions: number
}
