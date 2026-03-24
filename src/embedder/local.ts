/**
 * Vector Memory Engine — Local Embedder
 *
 * Implements the Embedder interface using @huggingface/transformers
 * with Xenova/all-MiniLM-L6-v2 (384-dimensional, L2-normalized).
 *
 * Key design:
 * - Model loaded LAZILY on first embed()/embedBatch() call
 * - Model cached at configurable path (default: ~/.vector/models/)
 * - Returns Float32Array — consumers never deal with raw number[]
 * - Deterministic: same input -> same output within same process
 * - Throws EmptyInputError for empty/whitespace text
 */

import { pipeline, env, type FeatureExtractionPipeline } from '@huggingface/transformers'
import { EmptyInputError, ModelLoadError } from '../errors.js'
import type { Embedder } from './interface.js'

// ============================================================================
// Configuration
// ============================================================================

export interface LocalEmbedderOptions {
  readonly model: string
  readonly cachePath: string
}

// ============================================================================
// Constants
// ============================================================================

const EMBEDDING_DIMENSIONS = 384

// ============================================================================
// LocalEmbedder
// ============================================================================

export class LocalEmbedder implements Embedder {
  private readonly model: string
  private readonly cachePath: string
  private pipe: FeatureExtractionPipeline | null = null
  private loading: Promise<FeatureExtractionPipeline> | null = null

  constructor(options: LocalEmbedderOptions) {
    this.model = options.model
    this.cachePath = options.cachePath
  }

  /** Fixed embedding dimensions for all-MiniLM-L6-v2 */
  get dimensions(): number {
    return EMBEDDING_DIMENSIONS
  }

  /**
   * Embed a single text string into a 384-dimensional Float32Array.
   *
   * @throws EmptyInputError if text is empty or whitespace-only
   * @throws ModelLoadError if the model fails to load
   */
  async embed(text: string): Promise<Float32Array> {
    this.validateInput(text)
    const pipe = await this.ensurePipeline()
    const output = await pipe(text, { pooling: 'mean', normalize: true })
    return new Float32Array(output.data as Float32Array)
  }

  /**
   * Embed multiple texts in a single batch operation.
   * Returns one Float32Array per input text.
   *
   * Batch processing is more efficient than individual embed() calls —
   * one model invocation for N texts vs N invocations.
   *
   * @throws EmptyInputError if any text is empty or whitespace-only
   * @throws ModelLoadError if the model fails to load
   */
  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) {
      return []
    }

    for (const text of texts) {
      this.validateInput(text)
    }

    const pipe = await this.ensurePipeline()
    const output = await pipe(texts, { pooling: 'mean', normalize: true })
    const rawData = output.data as Float32Array

    // Tensor shape is [N, 384] — slice into individual Float32Arrays
    const results: Float32Array[] = []
    for (let i = 0; i < texts.length; i++) {
      const start = i * EMBEDDING_DIMENSIONS
      const end = start + EMBEDDING_DIMENSIONS
      results.push(new Float32Array(rawData.slice(start, end)))
    }

    return results
  }

  // ==========================================================================
  // Private
  // ==========================================================================

  /**
   * Validate input text. Throws EmptyInputError for empty or whitespace-only strings.
   */
  private validateInput(text: string): void {
    if (text.trim().length === 0) {
      throw new EmptyInputError('Embedding input must not be empty or whitespace-only')
    }
  }

  /**
   * Lazily load the model pipeline. Subsequent calls return the cached pipeline.
   * Uses a shared promise to prevent duplicate model loads from concurrent calls.
   */
  private async ensurePipeline(): Promise<FeatureExtractionPipeline> {
    if (this.pipe) {
      return this.pipe
    }

    // Prevent duplicate concurrent loads
    if (this.loading) {
      return this.loading
    }

    this.loading = this.loadPipeline()

    try {
      this.pipe = await this.loading
      return this.pipe
    } catch (error: unknown) {
      // Reset loading so retry is possible
      this.loading = null
      const message = error instanceof Error ? error.message : String(error)
      throw new ModelLoadError(
        `Failed to load model '${this.model}': ${message}`,
      )
    }
  }

  /**
   * Load the HuggingFace pipeline for feature extraction.
   */
  private async loadPipeline(): Promise<FeatureExtractionPipeline> {
    env.cacheDir = this.cachePath
    return await pipeline('feature-extraction', this.model, {
      dtype: 'fp32',
    })
  }
}
