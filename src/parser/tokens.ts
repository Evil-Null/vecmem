/**
 * Vector Memory Engine — Token Counter
 *
 * Simple word-based token estimator: ~4 chars per token.
 * This matches the spec's guideline and is used by the chunker
 * to enforce maxChunkTokens / minChunkTokens boundaries.
 *
 * The estimate is intentionally simple — precise token counting
 * would require the actual tokenizer (SentencePiece for MiniLM),
 * adding ~5MB of dependencies for marginal accuracy improvement.
 * For chunk boundary decisions, ~4 chars/token is sufficient.
 */

/**
 * Estimate the number of tokens in a text string.
 *
 * Algorithm: trim whitespace, if empty return 0.
 * Otherwise ceil(text.length / 4).
 *
 * @param text - The text to count tokens for
 * @returns Estimated token count (0 for empty/whitespace-only input)
 */
export function countTokens(text: string): number {
  const trimmed = text.trim()
  if (trimmed.length === 0) {
    return 0
  }
  return Math.ceil(trimmed.length / 4)
}
