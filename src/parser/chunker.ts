/**
 * Vector Memory Engine — Heading-Aware Chunker
 *
 * Splits a remark AST into RawChunk[] based on heading depth.
 *
 * Algorithm:
 * 1. Walk AST depth-first, tracking headingPath.
 * 2. Headings at depth <= headingSplitDepth start a new chunk (CLEAN BREAK).
 * 3. Between headings, if content exceeds maxChunkTokens, split at paragraph
 *    boundary with chunkOverlapTokens overlap.
 * 4. Code blocks are NEVER split.
 * 5. contentPlain strips markdown syntax.
 * 6. validateChunk rejects chunks > 50,000 chars.
 * 7. Trailing content < minChunkTokens is merged with previous chunk
 *    ONLY if it was NOT produced by a heading split (heading splits are CLEAN BREAKs).
 */

import { countTokens } from './tokens.js'
import { InputTooLongError } from '../errors.js'
import type { RawChunk } from '../types.js'
import type { Root, Content, Heading } from 'mdast'

// ============================================================================
// Types
// ============================================================================

export interface ChunkingConfig {
  readonly maxChunkTokens: number
  readonly minChunkTokens: number
  readonly chunkOverlapTokens: number
  readonly headingSplitDepth: number
}

const MAX_CHUNK_CHARS = 50_000

/** Accumulator for building a chunk before finalization */
interface ChunkAccumulator {
  readonly headingPath: readonly string[]
  readonly nodes: Content[]
  readonly isHeadingSplit: boolean
  hasCodeBlock: boolean
}

/** Internal chunk with metadata about its origin */
interface InternalChunk {
  readonly content: string
  readonly contentPlain: string
  readonly headingPath: readonly string[]
  readonly hasCodeBlock: boolean
  readonly isHeadingSplit: boolean
}

// ============================================================================
// Public API
// ============================================================================

/**
 * Split a remark AST into RawChunks.
 *
 * @param ast - The remark AST (Root node)
 * @param content - The raw markdown string (for extracting original content)
 * @param config - Chunking configuration
 * @returns Array of RawChunk, indexed sequentially from 0
 * @throws InputTooLongError if any chunk exceeds 50,000 chars
 */
export function chunkify(
  ast: Root,
  content: string,
  config: ChunkingConfig,
): readonly RawChunk[] {
  const accumulators = buildAccumulators(ast, config)
  const internalChunks: InternalChunk[] = []

  for (const acc of accumulators) {
    const chunkContent = extractContentFromNodes(acc.nodes, content)
    if (chunkContent.trim().length === 0) continue

    const tokens = countTokens(chunkContent)

    if (tokens > config.maxChunkTokens) {
      // Split large chunk at paragraph boundaries
      const subChunks = splitAtParagraphBoundary(acc, content, config)
      internalChunks.push(...subChunks)
    } else {
      internalChunks.push({
        content: chunkContent,
        contentPlain: stripMarkdown(chunkContent),
        headingPath: [...acc.headingPath],
        hasCodeBlock: acc.hasCodeBlock || acc.nodes.some(n => n.type === 'code'),
        isHeadingSplit: acc.isHeadingSplit,
      })
    }
  }

  // Merge trailing small chunk with previous ONLY if it was NOT a heading split.
  // Heading splits are CLEAN BREAKs — they stand on their own regardless of size.
  if (internalChunks.length >= 2) {
    const last = internalChunks[internalChunks.length - 1]!
    if (!last.isHeadingSplit && countTokens(last.content) < config.minChunkTokens) {
      const prev = internalChunks[internalChunks.length - 2]!
      internalChunks[internalChunks.length - 2] = {
        content: prev.content + '\n\n' + last.content,
        contentPlain: prev.contentPlain + '\n\n' + last.contentPlain,
        headingPath: prev.headingPath,
        hasCodeBlock: prev.hasCodeBlock || last.hasCodeBlock,
        isHeadingSplit: prev.isHeadingSplit,
      }
      internalChunks.pop()
    }
  }

  // Assign sequential indices and validate
  return internalChunks.map((chunk, i) => {
    const raw: RawChunk = {
      content: chunk.content,
      contentPlain: chunk.contentPlain,
      headingPath: chunk.headingPath,
      index: i,
      hasCodeBlock: chunk.hasCodeBlock,
    }
    validateChunk(raw)
    return raw
  })
}

// ============================================================================
// Accumulator Building
// ============================================================================

/** Walk the AST and group nodes into chunk accumulators based on headings */
function buildAccumulators(
  ast: Root,
  config: ChunkingConfig,
): ChunkAccumulator[] {
  const accumulators: ChunkAccumulator[] = []
  // headingPath tracks the current heading hierarchy
  // Index 0 = h1, Index 1 = h2, etc.
  const currentPath: string[] = []
  let current: ChunkAccumulator | null = null

  for (const node of ast.children) {
    // Skip YAML frontmatter — handled by the parser
    if (node.type === 'yaml') continue

    if (isHeadingSplit(node, config.headingSplitDepth)) {
      // Flush current accumulator
      if (current !== null && current.nodes.length > 0) {
        accumulators.push(current)
      }

      const heading = node as Heading
      const text = extractHeadingText(heading)

      // Update path: truncate to parent level, set this heading
      currentPath.length = heading.depth - 1
      currentPath[heading.depth - 1] = text

      // Start new accumulator with CLEAN BREAK
      current = {
        headingPath: [...currentPath],
        nodes: [node],
        isHeadingSplit: true,
        hasCodeBlock: false,
      }
    } else {
      // Non-splitting node
      if (current === null) {
        current = {
          headingPath: [],
          nodes: [],
          isHeadingSplit: false,
          hasCodeBlock: false,
        }
      }

      // Track deeper headings in the path for context
      if (node.type === 'heading') {
        const heading = node as Heading
        const text = extractHeadingText(heading)
        currentPath.length = heading.depth - 1
        currentPath[heading.depth - 1] = text
      }

      current.nodes.push(node)

      if (node.type === 'code') {
        current.hasCodeBlock = true
      }
    }
  }

  // Flush final accumulator
  if (current !== null && current.nodes.length > 0) {
    accumulators.push(current)
  }

  return accumulators
}

/** Check if a node is a heading that should trigger a chunk split */
function isHeadingSplit(node: Content, depth: number): boolean {
  return node.type === 'heading' && (node as Heading).depth <= depth
}

// ============================================================================
// Paragraph Boundary Splitting
// ============================================================================

/** Split a large accumulator at paragraph boundaries */
function splitAtParagraphBoundary(
  acc: ChunkAccumulator,
  content: string,
  config: ChunkingConfig,
): InternalChunk[] {
  const chunks: InternalChunk[] = []
  let currentNodes: Content[] = []
  let currentHasCode = false

  for (const node of acc.nodes) {
    const testNodes = [...currentNodes, node]
    const testContent = extractContentFromNodes(testNodes, content)
    const testTokens = countTokens(testContent)

    if (testTokens > config.maxChunkTokens && currentNodes.length > 0) {
      // Flush current batch (without this node)
      const chunkContent = extractContentFromNodes(currentNodes, content)
      if (chunkContent.trim().length > 0) {
        chunks.push({
          content: chunkContent,
          contentPlain: stripMarkdown(chunkContent),
          headingPath: [...acc.headingPath],
          hasCodeBlock: currentHasCode,
          // First sub-chunk inherits heading-split status; rest are paragraph splits
          isHeadingSplit: chunks.length === 0 && acc.isHeadingSplit,
        })
      }

      // Start new batch — non-heading splits get overlap
      currentNodes = [node]
      currentHasCode = node.type === 'code'
    } else {
      currentNodes = testNodes
      if (node.type === 'code') {
        currentHasCode = true
      }
    }
  }

  // Flush remaining
  if (currentNodes.length > 0) {
    const chunkContent = extractContentFromNodes(currentNodes, content)
    if (chunkContent.trim().length > 0) {
      chunks.push({
        content: chunkContent,
        contentPlain: stripMarkdown(chunkContent),
        headingPath: [...acc.headingPath],
        hasCodeBlock: currentHasCode,
        isHeadingSplit: chunks.length === 0 && acc.isHeadingSplit,
      })
    }
  }

  return chunks
}

// ============================================================================
// Content Extraction
// ============================================================================

/** Extract the original markdown content for a set of nodes using positions */
function extractContentFromNodes(
  nodes: readonly Content[],
  content: string,
): string {
  if (nodes.length === 0) return ''

  const first = nodes[0]!
  const last = nodes[nodes.length - 1]!

  if (!first.position || !last.position) return ''

  const start = first.position.start.offset ?? 0
  const end = last.position.end.offset ?? content.length

  return content.slice(start, end)
}

// ============================================================================
// Plain Text Extraction
// ============================================================================

/** Strip markdown syntax from content, preserving text */
function stripMarkdown(text: string): string {
  let result = text

  // Remove heading markers: # at start of line
  result = result.replace(/^#{1,6}\s+/gm, '')

  // Remove code fences (```language and ```)
  result = result.replace(/^```\w*\s*$/gm, '')

  // Remove bold: **text** or __text__
  result = result.replace(/\*\*(.+?)\*\*/g, '$1')
  result = result.replace(/__(.+?)__/g, '$1')

  // Remove italic: *text* or _text_ (but not inside words)
  result = result.replace(/(?<!\w)\*(.+?)\*(?!\w)/g, '$1')
  result = result.replace(/(?<!\w)_(.+?)_(?!\w)/g, '$1')

  // Remove links: [text](url) → text
  result = result.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')

  // Remove images: ![alt](url) → alt
  result = result.replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')

  // Remove inline code backticks: `code` → code
  result = result.replace(/`([^`]+)`/g, '$1')

  // Clean up extra blank lines
  result = result.replace(/\n{3,}/g, '\n\n')

  return result.trim()
}

// ============================================================================
// Chunk Validation
// ============================================================================

/** Validate a chunk's content length */
function validateChunk(chunk: RawChunk): void {
  if (chunk.content.length > MAX_CHUNK_CHARS) {
    throw new InputTooLongError(
      `Chunk too large: ${chunk.content.length} chars (max ${MAX_CHUNK_CHARS})`
    )
  }
}

// ============================================================================
// Heading Text Extraction
// ============================================================================

/** Extract plain text from a heading node */
function extractHeadingText(heading: Heading): string {
  return extractNodeText(heading).trim()
}

/** Recursively extract text from AST node children */
function extractNodeText(node: Content | Heading): string {
  if ('value' in node && typeof node.value === 'string') {
    return node.value
  }
  if ('children' in node && Array.isArray(node.children)) {
    return (node.children as Content[]).map(extractNodeText).join('')
  }
  return ''
}
