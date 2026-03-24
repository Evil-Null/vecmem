/**
 * Vector Memory Engine — Markdown Parser
 *
 * Parses .md files into DocumentMeta + RawChunk[] using remark AST.
 * Computes contentHash (SHA256 of raw file bytes) — SINGLE SOURCE OF TRUTH.
 *
 * Security: validateFilePath() prevents path traversal.
 * Error handling: FileNotFoundError, InvalidMarkdownError, SecurityError.
 */

import { readFileSync, existsSync } from 'node:fs'
import { resolve, basename, extname } from 'node:path'
import { createHash } from 'node:crypto'
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkFrontmatter from 'remark-frontmatter'
import { parse as parseYaml } from 'yaml'
import { chunkify } from './chunker.js'
import { FileNotFoundError, SecurityError } from '../errors.js'
import { DEFAULT_CONFIG } from '../config.js'
import type { Parser, ParseResult, DocumentMeta, RawChunk } from '../types.js'
import { createProjectId } from '../types.js'
import type { Root, Content } from 'mdast'

// ============================================================================
// Path Validation
// ============================================================================

/**
 * Validate a file path against a project root to prevent path traversal.
 * Resolves the path, checks it starts with projectRoot, and verifies the file exists.
 *
 * @param filePath - The path to validate
 * @param projectRoot - The root directory that the path must be within
 * @returns The resolved absolute path
 * @throws SecurityError if the path escapes projectRoot
 * @throws FileNotFoundError if the file does not exist
 */
export function validateFilePath(filePath: string, projectRoot: string): string {
  const resolved = resolve(filePath)
  const resolvedRoot = resolve(projectRoot)

  if (!resolved.startsWith(resolvedRoot)) {
    throw new SecurityError(`Path traversal detected: ${filePath}`)
  }

  if (!existsSync(resolved)) {
    throw new FileNotFoundError(`File not found: ${resolved}`)
  }

  return resolved
}

// ============================================================================
// Remark Processor
// ============================================================================

const processor = unified()
  .use(remarkParse)
  .use(remarkFrontmatter, ['yaml'])

// ============================================================================
// MarkdownParser
// ============================================================================

export class MarkdownParser implements Parser {
  private readonly config: {
    readonly maxChunkTokens: number
    readonly minChunkTokens: number
    readonly chunkOverlapTokens: number
    readonly headingSplitDepth: number
    readonly project: string
  }

  constructor(config?: {
    readonly maxChunkTokens?: number
    readonly minChunkTokens?: number
    readonly chunkOverlapTokens?: number
    readonly headingSplitDepth?: number
    readonly project?: string
  }) {
    this.config = {
      maxChunkTokens: config?.maxChunkTokens ?? DEFAULT_CONFIG.maxChunkTokens,
      minChunkTokens: config?.minChunkTokens ?? DEFAULT_CONFIG.minChunkTokens,
      chunkOverlapTokens: config?.chunkOverlapTokens ?? DEFAULT_CONFIG.chunkOverlapTokens,
      headingSplitDepth: config?.headingSplitDepth ?? DEFAULT_CONFIG.headingSplitDepth,
      project: config?.project ?? DEFAULT_CONFIG.project,
    }
  }

  /**
   * Parse a markdown file into DocumentMeta + RawChunk[].
   *
   * @param filePath - Absolute path to the .md file
   * @returns ParseResult with document metadata and raw chunks
   * @throws FileNotFoundError if file does not exist
   * @throws InputTooLongError if any chunk exceeds 50,000 chars
   */
  parse(filePath: string): ParseResult {
    const resolvedPath = resolve(filePath)

    if (!existsSync(resolvedPath)) {
      throw new FileNotFoundError(`File not found: ${resolvedPath}`)
    }

    // Read raw bytes for contentHash (single source of truth)
    const rawBytes = readFileSync(resolvedPath)
    const contentHash = createHash('sha256').update(rawBytes).digest('hex')
    const content = rawBytes.toString('utf-8')
    const fileSize = rawBytes.byteLength

    // Parse AST with remark
    const ast = processor.parse(content) as Root

    // Extract frontmatter
    const { frontmatter, title: fmTitle, tags } = extractFrontmatter(ast)

    // Extract title: frontmatter title > first h1 > filename
    // Empty or whitespace-only titles fall through to next source
    const h1Title = extractFirstH1(ast)
    const title = (fmTitle && fmTitle.trim().length > 0 ? fmTitle : null)
      ?? (h1Title && h1Title.trim().length > 0 ? h1Title : null)
      ?? basename(resolvedPath, extname(resolvedPath))

    // Build chunks
    const chunks: readonly RawChunk[] = chunkify(ast, content, {
      maxChunkTokens: this.config.maxChunkTokens,
      minChunkTokens: this.config.minChunkTokens,
      chunkOverlapTokens: this.config.chunkOverlapTokens,
      headingSplitDepth: this.config.headingSplitDepth,
    })

    const document: DocumentMeta = {
      title,
      filePath: resolvedPath,
      project: createProjectId(this.config.project),
      contentHash,
      tags,
      frontmatter,
      indexedAt: new Date(),
      fileSize,
    }

    return { document, chunks }
  }
}

// ============================================================================
// Internal helpers
// ============================================================================

interface FrontmatterResult {
  readonly frontmatter: Readonly<Record<string, unknown>>
  readonly title: string | undefined
  readonly tags: readonly string[]
}

/** Extract frontmatter YAML from the AST */
function extractFrontmatter(ast: Root): FrontmatterResult {
  const yamlNode = ast.children.find(
    (node): node is Content & { type: 'yaml'; value: string } =>
      node.type === 'yaml'
  )

  if (!yamlNode) {
    return { frontmatter: {}, title: undefined, tags: [] }
  }

  const parsed = parseYaml(yamlNode.value) as Record<string, unknown> | null

  if (!parsed || typeof parsed !== 'object') {
    return { frontmatter: {}, title: undefined, tags: [] }
  }

  const title = typeof parsed['title'] === 'string' ? parsed['title'] : undefined

  const rawTags = parsed['tags']
  const tags: readonly string[] = Array.isArray(rawTags)
    ? rawTags.filter((t): t is string => typeof t === 'string')
    : []

  return { frontmatter: parsed, title, tags }
}

/** Extract the text of the first h1 heading in the AST */
function extractFirstH1(ast: Root): string | undefined {
  for (const node of ast.children) {
    if (node.type === 'heading' && node.depth === 1) {
      return extractTextContent(node)
    }
  }
  return undefined
}

/** Recursively extract plain text from an AST node */
function extractTextContent(node: Content | Root): string {
  if ('value' in node && typeof node.value === 'string') {
    return node.value
  }
  if ('children' in node && Array.isArray(node.children)) {
    return (node.children as Content[]).map(extractTextContent).join('')
  }
  return ''
}
