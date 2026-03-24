/**
 * Vector Memory Engine — Chunker Unit Tests
 *
 * Tests the markdown parser and heading-aware chunker.
 * Covers: heading-based splitting, headingPath extraction, code block preservation,
 * frontmatter extraction, empty files, contentHash, chunk indices, contentPlain,
 * hasCodeBlock, and security (path traversal).
 */

import { describe, test, expect } from 'vitest'
import { join, resolve } from 'node:path'
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { MarkdownParser, validateFilePath } from '../../src/parser/markdown.js'
import { SecurityError, InputTooLongError, FileNotFoundError } from '../../src/errors.js'
import type { RawChunk } from '../../src/types.js'

const FIXTURES = resolve(import.meta.dirname, '../fixtures')

function createParser(): MarkdownParser {
  return new MarkdownParser()
}

describe('MarkdownParser', () => {
  // ========================================================================
  // Heading-based splitting
  // ========================================================================

  describe('heading-based splitting', () => {
    test('h1 creates a new chunk', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'simple.md'))
      // simple.md: # Simple Document + content
      // Should produce at least 1 chunk starting with h1
      expect(result.chunks.length).toBeGreaterThanOrEqual(1)
      expect(result.chunks[0]!.content).toContain('# Simple Document')
    })

    test('h2 creates a new chunk (headingSplitDepth: 2)', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'frontmatter.md'))
      // frontmatter.md: # Frontmatter Document, ## Section One
      // With headingSplitDepth: 2, h2 splits into a new chunk
      const sectionChunk = result.chunks.find(c =>
        c.content.includes('## Section One')
      )
      expect(sectionChunk).toBeDefined()
    })

    test('h3 does NOT create a new chunk with headingSplitDepth: 2', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'deep-headings.md'))
      // deep-headings.md has h1, h2, h3, h4 headings
      // h3 and h4 should NOT be split points — they stay within their h2 parent chunk
      const level2Chunk = result.chunks.find(c =>
        c.content.includes('## Level 2')
      )
      expect(level2Chunk).toBeDefined()
      // The h3 content should be in the same chunk as its h2 parent
      expect(level2Chunk!.content).toContain('### Level 3')
      expect(level2Chunk!.content).toContain('Third level content.')
    })

    test('deep-headings produces correct chunk count', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'deep-headings.md'))
      // Expected chunks: # Level 1, ## Level 2 (with ### and ####), ## Another Level 2 (with ###)
      expect(result.chunks.length).toBe(3)
    })
  })

  // ========================================================================
  // headingPath extraction
  // ========================================================================

  describe('headingPath extraction', () => {
    test('chunk under h2 under h1 has correct headingPath', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'frontmatter.md'))
      // # Frontmatter Document → ## Section One
      const sectionChunk = result.chunks.find(c =>
        c.content.includes('## Section One')
      )
      expect(sectionChunk).toBeDefined()
      expect(sectionChunk!.headingPath).toEqual(['Frontmatter Document', 'Section One'])
    })

    test('h1-only chunk has single-element headingPath', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'simple.md'))
      expect(result.chunks[0]!.headingPath).toEqual(['Simple Document'])
    })

    test('headingPath resets when a new h1 appears', () => {
      const tmpDir = mkdtempSync(join(tmpdir(), 'vector-test-'))
      const filePath = join(tmpDir, 'two-h1s.md')
      try {
        writeFileSync(filePath, '# First\n\nContent.\n\n# Second\n\nMore content.\n')
        const parser = createParser()
        const result = parser.parse(filePath)
        expect(result.chunks.length).toBe(2)
        expect(result.chunks[0]!.headingPath).toEqual(['First'])
        expect(result.chunks[1]!.headingPath).toEqual(['Second'])
      } finally {
        rmSync(tmpDir, { recursive: true })
      }
    })

    test('headingPath for nested h2 under different h1s', () => {
      const tmpDir = mkdtempSync(join(tmpdir(), 'vector-test-'))
      const filePath = join(tmpDir, 'nested.md')
      try {
        writeFileSync(filePath, [
          '# Auth',
          '',
          'Introduction.',
          '',
          '## Login',
          '',
          'Login content.',
          '',
          '# Settings',
          '',
          '## Profile',
          '',
          'Profile content.',
        ].join('\n'))
        const parser = createParser()
        const result = parser.parse(filePath)
        const loginChunk = result.chunks.find(c => c.content.includes('## Login'))
        const profileChunk = result.chunks.find(c => c.content.includes('## Profile'))
        expect(loginChunk!.headingPath).toEqual(['Auth', 'Login'])
        expect(profileChunk!.headingPath).toEqual(['Settings', 'Profile'])
      } finally {
        rmSync(tmpDir, { recursive: true })
      }
    })

    test('deep-headings headingPath tracks nested structure', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'deep-headings.md'))
      // ## Another Level 2 is under # Level 1
      const anotherL2 = result.chunks.find(c =>
        c.content.includes('## Another Level 2')
      )
      expect(anotherL2).toBeDefined()
      expect(anotherL2!.headingPath).toEqual(['Level 1', 'Another Level 2'])
    })
  })

  // ========================================================================
  // Code block preservation
  // ========================================================================

  describe('code block preservation', () => {
    test('code block is never split mid-block', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'code-blocks.md'))
      // The typescript code block should be entirely in one chunk
      const tsChunk = result.chunks.find(c =>
        c.content.includes('function greet(name: string)')
      )
      expect(tsChunk).toBeDefined()
      expect(tsChunk!.content).toContain('```typescript')
      expect(tsChunk!.content).toContain('console.log(result)')
      expect(tsChunk!.content).toContain('```')
    })

    test('hasCodeBlock is true when chunk contains fenced code block', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'code-blocks.md'))
      const codeChunk = result.chunks.find(c =>
        c.content.includes('```typescript')
      )
      expect(codeChunk).toBeDefined()
      expect(codeChunk!.hasCodeBlock).toBe(true)
    })

    test('hasCodeBlock is false when chunk has no code block', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'simple.md'))
      // simple.md has no code blocks
      for (const chunk of result.chunks) {
        expect(chunk.hasCodeBlock).toBe(false)
      }
    })
  })

  // ========================================================================
  // Frontmatter extraction
  // ========================================================================

  describe('frontmatter extraction', () => {
    test('YAML frontmatter parsed into DocumentMeta', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'frontmatter.md'))
      expect(result.document.frontmatter).toEqual({
        title: 'Frontmatter Test',
        tags: ['test', 'markdown', 'fixture'],
        author: 'Vector',
      })
    })

    test('tags extracted from frontmatter', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'frontmatter.md'))
      expect(result.document.tags).toEqual(['test', 'markdown', 'fixture'])
    })

    test('title from frontmatter takes precedence', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'frontmatter.md'))
      expect(result.document.title).toBe('Frontmatter Test')
    })

    test('title from first h1 when no frontmatter title', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'simple.md'))
      expect(result.document.title).toBe('Simple Document')
    })

    test('title falls back to filename when no h1 and no frontmatter', () => {
      const tmpDir = mkdtempSync(join(tmpdir(), 'vector-test-'))
      const filePath = join(tmpDir, 'no-heading.md')
      try {
        writeFileSync(filePath, 'Just some text without any heading.\n')
        const parser = createParser()
        const result = parser.parse(filePath)
        expect(result.document.title).toBe('no-heading')
      } finally {
        rmSync(tmpDir, { recursive: true })
      }
    })

    test('large fixture frontmatter parsed correctly', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'large.md'))
      expect(result.document.frontmatter).toMatchObject({
        title: 'Nexus API Documentation',
        version: '2.4.0',
        tags: ['api', 'rest', 'authentication', 'webhooks', 'rate-limiting'],
        author: 'Platform Team',
      })
      expect(result.document.tags).toEqual(['api', 'rest', 'authentication', 'webhooks', 'rate-limiting'])
    })

    test('file without frontmatter has empty frontmatter and tags', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'simple.md'))
      expect(result.document.frontmatter).toEqual({})
      expect(result.document.tags).toEqual([])
    })
  })

  // ========================================================================
  // Empty file handling
  // ========================================================================

  describe('empty file handling', () => {
    test('empty file returns empty chunks array', () => {
      const tmpDir = mkdtempSync(join(tmpdir(), 'vector-test-'))
      const filePath = join(tmpDir, 'empty.md')
      try {
        writeFileSync(filePath, '')
        const parser = createParser()
        const result = parser.parse(filePath)
        expect(result.chunks).toEqual([])
      } finally {
        rmSync(tmpDir, { recursive: true })
      }
    })

    test('file with only whitespace returns empty chunks array', () => {
      const tmpDir = mkdtempSync(join(tmpdir(), 'vector-test-'))
      const filePath = join(tmpDir, 'whitespace.md')
      try {
        writeFileSync(filePath, '   \n\n  \n')
        const parser = createParser()
        const result = parser.parse(filePath)
        expect(result.chunks).toEqual([])
      } finally {
        rmSync(tmpDir, { recursive: true })
      }
    })
  })

  // ========================================================================
  // File with only frontmatter
  // ========================================================================

  describe('file with only frontmatter', () => {
    test('returns document with metadata, empty chunks', () => {
      const tmpDir = mkdtempSync(join(tmpdir(), 'vector-test-'))
      const filePath = join(tmpDir, 'only-frontmatter.md')
      try {
        writeFileSync(filePath, '---\ntitle: Just Meta\ntags: [meta]\n---\n')
        const parser = createParser()
        const result = parser.parse(filePath)
        expect(result.document.title).toBe('Just Meta')
        expect(result.document.tags).toEqual(['meta'])
        expect(result.chunks).toEqual([])
      } finally {
        rmSync(tmpDir, { recursive: true })
      }
    })
  })

  // ========================================================================
  // contentHash computation
  // ========================================================================

  describe('contentHash computation', () => {
    test('contentHash is SHA256 of raw file bytes', () => {
      const parser = createParser()
      const filePath = join(FIXTURES, 'simple.md')
      const result = parser.parse(filePath)
      const rawBytes = readFileSync(filePath)
      const expectedHash = createHash('sha256').update(rawBytes).digest('hex')
      expect(result.document.contentHash).toBe(expectedHash)
    })

    test('contentHash is deterministic — same file, same hash', () => {
      const parser = createParser()
      const filePath = join(FIXTURES, 'frontmatter.md')
      const result1 = parser.parse(filePath)
      const result2 = parser.parse(filePath)
      expect(result1.document.contentHash).toBe(result2.document.contentHash)
    })
  })

  // ========================================================================
  // Chunk indices
  // ========================================================================

  describe('chunk indices', () => {
    test('chunk indices are sequential starting from 0', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'deep-headings.md'))
      for (let i = 0; i < result.chunks.length; i++) {
        expect(result.chunks[i]!.index).toBe(i)
      }
    })

    test('single chunk has index 0', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'simple.md'))
      expect(result.chunks.length).toBe(1)
      expect(result.chunks[0]!.index).toBe(0)
    })

    test('large file has sequential indices', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'large.md'))
      for (let i = 0; i < result.chunks.length; i++) {
        expect(result.chunks[i]!.index).toBe(i)
      }
      // large.md has many h2 sections — should produce multiple chunks
      expect(result.chunks.length).toBeGreaterThan(3)
    })
  })

  // ========================================================================
  // contentPlain
  // ========================================================================

  describe('contentPlain', () => {
    test('markdown syntax stripped, text content preserved', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'simple.md'))
      const plain = result.chunks[0]!.contentPlain
      // # markers stripped
      expect(plain).not.toContain('# ')
      // Text preserved
      expect(plain).toContain('Simple Document')
      expect(plain).toContain('Item one')
    })

    test('bold and italic markers stripped', () => {
      const tmpDir = mkdtempSync(join(tmpdir(), 'vector-test-'))
      const filePath = join(tmpDir, 'formatting.md')
      try {
        writeFileSync(filePath, '# Test\n\n**bold** and *italic* text.\n')
        const parser = createParser()
        const result = parser.parse(filePath)
        const plain = result.chunks[0]!.contentPlain
        expect(plain).toContain('bold')
        expect(plain).toContain('italic')
        expect(plain).not.toContain('**')
        expect(plain).not.toContain('*italic*')
      } finally {
        rmSync(tmpDir, { recursive: true })
      }
    })

    test('link syntax stripped, text preserved', () => {
      const tmpDir = mkdtempSync(join(tmpdir(), 'vector-test-'))
      const filePath = join(tmpDir, 'links.md')
      try {
        writeFileSync(filePath, '# Links\n\nVisit [Google](https://google.com) for search.\n')
        const parser = createParser()
        const result = parser.parse(filePath)
        const plain = result.chunks[0]!.contentPlain
        expect(plain).toContain('Google')
        expect(plain).not.toContain('[Google]')
        expect(plain).not.toContain('(https://google.com)')
      } finally {
        rmSync(tmpDir, { recursive: true })
      }
    })

    test('code fences stripped from contentPlain', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'code-blocks.md'))
      const chunk = result.chunks[0]!
      expect(chunk.contentPlain).not.toContain('```')
      // But code text content is preserved
      expect(chunk.contentPlain).toContain('function greet')
    })
  })

  // ========================================================================
  // hasCodeBlock
  // ========================================================================

  describe('hasCodeBlock', () => {
    test('true when chunk contains fenced code block', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'code-blocks.md'))
      // All chunks in code-blocks.md should have code blocks (single chunk with all content)
      const codeChunks = result.chunks.filter(c => c.hasCodeBlock)
      expect(codeChunks.length).toBeGreaterThanOrEqual(1)
    })

    test('false when chunk has no code block', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'simple.md'))
      expect(result.chunks.every(c => !c.hasCodeBlock)).toBe(true)
    })
  })

  // ========================================================================
  // Large file chunking
  // ========================================================================

  describe('large file chunking', () => {
    test('large.md produces multiple chunks per h2 section', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'large.md'))
      // large.md has h1 + several h2 sections
      expect(result.chunks.length).toBeGreaterThan(5)
    })

    test('large.md headingPaths are correct', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'large.md'))
      // First chunk is the h1 intro
      expect(result.chunks[0]!.headingPath).toEqual(['Nexus API Reference'])
      // Find the Authentication chunk
      const authChunk = result.chunks.find(c =>
        c.headingPath.includes('Authentication') && c.headingPath.length === 2
      )
      expect(authChunk).toBeDefined()
      expect(authChunk!.headingPath[0]).toBe('Nexus API Reference')
    })

    test('code blocks in large.md are not split', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'large.md'))
      // Every chunk with hasCodeBlock should have complete code blocks
      for (const chunk of result.chunks) {
        if (chunk.hasCodeBlock) {
          const openFences = (chunk.content.match(/```/g) ?? []).length
          // Code fences come in pairs (open + close)
          expect(openFences % 2).toBe(0)
        }
      }
    })
  })

  // ========================================================================
  // validateChunk — InputTooLongError
  // ========================================================================

  describe('chunk validation', () => {
    test('chunk exceeding 50,000 chars throws InputTooLongError', () => {
      const tmpDir = mkdtempSync(join(tmpdir(), 'vector-test-'))
      const filePath = join(tmpDir, 'huge.md')
      try {
        // Create a file with a single code block exceeding 50K chars
        const hugeContent = '# Huge\n\n```\n' + 'x'.repeat(51_000) + '\n```\n'
        writeFileSync(filePath, hugeContent)
        const parser = createParser()
        expect(() => parser.parse(filePath)).toThrow(InputTooLongError)
      } finally {
        rmSync(tmpDir, { recursive: true })
      }
    })
  })

  // ========================================================================
  // validateFilePath — SecurityError
  // ========================================================================

  describe('validateFilePath', () => {
    test('path traversal throws SecurityError', () => {
      expect(() =>
        validateFilePath('/some/project/../../../etc/passwd', '/some/project')
      ).toThrow(SecurityError)
    })

    test('valid path within project root resolves correctly', () => {
      const projectRoot = FIXTURES
      const result = validateFilePath(
        join(FIXTURES, 'simple.md'),
        projectRoot
      )
      expect(result).toBe(resolve(FIXTURES, 'simple.md'))
    })

    test('non-existent file throws FileNotFoundError', () => {
      expect(() =>
        validateFilePath(
          join(FIXTURES, 'nonexistent.md'),
          FIXTURES
        )
      ).toThrow(FileNotFoundError)
    })
  })

  // ========================================================================
  // DocumentMeta fields
  // ========================================================================

  describe('DocumentMeta fields', () => {
    test('fileSize matches actual file size', () => {
      const parser = createParser()
      const filePath = join(FIXTURES, 'simple.md')
      const result = parser.parse(filePath)
      const stat = readFileSync(filePath)
      expect(result.document.fileSize).toBe(stat.byteLength)
    })

    test('filePath is set correctly', () => {
      const parser = createParser()
      const filePath = join(FIXTURES, 'simple.md')
      const result = parser.parse(filePath)
      expect(result.document.filePath).toBe(filePath)
    })

    test('indexedAt is a Date', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'simple.md'))
      expect(result.document.indexedAt).toBeInstanceOf(Date)
    })
  })

  // ========================================================================
  // Chunk overlap
  // ========================================================================

  describe('chunk overlap', () => {
    test('heading splits do NOT add overlap', () => {
      const parser = createParser()
      const result = parser.parse(join(FIXTURES, 'deep-headings.md'))
      // Each h2 chunk starts with its own heading — no overlap from previous
      const secondChunk = result.chunks[1]!
      expect(secondChunk.content.startsWith('## Level 2')).toBe(true)
    })
  })

  // ========================================================================
  // minChunkTokens merging
  // ========================================================================

  describe('minChunkTokens merging', () => {
    test('heading-split chunks are CLEAN BREAKs — not merged even if small', () => {
      const tmpDir = mkdtempSync(join(tmpdir(), 'vector-test-'))
      const filePath = join(tmpDir, 'short-trail.md')
      try {
        // Heading-split chunks maintain independence regardless of size
        writeFileSync(filePath, [
          '# Main',
          '',
          'This is the main content section with enough text to be substantial.',
          'It contains multiple sentences to ensure it meets the minimum token count.',
          'We want to make sure the chunker processes this correctly.',
          '',
          '## Short',
          '',
          'Tiny.',
        ].join('\n'))
        const parser = createParser()
        const result = parser.parse(filePath)
        // Heading splits are CLEAN BREAKs — ## Short stays as its own chunk
        expect(result.chunks.length).toBe(2)
        expect(result.chunks[1]!.headingPath).toEqual(['Main', 'Short'])
      } finally {
        rmSync(tmpDir, { recursive: true })
      }
    })

    test('paragraph-boundary trailing small content is merged', () => {
      const tmpDir = mkdtempSync(join(tmpdir(), 'vector-test-'))
      const filePath = join(tmpDir, 'para-merge.md')
      try {
        // Create a single heading section with enough content to trigger
        // paragraph-boundary splitting, where the last split produces < minChunkTokens
        const longParagraph = 'This is a paragraph with enough content to fill tokens. '.repeat(40)
        const tinyTrail = 'End.'
        writeFileSync(filePath, `# Big Section\n\n${longParagraph}\n\n${tinyTrail}\n`)
        const parser = createParser()
        const result = parser.parse(filePath)
        // The tiny trailing "End." should be merged into the previous paragraph-split chunk
        const lastChunk = result.chunks[result.chunks.length - 1]!
        expect(lastChunk.content).toContain('End.')
        // And it shouldn't be a standalone chunk of just "End."
        expect(lastChunk.content.trim()).not.toBe('End.')
      } finally {
        rmSync(tmpDir, { recursive: true })
      }
    })
  })
})
