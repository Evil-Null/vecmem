/**
 * Vector Memory Engine — Parser Property Tests
 *
 * Property-based tests using fast-check to prove parser properties
 * hold for all valid markdown inputs.
 *
 * Properties tested:
 * 1. Chunk indices are sequential starting from 0
 * 2. Every chunk has non-empty content
 * 3. headingPath is always an array (may be empty)
 * 4. contentPlain never contains markdown heading markers at line start
 * 5. hasCodeBlock is consistent with content
 */

import { describe, test, afterAll } from 'vitest'
import fc from 'fast-check'
import { writeFileSync, mkdtempSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { MarkdownParser } from '../../src/parser/markdown.js'

// Shared temp directory for all property tests
const tmpDir = mkdtempSync(join(tmpdir(), 'vector-prop-'))

afterAll(() => {
  rmSync(tmpDir, { recursive: true, force: true })
})

let fileCounter = 0

/** Write markdown to a temp file and parse it */
function parseMarkdown(content: string) {
  const filePath = join(tmpDir, `prop-${fileCounter++}.md`)
  writeFileSync(filePath, content)
  const parser = new MarkdownParser()
  return parser.parse(filePath)
}

// Arbitrary for generating valid markdown headings
const headingArb = fc.tuple(
  fc.integer({ min: 1, max: 3 }),
  fc.stringOf(fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz '.split('')), { minLength: 1, maxLength: 30 })
).map(([depth, text]) => '#'.repeat(depth) + ' ' + text.trim())

// Arbitrary for generating markdown paragraphs
const paragraphArb = fc.stringOf(
  fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz 0123456789.,!? '.split('')),
  { minLength: 10, maxLength: 200 }
).map(s => s.trim()).filter(s => s.length > 5)

// Arbitrary for generating code blocks
const codeBlockArb = fc.tuple(
  fc.constantFrom('typescript', 'python', 'javascript', ''),
  fc.stringOf(
    fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz(){}= '.split('')),
    { minLength: 5, maxLength: 100 }
  )
).map(([lang, code]) => '```' + lang + '\n' + code.trim() + '\n```')

// Arbitrary for generating complete markdown documents
const markdownDocArb = fc.tuple(
  // Optional h1
  fc.option(headingArb.filter(h => h.startsWith('# ') && !h.startsWith('## '))),
  // Body sections
  fc.array(
    fc.oneof(
      { weight: 3, arbitrary: paragraphArb },
      { weight: 2, arbitrary: headingArb },
      { weight: 1, arbitrary: codeBlockArb },
    ),
    { minLength: 1, maxLength: 10 },
  ),
).map(([h1, sections]) => {
  const parts: string[] = []
  if (h1 !== null) parts.push(h1)
  parts.push(...sections)
  return parts.join('\n\n')
})

describe('Parser Properties', () => {
  test('Property 1: Chunk indices are sequential starting from 0', () => {
    fc.assert(
      fc.property(markdownDocArb, (markdown) => {
        const result = parseMarkdown(markdown)
        for (let i = 0; i < result.chunks.length; i++) {
          if (result.chunks[i]!.index !== i) return false
        }
        return true
      }),
      { numRuns: 500 },
    )
  })

  test('Property 2: Every chunk has non-empty content', () => {
    fc.assert(
      fc.property(markdownDocArb, (markdown) => {
        const result = parseMarkdown(markdown)
        return result.chunks.every(chunk => chunk.content.trim().length > 0)
      }),
      { numRuns: 500 },
    )
  })

  test('Property 3: headingPath is always an array (may be empty)', () => {
    fc.assert(
      fc.property(markdownDocArb, (markdown) => {
        const result = parseMarkdown(markdown)
        return result.chunks.every(chunk => Array.isArray(chunk.headingPath))
      }),
      { numRuns: 500 },
    )
  })

  test('Property 4: contentPlain never contains heading markers at line start', () => {
    fc.assert(
      fc.property(markdownDocArb, (markdown) => {
        const result = parseMarkdown(markdown)
        return result.chunks.every(chunk => {
          const lines = chunk.contentPlain.split('\n')
          return lines.every(line => !line.match(/^#{1,6}\s/))
        })
      }),
      { numRuns: 500 },
    )
  })

  test('Property 5: hasCodeBlock is consistent with content containing code fences', () => {
    fc.assert(
      fc.property(markdownDocArb, (markdown) => {
        const result = parseMarkdown(markdown)
        return result.chunks.every(chunk => {
          const hasFences = chunk.content.includes('```')
          // If hasCodeBlock is true, content must contain fences
          // If content has fences, hasCodeBlock must be true
          return chunk.hasCodeBlock === hasFences
        })
      }),
      { numRuns: 500 },
    )
  })

  test('Property 6: contentHash is deterministic', () => {
    fc.assert(
      fc.property(markdownDocArb, (markdown) => {
        const result1 = parseMarkdown(markdown)
        // Re-write with same content and parse again
        const result2 = parseMarkdown(markdown)
        return result1.document.contentHash === result2.document.contentHash
      }),
      { numRuns: 200 },
    )
  })

  test('Property 7: chunk count is zero for empty-like content', () => {
    fc.assert(
      fc.property(
        fc.stringOf(fc.constantFrom(' ', '\n', '\t'), { minLength: 0, maxLength: 50 }),
        (whitespace) => {
          const result = parseMarkdown(whitespace)
          return result.chunks.length === 0
        },
      ),
      { numRuns: 100 },
    )
  })

  test('Property 8: document title is always a non-empty string', () => {
    fc.assert(
      fc.property(markdownDocArb, (markdown) => {
        const result = parseMarkdown(markdown)
        return typeof result.document.title === 'string' && result.document.title.length > 0
      }),
      { numRuns: 500 },
    )
  })
})
