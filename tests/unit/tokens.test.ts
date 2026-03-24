import { describe, test, expect } from 'vitest'
import { countTokens } from '../../src/parser/tokens.js'

describe('countTokens', () => {
  test('empty string returns 0', () => {
    expect(countTokens('')).toBe(0)
  })

  test('simple text uses ~4 chars per token estimate', () => {
    const text = 'Hello world this is a test'  // 26 chars
    const tokens = countTokens(text)
    expect(tokens).toBeGreaterThan(4)
    expect(tokens).toBeLessThan(10)
  })

  test('code blocks count tokens', () => {
    const code = 'function hello() { return "world" }'
    expect(countTokens(code)).toBeGreaterThan(0)
  })

  test('whitespace-only returns 0', () => {
    expect(countTokens('   \n\t  ')).toBe(0)
  })
})
