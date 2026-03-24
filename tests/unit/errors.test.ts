import { describe, test, expect } from 'vitest'
import {
  VectorError,
  FileNotFoundError,
  InvalidMarkdownError,
  ModelLoadError,
  InputTooLongError,
  EmptyInputError,
  DatabaseCorruptedError,
  TransactionFailedError,
  InvariantViolation,
  SecurityError,
} from '../../src/errors.js'

describe('VectorError hierarchy', () => {
  test('all errors extend VectorError', () => {
    expect(new FileNotFoundError('/a.md')).toBeInstanceOf(VectorError)
    expect(new InvalidMarkdownError('bad')).toBeInstanceOf(VectorError)
    expect(new ModelLoadError('fail')).toBeInstanceOf(VectorError)
    expect(new InputTooLongError('long')).toBeInstanceOf(VectorError)
    expect(new EmptyInputError('empty')).toBeInstanceOf(VectorError)
    expect(new DatabaseCorruptedError('corrupt')).toBeInstanceOf(VectorError)
    expect(new TransactionFailedError('fail')).toBeInstanceOf(VectorError)
    expect(new InvariantViolation('violated')).toBeInstanceOf(VectorError)
    expect(new SecurityError('insecure')).toBeInstanceOf(VectorError)
  })

  test('all errors extend Error', () => {
    expect(new FileNotFoundError('/a.md')).toBeInstanceOf(Error)
    expect(new InvariantViolation('x')).toBeInstanceOf(Error)
  })

  test('error codes are correct', () => {
    expect(new FileNotFoundError('/a.md').code).toBe('PARSE_FILE_NOT_FOUND')
    expect(new InvalidMarkdownError('bad').code).toBe('PARSE_INVALID_MARKDOWN')
    expect(new ModelLoadError('fail').code).toBe('EMBED_MODEL_LOAD')
    expect(new InputTooLongError('long').code).toBe('EMBED_INPUT_TOO_LONG')
    expect(new EmptyInputError('empty').code).toBe('EMBED_EMPTY_INPUT')
    expect(new DatabaseCorruptedError('corrupt').code).toBe('STORE_CORRUPTED')
    expect(new TransactionFailedError('fail').code).toBe('STORE_TRANSACTION_FAILED')
    expect(new InvariantViolation('violated').code).toBe('INVARIANT_VIOLATION')
    expect(new SecurityError('insecure').code).toBe('SECURITY_ERROR')
  })

  test('recoverable flags are correct', () => {
    // Non-recoverable
    expect(new FileNotFoundError('/a.md').recoverable).toBe(false)
    expect(new InvalidMarkdownError('bad').recoverable).toBe(false)
    expect(new InputTooLongError('long').recoverable).toBe(false)
    expect(new EmptyInputError('empty').recoverable).toBe(false)
    expect(new DatabaseCorruptedError('corrupt').recoverable).toBe(false)
    expect(new InvariantViolation('violated').recoverable).toBe(false)
    expect(new SecurityError('insecure').recoverable).toBe(false)

    // Recoverable
    expect(new ModelLoadError('fail').recoverable).toBe(true)
    expect(new TransactionFailedError('fail').recoverable).toBe(true)
  })

  test('error messages are preserved', () => {
    const err = new FileNotFoundError('/path/to/missing.md')
    expect(err.message).toBe('/path/to/missing.md')
  })

  test('error name matches class name', () => {
    expect(new FileNotFoundError('x').name).toBe('FileNotFoundError')
    expect(new ModelLoadError('x').name).toBe('ModelLoadError')
    expect(new InvariantViolation('x').name).toBe('InvariantViolation')
    expect(new SecurityError('x').name).toBe('SecurityError')
  })

  test('errors have stack traces', () => {
    const err = new InvariantViolation('test')
    expect(err.stack).toBeDefined()
    expect(err.stack).toContain('InvariantViolation')
  })
})
