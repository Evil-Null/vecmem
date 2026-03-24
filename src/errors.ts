/**
 * Vector Memory Engine — Error Hierarchy
 *
 * All errors extend VectorError with:
 * - `code`: machine-readable error code
 * - `recoverable`: whether the caller should retry
 *
 * No error is silently swallowed. No error crashes the entire system.
 */

export abstract class VectorError extends Error {
  abstract readonly code: string
  abstract readonly recoverable: boolean
}

// --- Parser errors ---

export class FileNotFoundError extends VectorError {
  readonly code = 'PARSE_FILE_NOT_FOUND' as const
  readonly recoverable = false as const

  constructor(message: string) {
    super(message)
    this.name = 'FileNotFoundError'
  }
}

export class InvalidMarkdownError extends VectorError {
  readonly code = 'PARSE_INVALID_MARKDOWN' as const
  readonly recoverable = false as const

  constructor(message: string) {
    super(message)
    this.name = 'InvalidMarkdownError'
  }
}

// --- Embedder errors ---

export class ModelLoadError extends VectorError {
  readonly code = 'EMBED_MODEL_LOAD' as const
  readonly recoverable = true as const

  constructor(message: string) {
    super(message)
    this.name = 'ModelLoadError'
  }
}

export class InputTooLongError extends VectorError {
  readonly code = 'EMBED_INPUT_TOO_LONG' as const
  readonly recoverable = false as const

  constructor(message: string) {
    super(message)
    this.name = 'InputTooLongError'
  }
}

export class EmptyInputError extends VectorError {
  readonly code = 'EMBED_EMPTY_INPUT' as const
  readonly recoverable = false as const

  constructor(message: string) {
    super(message)
    this.name = 'EmptyInputError'
  }
}

// --- Store errors ---

export class DatabaseCorruptedError extends VectorError {
  readonly code = 'STORE_CORRUPTED' as const
  readonly recoverable = false as const

  constructor(message: string) {
    super(message)
    this.name = 'DatabaseCorruptedError'
  }
}

export class TransactionFailedError extends VectorError {
  readonly code = 'STORE_TRANSACTION_FAILED' as const
  readonly recoverable = true as const

  constructor(message: string) {
    super(message)
    this.name = 'TransactionFailedError'
  }
}

// --- System errors ---

export class InvariantViolation extends VectorError {
  readonly code = 'INVARIANT_VIOLATION' as const
  readonly recoverable = false as const

  constructor(message: string) {
    super(message)
    this.name = 'InvariantViolation'
  }
}

export class SecurityError extends VectorError {
  readonly code = 'SECURITY_ERROR' as const
  readonly recoverable = false as const

  constructor(message: string) {
    super(message)
    this.name = 'SecurityError'
  }
}
