/**
 * Vector Memory Engine — Structured Logger
 *
 * Writes to stderr (stdout is for command output).
 * Event-based API with structured data.
 *
 * verbose: true  → shows debug, info, warn, error
 * verbose: false → shows info, warn, error only
 *
 * Events: file.indexed, file.skipped, search.completed, search.degraded,
 *         invariant.violated, dir.indexed
 */

import type { VectorError } from './errors.js'
import type { Logger } from './types.js'

// ============================================================================
// Logger Factory
// ============================================================================

/**
 * Create a structured logger that writes to stderr.
 *
 * @param verbose - If true, includes debug-level messages
 * @returns Logger instance
 */
export function createLogger(verbose: boolean = false): Logger {
  return {
    debug(event: string, data?: Record<string, unknown>): void {
      if (!verbose) return
      writeLog('DEBUG', event, data)
    },

    info(event: string, data?: Record<string, unknown>): void {
      writeLog('INFO', event, data)
    },

    warn(event: string, data?: Record<string, unknown>): void {
      writeLog('WARN', event, data)
    },

    error(event: string, error: VectorError): void {
      writeLog('ERROR', event, {
        code: error.code,
        message: error.message,
        recoverable: error.recoverable,
      })
    },
  }
}

// ============================================================================
// Internal — write to stderr
// ============================================================================

function writeLog(
  level: string,
  event: string,
  data?: Record<string, unknown>,
): void {
  const timestamp = new Date().toISOString()
  const entry: Record<string, unknown> = {
    ts: timestamp,
    level,
    event,
  }

  if (data !== undefined) {
    entry['data'] = data
  }

  process.stderr.write(JSON.stringify(entry) + '\n')
}
