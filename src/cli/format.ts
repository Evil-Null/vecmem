/**
 * Vector Memory Engine — CLI Output Formatting
 *
 * Central formatting module — the ONLY file that deals with ANSI colors and box drawing.
 * CLI commands call these functions; they never output raw ANSI codes directly.
 *
 * Color-aware: detects process.stdout.isTTY and strips colors when piped.
 * Uses ANSI escape codes directly — no external color library for v1.
 */

import type { SearchResult } from '../types.js'

// ============================================================================
// TTY Detection
// ============================================================================

/** Whether stdout supports colors (false when piped) */
const isTTY = process.stdout.isTTY === true

// ============================================================================
// ANSI Color Codes
// ============================================================================

const RESET = '\x1b[0m'
const BOLD = '\x1b[1m'
const DIM = '\x1b[2m'
const GREEN = '\x1b[32m'
const YELLOW = '\x1b[33m'
const RED = '\x1b[31m'
const CYAN = '\x1b[36m'
const MAGENTA = '\x1b[35m'

function color(code: string, text: string): string {
  if (!isTTY) return text
  return `${code}${text}${RESET}`
}

function bold(text: string): string {
  return color(BOLD, text)
}

function dim(text: string): string {
  return color(DIM, text)
}

function green(text: string): string {
  return color(GREEN, text)
}

function yellow(text: string): string {
  return color(YELLOW, text)
}

function red(text: string): string {
  return color(RED, text)
}

function cyan(text: string): string {
  return color(CYAN, text)
}

function magenta(text: string): string {
  return color(MAGENTA, text)
}

// ============================================================================
// Status Messages
// ============================================================================

/** Format a success message with green checkmark */
export function formatSuccess(msg: string): string {
  return `  ${green('\u2713')} ${msg}`
}

/** Format a warning message with yellow triangle */
export function formatWarning(msg: string): string {
  return `  ${yellow('\u26A0')} ${msg}`
}

/** Format an error message with red cross */
export function formatError(msg: string): string {
  return `  ${red('\u2717')} ${msg}`
}

// ============================================================================
// Search Result Formatting
// ============================================================================

/**
 * Format a single search result as a box with path, score, and snippet.
 *
 * TTY output:
 * ```
 * ┌─ docs/auth/oauth.md ─── score: 0.94 ──────────┐
 * │ ## OAuth 2.0 Flow                               │
 * │ Authentication uses OAuth 2.0 with PKCE...      │
 * └─────────────────────────────────────────────────┘
 * ```
 *
 * Piped output (plain text, grep-friendly):
 * ```
 * [0.94] docs/auth/oauth.md
 *   ## OAuth 2.0 Flow
 *   Authentication uses OAuth 2.0 with PKCE...
 * ```
 */
export function formatSearchResult(result: SearchResult): string {
  const score = result.score.toFixed(2)
  const path = result.documentPath
  const snippet = truncateSnippet(result.highlight || result.chunk.contentPlain, 3)

  if (!isTTY) {
    // Plain text output for piping
    const lines = [`[${score}] ${path}`]
    for (const line of snippet) {
      lines.push(`  ${line}`)
    }
    return lines.join('\n')
  }

  // Box-drawn output for TTY
  const header = `${dim(path)} ${dim('\u2500\u2500\u2500')} score: ${cyan(score)}`
  const headerPlain = `${path} \u2500\u2500\u2500 score: ${score}`

  // Calculate box width (minimum 50, max 70)
  const contentWidth = Math.max(
    50,
    Math.min(70, ...snippet.map(l => l.length + 4), headerPlain.length + 6),
  )

  const topBorder = `  \u250C\u2500 ${header} ${dim('\u2500'.repeat(Math.max(1, contentWidth - headerPlain.length - 4)))}\u2510`
  const bottomBorder = `  \u2514${'\u2500'.repeat(contentWidth)}\u2518`

  const lines = [topBorder]
  for (const line of snippet) {
    const padded = line.padEnd(contentWidth - 4)
    lines.push(`  \u2502 ${padded} \u2502`)
  }
  lines.push(bottomBorder)

  return lines.join('\n')
}

// ============================================================================
// Progress Bar
// ============================================================================

/**
 * Format a progress bar string.
 *
 * TTY: `Indexing ████████████████████████░░░░  38/47 files`
 * Piped: `Indexing 38/47 files`
 */
export function formatProgress(current: number, total: number): string {
  const label = `${current}/${total} files`

  if (!isTTY || total === 0) {
    return `  Indexing ${label}`
  }

  const barWidth = 30
  const filled = Math.round((current / total) * barWidth)
  const empty = barWidth - filled
  const bar = '\u2588'.repeat(filled) + '\u2591'.repeat(empty)

  return `  Indexing ${bar}  ${label}`
}

// ============================================================================
// Index Summary
// ============================================================================

/** Format the index summary line */
export function formatIndexSummary(
  files: number,
  chunks: number,
  elapsedMs: number,
  dbPath: string,
  dbSizeBytes: number,
): string {
  const throughput = elapsedMs > 0 ? Math.round((chunks / elapsedMs) * 1000) : 0
  const elapsed = formatElapsed(elapsedMs)
  const dbSize = formatBytes(dbSizeBytes)

  const lines = [
    formatSuccess(`${files} files \u2192 ${chunks} chunks \u2192 ${chunks} embeddings`),
    `  ${magenta('\u23F1')} ${magenta(elapsed)} (${throughput} chunks/sec)`,
    `  ${dim('\uD83D\uDCBE')} ${dim(dbPath)} (${dbSize})`,
  ]

  return lines.join('\n')
}

// ============================================================================
// Doctor Check
// ============================================================================

/** Format a doctor check result */
export function formatDoctorCheck(name: string, ok: boolean, detail: string): string {
  if (ok) {
    return `  ${green('\u2713')} ${bold(name)}${' '.repeat(Math.max(1, 16 - name.length))}${detail}`
  }
  return `  ${red('\u2717')} ${bold(name)}${' '.repeat(Math.max(1, 16 - name.length))}${detail}`
}

// ============================================================================
// Utility Helpers
// ============================================================================

/** Format milliseconds into human-readable elapsed time */
export function formatElapsed(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`
  }
  return `${(ms / 1000).toFixed(1)}s`
}

/** Format bytes into human-readable size */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/** Truncate a text snippet to N lines, trimming long lines */
function truncateSnippet(text: string, maxLines: number): string[] {
  const allLines = text.split('\n').filter(l => l.trim().length > 0)
  const lines = allLines.slice(0, maxLines)
  return lines.map(l => (l.length > 66 ? l.slice(0, 63) + '...' : l))
}

/** Format a "time ago" string from a Date */
export function formatTimeAgo(date: Date): string {
  const now = Date.now()
  const diffMs = now - date.getTime()
  const diffSec = Math.floor(diffMs / 1000)
  const diffMin = Math.floor(diffSec / 60)
  const diffHrs = Math.floor(diffMin / 60)
  const diffDays = Math.floor(diffHrs / 24)

  if (diffSec < 60) return 'just now'
  if (diffMin < 60) return `${diffMin} minute${diffMin === 1 ? '' : 's'} ago`
  if (diffHrs < 24) return `${diffHrs} hour${diffHrs === 1 ? '' : 's'} ago`
  return `${diffDays} day${diffDays === 1 ? '' : 's'} ago`
}

/** Format a header line with the version */
export function formatHeader(version: string): string {
  return `  ${bold(`Vector v${version}`)}`
}
