/**
 * Vector Memory Engine — CLI Program
 *
 * Commander setup with global flags, version, and all subcommands.
 *
 * Global flags:
 *   --verbose       Show debug-level log output
 *   --project       Override project name
 *   --storage-path  Override database storage directory
 *
 * Registers subcommands: init, index, query, status, doctor, remove
 */

import { Command } from 'commander'
import { resolveConfig } from '../config.js'
import type { VectorConfig } from '../config.js'
import { runInit } from './init.js'
import { runIndex } from './index.js'
import { runQuery } from './query.js'
import { runStatus } from './status.js'
import { runDoctor } from './doctor.js'
import { runRemove } from './remove.js'
import { formatError } from './format.js'

// ============================================================================
// Version — read from package.json at build time
// ============================================================================

const VERSION = '0.1.0'

// ============================================================================
// Config Resolution Helper
// ============================================================================

/**
 * Resolve config from global flags (--project, --storage-path).
 * CLI flags override defaults and config files.
 */
function getConfig(opts: { project?: string; storagePath?: string }): VectorConfig {
  const overrides: Partial<VectorConfig> = {}
  if (opts.project !== undefined) {
    overrides.project = opts.project
  }
  if (opts.storagePath !== undefined) {
    overrides.storagePath = opts.storagePath
  }
  return resolveConfig(overrides)
}

// ============================================================================
// Program Definition
// ============================================================================

export const program = new Command()

program
  .name('vecmem')
  .version(VERSION)
  .description('Your notes become AI\'s memory \u2014 hybrid search for markdown files')
  .option('--verbose', 'Show debug-level output', false)
  .option('--project <name>', 'Override project name')
  .option('--storage-path <path>', 'Override database storage directory')

// --------------------------------------------------------------------------
// init
// --------------------------------------------------------------------------

program
  .command('init')
  .description('Initialize project \u2014 auto-discover .md files, create database')
  .addHelpText('after', `
Examples:
  $ vecmem init                    Initialize with defaults
  $ vecmem init --project myproj   Initialize with custom project name
`)
  .action(() => {
    const opts = program.opts<{ verbose: boolean; project?: string; storagePath?: string }>()

    try {
      const config = getConfig(opts)
      runInit(config, VERSION)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      process.stderr.write(formatError(message) + '\n')
      process.exitCode = 1
    }
  })

// --------------------------------------------------------------------------
// index
// --------------------------------------------------------------------------

program
  .command('index [path]')
  .description('Index markdown files \u2014 incremental, skips unchanged')
  .addHelpText('after', `
Examples:
  $ vecmem index                   Index current directory
  $ vecmem index ./docs            Index specific directory
  $ vecmem index --verbose         Index with debug output
`)
  .action(async (pathArg: string | undefined) => {
    const opts = program.opts<{ verbose: boolean; project?: string; storagePath?: string }>()

    try {
      const config = getConfig(opts)
      await runIndex(pathArg, config, opts.verbose)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      process.stderr.write(formatError(message) + '\n')
      process.exitCode = 1
    }
  })

// --------------------------------------------------------------------------
// query
// --------------------------------------------------------------------------

program
  .command('query <text>')
  .description('Search your notes \u2014 hybrid BM25 + vector + RRF')
  .option('-k, --top-k <n>', 'Max results to return', parseInt)
  .option('--all', 'Show all results (not just top 5)')
  .addHelpText('after', `
Examples:
  $ vecmem query "how does auth work?"
  $ vecmem query "deployment" --top-k 20
  $ vecmem query "OAuth" --all
`)
  .action(async (text: string, cmdOpts: { topK?: number; all?: boolean }) => {
    const opts = program.opts<{ verbose: boolean; project?: string; storagePath?: string }>()

    try {
      const config = getConfig(opts)
      await runQuery(
        text,
        { topK: cmdOpts.topK, project: opts.project, all: cmdOpts.all },
        config,
        opts.verbose,
      )
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      process.stderr.write(formatError(message) + '\n')
      process.exitCode = 1
    }
  })

// --------------------------------------------------------------------------
// status
// --------------------------------------------------------------------------

program
  .command('status')
  .description('Show index stats \u2014 document count, chunks, stale files')
  .addHelpText('after', `
Examples:
  $ vecmem status
  $ vecmem status --project myproj
`)
  .action(() => {
    const opts = program.opts<{ verbose: boolean; project?: string; storagePath?: string }>()

    try {
      const config = getConfig(opts)
      runStatus(config)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      process.stderr.write(formatError(message) + '\n')
      process.exitCode = 1
    }
  })

// --------------------------------------------------------------------------
// doctor
// --------------------------------------------------------------------------

program
  .command('doctor')
  .description('Health check \u2014 database, FTS sync, invariants')
  .addHelpText('after', `
Examples:
  $ vecmem doctor
`)
  .action(() => {
    const opts = program.opts<{ verbose: boolean; project?: string; storagePath?: string }>()

    try {
      const config = getConfig(opts)
      runDoctor(config, VERSION)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      process.stderr.write(formatError(message) + '\n')
      process.exitCode = 1
    }
  })

// --------------------------------------------------------------------------
// remove
// --------------------------------------------------------------------------

program
  .command('remove <path>')
  .description('Remove a document from the index')
  .addHelpText('after', `
Examples:
  $ vecmem remove docs/old-notes.md
  $ vecmem remove ./path/to/file.md
`)
  .action((filePath: string) => {
    const opts = program.opts<{ verbose: boolean; project?: string; storagePath?: string }>()

    try {
      const config = getConfig(opts)
      runRemove(filePath, config)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      process.stderr.write(formatError(message) + '\n')
      process.exitCode = 1
    }
  })
