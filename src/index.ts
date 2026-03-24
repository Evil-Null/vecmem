#!/usr/bin/env node
/**
 * Vector Memory Engine — Entry Point
 *
 * Dispatches to CLI (Commander) for terminal usage.
 * MCP server mode will be added separately.
 */
import { program } from './cli/program.js'

program.parse()
