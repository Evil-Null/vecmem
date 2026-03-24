import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    include: ['tests/**/*.{test,prop,perf}.ts'],
    testTimeout: 30000,
    pool: 'forks',
  },
})
