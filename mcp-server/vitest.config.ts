import { defineConfig } from "vitest/config";
import { resolve } from "node:path";

export default defineConfig({
  resolve: {
    alias: {
      // Resolve memosift to the built dist/ (file: link points to ../typescript).
      // This ensures vitest can resolve the package even when running in CI.
      memosift: resolve(__dirname, "../typescript/dist/index.js"),
    },
  },
  test: {
    include: ["../tests/mcp-server/**/*.test.ts"],
  },
});
