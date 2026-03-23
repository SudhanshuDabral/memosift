import { defineConfig } from "vitest/config";
import { resolve } from "node:path";

export default defineConfig({
  resolve: {
    alias: {
      // Resolve memosift to the TypeScript source so vitest can process it
      // without requiring a built dist/. The file: link in package.json
      // points to ../typescript but vitest needs the source for transforms.
      memosift: resolve(__dirname, "../typescript/src/index.ts"),
    },
  },
  test: {
    include: ["../tests/mcp-server/**/*.test.ts"],
  },
});
