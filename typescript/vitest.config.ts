import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  test: {
    include: ["../tests/typescript/**/*.test.ts"],
    globals: true,
  },
  resolve: {
    alias: {
      memosift: path.resolve(__dirname, "src"),
    },
  },
});
