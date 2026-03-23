#!/usr/bin/env node
// MemoSift MCP Server — stdio transport entry point.

import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { createMemoSiftServer } from "./server.js";

async function main(): Promise<void> {
  const { server, sessionManager } = createMemoSiftServer();
  const transport = new StdioServerTransport();

  // Cleanup on exit.
  process.on("SIGINT", () => {
    sessionManager.dispose();
    process.exit(0);
  });
  process.on("SIGTERM", () => {
    sessionManager.dispose();
    process.exit(0);
  });

  await server.connect(transport);
}

main().catch((err) => {
  console.error("MemoSift MCP server failed to start:", err);
  process.exit(1);
});
