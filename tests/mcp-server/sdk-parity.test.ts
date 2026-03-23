// SDK parity tests — verify MCP compress tool produces equivalent results to direct adapters.

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createMemoSiftServer } from "../../mcp-server/src/server.js";
import type { SessionManager } from "../../mcp-server/src/session-manager.js";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { compressOpenAIMessages, compressAnthropicMessages } from "memosift";

let client: Client;
let sessionManager: SessionManager;

function loadFixture(sdk: string, name: string): Record<string, unknown> {
  const path = resolve(
    __dirname,
    "../../benchmarks/sdk_integration/fixtures",
    sdk,
    `${name}.json`,
  );
  return JSON.parse(readFileSync(path, "utf-8"));
}

async function callTool(name: string, args: Record<string, unknown> = {}) {
  const result = await client.callTool({ name, arguments: args });
  const text = (result.content as Array<{ type: string; text: string }>)[0]?.text ?? "{}";
  return JSON.parse(text);
}

beforeAll(async () => {
  const { server, sessionManager: sm } = createMemoSiftServer();
  sessionManager = sm;
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  client = new Client({ name: "parity-test", version: "1.0.0" });
  await client.connect(clientTransport);
});

afterAll(() => sessionManager.dispose());

// ── OpenAI Fixtures ──────────────────────────────────────────────────────────

describe("OpenAI SDK parity", () => {
  const scenarios = ["coding_with_tools", "long_reasoning", "research_with_search"];

  for (const scenario of scenarios) {
    it(`${scenario}: MCP matches direct adapter`, async () => {
      const fixture = loadFixture("openai", scenario);
      const messages = fixture.messages as Record<string, unknown>[];

      // Direct adapter (TypeScript).
      const { messages: directCompressed, report: directReport } =
        await compressOpenAIMessages(messages);

      // MCP tool (also TypeScript, same pipeline).
      const mcpResult = await callTool("memosift_compress", {
        messages,
        framework: "openai",
        session_id: `openai-parity-${scenario}`,
      });

      expect(mcpResult.success).toBe(true);

      // MCP and direct adapter use the same TS pipeline, but the Session
      // facade's adapt_in/adapt_out path may count tokens slightly differently.
      // Allow ±1% tolerance on token counts.
      const tokenTolerance = Math.max(10, Math.round(directReport.originalTokens * 0.01));
      expect(mcpResult.report.original_tokens).toBeGreaterThanOrEqual(
        directReport.originalTokens - tokenTolerance,
      );
      expect(mcpResult.report.original_tokens).toBeLessThanOrEqual(
        directReport.originalTokens + tokenTolerance,
      );

      // Message count should match.
      expect(mcpResult.messages.length).toBe(directCompressed.length);
    });
  }
});

// ── Anthropic Fixtures ───────────────────────────────────────────────────────

describe("Anthropic SDK parity", () => {
  const scenarios = ["coding_with_tools", "extended_thinking", "long_multi_turn"];

  for (const scenario of scenarios) {
    it(`${scenario}: MCP matches direct adapter`, async () => {
      const fixture = loadFixture("anthropic", scenario);
      const messages = fixture.messages as Record<string, unknown>[];
      const system = (fixture.system as string) ?? undefined;

      // Direct adapter (TypeScript).
      const [directResult, directReport] = await compressAnthropicMessages(messages, {
        system,
      });

      // MCP tool (also TypeScript, same pipeline).
      const mcpResult = await callTool("memosift_compress", {
        messages,
        framework: "anthropic",
        system,
        session_id: `anthropic-parity-${scenario}`,
      });

      expect(mcpResult.success).toBe(true);

      // MCP route goes through MemoSiftSession which may count system prompt
      // tokens slightly differently. Allow ±1% tolerance on token counts.
      const tokenTolerance = Math.max(10, Math.round(directReport.originalTokens * 0.01));
      expect(mcpResult.report.original_tokens).toBeGreaterThanOrEqual(
        directReport.originalTokens - tokenTolerance,
      );
      expect(mcpResult.report.original_tokens).toBeLessThanOrEqual(
        directReport.originalTokens + tokenTolerance,
      );

      // Message count should match.
      expect(mcpResult.messages.length).toBe(directResult.messages.length);
    });
  }
});
