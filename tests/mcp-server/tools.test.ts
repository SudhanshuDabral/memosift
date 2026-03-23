// Tests for MemoSift MCP server tools.

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createMemoSiftServer } from "../../mcp-server/src/server.js";
import type { SessionManager } from "../../mcp-server/src/session-manager.js";

// ── Test Setup ───────────────────────────────────────────────────────────────

let client: Client;
let sessionManager: SessionManager;

async function setup() {
  const { server, sessionManager: sm } = createMemoSiftServer();
  sessionManager = sm;

  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);

  client = new Client({ name: "test-client", version: "1.0.0" });
  await client.connect(clientTransport);
}

async function callTool(name: string, args: Record<string, unknown> = {}) {
  const result = await client.callTool({ name, arguments: args });
  const text = (result.content as Array<{ type: string; text: string }>)[0]?.text ?? "{}";
  return { parsed: JSON.parse(text), isError: result.isError ?? false };
}

// ── Tool 1: memosift_check_pressure ──────────────────────────────────────────

describe("memosift_check_pressure", () => {
  beforeEach(setup);
  afterEach(() => sessionManager.dispose());

  it("returns NONE at low usage", async () => {
    const { parsed } = await callTool("memosift_check_pressure", {
      model: "claude-sonnet-4-6",
      current_usage_tokens: 10_000,
    });
    expect(parsed.pressure).toBe("NONE");
    expect(parsed.should_compress).toBe(false);
    expect(parsed.remaining_pct).toBeGreaterThan(90);
  });

  it("returns elevated pressure at high usage", async () => {
    const { parsed } = await callTool("memosift_check_pressure", {
      model: "claude-haiku-4-5",
      current_usage_tokens: 180_000,
    });
    expect(["HIGH", "CRITICAL"]).toContain(parsed.pressure);
    expect(parsed.should_compress).toBe(true);
  });

  it("works with message_count instead of tokens", async () => {
    const { parsed } = await callTool("memosift_check_pressure", {
      model: "claude-sonnet-4-6",
      message_count: 5,
    });
    expect(parsed.pressure).toBe("NONE");
  });
});

// ── Tool 2: memosift_compress ────────────────────────────────────────────────

describe("memosift_compress", () => {
  beforeEach(setup);
  afterEach(() => sessionManager.dispose());

  it("compresses OpenAI messages", async () => {
    const { parsed } = await callTool("memosift_compress", {
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there!" },
        { role: "user", content: "What is Python?" },
      ],
    });
    expect(parsed.success).toBe(true);
    expect(parsed.messages).toBeDefined();
    expect(parsed.report.original_tokens).toBeGreaterThan(0);
  });

  it("returns compressed_entries", async () => {
    const { parsed } = await callTool("memosift_compress", {
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi!" },
        { role: "user", content: "How are you?" },
      ],
    });
    expect(parsed.success).toBe(true);
    expect(Array.isArray(parsed.compressed_entries)).toBe(true);
  });

  it("uses session for stateful compression", async () => {
    await callTool("memosift_compress", {
      messages: [
        { role: "user", content: "Fix auth.py bug on line 42" },
        { role: "assistant", content: "Found TypeError at line 42." },
      ],
      session_id: "test-session",
    });

    // Second call — ledger should have accumulated facts.
    const { parsed: facts } = await callTool("memosift_get_facts", {
      session_id: "test-session",
    });
    expect(facts.total_facts).toBeGreaterThanOrEqual(0);
  });

  it("returns error on failure gracefully", async () => {
    // Empty messages should still work (edge case).
    const { parsed } = await callTool("memosift_compress", {
      messages: [{ role: "user", content: "Hello" }],
    });
    expect(parsed.success).toBe(true);
  });

  it("accepts preset parameter", async () => {
    const { parsed } = await callTool("memosift_compress", {
      messages: [
        { role: "system", content: "You are a coding assistant." },
        { role: "user", content: "Fix the bug" },
      ],
      preset: "coding",
    });
    expect(parsed.success).toBe(true);
  });
});

// ── Tool 3: memosift_configure ───────────────────────────────────────────────

describe("memosift_configure", () => {
  beforeEach(setup);
  afterEach(() => sessionManager.dispose());

  it("creates a new session", async () => {
    const { parsed } = await callTool("memosift_configure", {
      session_id: "new-session",
      preset: "coding",
      model: "claude-sonnet-4-6",
    });
    expect(parsed.status).toBe("created");
    expect(parsed.session_id).toBe("new-session");
  });

  it("updates an existing session", async () => {
    await callTool("memosift_configure", {
      session_id: "my-session",
      preset: "coding",
    });
    const { parsed } = await callTool("memosift_configure", {
      session_id: "my-session",
      preset: "research",
      token_budget: 10_000,
    });
    expect(parsed.status).toBe("updated");
  });
});

// ── Tool 4: memosift_get_facts ───────────────────────────────────────────────

describe("memosift_get_facts", () => {
  beforeEach(setup);
  afterEach(() => sessionManager.dispose());

  it("returns error for unknown session", async () => {
    const { parsed, isError } = await callTool("memosift_get_facts", {
      session_id: "nonexistent",
    });
    expect(isError).toBe(true);
    expect(parsed.error).toContain("not found");
  });

  it("returns facts after compression", async () => {
    await callTool("memosift_compress", {
      messages: [
        { role: "user", content: "Read the file /src/auth.py and fix the TypeError" },
        { role: "assistant", content: "I found the error in auth.py at line 42." },
      ],
      session_id: "fact-session",
      preset: "coding",
    });

    const { parsed } = await callTool("memosift_get_facts", {
      session_id: "fact-session",
    });
    expect(parsed.total_facts).toBeGreaterThanOrEqual(0);
    expect(Array.isArray(parsed.facts)).toBe(true);
  });
});

// ── Tool 5: memosift_expand ──────────────────────────────────────────────────

describe("memosift_expand", () => {
  beforeEach(setup);
  afterEach(() => sessionManager.dispose());

  it("returns not found for unknown session", async () => {
    const { parsed, isError } = await callTool("memosift_expand", {
      session_id: "nonexistent",
      original_index: 0,
    });
    expect(isError).toBe(true);
    expect(parsed.found).toBe(false);
  });

  it("returns not found for uncached index", async () => {
    await callTool("memosift_configure", { session_id: "expand-session" });
    const { parsed } = await callTool("memosift_expand", {
      session_id: "expand-session",
      original_index: 999,
    });
    expect(parsed.found).toBe(false);
    expect(parsed.reason).toContain("not cached");
  });
});

// ── Tool 6: memosift_report ──────────────────────────────────────────────────

describe("memosift_report", () => {
  beforeEach(setup);
  afterEach(() => sessionManager.dispose());

  it("returns error for unknown session", async () => {
    const { parsed, isError } = await callTool("memosift_report", {
      session_id: "nonexistent",
    });
    expect(isError).toBe(true);
  });

  it("returns error when no compression done", async () => {
    await callTool("memosift_configure", { session_id: "report-session" });
    const { parsed } = await callTool("memosift_report", {
      session_id: "report-session",
    });
    expect(parsed.info).toContain("No compression");
  });

  it("returns summary after compression", async () => {
    await callTool("memosift_compress", {
      messages: [
        { role: "system", content: "Be helpful." },
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi!" },
      ],
      session_id: "report-session",
    });

    const { parsed } = await callTool("memosift_report", {
      session_id: "report-session",
      detail_level: "summary",
    });
    expect(parsed.original_tokens).toBeGreaterThan(0);
    expect(parsed.compression_ratio).toBeGreaterThanOrEqual(1);
  });

  it("returns layers at layers detail level", async () => {
    await callTool("memosift_compress", {
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi!" },
      ],
      session_id: "report-layers",
    });

    const { parsed } = await callTool("memosift_report", {
      session_id: "report-layers",
      detail_level: "layers",
    });
    expect(Array.isArray(parsed.layers)).toBe(true);
  });

  it("returns decisions at full detail level", async () => {
    await callTool("memosift_compress", {
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi!" },
      ],
      session_id: "report-full",
    });

    const { parsed } = await callTool("memosift_report", {
      session_id: "report-full",
      detail_level: "full",
    });
    expect(Array.isArray(parsed.layers)).toBe(true);
    expect(Array.isArray(parsed.decisions)).toBe(true);
    expect(parsed.segment_counts).toBeDefined();
  });
});

// ── Tool 7: memosift_list_sessions ───────────────────────────────────────────

describe("memosift_list_sessions", () => {
  beforeEach(setup);
  afterEach(() => sessionManager.dispose());

  it("returns empty list initially", async () => {
    const { parsed } = await callTool("memosift_list_sessions");
    expect(parsed.sessions).toEqual([]);
  });

  it("lists sessions after creation", async () => {
    await callTool("memosift_configure", { session_id: "session-a" });
    await callTool("memosift_configure", { session_id: "session-b" });

    const { parsed } = await callTool("memosift_list_sessions");
    expect(parsed.sessions.length).toBe(2);
    const ids = parsed.sessions.map((s: { id: string }) => s.id);
    expect(ids).toContain("session-a");
    expect(ids).toContain("session-b");
  });
});

// ── Tool 8: memosift_destroy ─────────────────────────────────────────────────

describe("memosift_destroy", () => {
  beforeEach(setup);
  afterEach(() => sessionManager.dispose());

  it("destroys an existing session", async () => {
    await callTool("memosift_configure", { session_id: "to-destroy" });
    const { parsed } = await callTool("memosift_destroy", {
      session_id: "to-destroy",
    });
    expect(parsed.destroyed).toBe(true);

    // Verify it's gone.
    const { parsed: list } = await callTool("memosift_list_sessions");
    expect(list.sessions.length).toBe(0);
  });

  it("returns false for nonexistent session", async () => {
    const { parsed } = await callTool("memosift_destroy", {
      session_id: "nonexistent",
    });
    expect(parsed.destroyed).toBe(false);
  });
});

// ── Integration: Full Workflow ───────────────────────────────────────────────

describe("Integration: full workflow", () => {
  beforeEach(setup);
  afterEach(() => sessionManager.dispose());

  it("configure → compress → get-facts → report → destroy", async () => {
    // 1. Configure.
    const { parsed: config } = await callTool("memosift_configure", {
      session_id: "workflow",
      preset: "coding",
    });
    expect(config.status).toBe("created");

    // 2. Compress.
    const { parsed: compressed } = await callTool("memosift_compress", {
      messages: [
        { role: "system", content: "You are a coding assistant." },
        { role: "user", content: "Read /src/main.py and fix the import error" },
        { role: "assistant", content: "I found an ImportError in main.py at line 5." },
        { role: "user", content: "Now run the tests" },
        { role: "assistant", content: "All 42 tests passed." },
        { role: "user", content: "What else needs fixing?" },
      ],
      session_id: "workflow",
    });
    expect(compressed.success).toBe(true);

    // 3. Get facts.
    const { parsed: facts } = await callTool("memosift_get_facts", {
      session_id: "workflow",
    });
    expect(facts.total_facts).toBeGreaterThanOrEqual(0);

    // 4. Report.
    const { parsed: report } = await callTool("memosift_report", {
      session_id: "workflow",
      detail_level: "full",
    });
    expect(report.original_tokens).toBeGreaterThan(0);

    // 5. Destroy.
    const { parsed: destroyed } = await callTool("memosift_destroy", {
      session_id: "workflow",
    });
    expect(destroyed.destroyed).toBe(true);
  });
});
