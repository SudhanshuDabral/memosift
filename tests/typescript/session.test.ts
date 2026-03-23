// Tests for MemoSiftSession.

import { describe, it, expect } from "vitest";
import { MemoSiftSession } from "../../typescript/src/session.js";
import { Pressure } from "../../typescript/src/core/context-window.js";
import { createMessage } from "../../typescript/src/core/types.js";
import { CompressionReport } from "../../typescript/src/report.js";
import { writeFileSync, readFileSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// ── Helpers ──────────────────────────────────────────────────────────────────

function openaiMessages() {
  return [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Hello, what is Python?" },
    { role: "assistant", content: "Python is a programming language." },
    { role: "user", content: "Tell me more." },
    { role: "assistant", content: "It supports multiple paradigms." },
    { role: "user", content: "What about error handling?" },
  ];
}

function anthropicMessages() {
  return [
    { role: "user", content: [{ type: "text", text: "Hello" }] },
    { role: "assistant", content: [{ type: "text", text: "Hi there!" }] },
    { role: "user", content: [{ type: "text", text: "What is Python?" }] },
  ];
}

function memosiftMessages() {
  return [
    createMessage("system", "You are helpful."),
    createMessage("user", "Hello"),
    createMessage("assistant", "Hi!"),
    createMessage("user", "What is 2+2?"),
  ];
}

function tmpPath() {
  return join(tmpdir(), `memosift-test-${Date.now()}-${Math.random().toString(36).slice(2)}.json`);
}

// ── Constructor ──────────────────────────────────────────────────────────────

describe("MemoSiftSession constructor", () => {
  it("creates with defaults", () => {
    const session = new MemoSiftSession();
    expect(session.lastReport).toBeNull();
    expect(session.facts).toEqual([]);
    expect(session.system).toBeNull();
  });

  it("accepts preset and model", () => {
    const session = new MemoSiftSession("coding", { model: "claude-sonnet-4-6" });
    expect(session.lastReport).toBeNull();
  });

  it("accepts config overrides", () => {
    const session = new MemoSiftSession("general", {
      configOverrides: { tokenBudget: 50_000, recentTurns: 3 },
    });
    expect(session.lastReport).toBeNull();
  });

  it("throws on invalid framework", () => {
    expect(() => new MemoSiftSession("general", { framework: "bad" as never })).toThrow(
      "Unknown framework",
    );
  });

  it("throws on invalid config override", () => {
    expect(
      () => new MemoSiftSession("general", { configOverrides: { badField: true } as never }),
    ).toThrow("Unknown config field");
  });
});

// ── Compress ─────────────────────────────────────────────────────────────────

describe("MemoSiftSession.compress", () => {
  it("auto-detects OpenAI messages", async () => {
    const session = new MemoSiftSession();
    const { messages, report } = await session.compress(openaiMessages());
    expect(Array.isArray(messages)).toBe(true);
    expect(report).toBeInstanceOf(CompressionReport);
    expect(report.originalTokens).toBeGreaterThan(0);
  });

  it("auto-detects Anthropic messages", async () => {
    const session = new MemoSiftSession();
    const { messages, report } = await session.compress(anthropicMessages(), {
      system: "You are helpful.",
    });
    expect(Array.isArray(messages)).toBe(true);
    expect(session.system).not.toBeNull();
  });

  it("handles MemoSiftMessage passthrough", async () => {
    const session = new MemoSiftSession();
    const { messages } = await session.compress(memosiftMessages());
    expect(Array.isArray(messages)).toBe(true);
  });

  it("uses explicit framework", async () => {
    const session = new MemoSiftSession("general", { framework: "openai" });
    const { report } = await session.compress(openaiMessages());
    expect(report.originalTokens).toBeGreaterThan(0);
  });

  it("caches framework detection", async () => {
    const session = new MemoSiftSession();
    await session.compress(openaiMessages());
    // Second call should not re-detect.
    const { report } = await session.compress(openaiMessages());
    expect(report).toBeInstanceOf(CompressionReport);
  });

  it("works with model and usageTokens", async () => {
    const session = new MemoSiftSession("coding", { model: "claude-sonnet-4-6" });
    const { report } = await session.compress(openaiMessages(), { usageTokens: 150_000 });
    expect(report).toBeInstanceOf(CompressionReport);
  });

  it("works with task", async () => {
    const session = new MemoSiftSession();
    const { report } = await session.compress(openaiMessages(), { task: "explain Python" });
    expect(report).toBeInstanceOf(CompressionReport);
  });
});

// ── State Persistence ────────────────────────────────────────────────────────

describe("State persistence", () => {
  it("ledger accumulates across calls", async () => {
    const session = new MemoSiftSession("coding", { model: "claude-sonnet-4-6" });
    await session.compress(openaiMessages());
    const factsAfterFirst = session.facts.length;

    await session.compress([
      { role: "user", content: "Fix bug in auth.py line 42" },
      { role: "assistant", content: "Found TypeError at line 42." },
      { role: "user", content: "Now what about tests?" },
    ]);
    expect(session.facts.length).toBeGreaterThanOrEqual(factsAfterFirst);
  });

  it("expand returns undefined for unknown index", () => {
    const session = new MemoSiftSession();
    expect(session.expand(999)).toBeUndefined();
  });

  it("lastReport updates after each compress", async () => {
    const session = new MemoSiftSession();
    expect(session.lastReport).toBeNull();
    await session.compress(openaiMessages());
    expect(session.lastReport).not.toBeNull();
    const first = session.lastReport;
    await session.compress(openaiMessages());
    expect(session.lastReport).not.toBe(first);
  });
});

// ── Check Pressure ───────────────────────────────────────────────────────────

describe("checkPressure", () => {
  it("returns NONE with no model", () => {
    const session = new MemoSiftSession();
    expect(session.checkPressure()).toBe(Pressure.NONE);
  });

  it("returns NONE at low usage", () => {
    const session = new MemoSiftSession("general", { model: "claude-sonnet-4-6" });
    expect(session.checkPressure(10_000)).toBe(Pressure.NONE);
  });

  it("returns elevated pressure at high usage", () => {
    const session = new MemoSiftSession("general", { model: "claude-haiku-4-5" });
    const p = session.checkPressure(175_000);
    expect([Pressure.HIGH, Pressure.CRITICAL]).toContain(p);
  });
});

// ── Reconfigure ──────────────────────────────────────────────────────────────

describe("reconfigure", () => {
  it("preserves ledger across reconfigure", async () => {
    const session = new MemoSiftSession("coding", { model: "claude-sonnet-4-6" });
    await session.compress(openaiMessages());
    const factsBefore = session.facts.length;

    session.reconfigure("research", { tokenBudget: 10_000 });
    expect(session.facts.length).toBe(factsBefore);
  });

  it("throws on invalid field", () => {
    const session = new MemoSiftSession();
    expect(() => session.reconfigure("general", { badField: true } as never)).toThrow(
      "Unknown config field",
    );
  });
});

// ── Save / Load ──────────────────────────────────────────────────────────────

describe("saveState / loadState", () => {
  it("round-trips session state", async () => {
    const session = new MemoSiftSession("coding", {
      model: "claude-sonnet-4-6",
      framework: "openai",
    });
    await session.compress(openaiMessages());
    const factsBefore = session.facts.length;

    const path = tmpPath();
    try {
      session.saveState(path);

      const data = JSON.parse(readFileSync(path, "utf-8"));
      expect(data.framework).toBe("openai");
      expect(data.model).toBe("claude-sonnet-4-6");
      expect(data.config_preset).toBe("coding");

      const restored = MemoSiftSession.loadState(path);
      expect(restored.facts.length).toBe(factsBefore);
    } finally {
      try { unlinkSync(path); } catch {}
    }
  });

  it("expand returns undefined after loadState", async () => {
    const session = new MemoSiftSession("general", { framework: "openai" });
    await session.compress(openaiMessages());

    const path = tmpPath();
    try {
      session.saveState(path);
      const restored = MemoSiftSession.loadState(path);
      expect(restored.expand(0)).toBeUndefined();
    } finally {
      try { unlinkSync(path); } catch {}
    }
  });

  it("loadState with overrides", () => {
    const path = tmpPath();
    try {
      writeFileSync(path, JSON.stringify({
        ledger: { facts: [] },
        cross_window_hashes: [],
        framework: "openai",
        model: "gpt-4o",
        config_preset: "general",
      }));

      const restored = MemoSiftSession.loadState(path, "coding", {
        model: "claude-sonnet-4-6",
        configOverrides: { tokenBudget: 50_000 },
      });
      expect(restored.facts).toEqual([]);
    } finally {
      try { unlinkSync(path); } catch {}
    }
  });
});
