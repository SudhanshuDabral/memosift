// MemoSift TypeScript test suite — mirrors critical Python tests.

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  MODEL_BUDGET_DEFAULTS,
  MODEL_PRICING,
  createConfig,
  createPreset,
} from "../../typescript/src/config.js";
import { classifyMessages } from "../../typescript/src/core/classifier.js";
import { CompressionCache, compress, partitionZones } from "../../typescript/src/core/pipeline.js";
import {
  AnchorCategory,
  AnchorLedger,
  CompressionPolicy,
  ContentType,
  DEFAULT_POLICIES,
  createClassified,
  createCrossWindowState,
  createDependencyMap,
  createMessage,
  depMapAdd,
  depMapCanDrop,
  depMapDependentsOf,
} from "../../typescript/src/core/types.js";
import { HeuristicTokenCounter } from "../../typescript/src/providers/heuristic.js";
import { CompressionReport } from "../../typescript/src/report.js";

// ── Helpers ──────────────────────────────────────────────────────────────────

function loadVector(name: string): Record<string, unknown> {
  const path = resolve(__dirname, "../../spec/test-vectors", name);
  return JSON.parse(readFileSync(path, "utf-8"));
}

function sampleMessages() {
  return [
    createMessage("system", "You are a helpful assistant."),
    createMessage("user", "Hello"),
    createMessage("assistant", "Let me check.", {
      toolCalls: [
        {
          id: "tc1",
          type: "function",
          function: { name: "read_file", arguments: '{"path":"test.py"}' },
        },
      ],
    }),
    createMessage("tool", 'def hello():\n    print("hello")\n', {
      toolCallId: "tc1",
      name: "read_file",
    }),
    createMessage("assistant", "Here's the file content."),
    createMessage("user", "Thanks, what does it do?"),
  ];
}

// ── 1. Core Types ────────────────────────────────────────────────────────────

describe("MemoSiftMessage", () => {
  it("creates a message with defaults", () => {
    const msg = createMessage("user", "hello");
    expect(msg.role).toBe("user");
    expect(msg.content).toBe("hello");
    expect(msg.toolCalls).toBeNull();
    expect(msg.toolCallId).toBeNull();
    expect(msg._memosiftCompressed).toBe(false);
    expect(msg.metadata).toEqual({});
  });

  it("creates a message with tool calls", () => {
    const msg = createMessage("assistant", "checking", {
      toolCalls: [{ id: "tc1", type: "function", function: { name: "read", arguments: "{}" } }],
    });
    expect(msg.toolCalls).toHaveLength(1);
    expect(msg.toolCalls![0]!.id).toBe("tc1");
  });

  it("creates a tool result message", () => {
    const msg = createMessage("tool", "file content", { toolCallId: "tc1", name: "read_file" });
    expect(msg.toolCallId).toBe("tc1");
    expect(msg.name).toBe("read_file");
  });
});

describe("ContentType & CompressionPolicy enums", () => {
  it("has all 10 content types", () => {
    const types = Object.values(ContentType);
    expect(types).toHaveLength(10);
    expect(types).toContain("SYSTEM_PROMPT");
    expect(types).toContain("CODE_BLOCK");
    expect(types).toContain("ERROR_TRACE");
  });

  it("has all 7 compression policies", () => {
    const policies = Object.values(CompressionPolicy);
    expect(policies).toHaveLength(7);
    expect(policies).toContain("PRESERVE");
    expect(policies).toContain("AGGRESSIVE");
  });

  it("DEFAULT_POLICIES maps every content type", () => {
    for (const ct of Object.values(ContentType)) {
      expect(DEFAULT_POLICIES[ct]).toBeDefined();
    }
  });
});

describe("DependencyMap", () => {
  it("tracks references and blocks drops", () => {
    const dm = createDependencyMap();
    depMapAdd(dm, 5, 2); // message 5 references message 2
    expect(depMapCanDrop(dm, 2)).toBe(false);
    expect(depMapCanDrop(dm, 3)).toBe(true);
  });

  it("returns dependents of an index", () => {
    const dm = createDependencyMap();
    depMapAdd(dm, 5, 2);
    depMapAdd(dm, 7, 2);
    expect(depMapDependentsOf(dm, 2)).toEqual([5, 7]);
    expect(depMapDependentsOf(dm, 3)).toEqual([]);
  });
});

describe("AnchorLedger", () => {
  it("adds facts and deduplicates", () => {
    const ledger = new AnchorLedger();
    const added1 = ledger.add({
      category: AnchorCategory.FILES,
      content: "src/auth.ts",
      turn: 1,
      confidence: 1.0,
    });
    const added2 = ledger.add({
      category: AnchorCategory.FILES,
      content: "src/auth.ts",
      turn: 2,
      confidence: 1.0,
    });
    expect(added1).toBe(true);
    expect(added2).toBe(false); // duplicate
    expect(ledger.facts).toHaveLength(1);
  });

  it("renders markdown with section headers", () => {
    const ledger = new AnchorLedger();
    ledger.add({
      category: AnchorCategory.FILES,
      content: "src/auth.ts",
      turn: 1,
      confidence: 1.0,
    });
    ledger.add({
      category: AnchorCategory.ERRORS,
      content: "TypeError: undefined",
      turn: 2,
      confidence: 1.0,
    });
    const rendered = ledger.render();
    expect(rendered).toContain("## FILES TOUCHED");
    expect(rendered).toContain("## ERRORS ENCOUNTERED");
    expect(rendered).toContain("- src/auth.ts");
  });

  it("returns facts by category", () => {
    const ledger = new AnchorLedger();
    ledger.add({ category: AnchorCategory.FILES, content: "a.ts", turn: 1, confidence: 1.0 });
    ledger.add({ category: AnchorCategory.ERRORS, content: "err", turn: 2, confidence: 1.0 });
    ledger.add({ category: AnchorCategory.FILES, content: "b.ts", turn: 3, confidence: 1.0 });
    expect(ledger.factsByCategory(AnchorCategory.FILES)).toHaveLength(2);
    expect(ledger.factsByCategory(AnchorCategory.ERRORS)).toHaveLength(1);
  });

  it("estimates token count", () => {
    const ledger = new AnchorLedger();
    ledger.add({
      category: AnchorCategory.INTENT,
      content: "Fix the auth module bug",
      turn: 1,
      confidence: 1.0,
    });
    expect(ledger.tokenEstimate()).toBeGreaterThan(0);
  });
});

describe("CrossWindowState", () => {
  it("creates with empty hash set", () => {
    const state = createCrossWindowState();
    expect(state.contentHashes.size).toBe(0);
  });
});

// ── 2. Configuration ─────────────────────────────────────────────────────────

describe("createConfig", () => {
  it("creates default config with sensible values", () => {
    const config = createConfig();
    expect(config.recentTurns).toBe(2);
    expect(config.tokenBudget).toBeNull();
    expect(config.enableSummarization).toBe(false);
    expect(config.dedupSimilarityThreshold).toBe(0.8);
    expect(config.preBucketBypass).toBe(true);
  });

  it("accepts overrides", () => {
    const config = createConfig({ recentTurns: 5, tokenBudget: 50_000 });
    expect(config.recentTurns).toBe(5);
    expect(config.tokenBudget).toBe(50_000);
  });

  it("rejects invalid values", () => {
    expect(() => createConfig({ recentTurns: -1 })).toThrow();
    expect(() => createConfig({ tokenBudget: 10 })).toThrow();
    expect(() => createConfig({ dedupSimilarityThreshold: 2.0 })).toThrow();
    expect(() => createConfig({ performanceTier: "invalid" })).toThrow();
  });

  it("rejects out-of-order compression thresholds", () => {
    expect(() => createConfig({ softCompressionPct: 0.8, fullCompressionPct: 0.75 })).toThrow();
  });
});

describe("createPreset", () => {
  it("creates optimized coding preset", () => {
    const config = createPreset("coding");
    expect(config.recentTurns).toBe(2);
    expect(config.entropyThreshold).toBe(2.1);
    expect(config.codeKeepSignatures).toBe(true);
    expect(config.enableResolutionCompression).toBe(true);
  });

  it("creates all presets without error", () => {
    for (const name of ["coding", "research", "support", "data", "energy", "financial", "general", "auto"]) {
      expect(() => createPreset(name)).not.toThrow();
    }
  });

  it("rejects unknown preset", () => {
    expect(() => createPreset("nonexistent")).toThrow(/Unknown preset/);
  });

  it("allows overrides on top of presets", () => {
    const config = createPreset("coding", { tokenBudget: 80_000 });
    expect(config.tokenBudget).toBe(80_000);
    expect(config.recentTurns).toBe(2); // from optimized coding preset
  });
});

describe("MODEL_BUDGET_DEFAULTS & MODEL_PRICING", () => {
  it("has budget defaults for major models", () => {
    expect(MODEL_BUDGET_DEFAULTS["gpt-4o"]).toBe(80_000);
    expect(MODEL_BUDGET_DEFAULTS["claude-sonnet-4-6"]).toBe(120_000);
  });

  it("has pricing for major models", () => {
    expect(MODEL_PRICING["gpt-4o"]).toBe(0.0025);
    expect(MODEL_PRICING.default).toBe(0.003);
  });
});

// ── 3. Providers ─────────────────────────────────────────────────────────────

describe("HeuristicTokenCounter", () => {
  it("counts tokens based on character length", async () => {
    const counter = new HeuristicTokenCounter();
    const count = await counter.countTokens("Hello, world!");
    expect(count).toBeGreaterThan(0);
    expect(count).toBe(Math.ceil(13 / 3.5));
  });

  it("returns 0 for empty string", async () => {
    const counter = new HeuristicTokenCounter();
    expect(await counter.countTokens("")).toBe(0);
  });

  it("throws on generate()", async () => {
    const counter = new HeuristicTokenCounter();
    await expect(counter.generate("test", "prompt")).rejects.toThrow(/does not support/);
  });
});

// ── 4. Classifier ────────────────────────────────────────────────────────────

describe("classifyMessages", () => {
  it("classifies system prompt correctly", () => {
    const msgs = sampleMessages();
    const config = createConfig();
    const segments = classifyMessages(msgs, config);
    expect(segments[0]!.contentType).toBe(ContentType.SYSTEM_PROMPT);
  });

  it("classifies last user message as USER_QUERY", () => {
    const msgs = sampleMessages();
    const config = createConfig();
    const segments = classifyMessages(msgs, config);
    const lastSeg = segments[segments.length - 1]!;
    expect(lastSeg.contentType).toBe(ContentType.USER_QUERY);
  });

  it("matches classify-001 test vector", () => {
    const vector = loadVector("classify-001.json") as {
      config: Record<string, unknown>;
      input: Record<string, unknown>[];
      expected_classifications: { index: number; type: string }[];
    };
    const msgs = vector.input.map((d) =>
      createMessage(d.role as "system" | "user" | "assistant" | "tool", d.content as string, {
        toolCalls:
          (d.tool_calls as {
            id: string;
            type: string;
            function: { name: string; arguments: string };
          }[]) ?? null,
        toolCallId: (d.tool_call_id as string) ?? null,
        name: (d.name as string) ?? null,
      }),
    );
    const config = createConfig(vector.config as Partial<Record<string, unknown>>);
    const segments = classifyMessages(msgs, config);

    for (const exp of vector.expected_classifications) {
      expect(segments[exp.index]!.contentType).toBe(exp.type);
    }
  });
});

// ── 5. Pipeline ──────────────────────────────────────────────────────────────

describe("partitionZones", () => {
  it("separates system prompts into zone 1", () => {
    const msgs = sampleMessages();
    const [zone1, zone2, zone3] = partitionZones(msgs);
    expect(zone1).toHaveLength(1);
    expect(zone1[0]!.role).toBe("system");
    expect(zone2).toHaveLength(0);
    expect(zone3.length).toBeGreaterThan(0);
  });

  it("puts compressed messages into zone 2", () => {
    const msgs = [
      createMessage("system", "You are helpful."),
      createMessage("assistant", "Previously compressed.", { _memosiftCompressed: true }),
      createMessage("user", "New message."),
    ];
    const [zone1, zone2, zone3] = partitionZones(msgs);
    expect(zone1).toHaveLength(1);
    expect(zone2).toHaveLength(1);
    expect(zone3).toHaveLength(1);
  });
});

describe("compress (full pipeline)", () => {
  it("preserves system prompt and last user message", async () => {
    const msgs = sampleMessages();
    const { messages: compressed } = await compress(msgs);
    const systemMsgs = compressed.filter((m) => m.role === "system");
    const userMsgs = compressed.filter((m) => m.role === "user");
    expect(systemMsgs.length).toBeGreaterThanOrEqual(1);
    expect(userMsgs[userMsgs.length - 1]!.content).toBe("Thanks, what does it do?");
  });

  it("maintains tool call integrity", async () => {
    const msgs = sampleMessages();
    const { messages: compressed } = await compress(msgs);
    const tcIds = new Set<string>();
    const trIds = new Set<string>();
    for (const m of compressed) {
      if (m.toolCalls) for (const tc of m.toolCalls) tcIds.add(tc.id);
      if (m.toolCallId) trIds.add(m.toolCallId);
    }
    // Every tool_call id must have a matching tool_result
    for (const id of tcIds) {
      expect(trIds.has(id)).toBe(true);
    }
  });

  it("respects token budget", async () => {
    const msgs = sampleMessages();
    const { report } = await compress(msgs, { config: { tokenBudget: 500 } });
    expect(report.compressedTokens).toBeLessThanOrEqual(500);
  });

  it("returns compression report with metrics", async () => {
    const msgs = sampleMessages();
    const { report } = await compress(msgs);
    expect(report.originalTokens).toBeGreaterThan(0);
    expect(report.compressionRatio).toBeGreaterThanOrEqual(1.0);
    expect(report.performanceTier).toBeDefined();
  });

  it("skips compression when no zone 3 messages exist", async () => {
    const msgs = [createMessage("system", "You are helpful.")];
    const { messages: compressed, report } = await compress(msgs);
    expect(compressed).toHaveLength(1);
    expect(report.compressionRatio).toBe(1.0);
  });
});

// ── 6. CompressionReport ─────────────────────────────────────────────────────

describe("CompressionReport", () => {
  it("initializes with zeroed metrics", () => {
    const report = new CompressionReport();
    expect(report.originalTokens).toBe(0);
    expect(report.compressedTokens).toBe(0);
    expect(report.compressionRatio).toBe(1.0);
    expect(report.layers).toEqual([]);
    expect(report.decisions).toEqual([]);
  });

  it("computes ratio on finalize", () => {
    const report = new CompressionReport();
    report.finalize(1000, 250, 0.003);
    expect(report.compressionRatio).toBe(4.0);
    expect(report.tokensSaved).toBe(750);
    expect(report.estimatedCostSaved).toBeCloseTo(0.00225, 5);
  });
});

// ── 7. CompressionCache ──────────────────────────────────────────────────────

describe("CompressionCache", () => {
  it("stores and retrieves original content", () => {
    const cache = new CompressionCache();
    cache.store(3, "original long content here");
    expect(cache.has(3)).toBe(true);
    expect(cache.expand(3)).toBe("original long content here");
    expect(cache.expand(99)).toBeUndefined();
    expect(cache.size).toBe(1);
  });
});
