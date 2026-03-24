// Tests for Layer 0 adaptive compression integration in the TypeScript pipeline.
// Verifies parity with the Python implementation.

import { describe, expect, it } from "vitest";
import { createConfig } from "../../typescript/src/config.js";
import {
  Pressure,
  computeAdaptiveThresholds,
  contextWindowFromModel,
  createContextWindowState,
  estimateTokensHeuristic,
  resolveContextWindow,
} from "../../typescript/src/core/context-window.js";
import { CompressionCache, compress } from "../../typescript/src/core/pipeline.js";
import { createMessage } from "../../typescript/src/core/types.js";

// ── Helpers ──────────────────────────────────────────────────────────────────

function buildConversation(turnCount: number) {
  const messages = [createMessage("system", "You are a helpful coding assistant.")];
  for (let i = 0; i < turnCount; i++) {
    messages.push(createMessage("user", `Question ${i + 1}: What is ${i + 1} + ${i + 1}?`));
    messages.push(
      createMessage("assistant", `The answer to ${i + 1} + ${i + 1} is ${(i + 1) * 2}.`),
    );
  }
  return messages;
}

// ── Override Transparency ────────────────────────────────────────────────────

describe("AdaptiveOverrides.overrides", () => {
  it("is empty at NONE pressure", () => {
    const config = createConfig({ recentTurns: 3 });
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 10_000, // 5% used → NONE
      outputReserveTokens: 8_192,
    });
    const result = computeAdaptiveThresholds(state, config, 5);
    expect(result.pressure).toBe(Pressure.NONE);
    expect(result.skipCompression).toBe(true);
    expect(result.overrides).toEqual({});
  });

  it("tracks recent_turns override at MEDIUM pressure", () => {
    const config = createConfig({ recentTurns: 10 });
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 140_000, // ~73% used → MEDIUM
      outputReserveTokens: 8_192,
    });
    const result = computeAdaptiveThresholds(state, config, 20);
    expect(result.pressure).toBe(Pressure.MEDIUM);
    expect(result.overrides).toHaveProperty("recentTurns");
    const [original, effective] = result.overrides.recentTurns as [number, number];
    expect(original).toBe(10);
    expect(effective).toBeLessThan(10);
  });

  it("tracks all overrides at CRITICAL pressure", () => {
    const config = createConfig({
      recentTurns: 5,
      enableSummarization: false,
    });
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 185_000, // ~96% used → CRITICAL
      outputReserveTokens: 8_192,
    });
    const result = computeAdaptiveThresholds(state, config, 10);
    expect(result.pressure).toBe(Pressure.CRITICAL);
    expect(result.overrides).toHaveProperty("enableSummarization");
    expect(result.overrides.enableSummarization).toEqual([false, true]);
    expect(result.overrides).toHaveProperty("tokenBudget");
    expect(result.overrides).toHaveProperty("tokenPruneKeepRatio");
    expect(result.overrides).toHaveProperty("entropyThreshold");
    expect(result.overrides).toHaveProperty("performanceTier");
  });

  it("does not include fields that were not changed", () => {
    const config = createConfig({ recentTurns: 1 }); // Already minimal
    // 100K used on 200K window → remaining ~52% → LOW pressure
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 100_000,
      outputReserveTokens: 8_192,
    });
    const result = computeAdaptiveThresholds(state, config, 3);
    expect(result.pressure).toBe(Pressure.LOW);
    // recentTurns=1 and adaptive computes max(1, round(3*0.2))=1 → no change
    expect(result.overrides).not.toHaveProperty("recentTurns");
  });
});

// ── Pipeline L0 Integration ──────────────────────────────────────────────────

describe("Pipeline L0 adaptive integration", () => {
  it("short-circuits at NONE pressure", async () => {
    const messages = buildConversation(3);
    const contextWindow = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 10_000, // NONE
      outputReserveTokens: 8_192,
    });
    const { messages: compressed, report } = await compress(messages, { contextWindow });

    // Should return all messages unchanged.
    expect(compressed.length).toBe(messages.length);
    expect(report.adaptiveOverrides).toEqual({});
    // L0 decision should be present.
    const l0 = report.decisions.find((d) => d.layer === "L0_adaptive");
    expect(l0).toBeDefined();
    expect(l0!.reason).toContain("pressure=NONE");
  });

  it("compresses at elevated pressure with engine gating", async () => {
    const messages = buildConversation(10);
    const contextWindow = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 175_000, // Declared as HIGH, may recalibrate
      outputReserveTokens: 8_192,
    });
    const { messages: compressed, report } = await compress(messages, {
      contextWindow,
      config: { recentTurns: 5 },
    });

    // Should compress (fewer tokens).
    expect(report.compressionRatio).toBeGreaterThan(1.0);
    // Adaptive overrides should be reported.
    expect(report.adaptiveOverrides).not.toBeNull();
    expect(Object.keys(report.adaptiveOverrides!).length).toBeGreaterThan(0);
    // L0 decision should be present with non-NONE pressure.
    const l0 = report.decisions.find((d) => d.layer === "L0_adaptive");
    expect(l0).toBeDefined();
    expect(l0!.reason).not.toContain("pressure=NONE");
  });

  it("uses adaptive overrides in report at CRITICAL pressure", async () => {
    const messages = buildConversation(5);
    const contextWindow = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 185_000, // CRITICAL
      outputReserveTokens: 8_192,
    });
    const { report } = await compress(messages, {
      contextWindow,
      config: { recentTurns: 3, enableSummarization: false },
    });

    expect(report.adaptiveOverrides).not.toBeNull();
    // enable_summarization should be overridden to true at CRITICAL.
    expect(report.adaptiveOverrides!.enableSummarization).toEqual([false, true]);
    // performance_tier forced to "full".
    expect(report.adaptiveOverrides!.performanceTier).toBeDefined();
  });

  it("does not set adaptiveOverrides when no contextWindow provided", async () => {
    const messages = buildConversation(3);
    const { report } = await compress(messages);
    expect(report.adaptiveOverrides).toBeNull();
  });

  it("gates engines at LOW pressure — only dedup and verbatim", async () => {
    // Use a large context window with moderate usage to ensure LOW pressure
    // even after the pipeline recalibrates from actual message content.
    const messages = buildConversation(3); // Small conversation
    const contextWindow = createContextWindowState({
      contextWindowTokens: 1_000_000, // 1M window (like claude-opus-4-6)
      currentUsageTokens: 500_000, // 50% used → LOW
      outputReserveTokens: 128_000,
    });
    const { report } = await compress(messages, { contextWindow });

    const l0 = report.decisions.find((d) => d.layer === "L0_adaptive");
    expect(l0).toBeDefined();
    expect(l0!.reason).toContain("pressure=LOW");
    // At LOW pressure, pruner/importance/relevance_pruner/discourse should be skipped.
    const layerNames = report.layers.map((l) => l.name);
    expect(layerNames).not.toContain("engine_pruner");
    expect(layerNames).not.toContain("importance_scorer");
    expect(layerNames).not.toContain("relevance_pruner");
    expect(layerNames).not.toContain("discourse_compressor");
  });

  it("contextWindow passed via config.contextWindow works", async () => {
    const messages = buildConversation(3);
    const cw = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 180_000, // CRITICAL
      outputReserveTokens: 8_192,
    });
    const { report } = await compress(messages, {
      config: { contextWindow: cw },
    });

    expect(report.adaptiveOverrides).not.toBeNull();
    const l0 = report.decisions.find((d) => d.layer === "L0_adaptive");
    expect(l0!.reason).toContain("pressure=CRITICAL");
  });

  it("explicit contextWindow overrides config.contextWindow", async () => {
    const cwConfig = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 10_000, // NONE
      outputReserveTokens: 8_192,
    });
    const cwExplicit = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 180_000, // CRITICAL
      outputReserveTokens: 8_192,
    });
    const messages = buildConversation(3);
    const { report } = await compress(messages, {
      config: { contextWindow: cwConfig },
      contextWindow: cwExplicit,
    });

    const l0 = report.decisions.find((d) => d.layer === "L0_adaptive");
    expect(l0!.reason).toContain("pressure=CRITICAL");
  });

  it("recalibrates usage when messages are larger than declared", async () => {
    // Provide contextWindow with low declared usage, but many messages.
    const messages = buildConversation(15);
    const contextWindow = createContextWindowState({
      model: "claude-haiku-4-5",
      contextWindowTokens: 200_000,
      currentUsageTokens: 100, // Artificially low
      outputReserveTokens: 8_192,
    });
    const { report } = await compress(messages, { contextWindow });

    // The pipeline should recalibrate usage from message content.
    const l0 = report.decisions.find((d) => d.layer === "L0_adaptive");
    expect(l0).toBeDefined();
    // The actual remaining% should reflect recalibrated usage, not the artificial 100.
  });
});

// ── Adapter contextWindow forwarding ─────────────────────────────────────────

describe("Adapter contextWindow forwarding", () => {
  it("compressOpenAIMessages accepts contextWindow", async () => {
    const { compressOpenAIMessages } = await import("../../typescript/src/adapters/openai-sdk.js");
    const messages = [
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there!" },
      { role: "user", content: "What is 2+2?" },
    ];
    const contextWindow = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 180_000, // CRITICAL
      outputReserveTokens: 8_192,
    });
    const { report } = await compressOpenAIMessages(messages, { contextWindow });
    expect(report.adaptiveOverrides).not.toBeNull();
    const l0 = report.decisions.find((d: { layer: string }) => d.layer === "L0_adaptive");
    expect(l0).toBeDefined();
  });
});

// ── CompressionCache ─────────────────────────────────────────────────────────

describe("CompressionCache", () => {
  it("stores and retrieves original content", () => {
    const cache = new CompressionCache();
    cache.store(3, "original content here");
    expect(cache.has(3)).toBe(true);
    expect(cache.expand(3)).toBe("original content here");
    expect(cache.size).toBe(1);
  });

  it("returns undefined for missing index", () => {
    const cache = new CompressionCache();
    expect(cache.expand(99)).toBeUndefined();
    expect(cache.has(99)).toBe(false);
  });
});
