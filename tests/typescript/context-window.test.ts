// Tests for Layer 0: Context-aware adaptive compression.

import { describe, it, expect } from "vitest";
import {
  Pressure,
  createContextWindowState,
  contextWindowFromModel,
  effectiveCapacity,
  availableTokens,
  usageRatio,
  remainingRatio,
  pressure,
  lookupContextWindow,
  lookupOutputLimit,
  computeAdaptiveThresholds,
  resolveContextWindow,
  estimateTokensHeuristic,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_OUTPUT_RESERVE,
} from "../../typescript/src/core/context-window.js";
import { createConfig } from "../../typescript/src/config.js";

// ── Pressure Enum ────────────────────────────────────────────────────────

describe("Pressure", () => {
  it("has all five levels", () => {
    expect(Pressure.NONE).toBe("NONE");
    expect(Pressure.LOW).toBe("LOW");
    expect(Pressure.MEDIUM).toBe("MEDIUM");
    expect(Pressure.HIGH).toBe("HIGH");
    expect(Pressure.CRITICAL).toBe("CRITICAL");
  });
});

// ── ContextWindowState ───────────────────────────────────────────────────

describe("ContextWindowState", () => {
  it("has correct defaults", () => {
    const state = createContextWindowState();
    expect(state.model).toBeNull();
    expect(state.contextWindowTokens).toBe(DEFAULT_CONTEXT_WINDOW);
    expect(state.currentUsageTokens).toBe(0);
    expect(state.outputReserveTokens).toBe(DEFAULT_OUTPUT_RESERVE);
  });

  it("computes effective capacity", () => {
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      outputReserveTokens: 8_192,
    });
    expect(effectiveCapacity(state)).toBe(200_000 - 8_192);
  });

  it("clamps effective capacity to zero", () => {
    const state = createContextWindowState({
      contextWindowTokens: 1_000,
      outputReserveTokens: 5_000,
    });
    expect(effectiveCapacity(state)).toBe(0);
  });

  it("computes available tokens", () => {
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 150_000,
      outputReserveTokens: 8_192,
    });
    expect(availableTokens(state)).toBe(200_000 - 8_192 - 150_000);
  });

  it("clamps available tokens to zero", () => {
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 250_000,
      outputReserveTokens: 8_192,
    });
    expect(availableTokens(state)).toBe(0);
  });

  it("computes usage ratio at zero", () => {
    const state = createContextWindowState({ contextWindowTokens: 200_000, currentUsageTokens: 0 });
    expect(usageRatio(state)).toBeCloseTo(0.0);
  });

  it("computes usage ratio at half", () => {
    const cap = 200_000 - DEFAULT_OUTPUT_RESERVE;
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: Math.floor(cap / 2),
    });
    expect(usageRatio(state)).toBeCloseTo(0.5, 1);
  });

  it("clamps usage ratio at overflow", () => {
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 300_000,
    });
    expect(usageRatio(state)).toBe(1.0);
  });

  it("remaining + usage = 1.0", () => {
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 100_000,
    });
    expect(usageRatio(state) + remainingRatio(state)).toBeCloseTo(1.0);
  });

  it("pressure is NONE when mostly empty", () => {
    const cap = 200_000 - DEFAULT_OUTPUT_RESERVE;
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: Math.floor(cap * 0.3),
    });
    expect(pressure(state)).toBe(Pressure.NONE);
  });

  it("pressure is LOW at 50% usage", () => {
    const cap = 200_000 - DEFAULT_OUTPUT_RESERVE;
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: Math.floor(cap * 0.5),
    });
    expect(pressure(state)).toBe(Pressure.LOW);
  });

  it("pressure is MEDIUM at 67% usage", () => {
    const cap = 200_000 - DEFAULT_OUTPUT_RESERVE;
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: Math.floor(cap * 0.67),
    });
    expect(pressure(state)).toBe(Pressure.MEDIUM);
  });

  it("pressure is HIGH at 82% usage", () => {
    const cap = 200_000 - DEFAULT_OUTPUT_RESERVE;
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: Math.floor(cap * 0.82),
    });
    expect(pressure(state)).toBe(Pressure.HIGH);
  });

  it("pressure is CRITICAL at 95% usage", () => {
    const cap = 200_000 - DEFAULT_OUTPUT_RESERVE;
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: Math.floor(cap * 0.95),
    });
    expect(pressure(state)).toBe(Pressure.CRITICAL);
  });

  it("pressure boundary at 60% remaining is LOW", () => {
    const state = createContextWindowState({
      contextWindowTokens: 100_000,
      currentUsageTokens: 40_000,
      outputReserveTokens: 0,
    });
    expect(remainingRatio(state)).toBeCloseTo(0.6);
    expect(pressure(state)).toBe(Pressure.LOW);
  });

  it("is frozen (Object.freeze)", () => {
    const state = createContextWindowState();
    expect(Object.isFrozen(state)).toBe(true);
  });
});

// ── from_model ───────────────────────────────────────────────────────────

describe("contextWindowFromModel", () => {
  it("resolves known model", () => {
    const state = contextWindowFromModel("claude-sonnet-4-6", 50_000);
    expect(state.contextWindowTokens).toBe(1_000_000);
    expect(state.currentUsageTokens).toBe(50_000);
    expect(state.model).toBe("claude-sonnet-4-6");
    expect(state.outputReserveTokens).toBe(64_000);
  });

  it("falls back for unknown model", () => {
    const state = contextWindowFromModel("unknown-model-v2");
    expect(state.contextWindowTokens).toBe(DEFAULT_CONTEXT_WINDOW);
    expect(state.outputReserveTokens).toBe(DEFAULT_OUTPUT_RESERVE);
  });

  it("1M model with low usage has NONE pressure", () => {
    const state = contextWindowFromModel("claude-opus-4-6", 50_000);
    expect(pressure(state)).toBe(Pressure.NONE);
  });

  it("200K model with high usage has CRITICAL pressure", () => {
    const state = contextWindowFromModel("claude-haiku-4-5", 180_000);
    expect(pressure(state)).toBe(Pressure.CRITICAL);
  });
});

// ── Model Registry Lookup ────────────────────────────────────────────────

describe("lookupContextWindow", () => {
  it("exact match", () => {
    expect(lookupContextWindow("gpt-4o")).toBe(128_000);
  });

  it("prefix match", () => {
    expect(lookupContextWindow("gpt-4o-2024-08-06")).toBe(128_000);
  });

  it("case insensitive", () => {
    expect(lookupContextWindow("GPT-4o")).toBe(128_000);
    expect(lookupContextWindow("Claude-Sonnet-4-6")).toBe(1_000_000);
  });

  it("unknown model returns null", () => {
    expect(lookupContextWindow("totally-unknown-model")).toBeNull();
  });

  it("output limit lookup", () => {
    expect(lookupOutputLimit("claude-opus-4-6")).toBe(128_000);
    expect(lookupOutputLimit("gpt-4o")).toBe(16_384);
    expect(lookupOutputLimit("unknown-model")).toBeNull();
  });
});

// ── Adaptive Threshold Computation ───────────────────────────────────────

describe("computeAdaptiveThresholds", () => {
  it("NONE pressure skips compression", () => {
    const state = contextWindowFromModel("claude-opus-4-6", 50_000);
    const config = createConfig();
    const result = computeAdaptiveThresholds(state, config, 10);
    expect(result.pressure).toBe(Pressure.NONE);
    expect(result.skipCompression).toBe(true);
    expect(result.engineGates.size).toBe(0);
  });

  it("LOW pressure enables dedup and verbatim only", () => {
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 90_000,
      outputReserveTokens: 8_192,
    });
    expect(pressure(state)).toBe(Pressure.LOW);
    const config = createConfig();
    const result = computeAdaptiveThresholds(state, config, 20);
    expect(result.skipCompression).toBe(false);
    expect(result.engineGates.has("dedup")).toBe(true);
    expect(result.engineGates.has("verbatim")).toBe(true);
    expect(result.engineGates.has("pruner")).toBe(false);
  });

  it("HIGH pressure enables all deterministic + observation masking", () => {
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 160_000,
      outputReserveTokens: 8_192,
    });
    expect(pressure(state)).toBe(Pressure.HIGH);
    const config = createConfig();
    const result = computeAdaptiveThresholds(state, config, 20);
    expect(result.engineGates.has("importance")).toBe(true);
    expect(result.engineGates.has("relevance_pruner")).toBe(true);
    expect(result.enableObservationMasking).toBe(true);
    expect(result.effectiveConfig.performanceTier).toBe("full");
  });

  it("CRITICAL pressure includes summarizer", () => {
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 185_000,
      outputReserveTokens: 8_192,
    });
    expect(pressure(state)).toBe(Pressure.CRITICAL);
    const config = createConfig();
    const result = computeAdaptiveThresholds(state, config, 20);
    expect(result.engineGates.has("summarizer")).toBe(true);
    expect(result.effectiveConfig.enableSummarization).toBe(true);
  });

  it("recent turns are percentage-based", () => {
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 160_000,
      outputReserveTokens: 8_192,
    });
    const config = createConfig({ recentTurns: 10 });
    const result = computeAdaptiveThresholds(state, config, 100);
    expect(result.effectiveConfig.recentTurns).toBe(8); // 8% of 100
  });

  it("recent turns capped at config value", () => {
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 90_000,
      outputReserveTokens: 8_192,
    });
    const config = createConfig({ recentTurns: 2 });
    const result = computeAdaptiveThresholds(state, config, 100);
    expect(result.effectiveConfig.recentTurns).toBe(2);
  });

  it("explicit budget is respected", () => {
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 100_000,
      outputReserveTokens: 8_192,
    });
    const config = createConfig({ tokenBudget: 5_000 });
    const result = computeAdaptiveThresholds(state, config, 20);
    expect(result.effectiveConfig.tokenBudget).toBeLessThanOrEqual(5_000);
  });

  it("does not mutate input config", () => {
    const state = createContextWindowState({
      contextWindowTokens: 200_000,
      currentUsageTokens: 160_000,
      outputReserveTokens: 8_192,
    });
    const config = createConfig({ recentTurns: 5, tokenPruneKeepRatio: 0.5 });
    const origRecent = config.recentTurns;
    const origPrune = config.tokenPruneKeepRatio;
    computeAdaptiveThresholds(state, config, 50);
    expect(config.recentTurns).toBe(origRecent);
    expect(config.tokenPruneKeepRatio).toBe(origPrune);
  });
});

// ── resolveContextWindow ─────────────────────────────────────────────────

describe("resolveContextWindow", () => {
  it("explicit wins", () => {
    const explicit = createContextWindowState({ model: "test", contextWindowTokens: 500_000 });
    const result = resolveContextWindow(explicit, "claude-sonnet-4-6", 10_000);
    expect(result).toBe(explicit);
  });

  it("model name resolution", () => {
    const result = resolveContextWindow(null, "gpt-4o", 50_000);
    expect(result).not.toBeNull();
    expect(result!.contextWindowTokens).toBe(128_000);
    expect(result!.currentUsageTokens).toBe(50_000);
  });

  it("nothing returns null", () => {
    expect(resolveContextWindow(null, null, 0)).toBeNull();
  });
});

// ── estimateTokensHeuristic ──────────────────────────────────────────────

describe("estimateTokensHeuristic", () => {
  it("empty returns 0", () => {
    expect(estimateTokensHeuristic([])).toBe(0);
    expect(estimateTokensHeuristic([""])).toBe(0);
  });

  it("approximates ~3.5 chars per token", () => {
    const text = "a".repeat(350);
    const result = estimateTokensHeuristic([text]);
    expect(result).toBeGreaterThanOrEqual(90);
    expect(result).toBeLessThanOrEqual(110);
  });
});
