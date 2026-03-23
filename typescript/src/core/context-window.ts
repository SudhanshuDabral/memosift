// Layer 0: Context-aware adaptive compression — dynamic thresholds based on model context window.

import type { MemoSiftConfig } from "../config.js";

// ── Pressure levels ──────────────────────────────────────────────────────

/** Context window pressure level — drives adaptive compression thresholds. */
export enum Pressure {
  NONE = "NONE", // >60% remaining — no compression needed
  LOW = "LOW", // 40-60% remaining — light (dedup + verbatim only)
  MEDIUM = "MEDIUM", // 25-40% remaining — standard pipeline
  HIGH = "HIGH", // 10-25% remaining — aggressive
  CRITICAL = "CRITICAL", // <10% remaining — maximum compression
}

// ── Model context window registry ────────────────────────────────────────

/**
 * Maps model name prefixes to context window sizes (input tokens).
 * Lookup does longest-prefix matching so "gpt-4o-2024-08-06" matches "gpt-4o".
 */
export const MODEL_CONTEXT_WINDOWS: Readonly<Record<string, number>> = Object.freeze({
  // OpenAI
  "gpt-4o-mini": 128_000,
  "gpt-4o": 128_000,
  "gpt-4.1-nano": 1_047_576,
  "gpt-4.1-mini": 1_047_576,
  "gpt-4.1": 1_047_576,
  "o3-mini": 200_000,
  o3: 200_000,
  "o4-mini": 200_000,
  // Anthropic
  "claude-opus-4-6": 1_000_000,
  "claude-sonnet-4-6": 1_000_000,
  "claude-opus-4-5": 200_000,
  "claude-sonnet-4-5": 200_000,
  "claude-sonnet-4": 200_000,
  "claude-haiku-4-5": 200_000,
  // Google
  "gemini-2.5-pro": 1_048_576,
  "gemini-2.5-flash-lite": 1_048_576,
  "gemini-2.5-flash": 1_048_576,
});

/** Max output tokens per model. */
export const MODEL_OUTPUT_LIMITS: Readonly<Record<string, number>> = Object.freeze({
  "gpt-4o-mini": 16_384,
  "gpt-4o": 16_384,
  "gpt-4.1-nano": 32_768,
  "gpt-4.1-mini": 32_768,
  "gpt-4.1": 32_768,
  "o3-mini": 100_000,
  o3: 100_000,
  "o4-mini": 100_000,
  "claude-opus-4-6": 128_000,
  "claude-sonnet-4-6": 64_000,
  "claude-opus-4-5": 64_000,
  "claude-sonnet-4-5": 64_000,
  "claude-sonnet-4": 64_000,
  "claude-haiku-4-5": 64_000,
  "gemini-2.5-pro": 65_535,
  "gemini-2.5-flash-lite": 65_535,
  "gemini-2.5-flash": 65_535,
});

export const DEFAULT_CONTEXT_WINDOW = 200_000;
export const DEFAULT_OUTPUT_RESERVE = 8_192;

// ── Lookup functions ─────────────────────────────────────────────────────

/** Look up context window size for a model by longest-prefix match. */
export function lookupContextWindow(model: string): number | null {
  const lower = model.toLowerCase();
  const keys = Object.keys(MODEL_CONTEXT_WINDOWS).sort((a, b) => b.length - a.length);
  for (const prefix of keys) {
    if (lower.startsWith(prefix.toLowerCase())) {
      return MODEL_CONTEXT_WINDOWS[prefix]!;
    }
  }
  return null;
}

/** Look up max output tokens for a model by longest-prefix match. */
export function lookupOutputLimit(model: string): number | null {
  const lower = model.toLowerCase();
  const keys = Object.keys(MODEL_OUTPUT_LIMITS).sort((a, b) => b.length - a.length);
  for (const prefix of keys) {
    if (lower.startsWith(prefix.toLowerCase())) {
      return MODEL_OUTPUT_LIMITS[prefix]!;
    }
  }
  return null;
}

// ── ContextWindowState ───────────────────────────────────────────────────

/**
 * Immutable snapshot of model context window capacity and current utilization.
 *
 * Drives the adaptive compression system. Create a new instance when the
 * context changes (e.g., after each turn, or on model switch).
 */
export interface ContextWindowState {
  readonly model: string | null;
  readonly contextWindowTokens: number;
  readonly currentUsageTokens: number;
  readonly outputReserveTokens: number;
}

/** Create a new ContextWindowState with defaults. */
export function createContextWindowState(
  overrides?: Partial<ContextWindowState>,
): ContextWindowState {
  return Object.freeze({
    model: overrides?.model ?? null,
    contextWindowTokens: overrides?.contextWindowTokens ?? DEFAULT_CONTEXT_WINDOW,
    currentUsageTokens: overrides?.currentUsageTokens ?? 0,
    outputReserveTokens: overrides?.outputReserveTokens ?? DEFAULT_OUTPUT_RESERVE,
  });
}

/** Create a ContextWindowState from a model name using the registry. */
export function contextWindowFromModel(
  model: string,
  currentUsageTokens = 0,
  outputReserveTokens?: number,
): ContextWindowState {
  const window = lookupContextWindow(model) ?? DEFAULT_CONTEXT_WINDOW;
  const reserve = outputReserveTokens ?? lookupOutputLimit(model) ?? DEFAULT_OUTPUT_RESERVE;
  return createContextWindowState({
    model,
    contextWindowTokens: window,
    currentUsageTokens,
    outputReserveTokens: reserve,
  });
}

// ── Computed properties ──────────────────────────────────────────────────

/** Usable capacity: total window minus output reserve. */
export function effectiveCapacity(state: ContextWindowState): number {
  return Math.max(0, state.contextWindowTokens - state.outputReserveTokens);
}

/** Tokens remaining for input context. */
export function availableTokens(state: ContextWindowState): number {
  return Math.max(0, effectiveCapacity(state) - state.currentUsageTokens);
}

/** Fraction of effective capacity consumed (0.0–1.0). */
export function usageRatio(state: ContextWindowState): number {
  const cap = effectiveCapacity(state);
  if (cap <= 0) return 1.0;
  return Math.min(1.0, Math.max(0.0, state.currentUsageTokens / cap));
}

/** Fraction of effective capacity still available (0.0–1.0). */
export function remainingRatio(state: ContextWindowState): number {
  return 1.0 - usageRatio(state);
}

/** Context window pressure level derived from remaining capacity. */
export function pressure(state: ContextWindowState): Pressure {
  const r = remainingRatio(state);
  if (r > 0.6) return Pressure.NONE;
  if (r > 0.4) return Pressure.LOW;
  if (r > 0.25) return Pressure.MEDIUM;
  if (r > 0.1) return Pressure.HIGH;
  return Pressure.CRITICAL;
}

// ── Adaptive threshold computation ───────────────────────────────────────

const RECENT_TURN_RATIOS: Record<Pressure, number> = {
  [Pressure.NONE]: 0.3,
  [Pressure.LOW]: 0.2,
  [Pressure.MEDIUM]: 0.12,
  [Pressure.HIGH]: 0.08,
  [Pressure.CRITICAL]: 0.05,
};

const BUDGET_RATIOS: Record<Pressure, number | null> = {
  [Pressure.NONE]: null,
  [Pressure.LOW]: 0.9,
  [Pressure.MEDIUM]: 0.7,
  [Pressure.HIGH]: 0.5,
  [Pressure.CRITICAL]: 0.3,
};

const PRUNE_MULTIPLIERS: Record<Pressure, number> = {
  [Pressure.NONE]: 1.0,
  [Pressure.LOW]: 1.0,
  [Pressure.MEDIUM]: 0.85,
  [Pressure.HIGH]: 0.7,
  [Pressure.CRITICAL]: 0.5,
};

const ENTROPY_MULTIPLIERS: Record<Pressure, number> = {
  [Pressure.NONE]: 1.0,
  [Pressure.LOW]: 1.0,
  [Pressure.MEDIUM]: 0.9,
  [Pressure.HIGH]: 0.75,
  [Pressure.CRITICAL]: 0.55,
};

/** Computed adaptive thresholds from Layer 0 context assessment. */
export interface AdaptiveOverrides {
  readonly pressure: Pressure;
  readonly effectiveConfig: MemoSiftConfig;
  readonly skipCompression: boolean;
  readonly enableObservationMasking: boolean;
  readonly engineGates: ReadonlySet<string>;
  readonly contextWindow: ContextWindowState;
}

// Engine sets per pressure level.
const ENGINES_NONE = new Set<string>();
const ENGINES_LOW = new Set(["dedup", "verbatim"]);
const ENGINES_MEDIUM = new Set(["dedup", "verbatim", "pruner", "structural", "discourse"]);
const ENGINES_HIGH = new Set([
  "dedup",
  "verbatim",
  "pruner",
  "structural",
  "importance",
  "relevance_pruner",
  "discourse",
]);
const ENGINES_CRITICAL = new Set([
  "dedup",
  "verbatim",
  "pruner",
  "structural",
  "importance",
  "relevance_pruner",
  "discourse",
  "summarizer",
]);

const PRESSURE_ENGINES: Record<Pressure, ReadonlySet<string>> = {
  [Pressure.NONE]: ENGINES_NONE,
  [Pressure.LOW]: ENGINES_LOW,
  [Pressure.MEDIUM]: ENGINES_MEDIUM,
  [Pressure.HIGH]: ENGINES_HIGH,
  [Pressure.CRITICAL]: ENGINES_CRITICAL,
};

/**
 * Compute adaptive compression thresholds from context window state.
 *
 * Returns a new config with thresholds tuned to the current pressure level.
 * Never mutates the input config.
 */
export function computeAdaptiveThresholds(
  state: ContextWindowState,
  config: MemoSiftConfig,
  totalUserTurns = 0,
): AdaptiveOverrides {
  const p = pressure(state);

  // Skip compression at NONE pressure.
  if (p === Pressure.NONE) {
    return {
      pressure: p,
      effectiveConfig: config,
      skipCompression: true,
      enableObservationMasking: false,
      engineGates: ENGINES_NONE,
      contextWindow: state,
    };
  }

  // Recent turn protection (percentage-based).
  const ratio = RECENT_TURN_RATIOS[p];
  const adaptiveRecent =
    totalUserTurns > 0 ? Math.max(1, Math.round(totalUserTurns * ratio)) : config.recentTurns;
  const effectiveRecent = Math.min(adaptiveRecent, config.recentTurns);

  // Auto-budget from available capacity.
  const budgetRatio = BUDGET_RATIOS[p];
  let effectiveBudget: number | null;
  if (budgetRatio !== null) {
    const autoBudget = Math.max(100, Math.floor(availableTokens(state) * budgetRatio));
    effectiveBudget =
      config.tokenBudget !== null ? Math.min(config.tokenBudget, autoBudget) : autoBudget;
  } else {
    effectiveBudget = config.tokenBudget;
  }

  // Pruning and entropy thresholds.
  const effectivePrune = Math.max(0.1, config.tokenPruneKeepRatio * PRUNE_MULTIPLIERS[p]);
  const effectiveEntropy = Math.max(0.3, config.entropyThreshold * ENTROPY_MULTIPLIERS[p]);

  // Performance tier override.
  const effectiveTier =
    p === Pressure.HIGH || p === Pressure.CRITICAL ? "full" : config.performanceTier;

  // Auto-enable summarization at CRITICAL pressure.
  const effectiveSummarization = p === Pressure.CRITICAL ? true : config.enableSummarization;

  const effectiveConfig: MemoSiftConfig = {
    ...config,
    recentTurns: effectiveRecent,
    tokenBudget: effectiveBudget,
    tokenPruneKeepRatio: effectivePrune,
    entropyThreshold: effectiveEntropy,
    performanceTier: effectiveTier,
    enableSummarization: effectiveSummarization,
  };

  return {
    pressure: p,
    effectiveConfig,
    skipCompression: false,
    enableObservationMasking: p === Pressure.HIGH || p === Pressure.CRITICAL,
    engineGates: PRESSURE_ENGINES[p],
    contextWindow: state,
  };
}

// ── Resolution helpers ───────────────────────────────────────────────────

/**
 * Resolve a ContextWindowState from available information.
 *
 * Priority: explicit > model_name > null.
 */
export function resolveContextWindow(
  explicit: ContextWindowState | null | undefined,
  modelName: string | null | undefined,
  messagesTokenEstimate = 0,
): ContextWindowState | null {
  if (explicit) return explicit;
  if (modelName) return contextWindowFromModel(modelName, messagesTokenEstimate);
  return null;
}

/** Fast heuristic token estimation: ~3.5 chars per token. */
export function estimateTokensHeuristic(contents: string[]): number {
  let totalChars = 0;
  for (const c of contents) {
    if (c) totalChars += c.length;
  }
  return Math.max(0, Math.floor((totalChars * 10) / 35));
}
