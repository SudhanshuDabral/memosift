// Pipeline configuration with sensible defaults and domain presets.

import { CompressionPolicy, ContentType } from "./core/types.js";

export const MODEL_BUDGET_DEFAULTS: Record<string, number> = {
  "gpt-4o": 80_000,
  "gpt-4.1": 600_000,
  "claude-sonnet-4-6": 120_000,
  "claude-opus-4-6": 120_000,
  "gemini-2.5-pro": 600_000,
};

export const MODEL_PRICING: Record<string, number> = {
  "gpt-4o": 0.0025,
  "gpt-4.1": 0.002,
  "claude-sonnet-4-6": 0.003,
  "claude-opus-4-6": 0.015,
  "gemini-2.5-pro": 0.00125,
  default: 0.003,
};

export interface MemoSiftConfig {
  recentTurns: number;
  tokenBudget: number | null;
  enableSummarization: boolean;
  llmRelevanceScoring: boolean;
  reorderSegments: boolean;
  dedupSimilarityThreshold: number;
  entropyThreshold: number;
  tokenPruneKeepRatio: number;
  jsonArrayThreshold: number;
  codeKeepSignatures: boolean;
  relevanceDropThreshold: number;
  policies: Partial<Record<ContentType, CompressionPolicy>>;
  softCompressionPct: number;
  fullCompressionPct: number;
  aggressiveCompressionPct: number;
  enableAnchorLedger: boolean;
  anchorLedgerMaxTokens: number;
  coalesceShortMessages: boolean;
  coalesceCharThreshold: number;
  costPer1kTokens: number;
  modelName: string | null;
  deterministicSeed: number | null;
  performanceTier: string | null;
  preBucketBypass: boolean;
}

const PRESETS: Record<string, Partial<MemoSiftConfig>> = {
  coding: {
    recentTurns: 3,
    entropyThreshold: 2.5,
    tokenPruneKeepRatio: 0.7,
    codeKeepSignatures: true,
    dedupSimilarityThreshold: 0.9,
    relevanceDropThreshold: 0.03,
    jsonArrayThreshold: 3,
    enableAnchorLedger: true,
    policies: {
      [ContentType.ERROR_TRACE]: CompressionPolicy.PRESERVE,
      [ContentType.CODE_BLOCK]: CompressionPolicy.SIGNATURE,
    },
  },
  research: {
    recentTurns: 2,
    entropyThreshold: 1.8,
    tokenPruneKeepRatio: 0.5,
    codeKeepSignatures: false,
    jsonArrayThreshold: 3,
    dedupSimilarityThreshold: 0.8,
    relevanceDropThreshold: 0.08,
    enableAnchorLedger: true,
  },
  support: {
    recentTurns: 5,
    entropyThreshold: 1.5,
    tokenPruneKeepRatio: 0.4,
    codeKeepSignatures: false,
    enableSummarization: true,
    dedupSimilarityThreshold: 0.75,
    relevanceDropThreshold: 0.1,
    jsonArrayThreshold: 2,
    enableAnchorLedger: true,
  },
  data: {
    recentTurns: 3,
    entropyThreshold: 2.0,
    tokenPruneKeepRatio: 0.6,
    codeKeepSignatures: false,
    jsonArrayThreshold: 10,
    dedupSimilarityThreshold: 0.85,
    relevanceDropThreshold: 0.05,
    enableAnchorLedger: true,
  },
  general: {
    recentTurns: 2,
    entropyThreshold: 1.8,
    tokenPruneKeepRatio: 0.5,
    dedupSimilarityThreshold: 0.8,
    relevanceDropThreshold: 0.05,
    enableAnchorLedger: true,
  },
};

export function createConfig(overrides?: Partial<MemoSiftConfig>): MemoSiftConfig {
  const config: MemoSiftConfig = {
    recentTurns: 2,
    tokenBudget: null,
    enableSummarization: false,
    llmRelevanceScoring: false,
    reorderSegments: false,
    dedupSimilarityThreshold: 0.8,
    entropyThreshold: 1.8,
    tokenPruneKeepRatio: 0.5,
    jsonArrayThreshold: 5,
    codeKeepSignatures: true,
    relevanceDropThreshold: 0.05,
    policies: {},
    softCompressionPct: 0.6,
    fullCompressionPct: 0.75,
    aggressiveCompressionPct: 0.9,
    enableAnchorLedger: true,
    anchorLedgerMaxTokens: 5000,
    coalesceShortMessages: true,
    coalesceCharThreshold: 100,
    costPer1kTokens: 0.003,
    modelName: null,
    deterministicSeed: 42,
    performanceTier: null,
    preBucketBypass: true,
    ...overrides,
  };
  validateConfig(config);
  return config;
}

export function createPreset(name: string, overrides?: Partial<MemoSiftConfig>): MemoSiftConfig {
  if (!(name in PRESETS)) {
    throw new Error(
      `Unknown preset '${name}'. Available: ${Object.keys(PRESETS).sort().join(", ")}`,
    );
  }
  return createConfig({ ...PRESETS[name], ...overrides });
}

function validateConfig(config: MemoSiftConfig): void {
  if (config.recentTurns < 0)
    throw new Error(`recentTurns must be >= 0, got ${config.recentTurns}`);
  if (config.tokenBudget !== null && config.tokenBudget < 100)
    throw new Error(`tokenBudget must be >= 100 or null, got ${config.tokenBudget}`);
  if (config.dedupSimilarityThreshold < 0 || config.dedupSimilarityThreshold > 1)
    throw new Error("dedupSimilarityThreshold must be 0.0–1.0");
  if (config.tokenPruneKeepRatio < 0.1 || config.tokenPruneKeepRatio > 1)
    throw new Error("tokenPruneKeepRatio must be 0.1–1.0");
  if (config.softCompressionPct >= config.fullCompressionPct)
    throw new Error("softCompressionPct must be < fullCompressionPct");
  if (config.fullCompressionPct >= config.aggressiveCompressionPct)
    throw new Error("fullCompressionPct must be < aggressiveCompressionPct");
  const VALID_TIERS = new Set(["full", "standard", "fast", "ultra_fast"]);
  if (config.performanceTier !== null && !VALID_TIERS.has(config.performanceTier))
    throw new Error(
      `performanceTier must be one of ${[...VALID_TIERS].join(", ")} or null, got '${config.performanceTier}'`,
    );
}
