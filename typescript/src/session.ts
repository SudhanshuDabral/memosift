// MemoSiftSession — stateful compression session, the recommended entry point.

import { readFileSync, writeFileSync } from "node:fs";
import type { MemoSiftConfig } from "./config.js";
import { createConfig, createPreset } from "./config.js";
import type { ContextWindowState } from "./core/context-window.js";
import {
  Pressure,
  pressure as computePressure,
  contextWindowFromModel,
  estimateTokensHeuristic,
} from "./core/context-window.js";
import { CompressionCache, compress } from "./core/pipeline.js";
import type { CompressResult } from "./core/pipeline.js";
import { type CompressionState, createCompressionState } from "./core/state.js";
import {
  type AnchorCategory,
  AnchorLedger,
  createCrossWindowState,
  createMessage,
} from "./core/types.js";
import type { AnchorFact, CrossWindowState, MemoSiftMessage } from "./core/types.js";
import { VALID_FRAMEWORKS, detectFramework } from "./detect.js";
import type { Framework } from "./detect.js";
import type { MemoSiftLLMProvider } from "./providers/base.js";
import type { CompressionReport } from "./report.js";

import {
  adaptIn as anthropicAdaptIn,
  adaptOut as anthropicAdaptOut,
} from "./adapters/anthropic-sdk.js";
import type { AnthropicCompressedResult } from "./adapters/anthropic-sdk.js";
import {
  adaptIn as agentSdkAdaptIn,
  adaptOut as agentSdkAdaptOut,
} from "./adapters/claude-agent-sdk.js";
import { adaptIn as adkAdaptIn, adaptOut as adkAdaptOut } from "./adapters/google-adk.js";
import {
  adaptIn as langchainAdaptIn,
  adaptOut as langchainAdaptOut,
} from "./adapters/langchain.js";
// Adapters — adaptIn/adaptOut use duck typing on plain objects, no framework SDK imports needed.
import { adaptIn as openaiAdaptIn, adaptOut as openaiAdaptOut } from "./adapters/openai-sdk.js";
import { adaptIn as vercelAdaptIn, adaptOut as vercelAdaptOut } from "./adapters/vercel-ai.js";

/** Valid config field names for override validation. */
const CONFIG_FIELDS = new Set([
  "recentTurns",
  "tokenBudget",
  "enableSummarization",
  "llmRelevanceScoring",
  "reorderSegments",
  "dedupSimilarityThreshold",
  "entropyThreshold",
  "tokenPruneKeepRatio",
  "jsonArrayThreshold",
  "codeKeepSignatures",
  "relevanceDropThreshold",
  "policies",
  "softCompressionPct",
  "fullCompressionPct",
  "aggressiveCompressionPct",
  "coalesceShortMessages",
  "coalesceCharThreshold",
  "enableAnchorLedger",
  "anchorLedgerMaxTokens",
  "costPer1kTokens",
  "modelName",
  "deterministicSeed",
  "performanceTier",
  "preBucketBypass",
  "contextWindow",
]);

export interface MemoSiftSessionOptions {
  model?: string;
  llm?: MemoSiftLLMProvider;
  framework?: Framework;
  incremental?: boolean;
  configOverrides?: Partial<MemoSiftConfig>;
}

export interface SessionCompressOptions {
  task?: string;
  usageTokens?: number;
  system?: string;
}

/**
 * Stateful compression session — owns ledger, dedup state, and cache.
 *
 * Collapses the raw `compress()` API's 7 objects + 51 knobs into a single
 * constructor + a single `compress()` call.
 *
 * ```typescript
 * const session = new MemoSiftSession("coding", { model: "claude-sonnet-4-6" });
 * const { messages, report } = await session.compress(openaiMessages, { usageTokens: 150_000 });
 * ```
 */
export class MemoSiftSession {
  private _model: string | null;
  private _llm: MemoSiftLLMProvider | null;
  private _framework: Framework | null;
  private _frameworkDetected: boolean;
  private _preset: string;
  private _config: MemoSiftConfig;
  private _ledger: AnchorLedger;
  private _crossWindow: CrossWindowState;
  private _cache: CompressionCache;
  private _incremental: boolean;
  private _state: CompressionState | null;
  private _lastReport: CompressionReport | null = null;
  private _system: string | null = null;

  constructor(preset = "general", options?: MemoSiftSessionOptions) {
    const model = options?.model ?? null;
    const llm = options?.llm ?? null;
    const framework = options?.framework ?? null;
    const overrides = options?.configOverrides;

    // Validate framework.
    if (framework !== null && !VALID_FRAMEWORKS.has(framework)) {
      throw new Error(
        `Unknown framework "${framework}". Valid: ${[...VALID_FRAMEWORKS].sort().join(", ")}`,
      );
    }

    // Validate config overrides.
    if (overrides) {
      for (const key of Object.keys(overrides)) {
        if (!CONFIG_FIELDS.has(key)) {
          throw new Error(
            `Unknown config field "${key}". Valid: ${[...CONFIG_FIELDS].sort().join(", ")}`,
          );
        }
      }
    }

    this._model = model;
    this._llm = llm;
    this._framework = framework;
    this._frameworkDetected = framework !== null;
    this._preset = preset;

    // Build config from preset + overrides.
    const base = preset !== "general" ? createPreset(preset) : createConfig();
    this._config = overrides ? createConfig({ ...base, ...overrides }) : base;

    // Persistent state.
    this._ledger = new AnchorLedger();
    this._crossWindow = createCrossWindowState();
    this._cache = new CompressionCache();
    this._incremental = options?.incremental ?? false;
    this._state = this._incremental ? createCompressionState() : null;
  }

  /**
   * Compress messages through the pipeline.
   *
   * Accepts framework-native messages (auto-detected or per `framework`
   * option on constructor). Returns compressed messages in the same format.
   */
  async compress(
    messages: unknown[],
    options?: SessionCompressOptions,
  ): Promise<{ messages: unknown[]; report: CompressionReport }> {
    const task = options?.task ?? null;
    const usageTokens = options?.usageTokens ?? null;
    const system = options?.system ?? null;

    // Framework detection (cached after first call).
    if (!this._frameworkDetected) {
      this._framework = detectFramework(messages);
      this._frameworkDetected = true;
    }

    // Adapt in.
    const internal = this.adaptIn(messages, system);

    // Build context window state.
    let contextWindow: ContextWindowState | null = null;
    if (this._model !== null) {
      const tokens = usageTokens ?? estimateTokensHeuristic(internal.map((m) => m.content));
      contextWindow = contextWindowFromModel(this._model, tokens);
    }

    // Compress.
    const { messages: compressed, report } = await compress(internal, {
      llm: this._llm,
      config: this._config,
      task,
      ledger: this._ledger,
      crossWindow: this._crossWindow,
      cache: this._cache,
      contextWindow,
      state: this._state,
    });

    this._lastReport = report;

    // Adapt out.
    const result = this.adaptOut(compressed, system);
    return { messages: result, report };
  }

  private adaptIn(messages: unknown[], system: string | null): MemoSiftMessage[] {
    const fw = this._framework;
    if (fw === "memosift") return messages as MemoSiftMessage[];
    if (fw === "openai") return openaiAdaptIn(messages as Record<string, unknown>[]);
    if (fw === "anthropic") return anthropicAdaptIn(messages as Record<string, unknown>[], system);
    if (fw === "agent_sdk") return agentSdkAdaptIn(messages);
    if (fw === "adk") return adkAdaptIn(messages as Record<string, unknown>[]);
    if (fw === "langchain") return langchainAdaptIn(messages);
    if (fw === "vercel_ai") return vercelAdaptIn(messages as Record<string, unknown>[]);
    return openaiAdaptIn(messages as Record<string, unknown>[]);
  }

  private adaptOut(messages: MemoSiftMessage[], system: string | null): unknown[] {
    const fw = this._framework;
    if (fw === "memosift") return messages;
    if (fw === "openai") return openaiAdaptOut(messages);
    if (fw === "anthropic") {
      const result: AnthropicCompressedResult = anthropicAdaptOut(messages);
      this._system = result.system;
      return result.messages;
    }
    if (fw === "agent_sdk") return agentSdkAdaptOut(messages);
    if (fw === "adk") return adkAdaptOut(messages);
    if (fw === "langchain") return langchainAdaptOut(messages);
    if (fw === "vercel_ai") return vercelAdaptOut(messages);
    return openaiAdaptOut(messages);
  }

  /** Check current context window pressure without compressing. */
  checkPressure(usageTokens?: number): Pressure {
    if (this._model === null) return Pressure.NONE;
    const state = contextWindowFromModel(this._model, usageTokens ?? 0);
    return computePressure(state);
  }

  /** The model name this session was created with. */
  get model(): string | null {
    return this._model;
  }

  /** The current config preset name. */
  get preset(): string {
    return this._preset;
  }

  /** The detected or configured framework. */
  get framework(): Framework | null {
    return this._framework;
  }

  /** Set the framework explicitly (skips auto-detection). */
  setFramework(framework: Framework): void {
    this._framework = framework;
    this._frameworkDetected = true;
  }

  /** The session's anchor ledger (accumulates facts across compress calls). */
  get ledger(): AnchorLedger {
    return this._ledger;
  }

  /** All extracted anchor facts. */
  get facts(): readonly AnchorFact[] {
    return this._ledger.facts;
  }

  /** Compression report from the most recent compress() call. */
  get lastReport(): CompressionReport | null {
    return this._lastReport;
  }

  /** Whether incremental compression is enabled. */
  get incremental(): boolean {
    return this._incremental;
  }

  /** The CompressionState for incremental mode, or null if disabled. */
  get state(): CompressionState | null {
    return this._state;
  }

  /** Anthropic system prompt from the most recent compression. */
  get system(): string | null {
    return this._system;
  }

  /**
   * Re-expand a previously compressed message.
   *
   * Original content is only available within the same session lifecycle.
   * Cache is not persisted across saveState/loadState.
   */
  expand(originalIndex: number): string | undefined {
    return this._cache.expand(originalIndex);
  }

  /**
   * Change config while preserving session state (ledger, dedup, cache).
   */
  reconfigure(preset?: string, overrides?: Partial<MemoSiftConfig>): void {
    if (overrides) {
      for (const key of Object.keys(overrides)) {
        if (!CONFIG_FIELDS.has(key)) {
          throw new Error(
            `Unknown config field "${key}". Valid: ${[...CONFIG_FIELDS].sort().join(", ")}`,
          );
        }
      }
    }

    if (preset !== undefined) {
      this._preset = preset;
    }

    const base =
      preset !== undefined
        ? preset !== "general"
          ? createPreset(preset)
          : createConfig()
        : this._config;

    this._config = overrides ? createConfig({ ...base, ...overrides }) : base;
  }

  /**
   * Persist session state (ledger + dedup hashes) to a JSON file.
   *
   * CompressionCache is NOT serialized — original content is only
   * available within the same session lifecycle.
   */
  saveState(path: string): void {
    const state = {
      version: 1,
      ledger: {
        facts: this._ledger.facts.map((f) => ({
          category: f.category,
          content: f.content,
          turn: f.turn,
          confidence: f.confidence,
        })),
      },
      cross_window_hashes: [...this._crossWindow.contentHashes].sort(),
      framework: this._framework,
      model: this._model,
      config_preset: this._preset,
    };
    writeFileSync(path, JSON.stringify(state, null, 2), "utf-8");
  }

  /**
   * Restore a session from saved state.
   *
   * CompressionCache is NOT restored.
   */
  static loadState(
    path: string,
    preset = "general",
    options?: MemoSiftSessionOptions,
  ): MemoSiftSession {
    const data = JSON.parse(readFileSync(path, "utf-8"));

    const effectivePreset = preset !== "general" ? preset : (data.config_preset ?? "general");
    const effectiveModel = options?.model ?? data.model ?? null;
    const effectiveFramework = options?.framework ?? data.framework ?? null;

    const session = new MemoSiftSession(effectivePreset, {
      model: effectiveModel,
      llm: options?.llm,
      framework: effectiveFramework,
      configOverrides: options?.configOverrides,
    });

    // Restore ledger facts.
    for (const f of data.ledger?.facts ?? []) {
      session._ledger.add({
        category: f.category as AnchorCategory,
        content: f.content,
        turn: f.turn ?? 0,
        confidence: f.confidence ?? 1.0,
      });
    }

    // Restore cross-window hashes.
    for (const h of data.cross_window_hashes ?? []) {
      session._crossWindow.contentHashes.add(h);
    }

    return session;
  }
}
