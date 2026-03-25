// MemoSift — Framework-agnostic context compaction engine for agentic AI systems.

export {
  Pressure,
  MODEL_CONTEXT_WINDOWS,
  MODEL_OUTPUT_LIMITS,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_OUTPUT_RESERVE,
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
} from "./core/context-window.js";
export type {
  ContextWindowState,
  AdaptiveOverrides,
  OverrideEntry,
} from "./core/context-window.js";

export { compress, CompressionCache } from "./core/pipeline.js";
export type { CompressOptions, CompressResult } from "./core/pipeline.js";
export { createCompressionState } from "./core/state.js";
export type { CompressionState } from "./core/state.js";
export { partitionZones } from "./core/pipeline.js";

export { ConversationPhase, PHASE_KEEP_MULTIPLIERS, detectPhase } from "./core/phase-detector.js";

export {
  ContentType,
  CompressionPolicy,
  DEFAULT_POLICIES,
  AnchorCategory,
  AnchorLedger,
  createMessage,
  createClassified,
  createAnchorFact,
  createDependencyMap,
  createCrossWindowState,
  saveCrossWindowState,
  loadCrossWindowState,
  getContent,
  depMapAdd,
  depMapCanDrop,
  depMapDependentsOf,
} from "./core/types.js";
export type {
  MemoSiftMessage,
  ClassifiedMessage,
  ToolCall,
  ToolCallFunction,
  AnchorFact,
  DependencyMap,
  CrossWindowState,
} from "./core/types.js";

export { createConfig, createPreset, MODEL_BUDGET_DEFAULTS, MODEL_PRICING } from "./config.js";
export type { MemoSiftConfig } from "./config.js";

export { CompressionReport } from "./report.js";
export type { Decision, LayerReport } from "./report.js";

export type { MemoSiftLLMProvider, LLMResponse } from "./providers/base.js";
export { HeuristicTokenCounter } from "./providers/heuristic.js";

// Session Facade — the recommended entry point.
export { MemoSiftSession } from "./session.js";
export type { MemoSiftSessionOptions, SessionCompressOptions } from "./session.js";

// Streaming compression.
export { MemoSiftStream } from "./stream.js";
export type { StreamEvent } from "./stream.js";

// Framework auto-detection.
export { detectFramework, VALID_FRAMEWORKS } from "./detect.js";
export type { Framework } from "./detect.js";

// Adapters — re-exported for convenience, also available via deep imports.
export {
  adaptIn as openaiAdaptIn,
  adaptOut as openaiAdaptOut,
  compressOpenAIMessages,
  OpenAILLMProvider,
} from "./adapters/openai-sdk.js";
export {
  adaptIn as anthropicAdaptIn,
  adaptOut as anthropicAdaptOut,
  compressAnthropicMessages,
  AnthropicLLMProvider,
} from "./adapters/anthropic-sdk.js";
export type { AnthropicCompressedResult } from "./adapters/anthropic-sdk.js";
export {
  adaptIn as langchainAdaptIn,
  adaptOut as langchainAdaptOut,
  compressLangChainMessages,
  LangChainLLMProvider,
} from "./adapters/langchain.js";
export {
  adaptIn as agentSdkAdaptIn,
  adaptOut as agentSdkAdaptOut,
  compressAgentSdkMessages,
  ClaudeAgentLLMProvider,
} from "./adapters/claude-agent-sdk.js";
export {
  adaptIn as adkAdaptIn,
  adaptOut as adkAdaptOut,
  compressAdkEvents,
} from "./adapters/google-adk.js";
export {
  adaptIn as vercelAdaptIn,
  adaptOut as vercelAdaptOut,
  compressVercelMessages,
  VercelAILLMProvider,
} from "./adapters/vercel-ai.js";
