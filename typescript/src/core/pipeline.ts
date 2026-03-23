// Pipeline orchestrator — runs the 6-layer compression pipeline with Three-Zone Model.

import type { MemoSiftConfig } from "../config.js";
import { createConfig } from "../config.js";
import type { MemoSiftLLMProvider } from "../providers/base.js";
import { HeuristicTokenCounter } from "../providers/heuristic.js";
import { CompressionReport } from "../report.js";
import { extractAnchorsFromSegments, extractReasoningChains } from "./anchor-extractor.js";
import { enforceBudget } from "./budget.js";
import { classifyMessages } from "./classifier.js";
import { coalesceShortMessages } from "./coalescer.js";
import { deduplicate } from "./deduplicator.js";
import { elaborateCompress } from "./engines/discourse-compressor.js";
import { scoreImportance } from "./engines/importance.js";
import { pruneTokens } from "./engines/pruner.js";
import { queryRelevancePrune } from "./engines/relevance-pruner.js";
import { structuralCompress } from "./engines/structural.js";
import { summarizeSegments } from "./engines/summarizer.js";
import { verbatimCompress } from "./engines/verbatim.js";
import { PHASE_KEEP_MULTIPLIERS, detectPhase } from "./phase-detector.js";
import { optimizePosition } from "./positioner.js";
import { scoreRelevance, scoreRelevanceLlm } from "./scorer.js";
import {
  type AnchorLedger,
  type ClassifiedMessage,
  CompressionPolicy,
  ContentType,
  type CrossWindowState,
  type DependencyMap,
  type MemoSiftMessage,
  createClassified,
  createDependencyMap,
  createMessage,
} from "./types.js";

/** Resolve the performance tier from config or auto-detect from message count. */
export function resolveTier(config: MemoSiftConfig, messageCount: number): string {
  if (config.performanceTier !== null) return config.performanceTier;
  if (messageCount <= 50) return "full";
  if (messageCount <= 150) return "standard";
  if (messageCount <= 300) return "fast";
  return "ultra_fast";
}

/** Content types that bypass compression layers when pre-bucketing is enabled. */
const _BYPASS_TYPES: ReadonlySet<ContentType> = new Set([
  ContentType.SYSTEM_PROMPT,
  ContentType.USER_QUERY,
  ContentType.RECENT_TURN,
  ContentType.PREVIOUSLY_COMPRESSED,
]);

/** Split segments into bypass (skip compression) and compress (run through engines) buckets. */
export function preBucket(
  segments: ClassifiedMessage[],
): [ClassifiedMessage[], ClassifiedMessage[]] {
  const bypass: ClassifiedMessage[] = [];
  const compress: ClassifiedMessage[] = [];
  for (const seg of segments) {
    if (_BYPASS_TYPES.has(seg.contentType as ContentType) || seg.protected) {
      bypass.push(seg);
    } else {
      compress.push(seg);
    }
  }
  return [bypass, compress];
}

/**
 * Stores original content for messages collapsed during compression.
 *
 * Enables selective re-expansion when the agent needs more context
 * (e.g., a re-read request). Keyed by message originalIndex.
 */
export class CompressionCache {
  private readonly originals = new Map<number, string>();

  /** Store original content before collapse. */
  store(originalIndex: number, content: string): void {
    this.originals.set(originalIndex, content);
  }

  /** Retrieve original content for a collapsed message. Returns undefined if not stored. */
  expand(originalIndex: number): string | undefined {
    return this.originals.get(originalIndex);
  }

  /** Return true if original content is stored for this index. */
  has(originalIndex: number): boolean {
    return this.originals.has(originalIndex);
  }

  /** Number of stored originals. */
  get size(): number {
    return this.originals.size;
  }
}

export interface CompressOptions {
  llm?: MemoSiftLLMProvider | null;
  config?: Partial<MemoSiftConfig> | null;
  task?: string | null;
  ledger?: AnchorLedger | null;
  crossWindow?: CrossWindowState | null;
  cache?: CompressionCache | null;
}

export interface CompressResult {
  messages: MemoSiftMessage[];
  report: CompressionReport;
}

export async function compress(
  messages: MemoSiftMessage[],
  options?: CompressOptions,
): Promise<CompressResult> {
  const config = options?.config ? createConfig(options.config) : createConfig();
  const llm = options?.llm ?? null;
  const task = options?.task ?? null;
  const ledger = options?.ledger ?? null;
  const crossWindow = options?.crossWindow ?? null;
  const report = new CompressionReport();
  const counter: MemoSiftLLMProvider = llm ?? new HeuristicTokenCounter();

  // ── Three-Zone Partitioning ──
  const [zone1, zone2, zone3] = partitionZones(messages);

  let originalTokens = 0;
  for (const m of messages) originalTokens += await counter.countTokens(m.content);

  if (zone3.length === 0) {
    report.finalize(originalTokens, originalTokens, config.costPer1kTokens);
    return { messages: [...messages], report };
  }

  // ── Layer 1: Classify (Zone 3 only) ──
  let segments = await runLayer(
    "classifier",
    async () => classifyMessages(zone3, config),
    [],
    report,
  );

  if (!segments) {
    segments = zone3.map((m, i) =>
      createClassified(
        m,
        m.role === "system" ? ContentType.SYSTEM_PROMPT : ContentType.OLD_CONVERSATION,
        m.role === "system" ? CompressionPolicy.PRESERVE : CompressionPolicy.MODERATE,
        { originalIndex: i },
      ),
    );
  }

  // ── Anchor Extraction (before compression, after classification) ──
  if (ledger && config.enableAnchorLedger) {
    extractAnchorsFromSegments(segments, ledger);
  }

  // ── Reasoning Chain Tracking (after anchor extraction) ──
  const deps: DependencyMap = createDependencyMap();
  extractReasoningChains(segments, deps);

  for (const seg of segments) {
    const key = seg.contentType;
    report.segmentCounts[key] = (report.segmentCounts[key] ?? 0) + 1;
  }

  // ── Tier resolution ──
  const tier = resolveTier(config, segments.length);
  report.performanceTier = tier;

  // ── Pre-bucketing: route bypass-eligible segments past compression ──
  let bypassSegments: ClassifiedMessage[] = [];
  if (config.preBucketBypass) {
    [bypassSegments, segments] = preBucket(segments);
  }

  // ── Layer 2: Deduplicate ──
  // deps already initialized above with reasoning chain edges.
  const dedupExactOnly = tier === "ultra_fast";
  const dedupResult = await runLayer(
    "deduplicator",
    async () => deduplicate(segments!, config, crossWindow, dedupExactOnly),
    segments,
    report,
  );
  if (dedupResult) {
    segments = dedupResult.segments;
    // Merge dedup deps into our deps (which already has logical edges).
    for (const [k, v] of dedupResult.deps.references) {
      deps.references.set(k, v);
    }
  }

  // ── Layer 2.5: Coalesce short messages ──
  if (config.coalesceShortMessages && tier !== "ultra_fast") {
    segments =
      (await runLayer(
        "coalescer",
        async () => coalesceShortMessages(segments!, config),
        segments,
        report,
      )) ?? segments;
  }

  // ── Layer 3: Engines ──
  // Track content hashes for first-read vs re-read detection (Item 2.2).
  const seenContentHashes = new Map<string, number>();
  segments =
    (await runLayer(
      "engine_verbatim",
      async () => verbatimCompress(segments!, config, ledger, seenContentHashes),
      segments,
      report,
    )) ?? segments;

  if (tier !== "ultra_fast") {
    segments =
      (await runLayer(
        "engine_pruner",
        async () => pruneTokens(segments!, config, ledger),
        segments,
        report,
      )) ?? segments;
  }

  segments =
    (await runLayer(
      "engine_structural",
      async () => structuralCompress(segments!, config, ledger),
      segments,
      report,
    )) ?? segments;

  // ── Conversation Phase Detection ──
  const phase = detectPhase(segments);
  const phaseMult = PHASE_KEEP_MULTIPLIERS.get(phase) ?? 1.0;

  // ── Layer 3G: Importance Scoring (scoring only, no deletion) ──
  if (!["fast", "ultra_fast"].includes(tier)) {
    segments =
      (await runLayer(
        "importance_scorer",
        async () => scoreImportance(segments!, config, ledger, phaseMult),
        segments,
        report,
      )) ?? segments;
  }

  // ── Layer 3E: Query-Relevance Pruning (uses shields from L3G) ──
  if (!["fast", "ultra_fast"].includes(tier)) {
    segments =
      (await runLayer(
        "relevance_pruner",
        async () => queryRelevancePrune(segments!, config, deps, ledger),
        segments,
        report,
      )) ?? segments;
  }

  // ── Layer 3F: Elaboration Compression (uses shields from L3G) ──
  if (tier !== "ultra_fast") {
    segments =
      (await runLayer(
        "discourse_compressor",
        async () => elaborateCompress(segments!, config, ledger),
        segments,
        report,
      )) ?? segments;
  }

  // Engine D: Summarization (LLM-dependent, opt-in).
  if (config.enableSummarization && llm) {
    segments =
      (await runLayer(
        "engine_summarizer",
        async () => summarizeSegments(segments!, config, llm),
        segments,
        report,
      )) ?? segments;
  }

  // ── Merge bypass segments back (budget and scorer need the full set) ──
  // Merge by originalIndex to maintain message ordering.
  if (bypassSegments.length > 0) {
    segments = [...bypassSegments, ...segments].sort((a, b) => a.originalIndex - b.originalIndex);
  }

  // ── Layer 4: Score relevance ──
  if (config.llmRelevanceScoring && llm && task) {
    segments =
      (await runLayer(
        "scorer_llm",
        async () => scoreRelevanceLlm(segments!, config, task, llm),
        segments,
        report,
      )) ?? segments;
  } else {
    segments =
      (await runLayer(
        "scorer",
        async () => scoreRelevance(segments!, config, task, ledger),
        segments,
        report,
      )) ?? segments;
  }

  // ── Layer 5: Position optimization ──
  segments =
    (await runLayer(
      "positioner",
      async () => optimizePosition(segments!, config),
      segments,
      report,
    )) ?? segments;

  // ── Layer 6: Budget enforcement ──
  let budgetConfig = config;
  if (config.tokenBudget !== null) {
    let z1Tokens = 0;
    for (const m of zone1) z1Tokens += await counter.countTokens(m.content);
    let z2Tokens = 0;
    for (const m of zone2) z2Tokens += await counter.countTokens(m.content);
    const effectiveBudget = Math.max(100, config.tokenBudget - z1Tokens - z2Tokens);
    budgetConfig = createConfig({ ...config, tokenBudget: effectiveBudget });
  }
  segments =
    (await runLayer(
      "budget",
      async () => enforceBudget(segments!, budgetConfig, deps, counter, ledger),
      segments,
      report,
    )) ?? segments;

  // ── Enforce tool call integrity ──
  segments = enforceToolCallIntegrity(segments, report);

  // ── Tag and reassemble ──
  const compressedZone3 = toMessages(segments);
  for (const msg of compressedZone3) msg._memosiftCompressed = true;

  const compressed = reassembleZones(zone1, zone2, compressedZone3);

  let compressedTokens = 0;
  for (const m of compressed) compressedTokens += await counter.countTokens(m.content);

  report.finalize(originalTokens, compressedTokens, config.costPer1kTokens);
  return { messages: compressed, report };
}

// ── Three-Zone helpers ──────────────────────────────────────────────────────

export function partitionZones(
  messages: MemoSiftMessage[],
): [MemoSiftMessage[], MemoSiftMessage[], MemoSiftMessage[]] {
  const zone1: MemoSiftMessage[] = [];
  const zone2: MemoSiftMessage[] = [];
  const zone3: MemoSiftMessage[] = [];
  for (const msg of messages) {
    if (msg.role === "system") zone1.push(msg);
    else if (msg._memosiftCompressed) zone2.push(msg);
    else zone3.push(msg);
  }
  return [zone1, zone2, zone3];
}

function reassembleZones(
  zone1: MemoSiftMessage[],
  zone2: MemoSiftMessage[],
  zone3: MemoSiftMessage[],
): MemoSiftMessage[] {
  return [...zone1, ...zone2, ...zone3];
}

// ── Layer execution ─────────────────────────────────────────────────────────

async function runLayer<T>(
  name: string,
  fn: () => Promise<T>,
  segments: ClassifiedMessage[],
  report: CompressionReport,
): Promise<T | null> {
  const start = performance.now();
  try {
    const result = await fn();
    const elapsed = performance.now() - start;
    const inTokens = segments.reduce((s, seg) => s + seg.estimatedTokens, 0);
    const resultSegs = Array.isArray(result)
      ? (result as ClassifiedMessage[])
      : result && typeof result === "object" && "segments" in result
        ? (result as { segments: ClassifiedMessage[] }).segments
        : segments;
    const outTokens = resultSegs.reduce(
      (s: number, seg: ClassifiedMessage) => s + seg.estimatedTokens,
      0,
    );
    report.addLayer(name, inTokens, outTokens, elapsed);
    return result;
  } catch (e) {
    const elapsed = performance.now() - start;
    report.addLayerFailure(name, String(e), elapsed);
    return null;
  }
}

function enforceToolCallIntegrity(
  segments: ClassifiedMessage[],
  report: CompressionReport,
): ClassifiedMessage[] {
  const callIds = new Set<string>();
  const resultIds = new Set<string>();
  for (const seg of segments) {
    if (seg.message.toolCalls) for (const tc of seg.message.toolCalls) callIds.add(tc.id);
    if (seg.message.toolCallId) resultIds.add(seg.message.toolCallId);
  }
  const orphanedCalls = new Set([...callIds].filter((id) => !resultIds.has(id)));
  const orphanedResults = new Set([...resultIds].filter((id) => !callIds.has(id)));

  if (orphanedCalls.size === 0 && orphanedResults.size === 0) return segments;

  return segments.filter((seg) => {
    if (seg.message.toolCallId && orphanedResults.has(seg.message.toolCallId)) {
      report.addDecision(
        "pipeline",
        "dropped",
        seg.originalIndex,
        seg.estimatedTokens,
        0,
        `Orphaned tool result: ${seg.message.toolCallId}`,
      );
      return false;
    }
    if (seg.message.toolCalls) {
      const remaining = seg.message.toolCalls.filter((tc) => !orphanedCalls.has(tc.id));
      if (remaining.length !== seg.message.toolCalls.length) {
        seg.message.toolCalls = remaining.length > 0 ? remaining : null;
      }
    }
    return true;
  });
}

function toMessages(segments: ClassifiedMessage[]): MemoSiftMessage[] {
  return segments.map((seg) => {
    seg.message._memosiftContentType = seg.contentType;
    return seg.message;
  });
}
