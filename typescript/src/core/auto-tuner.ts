// Level 2 Content Detection — auto-tunes config based on incoming message analysis.
//
// Replaces static presets with data-driven configuration. Scans message content
// to detect conversation type (code-heavy, data-heavy, tool-heavy, etc.) and
// sets optimal compression parameters accordingly.
//
// Design principles:
// - Runs ONCE per session (or per explicit retune), NOT per compress() call
// - Layer 0 (pressure-based) still runs per-call ON TOP of auto-tuned config
// - Explicit user overrides are never touched ("parameter locking")
// - All decisions logged to CompressionReport for transparency

import { type MemoSiftConfig, createConfig } from "../config.js";
import { CompressionPolicy, ContentType, type MemoSiftMessage } from "./types.js";

// ── Content signal patterns ──────────────────────────────────────────────

const CODE_RE =
  /\b(?:def |class |function |import |from |const |let |var |async |await |return |if \(|else \{|for \(|while \(|switch |try \{|catch )\b/;
const ERROR_RE =
  /\b(?:Error|Exception|Traceback|FAIL|TypeError|KeyError|ValueError|SyntaxError|ReferenceError|AttributeError|RuntimeError)\b/;
const JSON_START_RE = /^\s*[[\{]/m;
const FILE_PATH_RE =
  /(?:[a-zA-Z]:)?(?:[./\\])?(?:[\w.\-]+[/\\])+[\w.\-]+\.\w{1,10}/;
const NUMERIC_RE = /\b\d[\d,]*(?:\.\d+)?\b/g;
const TABLE_RE = /\|[^|]+\|[^|]+\|/;
const UNIT_RATIO_RE = /\b\d[\d,.]*\s+[A-Za-z]+\/[A-Za-z]+\b/;

/** Domain hint patterns — detected from content to auto-add metric_patterns. */
const DOMAIN_HINTS: Record<string, readonly string[]> = {
  energy: [
    "Mcf/d", "bbl/d", "STB/d", "psig", "psia", "bbl", "STB",
    "Mcf", "MMcf", "BOE", "BOPD", "Scf/STB", "STB/MMcf",
    "GOR", "WOR", "GLR", "WGR", "EUR", "API",
  ],
  financial: ["bps", "AUM", "NAV", "EPS", "EBITDA", "YoY", "QoQ"],
  tech: ["QPS", "RPS", "p99", "p95", "p50", "SLA"],
  medical: ["mg/dL", "mmHg", "BPM", "IU/L", "mEq/L"],
};

// ── Content profile ─────────────────────────────────────────────────────

/** Quantified profile of conversation content for auto-tuning decisions. */
export interface ContentProfile {
  readonly totalMessages: number;
  readonly totalChars: number;
  readonly userTurns: number;
  readonly toolResultCount: number;
  readonly toolCallCount: number;
  /** Fraction of messages containing code patterns. */
  readonly codeDensity: number;
  /** Fraction of messages containing error patterns. */
  readonly errorDensity: number;
  /** Fraction of messages containing JSON structures. */
  readonly jsonDensity: number;
  /** Fraction of messages with 3+ numeric values. */
  readonly numericDensity: number;
  /** Fraction of messages containing file paths. */
  readonly filePathDensity: number;
  /** Fraction of messages containing markdown tables. */
  readonly tableDensity: number;
  /** Fraction of messages with X/Y unit ratios. */
  readonly unitRatioDensity: number;
  /** Average message length in chars. */
  readonly avgMessageLength: number;
  /** Estimated duplicate content ratio. */
  readonly duplicateRatio: number;
  /** Auto-detected domain hints. */
  readonly detectedDomains: readonly string[];
}

/** Result of auto-tuning with full transparency. */
export interface AutoTuneResult {
  readonly profile: ContentProfile;
  /** param_name -> chosen value */
  readonly tunedParams: Record<string, unknown>;
  /** param_name -> human-readable reason */
  readonly reasons: Record<string, string>;
  /** Params the caller explicitly set (not touched). */
  readonly lockedParams: ReadonlySet<string>;
  /** "code", "data", "mixed", "conversation", etc. */
  readonly detectedStyle: string;
}

/**
 * Analyze message content to build a quantified profile.
 *
 * Single-pass analysis, fast (~1ms for 500 messages). Samples first 2000
 * chars of each message for pattern matching.
 */
export function profileMessages(messages: readonly MemoSiftMessage[]): ContentProfile {
  const total = messages.length;
  if (total === 0) {
    return {
      totalMessages: 0,
      totalChars: 0,
      userTurns: 0,
      toolResultCount: 0,
      toolCallCount: 0,
      codeDensity: 0,
      errorDensity: 0,
      jsonDensity: 0,
      numericDensity: 0,
      filePathDensity: 0,
      tableDensity: 0,
      unitRatioDensity: 0,
      avgMessageLength: 0,
      duplicateRatio: 0,
      detectedDomains: [],
    };
  }

  let userTurns = 0;
  let toolResults = 0;
  let toolCalls = 0;
  let codeCount = 0;
  let errorCount = 0;
  let jsonCount = 0;
  let numericCount = 0;
  let filePathCount = 0;
  let tableCount = 0;
  let unitRatioCount = 0;
  let totalChars = 0;
  const seenHashes = new Set<number>();
  let duplicates = 0;
  let emptyCount = 0;

  for (const msg of messages) {
    const content = msg.content ?? "";
    totalChars += content.length;

    if (msg.role === "user") userTurns++;
    if (msg.role === "tool") toolResults++;
    if (msg.toolCalls) toolCalls += msg.toolCalls.length;

    if (!content) {
      emptyCount++;
      continue;
    }

    // Quick duplicate detection via hash of first 200 chars.
    const h = simpleHash(content.slice(0, 200));
    if (seenHashes.has(h)) {
      duplicates++;
    } else {
      seenHashes.add(h);
    }

    // Pattern detection on first 2000 chars for speed.
    const sample = content.slice(0, 2000);
    if (CODE_RE.test(sample)) codeCount++;
    if (ERROR_RE.test(sample)) errorCount++;
    if (JSON_START_RE.test(sample)) jsonCount++;
    // Count numeric matches — need 3+ to flag.
    const numericMatches = sample.match(NUMERIC_RE);
    if (numericMatches && numericMatches.length >= 3) numericCount++;
    if (FILE_PATH_RE.test(sample)) filePathCount++;
    if (TABLE_RE.test(sample)) tableCount++;
    if (UNIT_RATIO_RE.test(sample)) unitRatioCount++;
  }

  const nonEmpty = Math.max(total - emptyCount, 1);

  // Detect domains from content of first 30 messages.
  const detectedDomains: string[] = [];
  const sampleText = messages
    .slice(0, 30)
    .map((m) => (m.content ?? "").slice(0, 500))
    .join(" ")
    .toLowerCase();
  for (const [domain, patterns] of Object.entries(DOMAIN_HINTS)) {
    if (patterns.some((p) => sampleText.includes(p.toLowerCase()))) {
      detectedDomains.push(domain);
    }
  }

  return {
    totalMessages: total,
    totalChars,
    userTurns,
    toolResultCount: toolResults,
    toolCallCount: toolCalls,
    codeDensity: codeCount / nonEmpty,
    errorDensity: errorCount / nonEmpty,
    jsonDensity: jsonCount / nonEmpty,
    numericDensity: numericCount / nonEmpty,
    filePathDensity: filePathCount / nonEmpty,
    tableDensity: tableCount / nonEmpty,
    unitRatioDensity: unitRatioCount / nonEmpty,
    avgMessageLength: Math.floor(totalChars / Math.max(total, 1)),
    duplicateRatio: duplicates / Math.max(total, 1),
    detectedDomains,
  };
}

// ── Auto-tuner ───────────────────────────────────────────────────────────

/**
 * Analyze messages and adapt config parameters based on content.
 *
 * Parameter locking: any param name in `lockedParams` is never changed.
 * Explicit user overrides (values that differ from MemoSiftConfig defaults)
 * are also treated as locked — the auto-tuner respects intentional choices.
 *
 * @param config - The base config (may have user overrides).
 * @param messages - The incoming messages to analyze.
 * @param lockedParams - Parameter names the caller explicitly wants preserved.
 * @returns Tuple of [tuned_config, result_with_transparency].
 */
export function autoTune(
  config: MemoSiftConfig,
  messages: readonly MemoSiftMessage[],
  lockedParams: ReadonlySet<string> = new Set(),
): [MemoSiftConfig, AutoTuneResult] {
  const profile = profileMessages(messages);
  if (profile.totalMessages === 0) {
    return [
      config,
      {
        profile,
        tunedParams: {},
        reasons: {},
        lockedParams,
        detectedStyle: "empty",
      },
    ];
  }

  // Detect which params the user explicitly set (differ from defaults).
  const defaults = getDefaults();
  const userLocked = new Set(lockedParams);
  for (const key of TUNABLE_PARAMS) {
    if (key === "policies" || key === "metricPatterns" || key === "contextWindow") continue;
    const current = config[key as keyof MemoSiftConfig];
    const defaultVal = defaults[key];
    if (current !== defaultVal && !userLocked.has(key)) {
      userLocked.add(key);
    }
  }

  const tuned: Record<string, unknown> = {};
  const reasons: Record<string, string> = {};

  function set(name: string, value: unknown, reason: string): void {
    if (!userLocked.has(name)) {
      tuned[name] = value;
      reasons[name] = reason;
    }
  }

  // ── Detect conversation style ──
  const isCode = profile.codeDensity > 0.25;
  const isData = profile.numericDensity > 0.35 || profile.tableDensity > 0.15;
  const isErrorHeavy = profile.errorDensity > 0.15;
  const isToolHeavy = profile.toolResultCount > 10;
  const isJsonHeavy = profile.jsonDensity > 0.4;
  const isDuplicateHeavy = profile.duplicateRatio > 0.08;
  const isLong = profile.totalMessages > 100;
  const isShort = profile.totalMessages < 30;
  const hasUnitRatios = profile.unitRatioDensity > 0.05;

  let style: string;
  if (isCode && isErrorHeavy) {
    style = "code_debug";
  } else if (isCode) {
    style = "code";
  } else if (isData) {
    style = "data";
  } else if (isToolHeavy && isJsonHeavy) {
    style = "tool_heavy";
  } else {
    style = "mixed";
  }

  // ── recentTurns ──
  if (isShort) {
    set("recentTurns", 2, `short conversation (${profile.userTurns} turns)`);
  } else if (isLong) {
    set("recentTurns", 2, `long conversation (${profile.userTurns} turns), ledger as safety net`);
  } else {
    set("recentTurns", 2, "balanced default");
  }

  // ── entropyThreshold ──
  if (isCode) {
    set(
      "entropyThreshold",
      2.3,
      `code-heavy (${pct(profile.codeDensity)} code density), preserve structure`,
    );
  } else if (isData || hasUnitRatios) {
    set(
      "entropyThreshold",
      1.9,
      `data-heavy (${pct(profile.numericDensity)} numeric), remove boilerplate`,
    );
  } else if (isJsonHeavy) {
    set("entropyThreshold", 2.0, `JSON-heavy (${pct(profile.jsonDensity)}), moderate filtering`);
  } else {
    set("entropyThreshold", 2.1, "balanced default");
  }

  // ── tokenPruneKeepRatio ──
  if (profile.filePathDensity > 0.3 && isErrorHeavy) {
    set(
      "tokenPruneKeepRatio",
      0.6,
      `file-path + error heavy (${pct(profile.filePathDensity)} paths, ` +
        `${pct(profile.errorDensity)} errors), preserve identifiers`,
    );
  } else if (isData || isJsonHeavy) {
    set("tokenPruneKeepRatio", 0.5, "data/JSON-heavy, numbers protected by anchor extractor");
  } else {
    set("tokenPruneKeepRatio", 0.55, "balanced default");
  }

  // ── dedupSimilarityThreshold ──
  if (isDuplicateHeavy) {
    set(
      "dedupSimilarityThreshold",
      0.8,
      `high duplicate ratio (${pct(profile.duplicateRatio)}), aggressive dedup`,
    );
  } else if (isToolHeavy) {
    set(
      "dedupSimilarityThreshold",
      0.82,
      `tool-heavy (${profile.toolResultCount} results), moderate dedup`,
    );
  } else if (isCode) {
    set(
      "dedupSimilarityThreshold",
      0.87,
      "code sessions have natural similarity, moderate threshold",
    );
  } else {
    set("dedupSimilarityThreshold", 0.85, "balanced default");
  }

  // ── relevanceDropThreshold ──
  if (isLong) {
    set(
      "relevanceDropThreshold",
      0.06,
      `long session (${profile.totalMessages} msgs), drop old irrelevant segments`,
    );
  } else if (isToolHeavy) {
    set("relevanceDropThreshold", 0.05, "tool-heavy, moderate relevance filtering");
  } else {
    set("relevanceDropThreshold", 0.04, "short/medium session, preserve more context");
  }

  // ── jsonArrayThreshold ──
  if (isData) {
    set(
      "jsonArrayThreshold",
      8,
      `data-heavy (${pct(profile.numericDensity)} numeric), preserve more array items`,
    );
  } else if (isJsonHeavy) {
    set("jsonArrayThreshold", 6, `JSON-heavy (${pct(profile.jsonDensity)}), moderate truncation`);
  } else {
    set("jsonArrayThreshold", 5, "balanced default");
  }

  // ── codeKeepSignatures ──
  set(
    "codeKeepSignatures",
    isCode || profile.codeDensity > 0.1,
    `code density ${pct(profile.codeDensity)}`,
  );

  // ── enableResolutionCompression ──
  if (profile.userTurns >= 5) {
    set(
      "enableResolutionCompression",
      true,
      `${profile.userTurns} user turns, compress resolved deliberation`,
    );
  }

  // ── enableAnchorLedger ──
  set("enableAnchorLedger", true, "always enabled for safety net");

  // ── ERROR_TRACE policy ──
  const policies: Partial<Record<ContentType, CompressionPolicy>> = { ...config.policies };
  if (
    !(ContentType.ERROR_TRACE in policies) ||
    !userLocked.has(`policy:${ContentType.ERROR_TRACE}`)
  ) {
    if (isErrorHeavy && isLong) {
      policies[ContentType.ERROR_TRACE] = CompressionPolicy.STACK;
      reasons["policy:ERROR_TRACE"] =
        `error-heavy (${pct(profile.errorDensity)}) + long session, ` +
        "anchor ledger preserves error type+message";
    } else if (isCode && !isLong) {
      policies[ContentType.ERROR_TRACE] = CompressionPolicy.PRESERVE;
      reasons["policy:ERROR_TRACE"] = "active code debugging, preserve full traces";
    } else {
      policies[ContentType.ERROR_TRACE] = CompressionPolicy.STACK;
      reasons["policy:ERROR_TRACE"] = "balanced default";
    }
  }

  if (isCode || profile.codeDensity > 0.1) {
    policies[ContentType.CODE_BLOCK] = CompressionPolicy.SIGNATURE;
    reasons["policy:CODE_BLOCK"] =
      `code present (${pct(profile.codeDensity)}), keep signatures`;
  }

  tuned["policies"] = policies;

  // ── metricPatterns ──
  const metricPatterns = [...config.metricPatterns];
  for (const domain of profile.detectedDomains) {
    const hints = DOMAIN_HINTS[domain] ?? [];
    for (const p of hints) {
      if (!metricPatterns.includes(p)) {
        metricPatterns.push(p);
      }
    }
  }
  if (metricPatterns.length !== config.metricPatterns.length) {
    tuned["metricPatterns"] = metricPatterns;
    reasons["metricPatterns"] = `auto-detected domains: ${profile.detectedDomains.join(", ")}`;
  }

  // ── Apply tuned parameters ──
  const replaceKwargs: Partial<MemoSiftConfig> = {};
  for (const [k, v] of Object.entries(tuned)) {
    (replaceKwargs as Record<string, unknown>)[k] = v;
  }
  replaceKwargs.policies = policies;
  if ("metricPatterns" in tuned) {
    replaceKwargs.metricPatterns = tuned["metricPatterns"] as string[];
  }

  const tunedConfig = createConfig({ ...config, ...replaceKwargs });

  return [
    tunedConfig,
    {
      profile,
      tunedParams: tuned,
      reasons,
      lockedParams: new Set(userLocked),
      detectedStyle: style,
    },
  ];
}

// ── Helpers ──────────────────────────────────────────────────────────────

/** Format a fraction as a percentage string (e.g., 0.25 -> "25%"). */
function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/** Simple numeric hash for fast duplicate detection. */
function simpleHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    hash = ((hash << 5) - hash + ch) | 0;
  }
  return hash;
}

/** Tunable parameter names from MemoSiftConfig. */
const TUNABLE_PARAMS: readonly string[] = [
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
  "softCompressionPct",
  "fullCompressionPct",
  "aggressiveCompressionPct",
  "enableAnchorLedger",
  "anchorLedgerMaxTokens",
  "coalesceShortMessages",
  "coalesceCharThreshold",
  "costPer1kTokens",
  "modelName",
  "deterministicSeed",
  "performanceTier",
  "preBucketBypass",
  "enableResolutionCompression",
  "autoTune",
];

/** Get default values from createConfig for detecting user overrides. */
function getDefaults(): Record<string, unknown> {
  const d = createConfig();
  const result: Record<string, unknown> = {};
  for (const key of TUNABLE_PARAMS) {
    result[key] = d[key as keyof MemoSiftConfig];
  }
  return result;
}
