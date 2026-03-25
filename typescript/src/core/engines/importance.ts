// Layer 3G: Multi-signal importance scoring (BudgetMem-inspired, 8 signals).

import type { MemoSiftConfig } from "../../config.js";
import {
  type AnchorLedger,
  type ClassifiedMessage,
  CompressionPolicy,
  Shield,
  createClassified,
} from "../types.js";

// Policies that skip importance scoring (already protected).
const SKIP_POLICIES: ReadonlySet<CompressionPolicy> = new Set([
  CompressionPolicy.PRESERVE,
  CompressionPolicy.LIGHT,
]);

// ── Signal detection patterns ──────────────────────────────────────────────

// Entity patterns: file paths, URLs, identifiers, function/class names.
const ENTITY_PATTERNS: readonly RegExp[] = [
  /(?:[A-Za-z]:)?[\w.\-]+(?:[/\\][\w.\-]+)+(?:\.\w+)?/g, // File paths
  /https?:\/\/\S+/g, // URLs
  /\b[a-z]+(?:[A-Z][a-z]+)+\b/g, // camelCase
  /\b[a-z]+(?:_[a-z]+)+\b/g, // snake_case
  /\bclass\s+\w+/g, // class names
  /\bdef\s+\w+/g, // function names
  /\bfunction\s+\w+/g, // JS function names
  /\b[A-Z][A-Z_]{2,}\b/g, // UPPER_CASE constants
];

// Numerical patterns: line numbers, ports, counts, versions, error codes.
const NUMERICAL_PATTERNS: readonly RegExp[] = [
  /\bline\s+\d+\b/gi, // line references
  /\bport\s+\d+\b/gi, // port references
  /\b\d+\s*(?:items?|rows?|records?|files?|bytes?|KB|MB|GB)\b/gi, // counts with units
  /\bv?\d+\.\d+(?:\.\d+)?\b/g, // Version numbers
  /\b(?:0x[0-9a-f]+|E\d{4,})\b/gi, // Error codes
  /\b\d{3,}\b/g, // 3+ digit numbers (ports, line numbers)
];

// Domain metric patterns: numbers followed by units with slashes or camelCase units.
const DOMAIN_METRIC_RE = /\b\d[\d,]*(?:\.\d+)?\s+(?:[A-Za-z]+\/[A-Za-z]+|[A-Z][a-z]*[A-Z])/gi;

// Combined patterns for single-pass matching (2 regex calls instead of 14).
const _ENTITY_COMBINED = new RegExp(ENTITY_PATTERNS.map((p) => p.source).join("|"), "g");
const _NUMERICAL_COMBINED = new RegExp(NUMERICAL_PATTERNS.map((p) => p.source).join("|"), "gi");

// Discourse markers: questions, conclusions, decisions.
const QUESTION_PATTERN = /\?\s*$/m;
const CONCLUSION_MARKERS =
  /\b(?:therefore|in conclusion|the result is|this means|in summary|to summarize|the key takeaway|ultimately|finally)\b/i;
const DECISION_MARKERS =
  /\b(?:decided to|chose|let's go with|I'll use|we'll use|the decision is|going with|selecting)\b/i;

// Instruction detection (graduated).
const ABSOLUTE_INSTRUCTIONS =
  /\b(?:must|never|always|required|mandatory|critical|essential|do not|don't ever|absolutely)\b/i;
const IMPERATIVE_INSTRUCTIONS =
  /\b(?:use|run|install|create|add|remove|delete|update|set|configure|ensure|make sure|implement|deploy|fix|change)\b/i;
const CONDITIONAL_INSTRUCTIONS =
  /\b(?:if\b.{1,40}\bthen|only when|unless|when\b.{1,40}\bshould|in case|provided that)\b/i;
const HEDGED_INSTRUCTIONS = /\b(?:maybe|consider|could|might|perhaps|possibly|optionally)\b/i;

// Stop words for TF-IDF.
const STOP_WORDS: ReadonlySet<string> = new Set([
  "a",
  "an",
  "the",
  "and",
  "or",
  "but",
  "in",
  "on",
  "at",
  "to",
  "for",
  "of",
  "with",
  "by",
  "from",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "have",
  "has",
  "had",
  "do",
  "does",
  "did",
  "will",
  "would",
  "could",
  "should",
  "this",
  "that",
  "it",
  "its",
  "i",
  "me",
  "my",
  "we",
  "you",
  "he",
  "she",
  "they",
  "not",
  "no",
  "so",
  "if",
  "then",
  "just",
  "also",
]);

/** Count all non-overlapping matches of a global regex in text. */
function countMatches(pattern: RegExp, text: string): number {
  // Reset lastIndex to ensure clean matching for global regexes.
  pattern.lastIndex = 0;
  let count = 0;
  while (pattern.exec(text) !== null) {
    count++;
  }
  return count;
}

/**
 * Compute graduated instruction strength (0.0-1.0).
 *
 * Higher for absolute constraints, lower for hedged suggestions.
 * User instructions weighted higher than assistant suggestions.
 */
export function computeInstructionStrength(text: string, role: string): number {
  let strength = 0.0;
  if (ABSOLUTE_INSTRUCTIONS.test(text)) {
    strength = 1.0;
  } else if (IMPERATIVE_INSTRUCTIONS.test(text)) {
    strength = 0.7;
  } else if (CONDITIONAL_INSTRUCTIONS.test(text)) {
    strength = 0.5;
  } else if (HEDGED_INSTRUCTIONS.test(text)) {
    strength = 0.2;
  }

  // Speaker weighting: user instructions > assistant suggestions.
  if (role === "user") {
    strength *= 1.0;
  } else if (role === "assistant") {
    strength *= 0.6;
  } else {
    strength *= 0.8;
  }

  return strength;
}

/** Compute IDF scores across the segment corpus. */
function computeCorpusIdf(segments: readonly ClassifiedMessage[]): Map<string, number> {
  const nDocs = segments.length;
  const docFreq = new Map<string, number>();

  for (const seg of segments) {
    const text = (seg.message.content ?? "").toLowerCase();
    const tokensInDoc = new Set<string>();
    for (const match of text.matchAll(/\b\w+\b/g)) {
      const token = match[0];
      if (!STOP_WORDS.has(token)) {
        tokensInDoc.add(token);
      }
    }
    for (const token of tokensInDoc) {
      docFreq.set(token, (docFreq.get(token) ?? 0) + 1);
    }
  }

  const idf = new Map<string, number>();
  for (const [token, freq] of docFreq) {
    idf.set(token, Math.log((nDocs + 1) / (freq + 1)) + 1);
  }
  return idf;
}

/** Compute mean TF-IDF score for tokens in text. */
function meanTfidfScore(text: string, corpusIdf: ReadonlyMap<string, number>): number {
  const allTokens: string[] = [];
  for (const match of text.toLowerCase().matchAll(/\b\w+\b/g)) {
    const token = match[0];
    if (!STOP_WORDS.has(token)) {
      allTokens.push(token);
    }
  }

  if (allTokens.length === 0) return 0.0;

  const tf = new Map<string, number>();
  for (const token of allTokens) {
    tf.set(token, (tf.get(token) ?? 0) + 1);
  }
  const total = allTokens.length;

  const uniqueTokens = new Set(allTokens);
  let scoreSum = 0.0;
  for (const token of uniqueTokens) {
    const tokenTf = (tf.get(token) ?? 0) / total;
    const tokenIdf = corpusIdf.get(token) ?? 1.0;
    scoreSum += tokenTf * tokenIdf;
  }

  return Math.min(scoreSum / Math.max(uniqueTokens.size, 1), 1.0);
}

/**
 * Compute what fraction of the segment's entities appear in the anchor ledger.
 *
 * Returns 0.0 if no ledger or no entities; 1.0 if all entities are covered.
 */
function computeAnchorCoverage(text: string, ledger: AnchorLedger | null): number {
  if (!ledger || !text) return 0.0;

  const protectedStrings = ledger.getProtectedStrings();
  if (protectedStrings.size === 0) return 0.0;

  // Extract entities from text.
  const entities = new Set<string>();

  // File paths.
  const filePathPattern = /(?:[A-Za-z]:)?[\w.\-]+(?:[/\\][\w.\-]+)+/g;
  for (const match of text.matchAll(filePathPattern)) {
    entities.add(match[0]);
  }

  // Code identifiers (camelCase).
  const camelCasePattern = /\b[a-z]+(?:[A-Z][a-z]+)+\b/g;
  for (const match of text.matchAll(camelCasePattern)) {
    entities.add(match[0]);
  }

  // Code identifiers (snake_case).
  const snakeCasePattern = /\b[a-z]+(?:_[a-z]+)+\b/g;
  for (const match of text.matchAll(snakeCasePattern)) {
    entities.add(match[0]);
  }

  if (entities.size === 0) return 0.0;

  let covered = 0;
  for (const entity of entities) {
    for (const p of protectedStrings) {
      if (p.includes(entity) || entity.includes(p)) {
        covered++;
        break;
      }
    }
  }
  return covered / entities.size;
}

/**
 * Score each segment's importance using 8 signals and assign shield levels.
 *
 * Signals (aligned with BudgetMem's validated feature set + instruction detection):
 * 1.  Entity density (file paths, IDs, code names per token) — weight 0.15
 * 2a. Generic numerical density (line numbers, ports, counts per token) — weight 0.05
 * 2b. Domain metric density (numbers with units like req/s, MiB) — weight 0.10
 * 3.  Discourse markers (questions, conclusions, decisions) — weight 0.15
 * 4.  Instruction detection (graduated: absolute > imperative > conditional > hedged) — weight 0.15
 * 5.  Position weight (recency bias -- closer to end = more important) — weight 0.15
 * 6.  TF-IDF importance (mean TF-IDF score of segment tokens) — weight 0.10
 * 7.  Anchor fact coverage (fraction of segment entities in anchor ledger) — weight 0.10
 *
 * Total weight: ~0.95 (with anchor coverage contributing to fidelity preservation).
 *
 * Shield assignment:
 * - importance > 0.7 -> PRESERVE
 * - importance > 0.3 -> MODERATE
 * - else -> COMPRESSIBLE
 *
 * @param segments - Classified messages from previous layers.
 * @param config - Pipeline configuration.
 * @param ledger - Optional anchor ledger for anchor fact coverage signal.
 * @returns Segments with importanceScore and shield populated.
 */
export function scoreImportance(
  segments: readonly ClassifiedMessage[],
  config: MemoSiftConfig,
  ledger?: AnchorLedger | null,
  phaseMultiplier = 1.0,
): ClassifiedMessage[] {
  if (segments.length === 0) return [];

  const totalSegments = segments.length;

  // Pre-compute TF-IDF scores only for scorable segments (skip PRESERVE/LIGHT).
  const scorableForIdf = segments.filter((seg) => !SKIP_POLICIES.has(seg.policy) && !seg.protected);
  const corpusIdf =
    scorableForIdf.length > 0 ? computeCorpusIdf(scorableForIdf) : new Map<string, number>();

  const result: ClassifiedMessage[] = [];

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i]!;

    if (SKIP_POLICIES.has(seg.policy) || seg.protected) {
      result.push(
        createClassified(seg.message, seg.contentType, seg.policy, {
          originalIndex: seg.originalIndex,
          relevanceScore: seg.relevanceScore,
          estimatedTokens: seg.estimatedTokens,
          protected: seg.protected,
          importanceScore: 1.0,
          shield: Shield.PRESERVE,
        }),
      );
      continue;
    }

    const text = seg.message.content ?? "";
    const tokenCount = Math.max(text.split(/\s+/).length, 1);

    // Signal 1: Entity density (weight: 0.15) — single combined regex.
    const entityCount = countMatches(_ENTITY_COMBINED, text);
    const entityDensity = Math.min(entityCount / tokenCount, 1.0);

    // Signal 2a: Generic numerical density (weight: 0.05) — single combined regex.
    const numCount = countMatches(_NUMERICAL_COMBINED, text);
    const genericNumericalDensity = Math.min(numCount / tokenCount, 1.0);

    // Signal 2b: Domain metric density (weight: 0.10) — numbers with units.
    const domainMetricCount = countMatches(DOMAIN_METRIC_RE, text);
    const domainMetricDensity = Math.min(domainMetricCount / tokenCount, 1.0);

    // Signal 3: Discourse markers (weight: 0.15)
    const hasQuestion = QUESTION_PATTERN.test(text);
    const hasConclusion = CONCLUSION_MARKERS.test(text);
    const hasDecision = DECISION_MARKERS.test(text);
    const discourseScore = hasQuestion || hasConclusion || hasDecision ? 1.0 : 0.0;

    // Signal 4: Instruction detection -- graduated (weight: 0.15)
    const instructionStrength = computeInstructionStrength(text, seg.message.role);

    // Signal 5: Position weight (weight: 0.15)
    const distanceFromEnd = totalSegments - 1 - i;
    const positionWeight = 1.0 / (1.0 + distanceFromEnd * 0.1);

    // Signal 6: TF-IDF importance (weight: 0.10)
    const tfidfImportance = meanTfidfScore(text, corpusIdf);

    // Signal 7: Anchor fact coverage (weight: 0.10)
    const anchorCoverage = computeAnchorCoverage(text, ledger ?? null);

    // Combined importance (BudgetMem-aligned weights + instruction + anchor coverage).
    let importance =
      entityDensity * 0.15 +
      genericNumericalDensity * 0.05 +
      domainMetricDensity * 0.1 +
      discourseScore * 0.15 +
      instructionStrength * 0.15 +
      positionWeight * 0.15 +
      tfidfImportance * 0.1 +
      anchorCoverage * 0.1;

    // Absolute override: hard constraints always PRESERVE.
    if (instructionStrength >= 0.7) {
      importance = Math.max(importance, 0.75);
    }

    // Apply phase multiplier (conversation phase detection).
    importance *= phaseMultiplier;

    // Assign shield.
    let shield: Shield;
    if (importance > 0.7) {
      shield = Shield.PRESERVE;
    } else if (importance > 0.3) {
      shield = Shield.MODERATE;
    } else {
      shield = Shield.COMPRESSIBLE;
    }

    result.push(
      createClassified(seg.message, seg.contentType, seg.policy, {
        originalIndex: seg.originalIndex,
        relevanceScore: seg.relevanceScore,
        estimatedTokens: seg.estimatedTokens,
        protected: seg.protected,
        importanceScore: importance,
        shield,
      }),
    );
  }

  return result;
}
