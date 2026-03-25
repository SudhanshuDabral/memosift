// Resolution tracker — audit-only detection of question→decision arcs in conversations.
//
// IMPORTANT: This module is READ-ONLY — it detects patterns and reports them
// but does NOT modify shields, scores, or any compression behavior. It exists
// to gather data on whether semantic detection would improve compression quality,
// without risking regressions from false positives.

import { CompressionPolicy, type ClassifiedMessage } from "./types.js";

// Question detection patterns.
const QUESTION_PATTERNS = [
  /\?\s*$/m,
  /\bshould (?:we|I)\b/i,
  /\bwhich (?:one|approach|option)\b/i,
  /\bhow (?:to|should|do)\b/i,
  /\bwhat (?:about|if)\b/i,
];

// Deliberation patterns.
const DELIBERATION_PATTERNS = [
  /\balternatively\b/i,
  /\bon (?:one|the other) hand\b/i,
  /\bpros and cons\b/i,
  /\boption [A-Z]\b/i,
  /\bcould (?:also |either )?use\b/i,
  /\bvs\.?\b/i,
  /\bcompare\b/i,
  /\btradeoff\b/i,
];

// Resolution patterns (mirrors anchor_extractor decision markers).
const RESOLUTION_PATTERNS = [
  /\bI'll use\b/i,
  /\bLet's go with\b/i,
  /\bchoosing\b.{1,60}\bbecause\b/i,
  /\bdecided to\b/i,
  /\bI'll go with\b/i,
  /\bwe(?:'ll| will) use\b/i,
  /\bgoing with\b/i,
  /\bthe (?:best|right) (?:choice|approach|option) is\b/i,
];

// Supersession patterns.
const SUPERSESSION_PATTERNS = [
  /\bactually\b/i,
  /\bturns out\b/i,
  /\bcorrection\b/i,
  /\bnot .{1,30} but\b/i,
  /\bupdated?\b.*\bnow\b/i,
  /\ball \d+ tests? pass/i,
];

const STOP_WORDS = new Set([
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
  "may",
  "might",
  "can",
  "not",
  "no",
  "if",
  "then",
  "else",
  "this",
  "that",
  "these",
  "those",
  "it",
  "its",
  "we",
  "you",
  "they",
  "use",
  "using",
  "used",
  "going",
  "think",
  "want",
  "need",
  "like",
]);

export interface ResolutionArc {
  readonly questionIndex: number;
  readonly deliberationIndices: readonly number[];
  readonly resolutionIndex: number | null;
  readonly resolved: boolean;
  readonly topicKeywords: ReadonlySet<string>;
}

export interface SupersessionSignal {
  readonly supersededIndex: number;
  readonly supersedingIndex: number;
  readonly reason: string;
  readonly sharedEntities: ReadonlySet<string>;
}

export interface ResolutionReport {
  readonly arcs: readonly ResolutionArc[];
  readonly supersessions: readonly SupersessionSignal[];
}

export interface ResolutionSignals {
  arcsDetected: number;
  arcsResolved: number;
  arcsUnresolved: number;
  supersessionsDetected: number;
  supersessionReasons: Record<string, number>;
}

/** Convert a ResolutionReport to a plain signals object for CompressionReport. */
export function toSignals(report: ResolutionReport): ResolutionSignals {
  const reasons: Record<string, number> = {};
  for (const s of report.supersessions) {
    reasons[s.reason] = (reasons[s.reason] ?? 0) + 1;
  }
  return {
    arcsDetected: report.arcs.length,
    arcsResolved: report.arcs.filter((a) => a.resolved).length,
    arcsUnresolved: report.arcs.filter((a) => !a.resolved).length,
    supersessionsDetected: report.supersessions.length,
    supersessionReasons: reasons,
  };
}

/**
 * Detect question→deliberation→decision arcs in classified messages.
 *
 * This is AUDIT-ONLY — it returns a report but does NOT modify any
 * segment's shield, relevance score, or compression behavior.
 */
export function detectResolutionArcs(segments: ClassifiedMessage[]): ResolutionReport {
  const questions: number[] = [];
  const deliberations: number[] = [];
  const resolutions: number[] = [];

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i]!;
    const content = seg.message.content;
    if (!content || content.length < 10) continue;
    const role = seg.message.role;

    if (role === "user" && matchesAny(content, QUESTION_PATTERNS)) {
      questions.push(i);
    }
    if (role === "assistant") {
      if (matchesAny(content, RESOLUTION_PATTERNS)) {
        resolutions.push(i);
      } else if (matchesAny(content, DELIBERATION_PATTERNS)) {
        deliberations.push(i);
      }
    }
  }

  // Link questions to resolutions via keyword overlap.
  const arcs: ResolutionArc[] = [];
  for (const qIdx of questions) {
    const qKeywords = extractKeywords(segments[qIdx]!.message.content);
    if (qKeywords.size < 2) continue;

    const nextQ = Math.min(...questions.filter((qi) => qi > qIdx), segments.length);
    const arcDelibs = deliberations.filter((d) => d > qIdx && d < nextQ);

    let arcResolution: number | null = null;
    for (const rIdx of resolutions) {
      if (rIdx <= qIdx) continue;
      const rKeywords = extractKeywords(segments[rIdx]!.message.content);
      const overlap = new Set([...qKeywords].filter((k) => rKeywords.has(k)));
      if (overlap.size >= 2) {
        arcResolution = rIdx;
        break;
      }
    }

    const topic = new Set(qKeywords);
    if (arcResolution !== null) {
      for (const k of extractKeywords(segments[arcResolution]!.message.content)) {
        topic.add(k);
      }
    }

    arcs.push({
      questionIndex: qIdx,
      deliberationIndices: arcDelibs,
      resolutionIndex: arcResolution,
      resolved: arcResolution !== null,
      topicKeywords: topic,
    });
  }

  // Detect supersessions.
  const supersessions: SupersessionSignal[] = [];
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i]!;
    if (seg.message.role !== "assistant" || seg.message.content.length < 20) continue;
    if (!matchesAny(seg.message.content, SUPERSESSION_PATTERNS)) continue;

    const iEntities = extractEntities(seg.message.content);
    if (iEntities.size === 0) continue;

    for (let j = Math.max(0, i - 20); j < i; j++) {
      const prev = segments[j]!;
      if (prev.message.role !== "assistant" && prev.message.role !== "tool") continue;
      const jEntities = extractEntities(prev.message.content);
      const shared = new Set([...iEntities].filter((e) => jEntities.has(e)));
      if (shared.size >= 1) {
        supersessions.push({
          supersededIndex: j,
          supersedingIndex: i,
          reason: classifySupersession(seg.message.content),
          sharedEntities: shared,
        });
        break;
      }
    }
  }

  return { arcs, supersessions };
}

function matchesAny(text: string, patterns: RegExp[]): boolean {
  return patterns.some((p) => p.test(text));
}

function extractKeywords(text: string): Set<string> {
  const tokens = new Set(text.toLowerCase().match(/\b[a-zA-Z]\w{2,}\b/g) ?? []);
  for (const sw of STOP_WORDS) tokens.delete(sw);
  return tokens;
}

function extractEntities(text: string): Set<string> {
  const entities = new Set<string>();
  for (const m of text.matchAll(/[\w./\\]+\.\w{1,8}/g)) entities.add(m[0].toLowerCase());
  for (const m of text.matchAll(/\b[a-z]+(?:[A-Z][a-z]+)+\b/g)) entities.add(m[0].toLowerCase());
  for (const m of text.matchAll(/\b(\w+)\s*\(/g)) {
    const name = m[1]!.toLowerCase();
    if (!STOP_WORDS.has(name) && name.length > 2) entities.add(name);
  }
  return entities;
}

function classifySupersession(text: string): string {
  const lower = text.toLowerCase();
  if (["actually", "turns out", "correction", "not", "but"].some((w) => lower.includes(w))) {
    return "correction";
  }
  if (["all", "pass", "now", "updated"].some((w) => lower.includes(w))) {
    return "status_update";
  }
  return "refinement";
}

/** Apply compression policy changes based on detected resolution arcs.
 * For resolved arcs: deliberation messages get AGGRESSIVE policy.
 * For superseded messages: get AGGRESSIVE policy.
 * Opt-in, gated by config.enableResolutionCompression. */
export function applyResolutionCompression(
  segments: ClassifiedMessage[],
  report: ResolutionReport,
): ClassifiedMessage[] {
  const result = [...segments];
  const aggressiveIndices = new Set<number>();

  for (const arc of report.arcs) {
    if (arc.resolved) {
      for (const dIdx of arc.deliberationIndices) {
        if (dIdx < result.length) aggressiveIndices.add(dIdx);
      }
    }
  }
  for (const sup of report.supersessions) {
    if (sup.supersededIndex < result.length) {
      aggressiveIndices.add(sup.supersededIndex);
    }
  }

  for (const idx of aggressiveIndices) {
    const seg = result[idx]!;
    if (seg.policy !== CompressionPolicy.PRESERVE && seg.policy !== CompressionPolicy.LIGHT) {
      result[idx] = { ...seg, policy: CompressionPolicy.AGGRESSIVE };
    }
  }
  return result;
}
