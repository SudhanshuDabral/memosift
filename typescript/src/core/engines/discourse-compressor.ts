// Layer 3F: Elaboration compression — compress satellite clauses, don't delete them.

import type { MemoSiftConfig } from "../../config.js";
import {
  type AnchorLedger,
  type ClassifiedMessage,
  CompressionPolicy,
  Shield,
  createClassified,
  createMessage,
} from "../types.js";

/** Policies that skip elaboration compression (already protected). */
const SKIP_POLICIES = new Set([CompressionPolicy.PRESERVE, CompressionPolicy.LIGHT]);

/** Satellite/elaboration markers — clauses starting with these are compressible. */
const SATELLITE_MARKERS: RegExp[] = [
  /^\s*(?:for example|e\.g\.|such as|in other words)/i,
  /^\s*(?:specifically|to clarify|note that|in particular)/i,
  /^\s*(?:because|since|as a result|as mentioned)/i,
  /^\s*(?:which is why|this means that|namely)/i,
];

/** Parenthetical pattern — content wrapped in parentheses. */
const PARENTHETICAL_RE = /^\s*\(.*\)\s*$/;

/** Numbered list elaboration — "1. First...", "2. Second..." style items. */
const NUMBERED_LIST_RE = /^\s*\d+\.\s+/;

/**
 * Compress elaboration/satellite clauses in eligible segments.
 *
 * Critical design: COMPRESS satellites, don't DELETE them. This is validated
 * by RST literature — satellites in dialogue often contain irreplaceable
 * reasoning rationale. We compress to ~20% of tokens, preserving key concepts.
 *
 * Only applies to:
 * - Non-recent turns (older messages)
 * - COMPRESSIBLE shield level (set by L3G importance scoring)
 * - MODERATE or AGGRESSIVE compression policy
 */
export function elaborateCompress(
  segments: readonly ClassifiedMessage[],
  config: MemoSiftConfig,
  ledger?: AnchorLedger | null,
): ClassifiedMessage[] {
  const recentBoundary = findRecentBoundary(segments, config.recentTurns);

  const result: ClassifiedMessage[] = [];
  for (const seg of segments) {
    // Skip protected/recent/shielded segments.
    if (SKIP_POLICIES.has(seg.policy) || seg.protected) {
      result.push(seg);
      continue;
    }
    if (seg.originalIndex >= recentBoundary) {
      result.push(seg);
      continue;
    }
    if (seg.shield !== Shield.COMPRESSIBLE) {
      result.push(seg);
      continue;
    }

    const compressed = compressElaborations(seg.message.content ?? "", ledger ?? null);
    if (compressed !== seg.message.content) {
      const newMsg = createMessage(seg.message.role, compressed, {
        name: seg.message.name,
        toolCallId: seg.message.toolCallId,
        toolCalls: seg.message.toolCalls,
        metadata: seg.message.metadata,
      });
      result.push(
        createClassified(newMsg, seg.contentType, seg.policy, {
          originalIndex: seg.originalIndex,
          relevanceScore: seg.relevanceScore,
          estimatedTokens: seg.estimatedTokens,
          protected: seg.protected,
          importanceScore: seg.importanceScore,
          shield: seg.shield,
        }),
      );
    } else {
      result.push(seg);
    }
  }

  return result;
}

/**
 * Find the original_index boundary for recent turns.
 *
 * Counts user messages from the end; returns the index of the Nth-from-last
 * user message. Messages at or after this index are "recent".
 */
function findRecentBoundary(segments: readonly ClassifiedMessage[], recentTurns: number): number {
  const userIndices = segments
    .filter((seg) => seg.message.role === "user")
    .map((seg) => seg.originalIndex)
    .sort((a, b) => a - b);

  if (userIndices.length <= recentTurns) {
    return 0; // All messages are recent.
  }
  return userIndices[userIndices.length - recentTurns]!;
}

/**
 * Compress satellite/elaboration clauses within text.
 *
 * Splits at sentence boundaries, identifies satellites, and compresses
 * them to ~20% of their tokens while keeping nucleus clauses intact.
 */
function compressElaborations(text: string, ledger: AnchorLedger | null): string {
  const clauses = splitAtSentenceBoundaries(text);
  const result: string[] = [];

  for (const clause of clauses) {
    const isSatellite = isSatelliteClause(clause);
    const isParenthetical = PARENTHETICAL_RE.test(clause);
    const isNumbered = NUMBERED_LIST_RE.test(clause);

    if (isSatellite || isParenthetical || isNumbered) {
      // Check if clause contains an anchor fact — preserve if so.
      if (ledger?.containsAnchorFact(clause)) {
        result.push(clause);
      } else {
        const compressed = pruneClause(clause, 0.2);
        result.push(compressed);
      }
    } else {
      // Nucleus clause — keep intact.
      result.push(clause);
    }
  }

  return result.join(" ");
}

/**
 * Split text at sentence boundaries (period, exclamation, newline).
 *
 * Preserves non-empty results from splitting to maintain structure.
 */
function splitAtSentenceBoundaries(text: string): string[] {
  // Split on sentence-ending punctuation followed by whitespace.
  const sentences = text.split(/(?<=[.!])\s+/);
  return sentences.filter((s) => s.trim().length > 0);
}

/** Return true if the clause is a satellite/elaboration. */
function isSatelliteClause(clause: string): boolean {
  return SATELLITE_MARKERS.some((marker) => marker.test(clause));
}

/**
 * Compress a clause by keeping only the most important tokens.
 *
 * Keeps the first few words (to maintain the marker/context) and
 * enough additional words to meet keepRatio.
 */
function pruneClause(clause: string, keepRatio = 0.2): string {
  const words = clause.split(/\s+/).filter((w) => w.length > 0);
  if (words.length <= 5) {
    return clause; // Too short to compress meaningfully.
  }

  const keepCount = Math.max(3, Math.floor(words.length * keepRatio));
  if (keepCount >= words.length) {
    return clause;
  }

  // Keep the first 2 words (marker context) + distribute remaining.
  const kept = words.slice(0, 2);
  const remainingBudget = keepCount - 2;
  if (remainingBudget > 0) {
    // Sample evenly from the rest.
    const rest = words.slice(2);
    const step = Math.max(1, Math.floor(rest.length / remainingBudget));
    for (let j = 0; j < rest.length; j += step) {
      kept.push(rest[j]!);
      if (kept.length >= keepCount) {
        break;
      }
    }
  }

  return kept.join(" ");
}
