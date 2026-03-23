// Engine B: IDF-based token pruning — remove low-information words.

import type { MemoSiftConfig } from "../../config.js";
import {
  type AnchorLedger,
  type ClassifiedMessage,
  CompressionPolicy,
  createClassified,
  createMessage,
} from "../types.js";

const TARGET_POLICIES = new Set([CompressionPolicy.MODERATE, CompressionPolicy.AGGRESSIVE]);

const PROTECTED_PATTERNS = [
  /^[\w.\-]+(?:[/\\][\w.\-]+)+(?::\d+)?$/, // file paths
  /^\d[\d.,:]*$/, // numbers
  /^[a-z]+(?:[A-Z][a-z]+)+$/, // camelCase
  /^[a-z]+(?:_[a-z]+)+$/, // snake_case
  /^[A-Z][A-Z_]+$/, // UPPER_CASE
  /^https?:\/\/\S+$/, // URLs
  /^\S+@\S+\.\S+$/, // emails
  /^[A-Za-z0-9]{10,}$/, // tracking numbers
  /^(?:ORD|RET|INV|TXN|REF)-[\w\-]+$/, // structured IDs
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i, // UUIDs
  /^[A-Z]{2,4}-\d{4,}$/, // Order IDs
  /^[A-Za-z0-9]{12,}$/, // Long alphanumeric identifiers
];

export function pruneTokens(
  segments: ClassifiedMessage[],
  config: MemoSiftConfig,
  ledger?: AnchorLedger | null,
): ClassifiedMessage[] {
  const allContents = segments.map((s) => s.message.content);
  const idfScores = computeIdfScores(allContents);

  // Auto-protect tokens from anchor ledger (Item 1.3).
  const ledgerLower = new Set<string>();
  if (ledger) {
    for (const s of ledger.getProtectedStrings()) {
      ledgerLower.add(s.toLowerCase());
    }
  }

  return segments.map((seg) => {
    if (!TARGET_POLICIES.has(seg.policy)) return seg;
    const pruned = pruneSegment(
      seg.message.content,
      idfScores,
      config.tokenPruneKeepRatio,
      ledgerLower,
    );
    if (pruned === seg.message.content) return seg;
    const newMsg = createMessage(seg.message.role, pruned, {
      name: seg.message.name,
      toolCallId: seg.message.toolCallId,
      toolCalls: seg.message.toolCalls,
      metadata: seg.message.metadata,
    });
    return createClassified(newMsg, seg.contentType, seg.policy, {
      originalIndex: seg.originalIndex,
      relevanceScore: seg.relevanceScore,
      protected: seg.protected,
      importanceScore: seg.importanceScore,
      shield: seg.shield,
    });
  });
}

function computeIdfScores(documents: string[]): Map<string, number> {
  const nDocs = documents.length;
  const docFreq = new Map<string, number>();
  for (const doc of documents) {
    const unique = new Set(doc.toLowerCase().match(/\b\w+\b/g) ?? []);
    for (const token of unique) docFreq.set(token, (docFreq.get(token) ?? 0) + 1);
  }
  const idf = new Map<string, number>();
  for (const [token, freq] of docFreq) {
    idf.set(token, Math.log((nDocs + 1) / (freq + 1)) + 1);
  }
  return idf;
}

function isProtectedToken(token: string, ledgerLower: ReadonlySet<string>): boolean {
  if (PROTECTED_PATTERNS.some((p) => p.test(token))) return true;
  if (ledgerLower.size > 0 && ledgerLower.has(token.toLowerCase())) return true;
  return false;
}

function pruneSegment(
  text: string,
  idfScores: Map<string, number>,
  keepRatio: number,
  ledgerLower: ReadonlySet<string>,
): string {
  return text
    .split("\n")
    .map((line) => {
      const stripped = line.trim();
      if (!stripped) return line;
      const words = stripped.split(/\s+/);
      if (words.length === 0) return line;

      const scored = words.map((w) => ({
        word: w,
        score: idfScores.get(w.toLowerCase().replace(/\W/g, "")) ?? 1.0,
      }));
      const sorted = [...scored].sort((a, b) => a.score - b.score);
      const keepCount = Math.max(1, Math.floor(sorted.length * keepRatio));
      if (keepCount >= sorted.length) return line;
      const threshold = sorted[sorted.length - keepCount]!.score;

      const kept = scored
        .filter((s) => isProtectedToken(s.word, ledgerLower) || s.score >= threshold)
        .map((s) => s.word);
      return kept.length > 0 ? kept.join(" ") : line;
    })
    .join("\n");
}
