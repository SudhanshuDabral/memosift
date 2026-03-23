// Layer 4: Task-aware relevance scoring — keyword mode + optional LLM mode.

import type { MemoSiftConfig } from "../config.js";
import type { MemoSiftLLMProvider } from "../providers/base.js";
import {
  type AnchorLedger,
  type ClassifiedMessage,
  ContentType,
  Shield,
  createClassified,
} from "./types.js";

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
  "being",
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
  "shall",
  "this",
  "that",
  "these",
  "those",
  "it",
  "its",
  "i",
  "me",
  "my",
  "we",
  "our",
  "you",
  "your",
  "he",
  "she",
  "they",
  "them",
  "their",
  "what",
  "which",
  "who",
  "when",
  "where",
  "why",
  "how",
  "not",
  "no",
  "so",
  "if",
  "then",
  "than",
  "just",
  "also",
  "very",
  "too",
  "all",
  "any",
  "some",
  "each",
]);

const PROTECTED_TYPES = new Set([
  ContentType.SYSTEM_PROMPT,
  ContentType.USER_QUERY,
  ContentType.RECENT_TURN,
  ContentType.PREVIOUSLY_COMPRESSED,
]);

export async function scoreRelevance(
  segments: ClassifiedMessage[],
  config: MemoSiftConfig,
  task?: string | null,
  ledger?: AnchorLedger | null,
): Promise<ClassifiedMessage[]> {
  if (!task) {
    return segments.map((seg) =>
      createClassified(seg.message, seg.contentType, seg.policy, {
        ...seg,
        relevanceScore: seg.protected ? 1.0 : 0.5,
      }),
    );
  }

  const taskKeywords = extractKeywords(task);
  if (taskKeywords.size === 0) {
    return segments.map((seg) =>
      createClassified(seg.message, seg.contentType, seg.policy, {
        ...seg,
        relevanceScore: seg.protected ? 1.0 : 0.5,
      }),
    );
  }

  // Use critical strings (FILES + ERRORS + high-value IDENTIFIERS) for rescue
  // to avoid broad IDENTIFIER matches that kill compression ratios.
  const criticalStrings = ledger ? ledger.getCriticalStrings() : new Set<string>();

  const result: ClassifiedMessage[] = [];
  for (const seg of segments) {
    if (PROTECTED_TYPES.has(seg.contentType)) {
      result.push(
        createClassified(seg.message, seg.contentType, seg.policy, { ...seg, relevanceScore: 1.0 }),
      );
      continue;
    }

    const contentKeywords = extractKeywords(seg.message.content);
    let score = 0;
    if (contentKeywords.size > 0) {
      let overlap = 0;
      for (const kw of taskKeywords) {
        if (contentKeywords.has(kw)) overlap++;
      }
      score = overlap / taskKeywords.size;
    }

    // Anchor rescue — segments containing critical facts get a floor
    // score to prevent dropping. Two tiers:
    //   a) Segments with critical strings (FILES, ERRORS, high-value IDs):
    //      floor at threshold (never dropped by L4).
    //   b) Segments with shield=PRESERVE from importance scoring:
    //      floor at threshold (importance scorer already validated).
    if (score < config.relevanceDropThreshold) {
      let rescued = false;

      // Tier a: critical strings rescue.
      if (criticalStrings.size > 0) {
        const textLower = seg.message.content.toLowerCase();
        for (const s of criticalStrings) {
          if (textLower.includes(s.toLowerCase())) {
            score = Math.max(score, config.relevanceDropThreshold);
            rescued = true;
            break;
          }
        }
      }

      // Tier b: importance shield rescue.
      if (!rescued && seg.shield === Shield.PRESERVE) {
        score = Math.max(score, config.relevanceDropThreshold);
      }
    }

    // Position-dependent compression (Lost in the Middle mitigation).
    // Multiply relevance by a U-shaped position factor: lighter compression
    // at start/end, heavier in the middle — exploits attention curve.
    // Uses originalIndex for stable positioning, not loop index.
    if (segments.length > 10) {
      const maxIdx = Math.max(...segments.map((s) => s.originalIndex)) || 1;
      const factor = positionFactor(seg.originalIndex, maxIdx);
      score *= factor;
    }

    if (score < config.relevanceDropThreshold) continue;
    result.push(
      createClassified(seg.message, seg.contentType, seg.policy, { ...seg, relevanceScore: score }),
    );
  }

  return result;
}

export async function scoreRelevanceLlm(
  segments: ClassifiedMessage[],
  config: MemoSiftConfig,
  task: string,
  llm: MemoSiftLLMProvider,
): Promise<ClassifiedMessage[]> {
  const result: ClassifiedMessage[] = [];
  for (const seg of segments) {
    if (PROTECTED_TYPES.has(seg.contentType)) {
      result.push(
        createClassified(seg.message, seg.contentType, seg.policy, { ...seg, relevanceScore: 1.0 }),
      );
      continue;
    }

    try {
      const prompt = `Rate the relevance of this SEGMENT to the given TASK on a scale of 0-10.\n\nTASK: ${task}\n\nSEGMENT:\n${seg.message.content.slice(0, 2000)}\n\nRespond with ONLY a JSON object: {"score": <0-10>, "reason": "<brief reason>"}`;
      const response = await llm.generate(prompt, { maxTokens: 100, temperature: 0 });
      const parsed = JSON.parse(response.text.trim()) as { score?: number };
      const llmScore = (parsed.score ?? 5) / 10;
      if (llmScore < 0.3) continue;
      result.push(
        createClassified(seg.message, seg.contentType, seg.policy, {
          ...seg,
          relevanceScore: llmScore,
        }),
      );
    } catch {
      const contentKeywords = extractKeywords(seg.message.content);
      const taskKeywords = extractKeywords(task);
      let score = 0.5;
      if (contentKeywords.size > 0 && taskKeywords.size > 0) {
        let overlap = 0;
        for (const kw of taskKeywords) {
          if (contentKeywords.has(kw)) overlap++;
        }
        score = overlap / taskKeywords.size;
      }
      if (score >= config.relevanceDropThreshold) {
        result.push(
          createClassified(seg.message, seg.contentType, seg.policy, {
            ...seg,
            relevanceScore: score,
          }),
        );
      }
    }
  }
  return result;
}

/**
 * U-shaped position factor for Lost in the Middle mitigation.
 *
 * First 15%: 1.1 (lighter compression — primacy effect).
 * Middle 70%: 0.9 (slightly heavier compression — model pays less attention).
 * Last 15%: 1.15 (lightest compression — recency matters most).
 *
 * Kept close to 1.0 to avoid large swings that tank compression ratio.
 */
function positionFactor(index: number, total: number): number {
  if (total <= 0) return 1.0;
  const positionPct = index / total;
  if (positionPct < 0.15) return 1.1;
  if (positionPct > 0.85) return 1.15;
  return 0.9;
}

function extractKeywords(text: string): Set<string> {
  const tokens = new Set(text.toLowerCase().match(/\b\w+\b/g) ?? []);
  for (const sw of STOP_WORDS) tokens.delete(sw);
  return tokens;
}
