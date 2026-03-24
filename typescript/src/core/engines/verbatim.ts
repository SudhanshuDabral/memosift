// Engine A: Verbatim deletion — remove noise lines while preserving surviving tokens.

import { createHash } from "node:crypto";
import type { MemoSiftConfig } from "../../config.js";
import {
  type AnchorLedger,
  type ClassifiedMessage,
  CompressionPolicy,
  ContentType,
  createClassified,
  createMessage,
} from "../types.js";

const TARGET_POLICIES = new Set([
  CompressionPolicy.MODERATE,
  CompressionPolicy.AGGRESSIVE,
  CompressionPolicy.STACK,
]);

const FILE_PATH_RE = /(?:[A-Za-z]:)?(?:[/\\][\w.\-]+)+(?:\.\w+)?(?::\d+)?/;
const REPETITION_THRESHOLD = 3;
const MAX_LINES = 100;

// Content types eligible for first-read vs re-read tracking.
const REREAD_TYPES = new Set([
  ContentType.TOOL_RESULT_TEXT,
  ContentType.TOOL_RESULT_JSON,
  ContentType.CODE_BLOCK,
]);

export function verbatimCompress(
  segments: ClassifiedMessage[],
  config: MemoSiftConfig,
  ledger?: AnchorLedger | null,
  seenContentHashes?: Map<string, number> | null,
): ClassifiedMessage[] {
  const seen = seenContentHashes ?? new Map<string, number>();

  return segments.map((seg) => {
    // Rule 0: First-read vs re-read detection for tool results.
    if (
      TARGET_POLICIES.has(seg.policy) &&
      seg.message.content &&
      seg.message.content.length > 200 &&
      REREAD_TYPES.has(seg.contentType)
    ) {
      const normalized = seg.message.content.replace(/\s+/g, " ").trim();
      const contentHash = createHash("sha256").update(normalized).digest("hex");
      if (seen.has(contentHash)) {
        const firstIndex = seen.get(contentHash)!;
        const label = extractContentLabel(seg.message.content);
        const collapsed = `[Previously read: ${label} — see message ${firstIndex}]`;
        const newMsg = createMessage(seg.message.role, collapsed, {
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
      }
      seen.set(contentHash, seg.originalIndex);
    }

    if (!TARGET_POLICIES.has(seg.policy)) return seg;
    const newContent = compressContent(seg.message.content, config.entropyThreshold);
    if (newContent === seg.message.content) return seg;
    const newMsg = createMessage(seg.message.role, newContent, {
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

function extractContentLabel(content: string): string {
  const match = FILE_PATH_RE.exec(content);
  if (match) return match[0];
  const firstLine = content.split("\n", 1)[0]!.trim();
  if (firstLine.length > 60) return `${firstLine.slice(0, 57)}...`;
  return firstLine || "content";
}

function compressContent(text: string, entropyThreshold: number): string {
  let lines = text.split("\n");
  lines = collapseBlankLines(lines);
  lines = removeLowEntropyLines(lines, entropyThreshold);
  lines = collapseRepetitivePatterns(lines);
  if (lines.length > MAX_LINES) lines = truncateWithMarker(lines, MAX_LINES);
  return lines.join("\n");
}

export function shannonEntropy(text: string): number {
  if (!text) return 0;
  const freq = new Map<string, number>();
  for (const ch of text) freq.set(ch, (freq.get(ch) ?? 0) + 1);
  let entropy = 0;
  for (const count of freq.values()) {
    const p = count / text.length;
    entropy -= p * Math.log2(p);
  }
  return entropy;
}

function isProtectedLine(line: string): boolean {
  const stripped = line.trim();
  if (!stripped) return false;
  if (FILE_PATH_RE.test(stripped)) return true;
  if (/\d+/.test(stripped)) return true;
  return false;
}

function collapseBlankLines(lines: string[]): string[] {
  const result: string[] = [];
  let blankCount = 0;
  for (const line of lines) {
    if (line.trim() === "") {
      blankCount++;
      if (blankCount <= 1) result.push(line);
    } else {
      blankCount = 0;
      result.push(line);
    }
  }
  return result;
}

function removeLowEntropyLines(lines: string[], threshold: number): string[] {
  return lines.filter((line) => {
    const stripped = line.trim();
    if (!stripped) return true;
    if (isProtectedLine(line)) return true;
    return shannonEntropy(stripped) >= threshold;
  });
}

function collapseRepetitivePatterns(lines: string[]): string[] {
  if (lines.length < REPETITION_THRESHOLD + 1) return lines;
  const result: string[] = [];
  let i = 0;
  while (i < lines.length) {
    const pattern = normalizePattern(lines[i]!);
    const runStart = i;
    while (i < lines.length && normalizePattern(lines[i]!) === pattern) i++;
    const runLength = i - runStart;
    if (runLength > REPETITION_THRESHOLD && pattern) {
      for (let j = runStart; j < runStart + REPETITION_THRESHOLD; j++) result.push(lines[j]!);
      result.push(`[... ${runLength - REPETITION_THRESHOLD} similar lines omitted ...]`);
    } else {
      for (let j = runStart; j < runStart + runLength; j++) result.push(lines[j]!);
    }
  }
  return result;
}

function normalizePattern(line: string): string {
  let s = line.trim();
  if (!s) return "";
  s = s.replace(/\d+/g, "#");
  s = s.replace(/['"].*?['"]/g, "STR");
  return s;
}

function truncateWithMarker(lines: string[], maxLines: number): string[] {
  const keep = Math.floor(maxLines / 2);
  const omitted = lines.length - maxLines;
  return [
    ...lines.slice(0, keep),
    `[... ${omitted} lines omitted — showing first ${keep} and last ${keep} lines ...]`,
    ...lines.slice(-keep),
  ];
}
