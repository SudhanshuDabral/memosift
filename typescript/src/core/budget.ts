// Layer 6: Token budget enforcement — ensure output fits within constraints.

import type { MemoSiftConfig } from "../config.js";
import type { MemoSiftLLMProvider } from "../providers/base.js";
import {
  type AnchorLedger,
  type ClassifiedMessage,
  ContentType,
  type DependencyMap,
  createClassified,
  createMessage,
  depMapCanDrop,
  depMapDependentsOf,
} from "./types.js";

// Per-domain compression caps: content types that should not be aggressively
// compressed beyond a certain ratio. Code and error traces need higher fidelity.
const DOMAIN_MAX_COMPRESSION: Partial<Record<ContentType, number>> = {
  [ContentType.CODE_BLOCK]: 4.0, // Cap at 4x — code needs fidelity
  [ContentType.ERROR_TRACE]: 3.0, // Cap at 3x — errors are critical
};

export async function enforceBudget(
  segments: ClassifiedMessage[],
  config: MemoSiftConfig,
  deps: DependencyMap,
  counter?: MemoSiftLLMProvider | null,
  ledger?: AnchorLedger | null,
): Promise<ClassifiedMessage[]> {
  if (config.tokenBudget === null) return segments;

  const estimated = await estimateTokens(segments, counter ?? null);
  let total = estimated.reduce((sum, seg) => sum + seg.estimatedTokens, 0);

  if (total <= config.tokenBudget) return estimated;

  const indexed = estimated.map((seg, i) => ({ i, seg }));
  const droppable = indexed
    .filter(({ seg }) => !seg.protected && canDrop(seg, estimated) && !exceedsDomainCap(seg))
    .sort(
      (a, b) =>
        a.seg.relevanceScore - b.seg.relevanceScore || a.seg.originalIndex - b.seg.originalIndex,
    );

  const droppedIndices = new Set<number>();
  for (const { i, seg } of droppable) {
    if (total <= config.tokenBudget) break;
    if (!depMapCanDrop(deps, seg.originalIndex)) {
      expandDependents(seg.originalIndex, estimated, deps);
    }
    total -= seg.estimatedTokens;
    droppedIndices.add(i);
  }

  let result = estimated.filter((_, i) => !droppedIndices.has(i));

  total = result.reduce((sum, seg) => sum + seg.estimatedTokens, 0);
  if (total > config.tokenBudget && result.length > 1) {
    result = truncateLargest(result, total - config.tokenBudget);
  }

  return result;
}

async function estimateTokens(
  segments: ClassifiedMessage[],
  counter: MemoSiftLLMProvider | null,
): Promise<ClassifiedMessage[]> {
  const result: ClassifiedMessage[] = [];
  for (const seg of segments) {
    const tokens = counter
      ? await counter.countTokens(seg.message.content)
      : heuristicCount(seg.message.content);
    result.push(
      createClassified(seg.message, seg.contentType, seg.policy, {
        ...seg,
        estimatedTokens: tokens,
      }),
    );
  }
  return result;
}

function heuristicCount(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 3.5);
}

function canDrop(seg: ClassifiedMessage, segments: ClassifiedMessage[]): boolean {
  if (seg.message.toolCallId) {
    for (const other of segments) {
      if (other.message.toolCalls) {
        for (const tc of other.message.toolCalls) {
          if (tc.id === seg.message.toolCallId) return false;
        }
      }
    }
  }
  if (seg.message.toolCalls) {
    const callIds = new Set(seg.message.toolCalls.map((tc) => tc.id));
    for (const other of segments) {
      if (other.message.toolCallId && callIds.has(other.message.toolCallId)) return false;
    }
  }
  return true;
}

function exceedsDomainCap(seg: ClassifiedMessage): boolean {
  const maxRatio = DOMAIN_MAX_COMPRESSION[seg.contentType];
  if (maxRatio === undefined) return false;

  // If we know the original token count, check the compression ratio.
  const original = seg.message._memosiftOriginalTokens;
  if (original !== undefined && original !== null && original > 0) {
    const current = Math.max(heuristicCount(seg.message.content), 1);
    const ratio = original / current;
    return ratio >= maxRatio;
  }

  // Without original tokens, protect all domain-capped types from dropping.
  // They can still be truncated but not fully removed.
  return true;
}

function expandDependents(
  originalIndex: number,
  segments: ClassifiedMessage[],
  deps: DependencyMap,
): void {
  const dependentIndices = depMapDependentsOf(deps, originalIndex);
  for (const depIdx of dependentIndices) {
    for (let i = 0; i < segments.length; i++) {
      if (segments[i]!.originalIndex === depIdx) {
        for (const orig of segments) {
          if (orig.originalIndex === originalIndex) {
            const truncated = headTailTruncate(orig.message.content);
            const newMsg = createMessage(segments[i]!.message.role, truncated, {
              name: segments[i]!.message.name,
              toolCallId: segments[i]!.message.toolCallId,
              toolCalls: segments[i]!.message.toolCalls,
              metadata: segments[i]!.message.metadata,
            });
            segments[i] = createClassified(newMsg, segments[i]!.contentType, segments[i]!.policy, {
              originalIndex: segments[i]!.originalIndex,
              relevanceScore: segments[i]!.relevanceScore,
              protected: segments[i]!.protected,
            });
            break;
          }
        }
        break;
      }
    }
    deps.references.delete(depIdx);
  }
}

function headTailTruncate(text: string): string {
  const lines = text.split("\n");
  if (lines.length <= 10) return text;
  const keep = Math.max(3, Math.floor(lines.length / 10));
  const omitted = lines.length - 2 * keep;
  return [
    ...lines.slice(0, keep),
    `[... ${omitted} lines omitted ...]`,
    ...lines.slice(-keep),
  ].join("\n");
}

function truncateLargest(
  segments: ClassifiedMessage[],
  overshootTokens: number,
): ClassifiedMessage[] {
  const candidates = segments
    .map((seg, i) => ({ i, seg }))
    .filter(({ seg }) => seg.contentType !== ContentType.SYSTEM_PROMPT && !seg.protected);

  if (candidates.length === 0) return segments;

  const largest = candidates.reduce((a, b) =>
    a.seg.estimatedTokens > b.seg.estimatedTokens ? a : b,
  );
  const charsToRemove = Math.floor(overshootTokens * 3.5);
  const content = largest.seg.message.content;
  let newContent: string;

  if (charsToRemove >= content.length) {
    newContent = "[Content removed to fit budget.]";
  } else {
    const keep = content.length - charsToRemove;
    const half = Math.floor(keep / 2);
    newContent = `${content.slice(0, half)}\n[... ${charsToRemove} characters omitted to fit budget ...]\n${content.slice(-half)}`;
  }

  const newMsg = createMessage(largest.seg.message.role, newContent, {
    name: largest.seg.message.name,
    toolCallId: largest.seg.message.toolCallId,
    toolCalls: largest.seg.message.toolCalls,
    metadata: largest.seg.message.metadata,
  });
  segments[largest.i] = createClassified(newMsg, largest.seg.contentType, largest.seg.policy, {
    originalIndex: largest.seg.originalIndex,
    relevanceScore: largest.seg.relevanceScore,
    protected: largest.seg.protected,
  });
  return segments;
}
