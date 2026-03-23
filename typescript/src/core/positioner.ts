// Layer 5: Position optimization — reorder segments for attention distribution.

import type { MemoSiftConfig } from "../config.js";
import { type ClassifiedMessage, ContentType } from "./types.js";

const HIGH_PRIORITY = new Set([ContentType.SYSTEM_PROMPT, ContentType.ERROR_TRACE]);
const END_TYPES = new Set([ContentType.RECENT_TURN, ContentType.USER_QUERY]);

export function optimizePosition(
  segments: ClassifiedMessage[],
  config: MemoSiftConfig,
): ClassifiedMessage[] {
  if (!config.reorderSegments || segments.length === 0) return segments;

  const blocks = buildBlocks(segments);
  const beginning: ClassifiedMessage[][] = [];
  const middle: ClassifiedMessage[][] = [];
  const end: ClassifiedMessage[][] = [];

  for (const block of blocks) {
    const primaryType = block[0]!.contentType;
    if (HIGH_PRIORITY.has(primaryType)) beginning.push(block);
    else if (END_TYPES.has(primaryType)) end.push(block);
    else middle.push(block);
  }

  const reordered = [...beginning, ...middle, ...end].flat();
  if (!isValidSequence(reordered)) return segments;
  return reordered;
}

function buildBlocks(segments: ClassifiedMessage[]): ClassifiedMessage[][] {
  const blocks: ClassifiedMessage[][] = [];
  let i = 0;
  while (i < segments.length) {
    const seg = segments[i]!;
    if (seg.message.toolCalls && seg.message.toolCalls.length > 0) {
      const block = [seg];
      const callIds = new Set(seg.message.toolCalls.map((tc) => tc.id));
      i++;
      while (
        i < segments.length &&
        segments[i]!.message.toolCallId &&
        callIds.has(segments[i]!.message.toolCallId!)
      ) {
        block.push(segments[i]!);
        callIds.delete(segments[i]!.message.toolCallId!);
        i++;
      }
      blocks.push(block);
    } else {
      blocks.push([seg]);
      i++;
    }
  }
  return blocks;
}

function isValidSequence(segments: ClassifiedMessage[]): boolean {
  const pending = new Set<string>();
  for (const seg of segments) {
    if (seg.message.toolCalls) {
      for (const tc of seg.message.toolCalls) pending.add(tc.id);
    } else if (seg.message.toolCallId) {
      if (!pending.has(seg.message.toolCallId)) return false;
      pending.delete(seg.message.toolCallId);
    } else if (pending.size > 0) {
      return false;
    }
  }
  return true;
}
