// Short message coalescence — merge consecutive tiny assistant messages into one.

import type { MemoSiftConfig } from "../config.js";
import {
  type ClassifiedMessage,
  CompressionPolicy,
  type MemoSiftMessage,
  createMessage,
} from "./types.js";

const COALESCEABLE_POLICIES = new Set([CompressionPolicy.MODERATE, CompressionPolicy.AGGRESSIVE]);

const MIN_RUN_LENGTH = 3;

export function coalesceShortMessages(
  segments: ClassifiedMessage[],
  config: MemoSiftConfig,
): ClassifiedMessage[] {
  if (!config.coalesceShortMessages) return segments;

  const threshold = config.coalesceCharThreshold;
  const result: ClassifiedMessage[] = [];
  let i = 0;

  while (i < segments.length) {
    const seg = segments[i]!;

    if (isCoalesceable(seg, threshold)) {
      const run: ClassifiedMessage[] = [seg];
      i++;
      while (i < segments.length && isCoalesceable(segments[i]!, threshold)) {
        run.push(segments[i]!);
        i++;
      }
      if (run.length >= MIN_RUN_LENGTH) {
        result.push(mergeRun(run));
      } else {
        result.push(...run);
      }
    } else {
      result.push(seg);
      i++;
    }
  }

  return result;
}

function isCoalesceable(seg: ClassifiedMessage, threshold: number): boolean {
  return (
    seg.message.role === "assistant" &&
    COALESCEABLE_POLICIES.has(seg.policy) &&
    !seg.protected &&
    !seg.message.toolCalls &&
    seg.message.content.length < threshold
  );
}

function mergeRun(run: ClassifiedMessage[]): ClassifiedMessage {
  const parts = run
    .map((seg) => seg.message.content.trim())
    .filter((t) => t.length > 0)
    .map((t) =>
      t.endsWith(".") || t.endsWith("!") || t.endsWith("?") || t.endsWith(";") || t.endsWith(":")
        ? t
        : `${t}.`,
    );

  const mergedContent = `[Assistant notes: ${parts.join(" ")}]`;
  const first = run[0]!;

  const mergedMsg = createMessage("assistant", mergedContent, {
    name: first.message.name,
    toolCallId: first.message.toolCallId,
    toolCalls: null,
    metadata: first.message.metadata,
  });

  return {
    ...first,
    message: mergedMsg,
  };
}
