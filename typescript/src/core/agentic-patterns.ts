// Layer 1.5: Agentic pattern detector — identifies and compresses waste patterns
// specific to AI agent conversations (duplicate tool calls, failed retries,
// large code arguments, thought process bloat, KPI restatement).

import { createHash } from "node:crypto";
import type { MemoSiftConfig } from "../config.js";
import {
  type AnchorLedger,
  type ClassifiedMessage,
  CompressionPolicy,
  ContentType,
  type MemoSiftMessage,
  type ToolCall,
  createMessage,
} from "./types.js";

// ── Pattern 1: Duplicate tool call detection ─────────────────────────────

/** Minimum content length to consider for dedup (skip tiny results). */
const MIN_DEDUP_CONTENT = 20;

function toolResultSignature(seg: ClassifiedMessage): string | null {
  if (seg.message.role !== "tool" || !seg.message.content) return null;
  if (seg.message.content.length < MIN_DEDUP_CONTENT) return null;
  return createHash("sha256").update(seg.message.content).digest("hex").slice(0, 16);
}

// ── Pattern 2: Failed + retried tool calls ───────────────────────────────

const ERROR_INDICATORS =
  /"exitCode"\s*:\s*1|"stderr"\s*:\s*"[^"]{5,}|Traceback|Error:|Exception:|FAILED|exitCode.*1/i;

function isErrorResult(seg: ClassifiedMessage): boolean {
  if (seg.message.role !== "tool" || !seg.message.content) return false;
  if (seg.contentType === ContentType.ERROR_TRACE) return true;
  return ERROR_INDICATORS.test(seg.message.content);
}

function findToolCallForResult(
  resultSeg: ClassifiedMessage,
  segments: readonly ClassifiedMessage[],
  resultIndex: number,
): number | null {
  if (!resultSeg.message.toolCallId) return null;
  for (let i = resultIndex - 1; i >= 0; i--) {
    const seg = segments[i]!;
    if (seg.message.toolCalls) {
      for (const tc of seg.message.toolCalls) {
        if (tc.id === resultSeg.message.toolCallId) return i;
      }
    }
  }
  return null;
}

function getToolNameFromResult(
  resultSeg: ClassifiedMessage,
  segments: readonly ClassifiedMessage[],
  resultIndex: number,
): string | null {
  const callIdx = findToolCallForResult(resultSeg, segments, resultIndex);
  if (callIdx !== null) {
    const seg = segments[callIdx]!;
    if (seg.message.toolCalls) {
      for (const tc of seg.message.toolCalls) {
        if (tc.id === resultSeg.message.toolCallId) return tc.function.name;
      }
    }
  }
  return null;
}

// ── Pattern 3: Large code arguments ──────────────────────────────────────

const CODE_INDICATORS =
  /\b(?:import |from |def |class |function |const |let |var |async )\b/;

const LARGE_CODE_THRESHOLD = 4000;

// ── Pattern 4: Thought process blocks ───────────────────────────────────

const THOUGHT_PATTERNS: RegExp[] = [
  /<thinking>/i,
  /<thought>/i,
  /\*\*Thought Process\*\*/i,
  /\*\*(?:Considering|Planning|Processing|Determining|Figuring)\b/,
  /^```\n\*\*(?:Considering|Planning|Analyzing)/m,
];

const THOUGHT_MIN_LENGTH = 200;

// ── Pattern 5: KPI restatement ───────────────────────────────────────────

const KPI_RESTATEMENT_THRESHOLD = 3;

// ── Helper: create a shallow copy of ClassifiedMessage with overrides ────

function cloneSegment(
  seg: ClassifiedMessage,
  overrides: Partial<ClassifiedMessage>,
): ClassifiedMessage {
  return { ...seg, ...overrides };
}

// ── Pattern 1 implementation ─────────────────────────────────────────────

function detectDuplicateToolResults(
  segments: readonly ClassifiedMessage[],
): ClassifiedMessage[] {
  const result: ClassifiedMessage[] = [];
  /** Map: result content hash -> first occurrence original_index. */
  const seenResults = new Map<string, number>();

  for (const seg of segments) {
    // Check tool result messages for duplicates.
    if (seg.message.role === "tool" && seg.message.content) {
      const sig = toolResultSignature(seg);
      if (sig !== null && seenResults.has(sig)) {
        // Duplicate result — replace content with back-reference.
        const firstIdx = seenResults.get(sig)!;
        const refContent = `[Identical result — same as tool output at message ${firstIdx}]`;
        const newMsg = createMessage(seg.message.role, refContent, {
          name: seg.message.name,
          toolCallId: seg.message.toolCallId,
          toolCalls: seg.message.toolCalls,
          metadata: seg.message.metadata,
        });
        result.push(cloneSegment(seg, { message: newMsg, policy: CompressionPolicy.AGGRESSIVE }));
        continue;
      } else if (sig !== null) {
        seenResults.set(sig, seg.originalIndex);
      }
    }

    result.push(seg);
  }

  return result;
}

// ── Pattern 2 implementation ─────────────────────────────────────────────

function detectFailedRetries(
  segments: readonly ClassifiedMessage[],
): ClassifiedMessage[] {
  const result = [...segments];
  const errorIndices: number[] = [];

  // Find all error results.
  for (let i = 0; i < segments.length; i++) {
    if (isErrorResult(segments[i]!)) {
      errorIndices.push(i);
    }
  }

  // For each error, check if a successful retry exists within 8 messages.
  for (const errIdx of errorIndices) {
    const errSeg = segments[errIdx]!;
    const errToolName = getToolNameFromResult(errSeg, segments, errIdx);
    if (!errToolName) continue;

    // Look forward for a success with the same tool name.
    let foundRetry = false;
    for (let j = errIdx + 1; j < Math.min(errIdx + 9, segments.length); j++) {
      const segJ = segments[j]!;
      if (segJ.message.role === "tool" && !isErrorResult(segJ)) {
        const retryName = getToolNameFromResult(segJ, segments, j);
        if (retryName === errToolName) {
          foundRetry = true;
          break;
        }
      }
    }

    if (foundRetry) {
      // Use MODERATE (not AGGRESSIVE) to allow pruning but avoid
      // full observation masking that could lose file paths in errors.
      // Only upgrade to AGGRESSIVE if the error content is very short
      // (< 200 chars — unlikely to contain unique file paths).
      const errContent = errSeg.message.content ?? "";
      const policy =
        errContent.length < 200
          ? CompressionPolicy.AGGRESSIVE
          : CompressionPolicy.MODERATE;
      result[errIdx] = cloneSegment(result[errIdx]!, { policy });
      // Also mark the corresponding tool call.
      const callIdx = findToolCallForResult(errSeg, segments, errIdx);
      if (callIdx !== null) {
        result[callIdx] = cloneSegment(result[callIdx]!, { policy });
      }
    }
  }

  return result;
}

// ── Pattern 3 implementation ─────────────────────────────────────────────

function detectLargeCodeArgs(
  segments: readonly ClassifiedMessage[],
): ClassifiedMessage[] {
  const result: ClassifiedMessage[] = [];

  for (const seg of segments) {
    if (!seg.message.toolCalls) {
      result.push(seg);
      continue;
    }

    let modified = false;
    const newToolCalls: ToolCall[] = [];
    for (const tc of seg.message.toolCalls) {
      const args = tc.function.arguments;
      if (args.length > LARGE_CODE_THRESHOLD && CODE_INDICATORS.test(args)) {
        // Truncate to first 500 chars + marker.
        const truncated =
          args.slice(0, 500) + `\n... [truncated ${args.length - 500} chars of code]`;
        const newTc: ToolCall = {
          id: tc.id,
          type: tc.type,
          function: {
            name: tc.function.name,
            arguments: truncated,
          },
        };
        newToolCalls.push(newTc);
        modified = true;
      } else {
        newToolCalls.push(tc);
      }
    }

    if (modified) {
      const newMsg = createMessage(seg.message.role, seg.message.content, {
        name: seg.message.name,
        toolCallId: seg.message.toolCallId,
        toolCalls: newToolCalls,
        metadata: seg.message.metadata,
      });
      result.push(cloneSegment(seg, { message: newMsg }));
    } else {
      result.push(seg);
    }
  }

  return result;
}

// ── Pattern 4 implementation ─────────────────────────────────────────────

function detectThoughtBlocks(
  segments: readonly ClassifiedMessage[],
): ClassifiedMessage[] {
  const result: ClassifiedMessage[] = [];

  for (const seg of segments) {
    if (seg.message.role !== "assistant" || !seg.message.content) {
      result.push(seg);
      continue;
    }

    const content = seg.message.content;
    if (content.length < THOUGHT_MIN_LENGTH) {
      result.push(seg);
      continue;
    }

    const isThought = THOUGHT_PATTERNS.some((p) => p.test(content));
    if (isThought) {
      result.push(
        cloneSegment(seg, {
          contentType: ContentType.ASSISTANT_REASONING,
          policy: CompressionPolicy.AGGRESSIVE,
        }),
      );
    } else {
      result.push(seg);
    }
  }

  return result;
}

// ── Pattern 5 implementation ─────────────────────────────────────────────

function detectKpiRestatement(
  segments: readonly ClassifiedMessage[],
  ledger: AnchorLedger,
): ClassifiedMessage[] {
  const criticalStrings = ledger.getCriticalStrings();
  if (criticalStrings.size === 0) return [...segments];

  // Find recent boundary — skip last 2 assistant messages.
  const assistantIndices: number[] = [];
  for (let i = 0; i < segments.length; i++) {
    if (segments[i]!.message.role === "assistant" && segments[i]!.message.content) {
      assistantIndices.push(i);
    }
  }
  if (assistantIndices.length <= 2) return [...segments];
  const recentBoundary = assistantIndices[assistantIndices.length - 2]!;

  const result: ClassifiedMessage[] = [];
  let firstRestatementSeen = false;

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i]!;
    if (
      seg.message.role !== "assistant" ||
      !seg.message.content ||
      i >= recentBoundary ||
      seg.policy === CompressionPolicy.PRESERVE ||
      seg.policy === CompressionPolicy.LIGHT
    ) {
      result.push(seg);
      continue;
    }

    // Count how many anchor facts appear in this message.
    const contentLower = seg.message.content.toLowerCase();
    let factCount = 0;
    for (const s of criticalStrings) {
      if (contentLower.includes(s.toLowerCase())) factCount++;
    }

    if (factCount >= KPI_RESTATEMENT_THRESHOLD) {
      if (!firstRestatementSeen) {
        // Keep the first restatement — it's the original synthesis.
        firstRestatementSeen = true;
        result.push(seg);
      } else {
        // Subsequent restatements -> MODERATE policy for harder pruning.
        result.push(cloneSegment(seg, { policy: CompressionPolicy.MODERATE }));
      }
    } else {
      result.push(seg);
    }
  }

  return result;
}

// ── Main detector ────────────────────────────────────────────────────────

/**
 * Detect and annotate agentic waste patterns in classified segments.
 *
 * Runs 5 deterministic pattern detectors:
 * 1. Duplicate tool calls — collapse identical call+result pairs
 * 2. Failed + retried tool calls — mark resolved errors as AGGRESSIVE
 * 3. Large code arguments — compress tool call args > 4KB containing code
 * 4. Thought process blocks — reclassify as ASSISTANT_REASONING + AGGRESSIVE
 * 5. KPI restatement — mark messages restating 3+ anchor facts as MODERATE
 *
 * This layer runs after L1 classification and anchor extraction, before L2
 * deduplication. It does NOT delete messages — it reclassifies and annotates
 * them so downstream layers (L2, L3, L6) compress more effectively.
 *
 * @param segments - Classified messages from Layer 1.
 * @param config - Pipeline configuration (unused currently, reserved for tuning).
 * @param ledger - Anchor ledger for KPI restatement detection.
 * @returns Segments with agentic patterns annotated via policy/content_type changes.
 */
export function detectAgenticPatterns(
  segments: readonly ClassifiedMessage[],
  config?: MemoSiftConfig | null,
  ledger?: AnchorLedger | null,
): ClassifiedMessage[] {
  let result = detectDuplicateToolResults(segments);
  result = detectFailedRetries(result);
  result = detectLargeCodeArgs(result);
  result = detectThoughtBlocks(result);

  if (ledger != null) {
    result = detectKpiRestatement(result, ledger);
  }

  return result;
}
