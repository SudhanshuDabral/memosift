// Conversation phase detection — lightweight heuristic based on ContentType distribution.

import { type ClassifiedMessage, ContentType } from "./types.js";

/** Detected conversation phase — affects compression aggressiveness. */
export enum ConversationPhase {
  EXPLORATION = "EXPLORATION", // Asking questions, browsing — most content compressible
  IMPLEMENTATION = "IMPLEMENTATION", // Writing/editing code — protect code diffs
  DEBUGGING = "DEBUGGING", // Fixing errors — protect error traces and stack frames
  REVIEW = "REVIEW", // Reviewing/testing — moderate protection
}

/** Phase-specific keep_ratio multipliers applied to L3G importance scoring.
 * Kept close to 1.0 to avoid compounding with position factors. */
export const PHASE_KEEP_MULTIPLIERS: ReadonlyMap<ConversationPhase, number> = new Map([
  [ConversationPhase.DEBUGGING, 1.1], // Slightly protect more during debugging
  [ConversationPhase.IMPLEMENTATION, 1.05], // Slight protection for code
  [ConversationPhase.REVIEW, 1.0], // Normal during review
  [ConversationPhase.EXPLORATION, 0.95], // Slightly more aggressive during exploration
]);

/**
 * Detect the current conversation phase from recent message types.
 *
 * Looks at the last `window` messages and determines the phase based
 * on the dominant content type distribution.
 *
 * @param segments - Classified messages from the pipeline.
 * @param window - Number of recent messages to analyze (default 10).
 * @returns The detected conversation phase.
 */
export function detectPhase(
  segments: readonly ClassifiedMessage[],
  window = 10,
): ConversationPhase {
  if (segments.length === 0) return ConversationPhase.EXPLORATION;

  const recent = segments.slice(-window);
  const typeCounts = new Map<ContentType, number>();
  for (const seg of recent) {
    typeCounts.set(seg.contentType, (typeCounts.get(seg.contentType) ?? 0) + 1);
  }

  // Check for debugging phase (error traces present).
  const errorCount = typeCounts.get(ContentType.ERROR_TRACE) ?? 0;
  if (errorCount >= 2 || (errorCount >= 1 && recent.length <= 5)) {
    return ConversationPhase.DEBUGGING;
  }

  // Check for implementation phase (code blocks dominant).
  const codeCount = typeCounts.get(ContentType.CODE_BLOCK) ?? 0;
  if (codeCount >= 3 || codeCount / Math.max(recent.length, 1) > 0.3) {
    return ConversationPhase.IMPLEMENTATION;
  }

  // Check for review phase (tool results dominant).
  const toolCount =
    (typeCounts.get(ContentType.TOOL_RESULT_TEXT) ?? 0) +
    (typeCounts.get(ContentType.TOOL_RESULT_JSON) ?? 0);
  if (toolCount >= 3) {
    return ConversationPhase.REVIEW;
  }

  return ConversationPhase.EXPLORATION;
}
