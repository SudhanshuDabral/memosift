# Conversation phase detection — lightweight heuristic based on ContentType distribution.
from __future__ import annotations

from enum import StrEnum

from memosift.core.types import ClassifiedMessage, ContentType


class ConversationPhase(StrEnum):
    """Detected conversation phase — affects compression aggressiveness."""

    EXPLORATION = "EXPLORATION"  # Asking questions, browsing — most content compressible
    IMPLEMENTATION = "IMPLEMENTATION"  # Writing/editing code — protect code diffs
    DEBUGGING = "DEBUGGING"  # Fixing errors — protect error traces and stack frames
    REVIEW = "REVIEW"  # Reviewing/testing — moderate protection


# Content type indicators for each phase.
_PHASE_INDICATORS: dict[ConversationPhase, set[ContentType]] = {
    ConversationPhase.DEBUGGING: {ContentType.ERROR_TRACE},
    ConversationPhase.IMPLEMENTATION: {ContentType.CODE_BLOCK},
    ConversationPhase.REVIEW: {ContentType.TOOL_RESULT_TEXT, ContentType.TOOL_RESULT_JSON},
    ConversationPhase.EXPLORATION: {ContentType.ASSISTANT_REASONING, ContentType.OLD_CONVERSATION},
}

# Phase-specific keep_ratio multipliers applied to L3G importance scoring.
# Kept close to 1.0 to avoid compounding with position factors.
PHASE_KEEP_MULTIPLIERS: dict[ConversationPhase, float] = {
    ConversationPhase.DEBUGGING: 1.1,  # Slightly protect more during debugging
    ConversationPhase.IMPLEMENTATION: 1.05,  # Slight protection for code
    ConversationPhase.REVIEW: 1.0,  # Normal during review
    ConversationPhase.EXPLORATION: 0.95,  # Slightly more aggressive during exploration
}


def detect_phase(
    segments: list[ClassifiedMessage],
    window: int = 10,
) -> ConversationPhase:
    """Detect the current conversation phase from recent message types.

    Looks at the last ``window`` messages and determines the phase based
    on the dominant content type distribution.

    Args:
        segments: Classified messages from the pipeline.
        window: Number of recent messages to analyze.

    Returns:
        The detected conversation phase.
    """
    if not segments:
        return ConversationPhase.EXPLORATION

    recent = segments[-window:]
    type_counts: dict[ContentType, int] = {}
    for seg in recent:
        type_counts[seg.content_type] = type_counts.get(seg.content_type, 0) + 1

    # Check for debugging phase (error traces present).
    error_count = type_counts.get(ContentType.ERROR_TRACE, 0)
    if error_count >= 2 or (error_count >= 1 and len(recent) <= 5):
        return ConversationPhase.DEBUGGING

    # Check for implementation phase (code blocks dominant).
    code_count = type_counts.get(ContentType.CODE_BLOCK, 0)
    if code_count >= 3 or code_count / max(len(recent), 1) > 0.3:
        return ConversationPhase.IMPLEMENTATION

    # Check for review phase (tool results dominant).
    tool_count = type_counts.get(ContentType.TOOL_RESULT_TEXT, 0) + type_counts.get(
        ContentType.TOOL_RESULT_JSON, 0
    )
    if tool_count >= 3:
        return ConversationPhase.REVIEW

    return ConversationPhase.EXPLORATION
