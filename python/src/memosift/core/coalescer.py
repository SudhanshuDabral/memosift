# Short message coalescence — merge consecutive tiny assistant messages into one.
from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from memosift.core.types import (
    ClassifiedMessage,
    CompressionPolicy,
    MemoSiftMessage,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig

# Only coalesce messages with these policies.
_COALESCEABLE_POLICIES = {
    CompressionPolicy.MODERATE,
    CompressionPolicy.AGGRESSIVE,
}

# Minimum consecutive short messages to trigger coalescence.
_MIN_RUN_LENGTH = 3


def coalesce_short_messages(
    segments: list[ClassifiedMessage],
    config: MemoSiftConfig,
) -> list[ClassifiedMessage]:
    """Merge consecutive short assistant messages into a single combined message.

    Short assistant messages like "Let me check.", "Done.", "Running tests..."
    consume full message slots but carry little information. Merging 3+ of them
    into one message is free compression without fact loss.

    Only messages meeting ALL criteria are coalesced:
    - Role is ``assistant``
    - Policy is MODERATE or AGGRESSIVE (not PRESERVE or LIGHT)
    - Content length < ``config.coalesce_char_threshold``
    - No ``tool_calls`` (tool-calling messages are structurally significant)
    - Not protected

    Args:
        segments: Classified messages from previous layers.
        config: Pipeline configuration.

    Returns:
        Segments with short runs merged.
    """
    if not config.coalesce_short_messages:
        return segments

    threshold = config.coalesce_char_threshold
    result: list[ClassifiedMessage] = []
    i = 0

    while i < len(segments):
        seg = segments[i]

        # Check if this starts a coalesceable run.
        if _is_coalesceable(seg, threshold):
            run: list[ClassifiedMessage] = [seg]
            i += 1

            while i < len(segments) and _is_coalesceable(segments[i], threshold):
                run.append(segments[i])
                i += 1

            if len(run) >= _MIN_RUN_LENGTH:
                # Merge the run into a single message.
                merged = _merge_run(run)
                result.append(merged)
            else:
                # Run too short — keep individual messages.
                result.extend(run)
        else:
            result.append(seg)
            i += 1

    return result


def _is_coalesceable(seg: ClassifiedMessage, threshold: int) -> bool:
    """Check if a segment can be coalesced."""
    return (
        seg.message.role == "assistant"
        and seg.policy in _COALESCEABLE_POLICIES
        and not seg.protected
        and not seg.message.tool_calls
        and len(seg.content) < threshold
    )


def _merge_run(run: list[ClassifiedMessage]) -> ClassifiedMessage:
    """Merge a run of short assistant messages into one."""
    parts = [seg.content.strip() for seg in run if seg.content.strip()]

    # Join with periods if the parts don't already end with punctuation.
    formatted_parts: list[str] = []
    for part in parts:
        if part and part[-1] not in ".!?;:":
            formatted_parts.append(part + ".")
        else:
            formatted_parts.append(part)

    merged_content = "[Assistant notes: " + " ".join(formatted_parts) + "]"

    # Use the first message's metadata and index.
    first = run[0]
    merged_msg = MemoSiftMessage(
        role="assistant",
        content=merged_content,
        name=first.message.name,
        tool_call_id=first.message.tool_call_id,
        tool_calls=None,
        metadata=dict(first.message.metadata),
    )

    return dc_replace(
        first,
        message=merged_msg,
    )
