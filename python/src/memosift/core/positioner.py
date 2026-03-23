# Layer 5: Position optimization — reorder segments for attention distribution.
from __future__ import annotations

from typing import TYPE_CHECKING

from memosift.core.types import (
    ClassifiedMessage,
    ContentType,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig

# High-priority types go to the beginning (high attention zone).
_HIGH_PRIORITY_TYPES: frozenset[ContentType] = frozenset(
    {
        ContentType.SYSTEM_PROMPT,
        ContentType.ERROR_TRACE,
    }
)

# Low-priority types go to the middle (low attention zone).
_LOW_PRIORITY_TYPES: frozenset[ContentType] = frozenset(
    {
        ContentType.OLD_CONVERSATION,
        ContentType.ASSISTANT_REASONING,
    }
)

# End types stay at the end (high attention zone).
_END_TYPES: frozenset[ContentType] = frozenset(
    {
        ContentType.RECENT_TURN,
        ContentType.USER_QUERY,
    }
)


def optimize_position(
    segments: list[ClassifiedMessage],
    config: MemoSiftConfig,
) -> list[ClassifiedMessage]:
    """Reorder segments to exploit the U-shaped attention curve.

    DISABLED by default (``config.reorder_segments = False``).
    When enabled, reorders "blocks" (assistant + its tool calls/results as
    an atomic unit). Never reorders individual messages within a block.

    HIGH ATTENTION (beginning): System prompt + error traces
    LOW ATTENTION (middle): Old conversation + assistant reasoning
    HIGH ATTENTION (end): Recent turns + current query

    Args:
        segments: Classified messages from previous layers.
        config: Pipeline configuration.

    Returns:
        Segments, potentially reordered for attention optimization.
    """
    if not config.reorder_segments:
        return segments

    if not segments:
        return segments

    # Group into blocks (assistant + associated tool calls/results).
    blocks = _build_blocks(segments)

    # Partition blocks by position priority.
    beginning: list[list[ClassifiedMessage]] = []
    middle: list[list[ClassifiedMessage]] = []
    end: list[list[ClassifiedMessage]] = []

    for block in blocks:
        primary_type = block[0].content_type
        if primary_type in _HIGH_PRIORITY_TYPES:
            beginning.append(block)
        elif primary_type in _END_TYPES:
            end.append(block)
        else:
            middle.append(block)

    # Validate the reordered sequence.
    reordered = _flatten(beginning + middle + end)
    if not _is_valid_sequence(reordered):
        # Invalid sequence — abort reordering.
        return segments

    return reordered


def _build_blocks(segments: list[ClassifiedMessage]) -> list[list[ClassifiedMessage]]:
    """Group messages into atomic blocks.

    A block is either:
    - A single message that's not part of a tool-call sequence
    - An assistant message + its associated tool results (matched by tool_call_id)
    """
    blocks: list[list[ClassifiedMessage]] = []
    i = 0

    while i < len(segments):
        seg = segments[i]
        if seg.message.tool_calls:
            # Start of a block: assistant with tool_calls.
            block = [seg]
            tool_call_ids = {tc.id for tc in seg.message.tool_calls}
            i += 1
            # Collect following tool results that match.
            while i < len(segments) and segments[i].message.tool_call_id in tool_call_ids:
                block.append(segments[i])
                tool_call_ids.discard(segments[i].message.tool_call_id)
                i += 1
            blocks.append(block)
        else:
            blocks.append([seg])
            i += 1

    return blocks


def _flatten(blocks: list[list[ClassifiedMessage]]) -> list[ClassifiedMessage]:
    """Flatten a list of blocks into a single list."""
    return [seg for block in blocks for seg in block]


def _is_valid_sequence(segments: list[ClassifiedMessage]) -> bool:
    """Validate that tool results immediately follow their tool calls.

    OpenAI/Anthropic APIs require this ordering — violating it causes errors.
    """
    pending_tool_ids: set[str] = set()

    for seg in segments:
        if seg.message.tool_calls:
            pending_tool_ids = {tc.id for tc in seg.message.tool_calls}
        elif seg.message.tool_call_id:
            if seg.message.tool_call_id not in pending_tool_ids:
                return False
            pending_tool_ids.discard(seg.message.tool_call_id)
        else:
            if pending_tool_ids:
                return False  # Non-tool message interrupts a tool sequence.

    return True
