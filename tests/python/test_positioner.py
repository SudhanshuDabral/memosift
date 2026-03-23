# Tests for Layer 5: Position Optimization.
from __future__ import annotations

import pytest

from memosift.config import MemoSiftConfig
from memosift.core.positioner import _build_blocks, _is_valid_sequence, optimize_position
from memosift.core.types import (
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    MemoSiftMessage,
    ToolCall,
    ToolCallFunction,
)


def _make_segment(
    content: str,
    content_type: ContentType,
    role: str = "assistant",
    tool_calls: list[ToolCall] | None = None,
    tool_call_id: str | None = None,
) -> ClassifiedMessage:
    return ClassifiedMessage(
        message=MemoSiftMessage(
            role=role, content=content, tool_calls=tool_calls, tool_call_id=tool_call_id,
        ),
        content_type=content_type,
        policy=CompressionPolicy.MODERATE,
    )


class TestDisabledByDefault:
    """Position optimization is disabled by default."""

    def test_disabled_returns_unchanged(self) -> None:
        segs = [
            _make_segment("old", ContentType.OLD_CONVERSATION),
            _make_segment("system", ContentType.SYSTEM_PROMPT, role="system"),
        ]
        config = MemoSiftConfig(reorder_segments=False)
        result = optimize_position(segs, config)
        assert [s.content for s in result] == ["old", "system"]


class TestPositionOptimization:
    """When enabled, segments are reordered by attention priority."""

    def test_system_prompt_moves_to_beginning(self) -> None:
        segs = [
            _make_segment("old stuff", ContentType.OLD_CONVERSATION),
            _make_segment("system", ContentType.SYSTEM_PROMPT, role="system"),
            _make_segment("query", ContentType.USER_QUERY, role="user"),
        ]
        config = MemoSiftConfig(reorder_segments=True)
        result = optimize_position(segs, config)
        assert result[0].content_type == ContentType.SYSTEM_PROMPT
        assert result[-1].content_type == ContentType.USER_QUERY

    def test_error_trace_in_beginning(self) -> None:
        segs = [
            _make_segment("old", ContentType.OLD_CONVERSATION),
            _make_segment("error", ContentType.ERROR_TRACE, role="tool"),
            _make_segment("recent", ContentType.RECENT_TURN),
        ]
        config = MemoSiftConfig(reorder_segments=True)
        result = optimize_position(segs, config)
        assert result[0].content_type == ContentType.ERROR_TRACE

    def test_recent_turns_at_end(self) -> None:
        segs = [
            _make_segment("recent", ContentType.RECENT_TURN),
            _make_segment("old", ContentType.OLD_CONVERSATION),
            _make_segment("system", ContentType.SYSTEM_PROMPT, role="system"),
        ]
        config = MemoSiftConfig(reorder_segments=True)
        result = optimize_position(segs, config)
        assert result[-1].content_type == ContentType.RECENT_TURN


class TestBlockBuilding:
    """Tool call blocks are kept atomic."""

    def test_tool_call_block(self) -> None:
        tc = ToolCall(id="tc1", function=ToolCallFunction(name="read", arguments="{}"))
        segs = [
            _make_segment("reading", ContentType.RECENT_TURN, tool_calls=[tc]),
            _make_segment("result", ContentType.TOOL_RESULT_TEXT, role="tool", tool_call_id="tc1"),
            _make_segment("standalone", ContentType.OLD_CONVERSATION),
        ]
        blocks = _build_blocks(segs)
        assert len(blocks) == 2
        assert len(blocks[0]) == 2  # assistant + tool result
        assert len(blocks[1]) == 1  # standalone


class TestSequenceValidation:
    """Invalid sequences abort reordering."""

    def test_valid_sequence(self) -> None:
        tc = ToolCall(id="tc1", function=ToolCallFunction(name="read", arguments="{}"))
        segs = [
            _make_segment("call", ContentType.RECENT_TURN, tool_calls=[tc]),
            _make_segment("result", ContentType.TOOL_RESULT_TEXT, role="tool", tool_call_id="tc1"),
        ]
        assert _is_valid_sequence(segs) is True

    def test_orphaned_tool_result(self) -> None:
        segs = [
            _make_segment("orphan", ContentType.TOOL_RESULT_TEXT, role="tool", tool_call_id="tc999"),
        ]
        assert _is_valid_sequence(segs) is False


class TestEdgeCases:
    """Edge cases."""

    def test_empty_input(self) -> None:
        result = optimize_position([], MemoSiftConfig(reorder_segments=True))
        assert result == []

    def test_single_segment(self) -> None:
        segs = [_make_segment("only", ContentType.SYSTEM_PROMPT, role="system")]
        result = optimize_position(segs, MemoSiftConfig(reorder_segments=True))
        assert len(result) == 1
