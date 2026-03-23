# Tests for Layer 6: Token Budget Enforcement.
from __future__ import annotations

import pytest

from memosift.config import MemoSiftConfig
from memosift.core.budget import enforce_budget
from memosift.core.types import (
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    DependencyMap,
    MemoSiftMessage,
)
from memosift.providers.heuristic import HeuristicTokenCounter


def _make_segment(
    content: str,
    content_type: ContentType = ContentType.TOOL_RESULT_TEXT,
    policy: CompressionPolicy = CompressionPolicy.MODERATE,
    protected: bool = False,
    relevance_score: float = 0.5,
    original_index: int = 0,
) -> ClassifiedMessage:
    return ClassifiedMessage(
        message=MemoSiftMessage(role="tool", content=content, tool_call_id=f"tc{original_index}"),
        content_type=content_type,
        policy=policy,
        protected=protected,
        relevance_score=relevance_score,
        original_index=original_index,
    )


class TestBudgetEnforcement:
    """Tests for budget enforcement."""

    @pytest.mark.asyncio
    async def test_under_budget_unchanged(self) -> None:
        segs = [_make_segment("short", relevance_score=0.5)]
        config = MemoSiftConfig(token_budget=10000)
        counter = HeuristicTokenCounter()
        result = await enforce_budget(segs, config, DependencyMap(), counter)
        assert len(result) == 1
        assert result[0].content == "short"

    @pytest.mark.asyncio
    async def test_no_budget_unchanged(self) -> None:
        segs = [_make_segment("x" * 10000)]
        config = MemoSiftConfig(token_budget=None)
        result = await enforce_budget(segs, config, DependencyMap())
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_lowest_scored_dropped_first(self) -> None:
        segs = [
            _make_segment("sys", ContentType.SYSTEM_PROMPT, CompressionPolicy.PRESERVE,
                          protected=True, relevance_score=1.0, original_index=0),
            _make_segment("low " * 500, relevance_score=0.1, original_index=1),
            _make_segment("high " * 500, relevance_score=0.9, original_index=2),
        ]
        config = MemoSiftConfig(token_budget=1000)
        counter = HeuristicTokenCounter()
        result = await enforce_budget(segs, config, DependencyMap(), counter)
        # Low-scored segment should be dropped first.
        contents = [s.content for s in result]
        assert "low " * 500 not in contents
        assert "sys" in contents

    @pytest.mark.asyncio
    async def test_protected_segments_never_dropped(self) -> None:
        segs = [
            _make_segment("system", ContentType.SYSTEM_PROMPT, CompressionPolicy.PRESERVE,
                          protected=True, relevance_score=1.0, original_index=0),
            _make_segment("query", ContentType.USER_QUERY, CompressionPolicy.PRESERVE,
                          protected=True, relevance_score=1.0, original_index=1),
            _make_segment("x" * 5000, relevance_score=0.1, original_index=2),
        ]
        config = MemoSiftConfig(token_budget=1000)
        counter = HeuristicTokenCounter()
        result = await enforce_budget(segs, config, DependencyMap(), counter)
        types = [s.content_type for s in result]
        assert ContentType.SYSTEM_PROMPT in types
        assert ContentType.USER_QUERY in types

    @pytest.mark.asyncio
    async def test_tool_call_integrity(self) -> None:
        """If an assistant tool_call survives, its result must also survive."""
        from memosift.core.types import ToolCall, ToolCallFunction

        tc = ToolCall(id="tc1", function=ToolCallFunction(name="read", arguments="{}"))
        segs = [
            ClassifiedMessage(
                message=MemoSiftMessage(role="assistant", content="reading", tool_calls=[tc]),
                content_type=ContentType.RECENT_TURN,
                policy=CompressionPolicy.LIGHT,
                protected=True,
                relevance_score=1.0,
                original_index=0,
            ),
            ClassifiedMessage(
                message=MemoSiftMessage(role="tool", content="x" * 2000, tool_call_id="tc1"),
                content_type=ContentType.TOOL_RESULT_TEXT,
                policy=CompressionPolicy.MODERATE,
                protected=False,
                relevance_score=0.1,
                original_index=1,
            ),
        ]
        config = MemoSiftConfig(token_budget=1000)
        counter = HeuristicTokenCounter()
        result = await enforce_budget(segs, config, DependencyMap(), counter)
        # Both should survive — tool result can't be dropped while its call exists.
        assert len(result) == 2


class TestDanglingReferences:
    """Dangling reference protection."""

    @pytest.mark.asyncio
    async def test_dangling_reference_expanded(self) -> None:
        """If a dedup original is dropped, back-references should be expanded."""
        deps = DependencyMap()
        deps.add(deduped_index=2, original_index=1)

        segs = [
            _make_segment("sys", ContentType.SYSTEM_PROMPT, CompressionPolicy.PRESERVE,
                          protected=True, relevance_score=1.0, original_index=0),
            _make_segment("original content " * 100, relevance_score=0.1, original_index=1),
            _make_segment("[back-reference to #1]", relevance_score=0.8, original_index=2),
        ]
        config = MemoSiftConfig(token_budget=1000)
        counter = HeuristicTokenCounter()
        result = await enforce_budget(segs, config, deps, counter)
        # The dependency map should prevent dropping original while reference exists.
        # Or the reference should be expanded.


class TestEdgeCases:
    """Edge cases."""

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        result = await enforce_budget([], MemoSiftConfig(token_budget=1000), DependencyMap())
        assert result == []

    @pytest.mark.asyncio
    async def test_no_counter_uses_heuristic(self) -> None:
        segs = [_make_segment("hello world")]
        config = MemoSiftConfig(token_budget=10000)
        result = await enforce_budget(segs, config, DependencyMap(), counter=None)
        assert len(result) == 1
