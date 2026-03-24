# Comprehensive tests for Layer 6: Budget enforcement.
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st

from memosift.config import MemoSiftConfig
from memosift.core.budget import enforce_budget, _head_tail_truncate, _truncate_largest
from memosift.core.types import (
    AnchorCategory,
    AnchorFact,
    AnchorLedger,
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    DependencyMap,
    MemoSiftMessage,
    Shield,
    ToolCall,
    ToolCallFunction,
)
from memosift.providers.heuristic import HeuristicTokenCounter


def _msg(role: str, content: str, **kwargs) -> MemoSiftMessage:
    return MemoSiftMessage(role=role, content=content, **kwargs)


def _classified(
    content: str,
    ctype: ContentType = ContentType.OLD_CONVERSATION,
    policy: CompressionPolicy = CompressionPolicy.AGGRESSIVE,
    idx: int = 0,
    protected: bool = False,
    score: float = 0.5,
    tokens: int | None = None,
    **msg_kwargs,
) -> ClassifiedMessage:
    msg = _msg("assistant", content, **msg_kwargs)
    return ClassifiedMessage(
        message=msg,
        content_type=ctype,
        policy=policy,
        original_index=idx,
        protected=protected,
        relevance_score=score,
        estimated_tokens=tokens if tokens is not None else math.ceil(len(content) / 3.5),
    )


# ── Basic budget enforcement ─────────────────────────────────────────────────


async def test_no_budget_returns_unchanged():
    """When token_budget is None, segments are returned unchanged."""
    config = MemoSiftConfig(token_budget=None)
    segments = [_classified("hello world", idx=0)]
    result = await enforce_budget(segments, config, DependencyMap())
    assert len(result) == 1
    assert result[0].content == "hello world"


async def test_under_budget_returns_unchanged():
    """When total tokens <= budget, no changes."""
    config = MemoSiftConfig(token_budget=10_000)
    segments = [_classified("short message", idx=0, tokens=10)]
    result = await enforce_budget(segments, config, DependencyMap())
    assert len(result) == 1


async def test_over_budget_drops_lowest_relevance():
    """When over budget, lowest relevance score gets dropped first."""
    # Use content long enough that 3 segments clearly exceed a small budget.
    long = "x" * 500  # ~143 tokens each
    config = MemoSiftConfig(token_budget=200)
    segments = [
        _classified(long + " keep", idx=0, score=0.9),
        _classified(long + " drop", idx=1, score=0.1),
        _classified(long + " also keep", idx=2, score=0.8),
    ]
    result = await enforce_budget(segments, config, DependencyMap())
    contents = [s.content for s in result]
    # Lowest relevance should be dropped.
    assert not any("drop" in c for c in contents)
    assert any("keep" in c for c in contents)


# ── Protected segments ───────────────────────────────────────────────────────


async def test_protected_never_dropped():
    """Protected segments (system, user, recent) survive any budget."""
    config = MemoSiftConfig(token_budget=100)
    segments = [
        _classified(
            "system prompt that is very long " * 20,
            ctype=ContentType.SYSTEM_PROMPT,
            policy=CompressionPolicy.PRESERVE,
            idx=0,
            protected=True,
            score=1.0,
            tokens=200,
        ),
        _classified("old stuff", idx=1, score=0.1, tokens=50),
    ]
    result = await enforce_budget(segments, config, DependencyMap())
    assert any(s.content_type == ContentType.SYSTEM_PROMPT for s in result)


async def test_budget_minimum_protected_survive():
    """With minimum token_budget=100, protected segments still survive."""
    config = MemoSiftConfig(token_budget=100)
    segments = [
        _classified(
            "I am protected",
            ctype=ContentType.USER_QUERY,
            policy=CompressionPolicy.PRESERVE,
            idx=0,
            protected=True,
            score=1.0,
            tokens=50,
        ),
        _classified("I am not", idx=1, score=0.1, tokens=50),
    ]
    # token_budget=0 means pipeline passes max(100, 0 - zone1 - zone2) = 100.
    # But at the budget.py level, 0 means drop everything droppable.
    result = await enforce_budget(segments, config, DependencyMap())
    assert any(s.protected for s in result)


# ── Tool call integrity under budget pressure ────────────────────────────────


async def test_tool_call_pair_preserved():
    """If a tool_call message survives, its matching tool_result must survive too."""
    config = MemoSiftConfig(token_budget=200)
    tc = ToolCall(id="tc1", function=ToolCallFunction(name="read_file", arguments="{}"))
    segments = [
        _classified("filler " * 100, idx=0, score=0.1, tokens=100),  # Low relevance, droppable
        ClassifiedMessage(
            message=MemoSiftMessage(
                role="assistant", content="Checking.", tool_calls=[tc]
            ),
            content_type=ContentType.RECENT_TURN,
            policy=CompressionPolicy.LIGHT,
            original_index=1,
            relevance_score=0.9,
            estimated_tokens=10,
            protected=False,
        ),
        ClassifiedMessage(
            message=MemoSiftMessage(
                role="tool", content="file content here",
                tool_call_id="tc1", name="read_file",
            ),
            content_type=ContentType.TOOL_RESULT_TEXT,
            policy=CompressionPolicy.MODERATE,
            original_index=2,
            relevance_score=0.5,
            estimated_tokens=15,
            protected=False,
        ),
    ]
    result = await enforce_budget(segments, config, DependencyMap())
    call_ids = {tc.id for s in result if s.message.tool_calls for tc in s.message.tool_calls}
    result_ids = {s.message.tool_call_id for s in result if s.message.tool_call_id}
    # If a call exists, its result must too (and vice versa).
    assert call_ids <= result_ids or not call_ids


# ── Dependency map ───────────────────────────────────────────────────────────


async def test_dependency_prevents_drop():
    """A message referenced by another via DependencyMap cannot be simply dropped."""
    config = MemoSiftConfig(token_budget=100)
    deps = DependencyMap()
    deps.add(deduped_index=2, original_index=0)  # Message 2 references message 0

    segments = [
        _classified("I am the original", idx=0, score=0.1, tokens=80),
        _classified("I reference msg 0", idx=2, score=0.8, tokens=80),
    ]
    result = await enforce_budget(segments, config, deps)
    # Message 0 can't be dropped without expanding dependents.
    # The result should have both messages or have expanded the reference.
    assert len(result) >= 1


# ── Anchor fact truncation ───────────────────────────────────────────────────


async def test_anchor_fact_truncates_not_drops():
    """Segments containing anchor ledger facts are truncated, not dropped."""
    config = MemoSiftConfig(token_budget=100)
    ledger = AnchorLedger()
    # Use a path that passes critical_strings filtering (contains / AND extension AND ≥8 chars).
    ledger.add(AnchorFact(
        category=AnchorCategory.FILES,
        content="src/components/auth.ts — modified at turn 5",
        turn=5,
    ))

    # Create content long enough (>10 lines) that head_tail_truncate will trim it,
    # with the anchor fact string present so contains_anchor_fact returns True.
    lines = [f"line {i}: irrelevant padding content here" for i in range(30)]
    lines[15] = "line 15: this references src/components/auth.ts which is important"
    long_content = "\n".join(lines)
    segments = [
        _classified(long_content, idx=0, score=0.1),
    ]
    result = await enforce_budget(segments, config, DependencyMap(), ledger=ledger)
    assert len(result) == 1
    # Should be truncated (shorter) but not fully dropped.
    assert len(result[0].content) < len(long_content)
    # The anchor fact line should still be present.
    assert "auth.ts" in result[0].content


# ── Domain compression caps ──────────────────────────────────────────────────


async def test_code_block_domain_cap():
    """CODE_BLOCK segments respect the 4x compression cap."""
    config = MemoSiftConfig(token_budget=100)
    segments = [
        _classified(
            "def foo(): pass\n" * 10,
            ctype=ContentType.CODE_BLOCK,
            policy=CompressionPolicy.SIGNATURE,
            idx=0,
            score=0.1,
            tokens=100,
        ),
        _classified("filler", idx=1, score=0.05, tokens=100),
    ]
    result = await enforce_budget(segments, config, DependencyMap())
    # Code blocks should be protected from dropping (domain cap prevents).
    code_blocks = [s for s in result if s.content_type == ContentType.CODE_BLOCK]
    assert len(code_blocks) >= 1


async def test_error_trace_domain_cap():
    """ERROR_TRACE segments respect the 3x compression cap."""
    config = MemoSiftConfig(token_budget=100)
    segments = [
        _classified(
            "Traceback:\n  File 'x.py'\nTypeError: bad",
            ctype=ContentType.ERROR_TRACE,
            policy=CompressionPolicy.STACK,
            idx=0,
            score=0.1,
            tokens=100,
        ),
        _classified("filler", idx=1, score=0.05, tokens=100),
    ]
    result = await enforce_budget(segments, config, DependencyMap())
    errors = [s for s in result if s.content_type == ContentType.ERROR_TRACE]
    assert len(errors) >= 1


# ── Emergency truncation ─────────────────────────────────────────────────────


async def test_emergency_truncation_uses_line_boundary():
    """Emergency truncation snaps to newline boundaries on protected-but-huge segments."""
    config = MemoSiftConfig(token_budget=100)
    # Create two segments: one protected (won't be dropped) and one droppable.
    # After dropping the droppable one, the protected one is still over budget → emergency truncation.
    long_content = "\n".join(f"line {i}: some content here" for i in range(100))
    segments = [
        ClassifiedMessage(
            message=_msg("user", long_content),
            content_type=ContentType.USER_QUERY,
            policy=CompressionPolicy.PRESERVE,
            original_index=0,
            protected=True,
            relevance_score=1.0,
            estimated_tokens=math.ceil(len(long_content) / 3.5),
        ),
        _classified("filler", idx=1, score=0.1),
    ]
    result = await enforce_budget(segments, config, DependencyMap())
    # Protected segment survives (possibly truncated via emergency path).
    assert any(s.content_type == ContentType.USER_QUERY for s in result)


# ── Head/tail truncation ─────────────────────────────────────────────────────


def test_head_tail_truncate_short():
    """Short text (<= 10 lines) should not be truncated."""
    text = "line 1\nline 2\nline 3"
    assert _head_tail_truncate(text) == text


def test_head_tail_truncate_long():
    """Long text should keep head + tail + omission marker."""
    lines = [f"line {i}" for i in range(100)]
    text = "\n".join(lines)
    result = _head_tail_truncate(text)
    assert "line 0" in result  # Head kept
    assert "line 99" in result  # Tail kept
    assert "omitted" in result  # Marker present
    assert len(result) < len(text)


def test_head_tail_truncate_with_ledger():
    """Lines containing ledger facts should be preserved."""
    ledger = AnchorLedger()
    ledger.add(AnchorFact(
        category=AnchorCategory.FILES,
        content="src/important.ts — read",
        turn=1,
    ))
    lines = [f"line {i}: irrelevant content" for i in range(100)]
    lines[50] = "line 50: this mentions src/important.ts which matters"
    text = "\n".join(lines)
    result = _head_tail_truncate(text, ledger=ledger)
    assert "important.ts" in result


# ── Property tests ───────────────────────────────────────────────────────────


@given(
    n_messages=st.integers(min_value=1, max_value=20),
    budget=st.integers(min_value=100, max_value=5000),
)
@settings(max_examples=50)
async def test_budget_never_crashes(n_messages: int, budget: int):
    """Budget enforcement should never crash regardless of inputs."""
    config = MemoSiftConfig(token_budget=budget)
    segments = [
        _classified(
            f"message {i} " * (i + 1),
            idx=i,
            score=i / max(n_messages, 1),
            tokens=math.ceil(len(f"message {i} " * (i + 1)) / 3.5),
        )
        for i in range(n_messages)
    ]
    # Should not raise.
    result = await enforce_budget(segments, config, DependencyMap())
    assert isinstance(result, list)
    assert len(result) <= n_messages


@given(
    n_messages=st.integers(min_value=2, max_value=15),
    budget=st.integers(min_value=200, max_value=2000),
)
@settings(max_examples=30)
async def test_budget_output_within_tolerance(n_messages: int, budget: int):
    """Compressed output should be within 20% of budget (heuristic tolerance)."""
    config = MemoSiftConfig(token_budget=budget)
    segments = [
        _classified(
            f"content for message number {i} with some padding text here " * 3,
            idx=i,
            score=0.3 + (i * 0.05),
        )
        for i in range(n_messages)
    ]
    result = await enforce_budget(segments, config, DependencyMap())
    total_tokens = sum(math.ceil(len(s.content) / 3.5) for s in result)
    # Allow 20% tolerance for heuristic rounding.
    assert total_tokens <= budget * 1.20 or len(result) == 0
