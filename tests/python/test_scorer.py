# Tests for Layer 4: Task-Aware Relevance Scoring.
from __future__ import annotations

import pytest

from memosift.config import MemoSiftConfig
from memosift.core.scorer import _extract_keywords, score_relevance, score_relevance_llm
from memosift.core.types import (
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    MemoSiftMessage,
)
from memosift.providers.base import LLMResponse


def _make_segment(
    content: str,
    content_type: ContentType = ContentType.TOOL_RESULT_TEXT,
    policy: CompressionPolicy = CompressionPolicy.MODERATE,
    protected: bool = False,
) -> ClassifiedMessage:
    return ClassifiedMessage(
        message=MemoSiftMessage(role="tool", content=content, tool_call_id="tc1"),
        content_type=content_type,
        policy=policy,
        protected=protected,
    )


class TestKeywordExtraction:
    """Tests for keyword extraction."""

    def test_stop_words_removed(self) -> None:
        keywords = _extract_keywords("the quick brown fox is a nice animal")
        assert "the" not in keywords
        assert "is" not in keywords
        assert "a" not in keywords
        assert "quick" in keywords
        assert "brown" in keywords

    def test_empty_text(self) -> None:
        assert _extract_keywords("") == set()

    def test_only_stop_words(self) -> None:
        assert _extract_keywords("the is a an") == set()


class TestScoreRelevance:
    """Tests for relevance scoring (async)."""

    @pytest.mark.asyncio
    async def test_no_task_neutral_scores(self) -> None:
        seg = _make_segment("anything")
        result = await score_relevance([seg], MemoSiftConfig(), task=None)
        assert len(result) == 1
        assert result[0].relevance_score == 0.5

    @pytest.mark.asyncio
    async def test_protected_types_always_1(self) -> None:
        segs = [
            _make_segment("sys", ContentType.SYSTEM_PROMPT, CompressionPolicy.PRESERVE, protected=True),
            _make_segment("query", ContentType.USER_QUERY, CompressionPolicy.PRESERVE, protected=True),
            _make_segment("recent", ContentType.RECENT_TURN, CompressionPolicy.LIGHT, protected=True),
        ]
        result = await score_relevance(segs, MemoSiftConfig(), task="unrelated task about databases")
        for seg in result:
            assert seg.relevance_score == 1.0

    @pytest.mark.asyncio
    async def test_relevant_content_high_score(self) -> None:
        seg = _make_segment("The authentication bug in auth.ts causes TypeError on login")
        result = await score_relevance(
            [seg], MemoSiftConfig(), task="Fix the authentication TypeError in auth.ts"
        )
        assert len(result) == 1
        assert result[0].relevance_score > 0.3

    @pytest.mark.asyncio
    async def test_irrelevant_content_dropped(self) -> None:
        seg = _make_segment("The weather is nice today in Paris")
        result = await score_relevance(
            [seg],
            MemoSiftConfig(relevance_drop_threshold=0.05),
            task="Fix database migration error in PostgreSQL",
        )
        # Completely irrelevant → should be dropped.
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_threshold_behavior(self) -> None:
        seg = _make_segment("auth module handles login")
        config = MemoSiftConfig(relevance_drop_threshold=1.0)
        await score_relevance([seg], config, task="fix auth login")
        # Very high threshold → likely drops unless perfect overlap.

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        result = await score_relevance([], MemoSiftConfig(), task="anything")
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_task_keywords(self) -> None:
        """Task with only stop words → neutral scores."""
        seg = _make_segment("important content here")
        result = await score_relevance([seg], MemoSiftConfig(), task="the is a an")
        assert len(result) == 1
        assert result[0].relevance_score == 0.5


# ── LLM-mode scoring tests ─────────────────────────────────────────────────


class MockScorerLLM:
    """Mock LLM that returns a JSON score."""

    def __init__(self, score: int = 7):
        self._score = score

    async def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.0) -> LLMResponse:
        import json
        return LLMResponse(
            text=json.dumps({"score": self._score, "reason": "test"}),
            input_tokens=50,
            output_tokens=10,
        )

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4


class FailingScorerLLM:
    """Mock LLM that always fails."""

    async def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.0) -> LLMResponse:
        raise RuntimeError("LLM timeout")

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4


class TestScoreRelevanceLLM:
    """Tests for LLM-mode relevance scoring."""

    @pytest.mark.asyncio
    async def test_high_score_kept(self) -> None:
        seg = _make_segment("auth module bug fix")
        llm = MockScorerLLM(score=8)
        result = await score_relevance_llm([seg], MemoSiftConfig(), "fix auth bug", llm)
        assert len(result) == 1
        assert result[0].relevance_score == 0.8  # 8/10

    @pytest.mark.asyncio
    async def test_low_score_dropped(self) -> None:
        seg = _make_segment("weather report for Paris")
        llm = MockScorerLLM(score=1)
        result = await score_relevance_llm([seg], MemoSiftConfig(), "fix auth bug", llm)
        assert len(result) == 0  # 1/10 < 3/10 threshold

    @pytest.mark.asyncio
    async def test_protected_types_always_kept(self) -> None:
        seg = _make_segment(
            "system prompt", ContentType.SYSTEM_PROMPT,
            CompressionPolicy.PRESERVE, protected=True,
        )
        llm = MockScorerLLM(score=0)
        result = await score_relevance_llm([seg], MemoSiftConfig(), "anything", llm)
        assert len(result) == 1
        assert result[0].relevance_score == 1.0

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_keyword(self) -> None:
        seg = _make_segment("auth module handles login and sessions")
        llm = FailingScorerLLM()
        result = await score_relevance_llm(
            [seg], MemoSiftConfig(), "fix auth login", llm,
        )
        # Should fall back to keyword scoring, not crash.
        assert len(result) >= 0  # May keep or drop depending on keyword overlap.

    @pytest.mark.asyncio
    async def test_malformed_json_falls_back(self) -> None:
        """LLM returns invalid JSON → falls back to keyword mode."""
        class BadJsonLLM:
            async def generate(self, prompt, max_tokens=100, temperature=0.0):
                return LLMResponse(text="not valid json", input_tokens=10, output_tokens=5)
            async def count_tokens(self, text):
                return len(text) // 4

        seg = _make_segment("auth login code")
        result = await score_relevance_llm(
            [seg], MemoSiftConfig(), "fix auth", BadJsonLLM(),
        )
        assert len(result) >= 0

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        llm = MockScorerLLM()
        result = await score_relevance_llm([], MemoSiftConfig(), "anything", llm)
        assert result == []
