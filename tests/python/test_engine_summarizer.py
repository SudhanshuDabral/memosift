# Tests for Engine D: Abstractive Summarization.
from __future__ import annotations

import pytest

from memosift.config import MemoSiftConfig
from memosift.core.engines.summarizer import (
    _extract_critical_facts,
    _is_valid_summary,
    summarize_segments,
)
from memosift.core.types import (
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    MemoSiftMessage,
)
from memosift.providers.base import LLMResponse


class MockLLMProvider:
    """Mock LLM that returns a shortened version of input."""

    def __init__(self, response_text: str = "Summary of content."):
        self._response_text = response_text
        self.call_count = 0

    async def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.0) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(text=self._response_text, input_tokens=100, output_tokens=20)

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4


class FailingLLMProvider:
    """Mock LLM that always raises."""

    async def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.0) -> LLMResponse:
        raise RuntimeError("LLM unavailable")

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4


def _make_segment(
    content: str,
    policy: CompressionPolicy = CompressionPolicy.AGGRESSIVE,
    content_type: ContentType = ContentType.OLD_CONVERSATION,
) -> ClassifiedMessage:
    return ClassifiedMessage(
        message=MemoSiftMessage(role="assistant", content=content),
        content_type=content_type,
        policy=policy,
    )


# ── Fact extraction tests ───────────────────────────────────────────────────


class TestExtractCriticalFacts:

    def test_extracts_file_paths(self) -> None:
        text = "The bug is in src/auth.ts and also affects lib/db.py"
        facts = _extract_critical_facts(text)
        assert "src/auth.ts" in facts["file_paths"]
        assert "lib/db.py" in facts["file_paths"]

    def test_extracts_error_messages(self) -> None:
        text = "Got TypeError: Cannot read properties of undefined (reading 'userId')"
        facts = _extract_critical_facts(text)
        assert len(facts["error_msgs"]) >= 1
        assert any("TypeError" in e for e in facts["error_msgs"])

    def test_extracts_line_references(self) -> None:
        text = "The error is on line 47 at auth.ts:47"
        facts = _extract_critical_facts(text)
        assert len(facts["line_refs"]) >= 1

    def test_empty_text(self) -> None:
        facts = _extract_critical_facts("")
        assert facts["file_paths"] == set()
        assert facts["error_msgs"] == set()
        assert facts["line_refs"] == set()


# ── Summary validation tests ───────────────────────────────────────────────


class TestIsValidSummary:

    def test_valid_summary(self) -> None:
        original = "The bug in src/auth.ts causes issues. " * 10
        summary = "Bug found in src/auth.ts causing issues."
        facts = _extract_critical_facts(original)
        assert _is_valid_summary(summary, original, facts) is True

    def test_rejects_empty_summary(self) -> None:
        original = "Content. " * 20
        facts = _extract_critical_facts(original)
        assert _is_valid_summary("", original, facts) is False

    def test_rejects_too_short_summary(self) -> None:
        original = "Content. " * 20
        facts = _extract_critical_facts(original)
        assert _is_valid_summary("Short.", original, facts) is False

    def test_rejects_longer_summary(self) -> None:
        original = "Content."
        facts = _extract_critical_facts(original)
        assert _is_valid_summary("Much longer content here.", original, facts) is False

    def test_rejects_missing_file_path(self) -> None:
        original = "The bug is in src/auth.ts and needs fixing. " * 5
        summary = "There's a bug that needs fixing in the auth module."
        facts = _extract_critical_facts(original)
        # Summary dropped "src/auth.ts" → should be rejected.
        assert _is_valid_summary(summary, original, facts) is False

    def test_rejects_missing_error_type(self) -> None:
        original = "Got TypeError: Cannot read properties. " * 5
        summary = "There was an error reading properties of an object."
        facts = _extract_critical_facts(original)
        # Summary dropped "TypeError" → should be rejected.
        assert _is_valid_summary(summary, original, facts) is False

    def test_accepts_summary_with_all_facts(self) -> None:
        original = "Found TypeError in src/auth.ts at line 47. The fix is to add null check. " * 3
        summary = "TypeError in src/auth.ts at line 47. Fix: add null check."
        facts = _extract_critical_facts(original)
        assert _is_valid_summary(summary, original, facts) is True


# ── Summarize segments tests ───────────────────────────────────────────────


class TestSummarizeSegments:

    @pytest.mark.asyncio
    async def test_summarizes_aggressive_policy(self) -> None:
        long_content = "The agent investigated the auth module. " * 20
        seg = _make_segment(long_content)
        config = MemoSiftConfig(enable_summarization=True)
        llm = MockLLMProvider("Summary of the auth investigation.")
        result = await summarize_segments([seg], config, llm)
        assert len(result) == 1
        assert result[0].content == "Summary of the auth investigation."
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self) -> None:
        seg = _make_segment("Long content " * 30)
        config = MemoSiftConfig(enable_summarization=False)
        llm = MockLLMProvider()
        result = await summarize_segments([seg], config, llm)
        assert result[0].content == seg.content
        assert llm.call_count == 0

    @pytest.mark.asyncio
    async def test_skips_preserve_policy(self) -> None:
        seg = _make_segment("System prompt", CompressionPolicy.PRESERVE, ContentType.SYSTEM_PROMPT)
        config = MemoSiftConfig(enable_summarization=True)
        llm = MockLLMProvider()
        result = await summarize_segments([seg], config, llm)
        assert result[0].content == "System prompt"
        assert llm.call_count == 0

    @pytest.mark.asyncio
    async def test_skips_short_content(self) -> None:
        seg = _make_segment("Short.")
        config = MemoSiftConfig(enable_summarization=True)
        llm = MockLLMProvider()
        result = await summarize_segments([seg], config, llm)
        assert result[0].content == "Short."
        assert llm.call_count == 0

    @pytest.mark.asyncio
    async def test_keeps_original_if_summary_longer(self) -> None:
        content = "Brief original."
        seg = _make_segment(content * 20)
        config = MemoSiftConfig(enable_summarization=True)
        llm = MockLLMProvider("A" * (len(content) * 20 + 100))
        result = await summarize_segments([seg], config, llm)
        assert result[0].content == content * 20

    @pytest.mark.asyncio
    async def test_handles_llm_failure(self) -> None:
        seg = _make_segment("Content to summarize. " * 20)
        config = MemoSiftConfig(enable_summarization=True)
        llm = FailingLLMProvider()
        result = await summarize_segments([seg], config, llm)
        assert result[0].content == seg.content

    @pytest.mark.asyncio
    async def test_preserves_metadata(self) -> None:
        msg = MemoSiftMessage(
            role="assistant",
            content="Long reasoning here. " * 20,
            metadata={"framework": "test"},
        )
        seg = ClassifiedMessage(
            message=msg,
            content_type=ContentType.ASSISTANT_REASONING,
            policy=CompressionPolicy.AGGRESSIVE,
        )
        config = MemoSiftConfig(enable_summarization=True)
        llm = MockLLMProvider("Summary of the long reasoning process.")
        result = await summarize_segments([seg], config, llm)
        assert result[0].message.metadata == {"framework": "test"}

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        config = MemoSiftConfig(enable_summarization=True)
        llm = MockLLMProvider()
        result = await summarize_segments([], config, llm)
        assert result == []

    @pytest.mark.asyncio
    async def test_rejects_summary_missing_file_path(self) -> None:
        """Summary that drops a file path should be rejected, original kept."""
        content = "The bug is in src/auth.ts and we need to fix the null check. " * 5
        seg = _make_segment(content)
        config = MemoSiftConfig(enable_summarization=True)
        # Summary deliberately omits src/auth.ts
        llm = MockLLMProvider("The auth module has a null check bug that needs fixing.")
        result = await summarize_segments([seg], config, llm)
        # Should keep original because summary dropped the file path.
        assert result[0].content == content

    @pytest.mark.asyncio
    async def test_accepts_summary_preserving_facts(self) -> None:
        """Summary preserving all critical facts should be accepted."""
        content = "The agent found a TypeError in src/auth.ts at line 47. " * 5
        seg = _make_segment(content)
        config = MemoSiftConfig(enable_summarization=True)
        llm = MockLLMProvider("TypeError found in src/auth.ts at line 47.")
        result = await summarize_segments([seg], config, llm)
        assert result[0].content == "TypeError found in src/auth.ts at line 47."

    @pytest.mark.asyncio
    async def test_rejects_empty_summary_from_llm(self) -> None:
        """LLM returning empty/whitespace should be rejected."""
        seg = _make_segment("Content to summarize. " * 20)
        config = MemoSiftConfig(enable_summarization=True)
        llm = MockLLMProvider("   ")
        result = await summarize_segments([seg], config, llm)
        assert result[0].content == seg.content
