# Integration tests for the pipeline orchestrator.
from __future__ import annotations

import pytest

from memosift import compress, MemoSiftConfig, MemoSiftMessage
from memosift.core.pipeline import validate_tool_call_integrity
from memosift.core.types import ToolCall, ToolCallFunction
from memosift.providers.base import LLMResponse
from conftest import messages_from_dicts


class TestPipelineSmoke:
    """Basic pipeline smoke tests."""

    @pytest.mark.asyncio
    async def test_compress_returns_messages_and_report(self, sample_messages) -> None:
        compressed, report = await compress(sample_messages)
        assert isinstance(compressed, list)
        assert len(compressed) > 0
        assert report.original_tokens > 0
        assert report.compressed_tokens > 0
        assert report.compression_ratio >= 0.8  # Small inputs may have slight overhead.

    @pytest.mark.asyncio
    async def test_compress_with_default_config(self, sample_messages) -> None:
        compressed, report = await compress(sample_messages)
        # System prompt preserved.
        assert compressed[0].role == "system"
        assert compressed[0].content == sample_messages[0].content
        # Last user message preserved.
        user_msgs = [m for m in compressed if m.role == "user"]
        assert user_msgs[-1].content == sample_messages[-1].content

    @pytest.mark.asyncio
    async def test_compress_empty_input(self) -> None:
        compressed, report = await compress([])
        assert compressed == []
        assert report.original_tokens == 0


class TestPipelineInvariants:
    """Pipeline invariants that must always hold."""

    @pytest.mark.asyncio
    async def test_system_prompt_preserved(self, sample_messages) -> None:
        compressed, _ = await compress(sample_messages)
        assert compressed[0].content == sample_messages[0].content

    @pytest.mark.asyncio
    async def test_last_user_message_preserved(self, sample_messages) -> None:
        compressed, _ = await compress(sample_messages)
        original_last_user = [m for m in sample_messages if m.role == "user"][-1]
        compressed_last_user = [m for m in compressed if m.role == "user"][-1]
        assert compressed_last_user.content == original_last_user.content

    @pytest.mark.asyncio
    async def test_message_count_non_increasing(self, sample_messages) -> None:
        compressed, _ = await compress(sample_messages)
        assert len(compressed) <= len(sample_messages)

    @pytest.mark.asyncio
    async def test_tool_call_integrity(self, sample_messages) -> None:
        compressed, _ = await compress(sample_messages)
        tool_call_ids: set[str] = set()
        tool_result_ids: set[str] = set()
        for m in compressed:
            if m.tool_calls:
                for tc in m.tool_calls:
                    tool_call_ids.add(tc.id)
            if m.tool_call_id:
                tool_result_ids.add(m.tool_call_id)
        assert tool_call_ids == tool_result_ids, "Orphaned tool calls or results!"


class TestPipelineWithBudget:
    """Tests with token budget enforcement."""

    @pytest.mark.asyncio
    async def test_budget_respected(self) -> None:
        """Create a large conversation and verify budget compliance."""
        msgs = [
            MemoSiftMessage(role="system", content="You are helpful."),
        ]
        # Add many old turns.
        for i in range(20):
            msgs.append(MemoSiftMessage(role="user", content=f"Question {i}: " + "x" * 200))
            msgs.append(MemoSiftMessage(role="assistant", content=f"Answer {i}: " + "y" * 200))
        msgs.append(MemoSiftMessage(role="user", content="Final question"))

        config = MemoSiftConfig(token_budget=2000, recent_turns=1)
        compressed, report = await compress(msgs, config=config)

        # Budget should be approximately respected (heuristic counting ±15%).
        total_chars = sum(len(m.content) for m in compressed)
        approx_tokens = total_chars / 3.5
        assert approx_tokens < 2000 * 1.15  # Allow 15% tolerance.


class TestPipelineWithTask:
    """Tests with task-aware scoring."""

    @pytest.mark.asyncio
    async def test_relevant_content_kept(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="You are a coding assistant."),
            MemoSiftMessage(role="user", content="What's the weather?"),
            MemoSiftMessage(role="assistant", content="I can't check weather."),
            MemoSiftMessage(role="user", content="Fix the auth bug in login.ts"),
            MemoSiftMessage(role="assistant", content="Looking at the auth module."),
            MemoSiftMessage(role="user", content="Fix the TypeError in authenticate()"),
        ]
        config = MemoSiftConfig(recent_turns=1)
        compressed, report = await compress(
            msgs, config=config, task="Fix TypeError in auth"
        )
        # System prompt and last user message always preserved.
        assert compressed[0].content == msgs[0].content
        assert compressed[-1].content == msgs[-1].content


class TestCompressVector:
    """Test against the compress-001 test vector."""

    @pytest.mark.asyncio
    async def test_compress_001(self, compress_vector: dict) -> None:
        msgs = messages_from_dicts(compress_vector["input"])
        config = MemoSiftConfig(**compress_vector["config"])

        compressed, report = await compress(msgs, config=config)
        invariants = compress_vector["expected_invariants"]

        # System prompt preserved.
        if invariants["system_prompt_preserved"]:
            assert compressed[0].content == msgs[0].content

        # Last user message preserved.
        if invariants["last_user_message_preserved"]:
            last_user = [m for m in compressed if m.role == "user"][-1]
            original_last = [m for m in msgs if m.role == "user"][-1]
            assert last_user.content == original_last.content

        # Tool call integrity.
        if invariants["tool_call_integrity"]:
            call_ids: set[str] = set()
            result_ids: set[str] = set()
            for m in compressed:
                if m.tool_calls:
                    for tc in m.tool_calls:
                        call_ids.add(tc.id)
                if m.tool_call_id:
                    result_ids.add(m.tool_call_id)
            assert call_ids == result_ids

        # Budget compliance.
        if invariants["compressed_tokens_within_budget"]:
            total_chars = sum(len(m.content) for m in compressed)
            approx_tokens = total_chars / 3.5
            assert approx_tokens < config.token_budget * 1.15


class TestPipelineReport:
    """Verify the report is populated correctly."""

    @pytest.mark.asyncio
    async def test_report_has_layers(self, sample_messages) -> None:
        _, report = await compress(sample_messages)
        assert len(report.layers) > 0
        # Should have entries for classifier, dedup, engines, scorer, positioner, budget.
        layer_names = [l.name for l in report.layers]
        assert "classifier" in layer_names

    @pytest.mark.asyncio
    async def test_report_has_segment_counts(self, sample_messages) -> None:
        _, report = await compress(sample_messages)
        assert len(report.segment_counts) > 0
        # System prompt is in Zone 1 (not compressed), so it won't appear
        # in segment_counts. Check for Zone 3 content types instead.
        assert "USER_QUERY" in report.segment_counts

    @pytest.mark.asyncio
    async def test_report_latency(self, sample_messages) -> None:
        _, report = await compress(sample_messages)
        assert report.total_latency_ms > 0
        assert report.total_latency_ms < 5000  # Should be well under 5s.


# ── Engine D integration test ───────────────────────────────────────────────


class MockSummarizerLLM:
    """Mock LLM for Engine D integration testing."""

    async def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.0) -> LLMResponse:
        return LLMResponse(
            text="Summary: auth module investigated, bug found at line 47.",
            input_tokens=100,
            output_tokens=20,
        )

    async def count_tokens(self, text: str) -> int:
        import math
        return math.ceil(len(text) / 3.5)


class TestPipelineWithEngineD:
    """Integration tests with Engine D (summarization) enabled."""

    @pytest.mark.asyncio
    async def test_engine_d_fires_when_enabled(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="Old question about the auth module"),
            MemoSiftMessage(role="assistant", content="The auth module has several issues. " * 20),
            MemoSiftMessage(role="user", content="Another question about the database"),
            MemoSiftMessage(role="assistant", content="The database connection pool. " * 20),
            MemoSiftMessage(role="user", content="What's the current status?"),
        ]
        config = MemoSiftConfig(enable_summarization=True, recent_turns=1)
        llm = MockSummarizerLLM()
        compressed, report = await compress(msgs, llm=llm, config=config)

        # Engine D should appear in the report layers.
        layer_names = [l.name for l in report.layers]
        assert "engine_summarizer" in layer_names

        # System prompt and last user message preserved.
        assert compressed[0].content == "System"
        user_msgs = [m for m in compressed if m.role == "user"]
        assert user_msgs[-1].content == "What's the current status?"

    @pytest.mark.asyncio
    async def test_engine_d_skipped_without_llm(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="Old question"),
            MemoSiftMessage(role="assistant", content="Old answer. " * 20),
            MemoSiftMessage(role="user", content="New question"),
        ]
        config = MemoSiftConfig(enable_summarization=True, recent_turns=1)
        # No LLM provided → Engine D should be skipped.
        compressed, report = await compress(msgs, config=config)
        layer_names = [l.name for l in report.layers]
        assert "engine_summarizer" not in layer_names

    @pytest.mark.asyncio
    async def test_engine_d_skipped_when_disabled(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="Question"),
        ]
        config = MemoSiftConfig(enable_summarization=False)
        llm = MockSummarizerLLM()
        compressed, report = await compress(msgs, llm=llm, config=config)
        layer_names = [l.name for l in report.layers]
        assert "engine_summarizer" not in layer_names
