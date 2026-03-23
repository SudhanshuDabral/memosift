# Smoke tests — verify the project installs and core types are importable.
from __future__ import annotations

import pytest

from memosift import (
    ClassifiedMessage,
    CompressionPolicy,
    CompressionReport,
    ContentType,
    DependencyMap,
    MemoSiftConfig,
    MemoSiftMessage,
    LLMResponse,
    ToolCall,
    ToolCallFunction,
)
from memosift.providers.heuristic import HeuristicTokenCounter


class TestMemoSiftMessage:
    """Tests for MemoSiftMessage construction and serialization."""

    def test_basic_construction(self) -> None:
        msg = MemoSiftMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.tool_call_id is None
        assert msg.tool_calls is None
        assert msg.metadata == {}

    def test_tool_message(self) -> None:
        msg = MemoSiftMessage(
            role="tool", content="result", tool_call_id="tc1", name="read_file"
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "tc1"
        assert msg.name == "read_file"

    def test_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(
            id="tc1",
            function=ToolCallFunction(name="read_file", arguments='{"path": "a.py"}'),
        )
        msg = MemoSiftMessage(role="assistant", content="Reading.", tool_calls=[tc])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "read_file"

    def test_round_trip_serialization(self) -> None:
        tc = ToolCall(
            id="tc1",
            function=ToolCallFunction(name="search", arguments='{"q": "test"}'),
        )
        original = MemoSiftMessage(
            role="assistant",
            content="Searching.",
            tool_calls=[tc],
            metadata={"source": "test"},
        )
        d = original.to_dict()
        restored = MemoSiftMessage.from_dict(d)
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.tool_calls is not None
        assert restored.tool_calls[0].id == "tc1"
        assert restored.metadata == {"source": "test"}

    def test_from_dict_missing_content_defaults_empty(self) -> None:
        msg = MemoSiftMessage.from_dict({"role": "assistant"})
        assert msg.content == ""


class TestContentType:
    """Tests for ContentType enum."""

    def test_all_types_exist(self) -> None:
        expected = {
            "SYSTEM_PROMPT", "USER_QUERY", "RECENT_TURN",
            "TOOL_RESULT_JSON", "TOOL_RESULT_TEXT", "CODE_BLOCK",
            "ERROR_TRACE", "ASSISTANT_REASONING", "OLD_CONVERSATION",
            "PREVIOUSLY_COMPRESSED",
        }
        assert {t.value for t in ContentType} == expected

    def test_string_enum(self) -> None:
        assert ContentType.SYSTEM_PROMPT == "SYSTEM_PROMPT"
        assert str(ContentType.CODE_BLOCK) == "CODE_BLOCK"


class TestCompressionPolicy:
    """Tests for CompressionPolicy enum."""

    def test_all_policies_exist(self) -> None:
        expected = {
            "PRESERVE", "LIGHT", "MODERATE", "STRUCTURAL",
            "STACK", "AGGRESSIVE", "SIGNATURE",
        }
        assert {p.value for p in CompressionPolicy} == expected


class TestClassifiedMessage:
    """Tests for ClassifiedMessage wrapper."""

    def test_construction(self) -> None:
        msg = MemoSiftMessage(role="system", content="You are helpful.")
        cm = ClassifiedMessage(
            message=msg,
            content_type=ContentType.SYSTEM_PROMPT,
            policy=CompressionPolicy.PRESERVE,
            original_index=0,
            protected=True,
        )
        assert cm.role == "system"
        assert cm.content == "You are helpful."
        assert cm.content_type == ContentType.SYSTEM_PROMPT
        assert cm.protected is True
        assert cm.relevance_score == 0.5
        assert cm.estimated_tokens == 0

    def test_content_setter(self) -> None:
        msg = MemoSiftMessage(role="tool", content="original")
        cm = ClassifiedMessage(
            message=msg,
            content_type=ContentType.TOOL_RESULT_TEXT,
            policy=CompressionPolicy.MODERATE,
        )
        cm.content = "compressed"
        assert cm.content == "compressed"
        assert cm.message.content == "compressed"


class TestMemoSiftConfig:
    """Tests for MemoSiftConfig defaults and validation."""

    def test_defaults(self) -> None:
        config = MemoSiftConfig()
        assert config.recent_turns == 2
        assert config.token_budget is None
        assert config.enable_summarization is False
        assert config.dedup_similarity_threshold == 0.80
        assert config.entropy_threshold == 1.8
        assert config.token_prune_keep_ratio == 0.5
        assert config.json_array_threshold == 5
        assert config.code_keep_signatures is True
        assert config.relevance_drop_threshold == 0.05
        assert config.policies == {}

    def test_custom_budget(self) -> None:
        config = MemoSiftConfig(token_budget=50_000)
        assert config.token_budget == 50_000

    def test_invalid_recent_turns(self) -> None:
        with pytest.raises(ValueError, match="recent_turns"):
            MemoSiftConfig(recent_turns=-1)

    def test_invalid_budget_too_low(self) -> None:
        with pytest.raises(ValueError, match="token_budget"):
            MemoSiftConfig(token_budget=50)

    def test_invalid_dedup_threshold(self) -> None:
        with pytest.raises(ValueError, match="dedup_similarity_threshold"):
            MemoSiftConfig(dedup_similarity_threshold=1.5)

    def test_invalid_prune_ratio(self) -> None:
        with pytest.raises(ValueError, match="token_prune_keep_ratio"):
            MemoSiftConfig(token_prune_keep_ratio=0.05)

    def test_policy_override(self) -> None:
        config = MemoSiftConfig(
            policies={ContentType.ERROR_TRACE: CompressionPolicy.PRESERVE}
        )
        assert config.policies[ContentType.ERROR_TRACE] == CompressionPolicy.PRESERVE


class TestCompressionReport:
    """Tests for CompressionReport construction and methods."""

    def test_empty_report(self) -> None:
        report = CompressionReport()
        assert report.original_tokens == 0
        assert report.compressed_tokens == 0
        assert report.compression_ratio == 1.0
        assert report.layers == []
        assert report.decisions == []

    def test_add_layer(self) -> None:
        report = CompressionReport()
        report.add_layer("classifier", input_tokens=1000, output_tokens=1000, latency_ms=2.5)
        assert len(report.layers) == 1
        assert report.layers[0].name == "classifier"
        assert report.layers[0].tokens_removed == 0
        assert report.total_latency_ms == 2.5

    def test_add_layer_failure(self) -> None:
        report = CompressionReport()
        report.add_layer_failure("scorer", error="LLM timeout", latency_ms=3000.0)
        assert len(report.layers) == 1
        assert len(report.decisions) == 1
        assert report.decisions[0].action == "skipped"
        assert "LLM timeout" in report.decisions[0].reason

    def test_finalize(self) -> None:
        report = CompressionReport()
        report.finalize(original_tokens=10_000, compressed_tokens=4_000)
        assert report.original_tokens == 10_000
        assert report.compressed_tokens == 4_000
        assert report.tokens_saved == 6_000
        assert report.compression_ratio == 2.5

    def test_finalize_zero_compressed(self) -> None:
        report = CompressionReport()
        report.finalize(original_tokens=1000, compressed_tokens=0)
        assert report.compression_ratio == float("inf")


class TestDependencyMap:
    """Tests for DependencyMap dedup tracking."""

    def test_add_and_can_drop(self) -> None:
        dm = DependencyMap()
        dm.add(deduped_index=5, original_index=2)
        assert dm.can_drop(2) is False
        assert dm.can_drop(5) is True
        assert dm.can_drop(0) is True

    def test_dependents_of(self) -> None:
        dm = DependencyMap()
        dm.add(5, 2)
        dm.add(8, 2)
        assert sorted(dm.dependents_of(2)) == [5, 8]
        assert dm.dependents_of(5) == []


class TestHeuristicTokenCounter:
    """Tests for HeuristicTokenCounter."""

    @pytest.mark.asyncio
    async def test_count_tokens(self) -> None:
        counter = HeuristicTokenCounter()
        count = await counter.count_tokens("Hello, world!")
        # 13 chars / 3.5 ≈ 3.71 → ceil = 4
        assert count == 4

    @pytest.mark.asyncio
    async def test_count_empty(self) -> None:
        counter = HeuristicTokenCounter()
        assert await counter.count_tokens("") == 0

    @pytest.mark.asyncio
    async def test_generate_raises(self) -> None:
        counter = HeuristicTokenCounter()
        with pytest.raises(NotImplementedError):
            await counter.generate("test prompt")


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_construction(self) -> None:
        resp = LLMResponse(text="Hello", input_tokens=5, output_tokens=1)
        assert resp.text == "Hello"
        assert resp.input_tokens == 5
        assert resp.output_tokens == 1


class TestTestVectorsLoadable:
    """Verify test vector files exist and parse correctly."""

    def test_classify_vector_loads(self, classify_vector: dict) -> None:
        assert classify_vector["description"]
        assert len(classify_vector["input"]) == 12
        assert len(classify_vector["expected_classifications"]) == 12

    def test_dedup_vector_loads(self, dedup_vector: dict) -> None:
        assert dedup_vector["description"]
        assert len(dedup_vector["input"]) == 4

    def test_compress_vector_loads(self, compress_vector: dict) -> None:
        assert compress_vector["description"]
        assert compress_vector["config"]["token_budget"] == 500
        assert len(compress_vector["input"]) == 6

    def test_classify_vector_messages_deserialize(self, classify_vector: dict) -> None:
        from conftest import messages_from_dicts
        msgs = messages_from_dicts(classify_vector["input"])
        assert len(msgs) == 12
        assert msgs[0].role == "system"
        assert msgs[2].tool_calls is not None
        assert msgs[3].tool_call_id == "tc1"
