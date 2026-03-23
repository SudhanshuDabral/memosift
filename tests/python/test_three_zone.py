# Tests for the Three-Zone Memory Model and pipeline integration.
from __future__ import annotations

import pytest

from memosift import compress, MemoSiftConfig, MemoSiftMessage
from memosift.core.pipeline import _partition_zones, _reassemble_zones
from memosift.core.classifier import classify_messages
from memosift.core.types import (
    AnchorCategory,
    AnchorFact,
    AnchorLedger,
    ContentType,
    ToolCall,
    ToolCallFunction,
)


# ── Zone partitioning tests ─────────────────────────────────────────────────


class TestPartitionZones:
    def test_basic_partitioning(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="Hello"),
            MemoSiftMessage(role="assistant", content="Hi"),
        ]
        zone1, zone2, zone3 = _partition_zones(msgs)
        assert len(zone1) == 1  # system
        assert len(zone2) == 0
        assert len(zone3) == 2  # user + assistant

    def test_no_compressed_messages(self) -> None:
        msgs = [
            MemoSiftMessage(role="user", content="Q"),
            MemoSiftMessage(role="assistant", content="A"),
        ]
        zone1, zone2, zone3 = _partition_zones(msgs)
        assert len(zone1) == 0
        assert len(zone2) == 0
        assert len(zone3) == 2

    def test_all_compressed(self) -> None:
        m1 = MemoSiftMessage(role="user", content="Q")
        m1._memosift_compressed = True
        m2 = MemoSiftMessage(role="assistant", content="A")
        m2._memosift_compressed = True
        zone1, zone2, zone3 = _partition_zones([m1, m2])
        assert len(zone1) == 0
        assert len(zone2) == 2
        assert len(zone3) == 0

    def test_mixed_zones(self) -> None:
        system = MemoSiftMessage(role="system", content="System")
        compressed = MemoSiftMessage(role="assistant", content="Old compressed answer")
        compressed._memosift_compressed = True
        raw = MemoSiftMessage(role="user", content="New question")

        zone1, zone2, zone3 = _partition_zones([system, compressed, raw])
        assert len(zone1) == 1
        assert zone1[0].role == "system"
        assert len(zone2) == 1
        assert zone2[0]._memosift_compressed is True
        assert len(zone3) == 1
        assert zone3[0].content == "New question"

    def test_system_prompt_never_in_zone2(self) -> None:
        """System prompts with _memosift_compressed should still be zone1."""
        system = MemoSiftMessage(role="system", content="System")
        system._memosift_compressed = True
        zone1, zone2, zone3 = _partition_zones([system])
        assert len(zone1) == 1
        assert len(zone2) == 0

    def test_empty_input(self) -> None:
        zone1, zone2, zone3 = _partition_zones([])
        assert zone1 == []
        assert zone2 == []
        assert zone3 == []


class TestReassembleZones:
    def test_order(self) -> None:
        z1 = [MemoSiftMessage(role="system", content="S")]
        z2 = [MemoSiftMessage(role="assistant", content="old")]
        z3 = [MemoSiftMessage(role="user", content="new")]
        result = _reassemble_zones(z1, z2, z3)
        assert len(result) == 3
        assert result[0].role == "system"
        assert result[1].content == "old"
        assert result[2].content == "new"


# ── Pipeline with Three-Zone Model ──────────────────────────────────────────


class TestThreeZonePipeline:
    @pytest.mark.asyncio
    async def test_zone2_passthrough(self) -> None:
        """Previously compressed messages should not be re-compressed."""
        compressed_msg = MemoSiftMessage(role="assistant", content="already compressed content")
        compressed_msg._memosift_compressed = True

        msgs = [
            MemoSiftMessage(role="system", content="System"),
            compressed_msg,
            MemoSiftMessage(role="user", content="New question"),
        ]
        result, report = await compress(msgs)
        # The compressed message should appear unchanged in output.
        assert any(m.content == "already compressed content" for m in result)

    @pytest.mark.asyncio
    async def test_zone3_tagged_on_output(self) -> None:
        """Zone 3 output messages should be tagged _memosift_compressed=True."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="Question"),
        ]
        result, _ = await compress(msgs)
        # The user message (zone3) should now be tagged.
        non_system = [m for m in result if m.role != "system"]
        for m in non_system:
            assert m._memosift_compressed is True

    @pytest.mark.asyncio
    async def test_consecutive_calls(self) -> None:
        """Second compress() call should only process new messages."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="First question"),
            MemoSiftMessage(role="assistant", content="First answer"),
        ]
        # First compression.
        result1, _ = await compress(msgs)

        # Add new messages to the compressed output.
        result1.append(MemoSiftMessage(role="user", content="Second question"))
        result1.append(MemoSiftMessage(role="assistant", content="Second answer"))
        result1.append(MemoSiftMessage(role="user", content="Third question"))

        # Second compression — zone2 should contain first call's output.
        result2, report2 = await compress(result1)

        # System prompt preserved.
        assert result2[0].content == "System"
        # Last user message preserved.
        user_msgs = [m for m in result2 if m.role == "user"]
        assert user_msgs[-1].content == "Third question"

    @pytest.mark.asyncio
    async def test_backward_compatible_without_ledger(self) -> None:
        """compress() without ledger should work identically to before."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="Hello"),
            MemoSiftMessage(role="assistant", content="Hi there"),
            MemoSiftMessage(role="user", content="How are you?"),
        ]
        result, report = await compress(msgs)
        assert len(result) > 0
        assert report.original_tokens > 0
        assert result[0].content == "System"

    @pytest.mark.asyncio
    async def test_with_ledger_extracts_facts(self) -> None:
        """When ledger is provided, facts should be extracted."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(
                role="tool",
                content="Contents of src/auth.ts:\nexport class Auth {}",
                tool_call_id="tc1",
                name="read_file",
            ),
            MemoSiftMessage(role="user", content="What's in it?"),
        ]
        ledger = AnchorLedger()
        result, _ = await compress(msgs, ledger=ledger)
        # Ledger should have extracted the file path.
        assert len(ledger.facts) > 0
        file_facts = ledger.facts_by_category(AnchorCategory.FILES)
        assert any("src/auth.ts" in f.content for f in file_facts)

    @pytest.mark.asyncio
    async def test_ledger_disabled_skips_extraction(self) -> None:
        """enable_anchor_ledger=False should skip extraction."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(
                role="tool", content="src/auth.ts: code here",
                tool_call_id="tc1", name="read_file",
            ),
            MemoSiftMessage(role="user", content="Q"),
        ]
        ledger = AnchorLedger()
        config = MemoSiftConfig(enable_anchor_ledger=False)
        await compress(msgs, config=config, ledger=ledger)
        assert len(ledger.facts) == 0

    @pytest.mark.asyncio
    async def test_ledger_dedup_across_calls(self) -> None:
        """Same file path seen twice should not duplicate in ledger."""
        from memosift.core.types import AnchorCategory

        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(
                role="tool", content="src/auth.ts: code",
                tool_call_id="tc1", name="read_file",
            ),
            MemoSiftMessage(role="user", content="Q"),
        ]
        ledger = AnchorLedger()
        await compress(msgs, ledger=ledger)
        file_facts_first = len(ledger.facts_by_category(AnchorCategory.FILES))

        # Same file in new messages.
        msgs2 = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(
                role="tool", content="src/auth.ts: code",
                tool_call_id="tc2", name="read_file",
            ),
            MemoSiftMessage(role="user", content="Q2"),
        ]
        await compress(msgs2, ledger=ledger)
        # File facts should not duplicate (same file path = same hash).
        file_facts_second = len(ledger.facts_by_category(AnchorCategory.FILES))
        assert file_facts_second == file_facts_first

    @pytest.mark.asyncio
    async def test_empty_zone3_returns_early(self) -> None:
        """When all messages are system or previously compressed, return as-is."""
        system = MemoSiftMessage(role="system", content="System")
        compressed = MemoSiftMessage(role="assistant", content="old")
        compressed._memosift_compressed = True

        result, report = await compress([system, compressed])
        assert len(result) == 2
        assert report.compressed_tokens == report.original_tokens

    @pytest.mark.asyncio
    async def test_effective_budget_accounts_for_zones(self) -> None:
        """Budget enforcement should subtract zone1+zone2 tokens from budget."""
        system = MemoSiftMessage(role="system", content="S" * 100)
        compressed = MemoSiftMessage(role="assistant", content="C" * 500)
        compressed._memosift_compressed = True

        # Zone 3 has lots of content.
        zone3_msgs: list[MemoSiftMessage] = []
        for i in range(10):
            zone3_msgs.append(MemoSiftMessage(role="user", content=f"Q{i} " + "x" * 200))
            zone3_msgs.append(MemoSiftMessage(role="assistant", content=f"A{i} " + "y" * 200))
        zone3_msgs.append(MemoSiftMessage(role="user", content="Final"))

        all_msgs = [system, compressed] + zone3_msgs
        config = MemoSiftConfig(token_budget=2000, recent_turns=1)
        result, report = await compress(all_msgs, config=config)

        # Total output should respect the budget (with heuristic ±15% tolerance).
        total_chars = sum(len(m.content) for m in result)
        approx_tokens = total_chars / 3.5
        assert approx_tokens < 2000 * 1.15


# ── Classifier PREVIOUSLY_COMPRESSED tests ──────────────────────────────────


class TestPreviouslyCompressedClassification:
    def test_compressed_message_classified(self) -> None:
        msg = MemoSiftMessage(role="assistant", content="compressed content")
        msg._memosift_compressed = True
        result = classify_messages([msg], MemoSiftConfig(recent_turns=0))
        assert result[0].content_type == ContentType.PREVIOUSLY_COMPRESSED
        assert result[0].protected is True

    def test_compressed_tool_result_classified(self) -> None:
        msg = MemoSiftMessage(role="tool", content="tool output", tool_call_id="tc1")
        msg._memosift_compressed = True
        result = classify_messages([msg], MemoSiftConfig(recent_turns=0))
        assert result[0].content_type == ContentType.PREVIOUSLY_COMPRESSED
