# Tests for Layer 1.5 agentic pattern detector using real trace patterns.
from __future__ import annotations

import pytest

from memosift.core.agentic_patterns import (
    detect_agentic_patterns,
    _detect_duplicate_tool_results,
    _detect_failed_retries,
    _detect_large_code_args,
    _detect_thought_blocks,
    _detect_kpi_restatement,
)
from memosift.core.types import (
    AnchorCategory,
    AnchorFact,
    AnchorLedger,
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    MemoSiftMessage,
    ToolCall,
    ToolCallFunction,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_tool_call_seg(
    tool_name: str,
    args: str,
    tc_id: str = "tc_1",
    index: int = 0,
) -> ClassifiedMessage:
    """Create a classified assistant message with a tool call."""
    msg = MemoSiftMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id=tc_id, function=ToolCallFunction(name=tool_name, arguments=args))],
    )
    return ClassifiedMessage(
        message=msg,
        content_type=ContentType.OLD_CONVERSATION,
        policy=CompressionPolicy.MODERATE,
        original_index=index,
    )


def _make_tool_result_seg(
    content: str,
    tc_id: str = "tc_1",
    index: int = 1,
    content_type: ContentType = ContentType.TOOL_RESULT_JSON,
) -> ClassifiedMessage:
    """Create a classified tool result message."""
    msg = MemoSiftMessage(role="tool", content=content, tool_call_id=tc_id)
    return ClassifiedMessage(
        message=msg,
        content_type=content_type,
        policy=CompressionPolicy.STRUCTURAL,
        original_index=index,
    )


def _make_assistant_seg(
    content: str,
    index: int = 0,
    policy: CompressionPolicy = CompressionPolicy.MODERATE,
) -> ClassifiedMessage:
    """Create a classified assistant message."""
    msg = MemoSiftMessage(role="assistant", content=content)
    return ClassifiedMessage(
        message=msg,
        content_type=ContentType.OLD_CONVERSATION,
        policy=policy,
        original_index=index,
    )


# ── Pattern 1: Duplicate Tool Calls ─────────────────────────────────────


class TestDuplicateToolCalls:
    """Pattern 1: Identical tool call results should be collapsed."""

    def test_identical_results_collapsed(self) -> None:
        """Two identical tool results -> second becomes back-reference."""
        segments = [
            _make_tool_call_seg("list_files", "{}", tc_id="tc_1", index=0),
            _make_tool_result_seg(
                '{"files":[{"name":"chart.png","size":500764}],"count":1}',
                tc_id="tc_1",
                index=1,
            ),
            _make_tool_call_seg("list_files", "{}", tc_id="tc_2", index=2),
            _make_tool_result_seg(
                '{"files":[{"name":"chart.png","size":500764}],"count":1}',
                tc_id="tc_2",
                index=3,
            ),
        ]
        result = _detect_duplicate_tool_results(segments)
        assert len(result) == 4  # Same count, but second result replaced
        assert "[Identical result" in (result[3].message.content or "")
        assert result[3].policy == CompressionPolicy.AGGRESSIVE

    def test_different_results_preserved(self) -> None:
        """Two different tool results should both be kept."""
        segments = [
            _make_tool_call_seg("analyze", '{"file":"a.xlsx"}', tc_id="tc_1", index=0),
            _make_tool_result_seg('{"gas_rate": 1992.32}', tc_id="tc_1", index=1),
            _make_tool_call_seg("analyze", '{"file":"b.xlsx"}', tc_id="tc_2", index=2),
            _make_tool_result_seg('{"gas_rate": 2261.28}', tc_id="tc_2", index=3),
        ]
        result = _detect_duplicate_tool_results(segments)
        assert all("[Identical" not in (r.message.content or "") for r in result)

    def test_five_identical_list_files(self) -> None:
        """From Trace 2: list_conversation_files called 5x identically."""
        content = '{"files":[{"fileRef":1,"fileName":"chart.png"}],"count":1}'
        segments = []
        for i in range(5):
            tc_id = f"tc_{i}"
            segments.append(_make_tool_call_seg("list_files", "{}", tc_id=tc_id, index=i * 2))
            segments.append(_make_tool_result_seg(content, tc_id=tc_id, index=i * 2 + 1))

        result = _detect_duplicate_tool_results(segments)
        collapsed = [r for r in result if "[Identical" in (r.message.content or "")]
        assert len(collapsed) == 4  # First kept, 4 collapsed


# ── Pattern 2: Failed + Retried Tool Calls ───────────────────────────────


class TestFailedRetries:
    """Pattern 2: Failed tool calls + successful retries -> drop errors."""

    def test_failed_then_success_marks_error_aggressive(self) -> None:
        """TypeError in analyze_spreadsheet, then retry succeeds."""
        segments = [
            _make_tool_call_seg("analyze_spreadsheet", '{"code":"buggy"}', tc_id="tc_1", index=0),
            _make_tool_result_seg(
                '{"exitCode":1,"stderr":"TypeError: arg must be a list"}',
                tc_id="tc_1",
                index=1,
                content_type=ContentType.ERROR_TRACE,
            ),
            _make_tool_call_seg("analyze_spreadsheet", '{"code":"fixed"}', tc_id="tc_2", index=2),
            _make_tool_result_seg(
                '{"exitCode":0,"stdout":"results"}', tc_id="tc_2", index=3
            ),
        ]
        result = _detect_failed_retries(segments)
        # Error result should be marked AGGRESSIVE.
        assert result[1].policy == CompressionPolicy.AGGRESSIVE
        # The failed call should also be marked AGGRESSIVE.
        assert result[0].policy == CompressionPolicy.AGGRESSIVE
        # Successful pair should keep original policy.
        assert result[2].policy == CompressionPolicy.MODERATE
        assert result[3].policy == CompressionPolicy.STRUCTURAL

    def test_error_without_retry_preserved(self) -> None:
        """Error with no subsequent retry should stay as-is."""
        segments = [
            _make_tool_call_seg("analyze", '{"code":"buggy"}', tc_id="tc_1", index=0),
            _make_tool_result_seg(
                '{"exitCode":1,"stderr":"ImportError: no module named xyz"}',
                tc_id="tc_1",
                index=1,
                content_type=ContentType.ERROR_TRACE,
            ),
        ]
        result = _detect_failed_retries(segments)
        # No retry found -> keep original policies.
        assert result[0].policy == CompressionPolicy.MODERATE
        assert result[1].policy == CompressionPolicy.STRUCTURAL

    def test_multiple_failures_then_success(self) -> None:
        """From Trace 1: 2 failures then 1 success for analyze_spreadsheet."""
        segments = [
            _make_tool_call_seg("analyze_spreadsheet", '{"v":1}', tc_id="tc_1", index=0),
            _make_tool_result_seg(
                '{"exitCode":1,"stderr":"TypeError: ..."}',
                tc_id="tc_1", index=1, content_type=ContentType.ERROR_TRACE,
            ),
            _make_tool_call_seg("analyze_spreadsheet", '{"v":2}', tc_id="tc_2", index=2),
            _make_tool_result_seg(
                '{"exitCode":1,"stderr":"KeyError: Real Time"}',
                tc_id="tc_2", index=3, content_type=ContentType.ERROR_TRACE,
            ),
            _make_tool_call_seg("analyze_spreadsheet", '{"v":3}', tc_id="tc_3", index=4),
            _make_tool_result_seg(
                '{"exitCode":0,"stdout":"success"}', tc_id="tc_3", index=5,
            ),
        ]
        result = _detect_failed_retries(segments)
        # Both failures should be AGGRESSIVE.
        assert result[0].policy == CompressionPolicy.AGGRESSIVE  # failed call 1
        assert result[1].policy == CompressionPolicy.AGGRESSIVE  # error result 1
        assert result[2].policy == CompressionPolicy.AGGRESSIVE  # failed call 2
        assert result[3].policy == CompressionPolicy.AGGRESSIVE  # error result 2
        # Success should be unchanged.
        assert result[4].policy == CompressionPolicy.MODERATE
        assert result[5].policy == CompressionPolicy.STRUCTURAL


# ── Pattern 3: Large Code Arguments ──────────────────────────────────────


class TestLargeCodeArguments:
    """Pattern 3: Large code in tool args -> signature-only compression."""

    def test_matplotlib_code_truncated(self) -> None:
        """From Trace 1: 9.7KB matplotlib code as render_document args."""
        code = (
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "def plot_rates():\n"
            + "    x = np.arange(100)\n" * 200  # ~5KB of code
        )
        args = f'{{"format":"chart","code":"{code}"}}'
        segments = [_make_tool_call_seg("render_document", args, tc_id="tc_1", index=0)]
        result = _detect_large_code_args(segments)

        # Arguments should be truncated.
        new_args = result[0].message.tool_calls[0].function.arguments
        assert len(new_args) < len(args)
        assert "truncated" in new_args
        # Tool name should be preserved.
        assert result[0].message.tool_calls[0].function.name == "render_document"

    def test_small_args_preserved(self) -> None:
        """Small tool call arguments should not be modified."""
        segments = [_make_tool_call_seg("list_files", "{}", tc_id="tc_1", index=0)]
        result = _detect_large_code_args(segments)
        assert result[0].message.tool_calls[0].function.arguments == "{}"

    def test_large_non_code_preserved(self) -> None:
        """Large arguments without code patterns should not be truncated."""
        # 5KB of JSON data, not code.
        data = '{"data":' + '"x" ' * 1500 + "}"
        segments = [_make_tool_call_seg("query_data", data, tc_id="tc_1", index=0)]
        result = _detect_large_code_args(segments)
        assert result[0].message.tool_calls[0].function.arguments == data


# ── Pattern 4: Thought Process Blocks ────────────────────────────────────


class TestThoughtProcessBlocks:
    """Pattern 4: Thought/reasoning blocks -> aggressive compression."""

    def test_thought_process_reclassified(self) -> None:
        """**Thought Process** blocks should become ASSISTANT_REASONING."""
        content = (
            "**Thought Process**:\n```\n**Analyzing files for comprehension**\n\n"
            + "I need to analyze some files.\n" * 20
            + "```\nHere are the results..."
        )
        segments = [_make_assistant_seg(content, index=0)]
        result = _detect_thought_blocks(segments)
        assert result[0].content_type == ContentType.ASSISTANT_REASONING
        assert result[0].policy == CompressionPolicy.AGGRESSIVE

    def test_thinking_tags_detected(self) -> None:
        """<thinking> tags should be detected."""
        content = "<thinking>\nI should consider the approach.\n" * 10 + "</thinking>"
        segments = [_make_assistant_seg(content, index=0)]
        result = _detect_thought_blocks(segments)
        assert result[0].content_type == ContentType.ASSISTANT_REASONING

    def test_short_content_not_reclassified(self) -> None:
        """Short assistant messages should not be reclassified."""
        segments = [_make_assistant_seg("**Thought Process**: Quick thought.", index=0)]
        result = _detect_thought_blocks(segments)
        assert result[0].content_type == ContentType.OLD_CONVERSATION

    def test_normal_assistant_preserved(self) -> None:
        """Normal assistant response should not be affected."""
        content = "Here are the analysis results:\n" + "- Gas rate: 1,992 Mcf/d\n" * 10
        segments = [_make_assistant_seg(content, index=0)]
        result = _detect_thought_blocks(segments)
        assert result[0].content_type == ContentType.OLD_CONVERSATION
        assert result[0].policy == CompressionPolicy.MODERATE


# ── Pattern 5: KPI Restatement ───────────────────────────────────────────


class TestKPIRestatement:
    """Pattern 5: Messages restating 3+ anchor facts -> MODERATE."""

    def test_restatement_detected(self) -> None:
        """Second restatement of 3+ facts should be marked MODERATE."""
        ledger = AnchorLedger()
        ledger.add(AnchorFact(category=AnchorCategory.IDENTIFIERS, content="Metric: 1,992 Mcf/d", turn=1))
        ledger.add(AnchorFact(category=AnchorCategory.IDENTIFIERS, content="Metric: 2,261 Mcf/d", turn=1))
        ledger.add(AnchorFact(category=AnchorCategory.IDENTIFIERS, content="Metric: 13.5%", turn=1))
        ledger.add(AnchorFact(category=AnchorCategory.IDENTIFIERS, content="ID: Roper North", turn=1))

        segments = [
            # First restatement — should be kept.
            _make_assistant_seg(
                "Gas rate: 1,992 Mcf/d for North, 2,261 Mcf/d for South. Delta: 13.5%. Roper North.",
                index=0,
            ),
            # Second restatement — should be MODERATE.
            _make_assistant_seg(
                "Roper North produced 1,992 Mcf/d vs South at 2,261 Mcf/d, a 13.5% difference.",
                index=1,
            ),
            # Third restatement — should also be MODERATE.
            _make_assistant_seg(
                "Summary: North 1,992, South 2,261 Mcf/d. 13.5% advantage for South. Roper North data.",
                index=2,
            ),
            # Recent message — should not be affected.
            _make_assistant_seg(
                "Final: 1,992 vs 2,261, delta 13.5%. Roper North is weaker.",
                index=3,
            ),
            _make_assistant_seg("Latest update.", index=4),
        ]
        result = _detect_kpi_restatement(segments, ledger)

        # First restatement kept as-is.
        assert result[0].policy == CompressionPolicy.MODERATE
        # Second restatement -> MODERATE (already MODERATE, but pattern detected).
        assert result[1].policy == CompressionPolicy.MODERATE
        # Recent messages preserved.
        assert result[3].policy == CompressionPolicy.MODERATE

    def test_no_restatement_without_facts(self) -> None:
        """Empty ledger should not trigger restatement detection."""
        ledger = AnchorLedger()
        segments = [
            _make_assistant_seg("Some analysis with numbers 1,992 and 2,261.", index=0),
        ]
        result = _detect_kpi_restatement(segments, ledger)
        assert result[0].policy == CompressionPolicy.MODERATE


# ── Integration: Full detector pipeline ──────────────────────────────────


class TestFullDetector:
    """End-to-end test with all 5 patterns active."""

    def test_mixed_patterns(self) -> None:
        """Conversation with multiple pattern types."""
        ledger = AnchorLedger()
        ledger.add(AnchorFact(category=AnchorCategory.IDENTIFIERS, content="Metric: 1,992 Mcf/d", turn=1))
        ledger.add(AnchorFact(category=AnchorCategory.IDENTIFIERS, content="Metric: 2,261 Mcf/d", turn=1))
        ledger.add(AnchorFact(category=AnchorCategory.IDENTIFIERS, content="Metric: 13.5%", turn=1))

        code = "import numpy as np\ndef f():\n" + "    x = 1\n" * 1500

        segments = [
            # Thought process (Pattern 4).
            _make_assistant_seg(
                "**Thought Process**:\n```\n**Analyzing**\n" + "thinking...\n" * 20 + "```",
                index=0,
            ),
            # Large code args (Pattern 3).
            _make_tool_call_seg("render", f'{{"code":"{code}"}}', tc_id="tc_1", index=1),
            _make_tool_result_seg('{"exitCode":0}', tc_id="tc_1", index=2),
            # Failed + retried (Pattern 2).
            _make_tool_call_seg("analyze", '{"v":1}', tc_id="tc_err", index=3),
            _make_tool_result_seg(
                '{"exitCode":1,"stderr":"TypeError: bad"}',
                tc_id="tc_err", index=4, content_type=ContentType.ERROR_TRACE,
            ),
            _make_tool_call_seg("analyze", '{"v":2}', tc_id="tc_ok", index=5),
            _make_tool_result_seg('{"exitCode":0,"stdout":"good"}', tc_id="tc_ok", index=6),
            # Duplicate (Pattern 1).
            _make_tool_call_seg("list_files", "{}", tc_id="tc_dup1", index=7),
            _make_tool_result_seg(
                '{"files":["a.png","b.png"],"count":2}', tc_id="tc_dup1", index=8,
            ),
            _make_tool_call_seg("list_files", "{}", tc_id="tc_dup2", index=9),
            _make_tool_result_seg(
                '{"files":["a.png","b.png"],"count":2}', tc_id="tc_dup2", index=10,
            ),
            # Recent assistant messages (not affected by restatement).
            _make_assistant_seg("Final results.", index=11),
            _make_assistant_seg("Done.", index=12),
        ]

        result = detect_agentic_patterns(segments, ledger=ledger)

        # Pattern 4: Thought block reclassified.
        assert result[0].content_type == ContentType.ASSISTANT_REASONING
        assert result[0].policy == CompressionPolicy.AGGRESSIVE

        # Pattern 3: Large code args truncated.
        new_args = result[1].message.tool_calls[0].function.arguments
        assert "truncated" in new_args

        # Pattern 2: Failed call + result marked AGGRESSIVE.
        assert result[3].policy == CompressionPolicy.AGGRESSIVE
        assert result[4].policy == CompressionPolicy.AGGRESSIVE

        # Pattern 1: Second list_files result is a back-reference.
        assert "[Identical result" in (result[10].message.content or "")

    def test_empty_segments(self) -> None:
        """Empty segment list should return empty."""
        result = detect_agentic_patterns([])
        assert result == []
