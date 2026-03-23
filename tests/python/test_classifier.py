# Tests for Layer 1: Content Classification.
from __future__ import annotations

import pytest

from memosift.config import MemoSiftConfig
from memosift.core.classifier import (
    _contains_code,
    _contains_error_trace,
    _find_nth_user_message_from_end,
    _is_valid_json,
    _last_user_index,
    classify_messages,
)
from memosift.core.types import (
    CompressionPolicy,
    ContentType,
    MemoSiftMessage,
    ToolCall,
    ToolCallFunction,
)
from conftest import messages_from_dicts


# ── Test vector validation ──────────────────────────────────────────────────


class TestClassifyVector:
    """Validate classify-001 test vector produces expected classifications."""

    def test_classify_001(self, classify_vector: dict) -> None:
        msgs = messages_from_dicts(classify_vector["input"])
        config = MemoSiftConfig(**classify_vector["config"])
        result = classify_messages(msgs, config)

        assert len(result) == len(classify_vector["expected_classifications"])
        for expected in classify_vector["expected_classifications"]:
            idx = expected["index"]
            expected_type = ContentType(expected["type"])
            actual_type = result[idx].content_type
            assert actual_type == expected_type, (
                f"Message {idx}: expected {expected_type}, got {actual_type}"
            )


# ── System prompt classification ────────────────────────────────────────────


class TestSystemPrompt:
    """System messages are always classified as SYSTEM_PROMPT."""

    def test_system_prompt_classified(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="You are helpful."),
            MemoSiftMessage(role="user", content="Hi"),
        ]
        result = classify_messages(msgs, MemoSiftConfig())
        assert result[0].content_type == ContentType.SYSTEM_PROMPT
        assert result[0].policy == CompressionPolicy.PRESERVE
        assert result[0].protected is True

    def test_multiple_system_prompts(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System 1"),
            MemoSiftMessage(role="system", content="System 2"),
            MemoSiftMessage(role="user", content="Hi"),
        ]
        result = classify_messages(msgs, MemoSiftConfig())
        assert result[0].content_type == ContentType.SYSTEM_PROMPT
        assert result[1].content_type == ContentType.SYSTEM_PROMPT


# ── User query classification ───────────────────────────────────────────────


class TestUserQuery:
    """The last user message is always USER_QUERY."""

    def test_last_user_is_query(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="First question"),
            MemoSiftMessage(role="assistant", content="Answer"),
            MemoSiftMessage(role="user", content="Second question"),
            MemoSiftMessage(role="assistant", content="Answer"),
            MemoSiftMessage(role="user", content="Third question"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=1))
        # Only the last user message should be USER_QUERY.
        query_indices = [i for i, r in enumerate(result) if r.content_type == ContentType.USER_QUERY]
        assert query_indices == [5]
        assert result[5].protected is True

    def test_single_user_message(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="Only question"),
        ]
        result = classify_messages(msgs, MemoSiftConfig())
        assert result[1].content_type == ContentType.USER_QUERY


# ── Recent turns boundary ───────────────────────────────────────────────────


class TestRecentTurns:
    """Recent turn detection counts by user messages, not message count."""

    def test_recent_turns_boundary(self) -> None:
        """With recent_turns=2, the boundary user message is OLD_CONVERSATION
        and everything after it is RECENT_TURN."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="Old question"),
            MemoSiftMessage(role="assistant", content="Old answer"),
            MemoSiftMessage(role="user", content="Boundary user msg"),
            MemoSiftMessage(role="assistant", content="Recent A1"),
            MemoSiftMessage(role="user", content="Current query"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=2))
        assert result[0].content_type == ContentType.SYSTEM_PROMPT
        assert result[1].content_type == ContentType.OLD_CONVERSATION
        assert result[2].content_type == ContentType.ASSISTANT_REASONING
        # Index 3 is the 2nd user msg from end → boundary. The boundary user
        # message itself is OLD_CONVERSATION; everything after is RECENT_TURN.
        assert result[3].content_type == ContentType.OLD_CONVERSATION
        assert result[4].content_type == ContentType.RECENT_TURN
        assert result[5].content_type == ContentType.USER_QUERY

    def test_agentic_turn_with_tools(self) -> None:
        """A single agentic 'turn' can be 6 messages: user → assistant(tool_call) →
        tool → assistant(tool_call) → tool → assistant. The boundary user message
        is OLD_CONVERSATION; messages after it are RECENT_TURN."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="Old msg"),
            MemoSiftMessage(role="assistant", content="Old reply"),
            MemoSiftMessage(role="user", content="Fix the bug"),
            MemoSiftMessage(
                role="assistant", content="Reading file.",
                tool_calls=[ToolCall(id="tc1", function=ToolCallFunction(name="read_file", arguments="{}"))]
            ),
            MemoSiftMessage(role="tool", content="file content", tool_call_id="tc1", name="read_file"),
            MemoSiftMessage(
                role="assistant", content="Running tests.",
                tool_calls=[ToolCall(id="tc2", function=ToolCallFunction(name="run_tests", arguments="{}"))]
            ),
            MemoSiftMessage(role="tool", content="all passed", tool_call_id="tc2", name="run_tests"),
            MemoSiftMessage(role="assistant", content="Done."),
            MemoSiftMessage(role="user", content="Thanks"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=2))
        # Index 3 is the 2nd user msg from end → boundary → OLD_CONVERSATION.
        assert result[3].content_type == ContentType.OLD_CONVERSATION
        # Index 4: assistant → RECENT_TURN
        assert result[4].content_type == ContentType.RECENT_TURN
        # Index 5: tool from read_file → CODE_BLOCK (notable type preserved in recent window)
        assert result[5].content_type == ContentType.CODE_BLOCK
        # Index 6: assistant → RECENT_TURN
        assert result[6].content_type == ContentType.RECENT_TURN
        # Index 7: tool with plain text → RECENT_TURN
        assert result[7].content_type == ContentType.RECENT_TURN
        # Index 8: assistant → RECENT_TURN
        assert result[8].content_type == ContentType.RECENT_TURN
        assert result[9].content_type == ContentType.USER_QUERY

    def test_recent_turns_zero_means_nothing_recent(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="Q1"),
            MemoSiftMessage(role="assistant", content="A1"),
            MemoSiftMessage(role="user", content="Q2"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=0))
        assert result[1].content_type == ContentType.OLD_CONVERSATION
        assert result[2].content_type == ContentType.ASSISTANT_REASONING
        assert result[3].content_type == ContentType.USER_QUERY  # Still current query

    def test_fewer_users_than_recent_turns(self) -> None:
        """If there are fewer user messages than recent_turns, everything is recent."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="Only user msg"),
            MemoSiftMessage(role="assistant", content="Reply"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=5))
        assert result[1].content_type == ContentType.USER_QUERY
        assert result[2].content_type == ContentType.RECENT_TURN


# ── Tool result sub-classification ──────────────────────────────────────────


class TestToolResultClassification:
    """Tool messages are sub-classified by content shape."""

    def test_json_tool_result(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(
                role="tool",
                content='{"users": [{"id": 1, "name": "Alice"}]}',
                tool_call_id="tc1",
            ),
            MemoSiftMessage(role="user", content="query"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=0))
        assert result[1].content_type == ContentType.TOOL_RESULT_JSON

    def test_json_array_tool_result(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="tool", content='[1, 2, 3]', tool_call_id="tc1"),
            MemoSiftMessage(role="user", content="query"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=0))
        assert result[1].content_type == ContentType.TOOL_RESULT_JSON

    def test_bare_string_not_json(self) -> None:
        """Bare strings that parse as JSON should NOT be TOOL_RESULT_JSON."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="tool", content='"hello"', tool_call_id="tc1"),
            MemoSiftMessage(role="user", content="query"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=0))
        assert result[1].content_type == ContentType.TOOL_RESULT_TEXT

    def test_bare_number_not_json(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="tool", content="42", tool_call_id="tc1"),
            MemoSiftMessage(role="user", content="query"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=0))
        assert result[1].content_type == ContentType.TOOL_RESULT_TEXT

    def test_error_trace_python(self) -> None:
        traceback = (
            "Traceback (most recent call last):\n"
            '  File "src/auth.py", line 47, in login\n'
            "    result = db.query(user)\n"
            '  File "src/db.py", line 12, in query\n'
            "    return cursor.fetchone()\n"
            "TypeError: 'NoneType' object is not subscriptable"
        )
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="tool", content=traceback, tool_call_id="tc1"),
            MemoSiftMessage(role="user", content="query"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=0))
        assert result[1].content_type == ContentType.ERROR_TRACE

    def test_error_trace_javascript(self) -> None:
        stack = (
            "TypeError: Cannot read properties of undefined (reading 'userId')\n"
            "    at AuthService.login (src/auth.ts:4:19)\n"
            "    at Object.<anonymous> (src/auth.test.ts:12:5)\n"
            "    at Module._compile (node:internal/modules/cjs/loader:1198:14)"
        )
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="tool", content=stack, tool_call_id="tc1"),
            MemoSiftMessage(role="user", content="query"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=0))
        assert result[1].content_type == ContentType.ERROR_TRACE

    def test_single_error_line_not_trace(self) -> None:
        """A single 'Error:' mention should NOT trigger ERROR_TRACE."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(
                role="tool",
                content="Error: file not found. Please check the path.",
                tool_call_id="tc1",
            ),
            MemoSiftMessage(role="user", content="query"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=0))
        assert result[1].content_type == ContentType.TOOL_RESULT_TEXT

    def test_code_block_from_fenced(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(
                role="tool",
                content="Here's the code:\n```python\ndef foo():\n    pass\n```",
                tool_call_id="tc1",
            ),
            MemoSiftMessage(role="user", content="query"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=0))
        assert result[1].content_type == ContentType.CODE_BLOCK

    def test_code_block_from_tool_name(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(
                role="tool",
                content="def hello():\n    print('hi')\n",
                tool_call_id="tc1",
                name="read_file",
            ),
            MemoSiftMessage(role="user", content="query"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=0))
        assert result[1].content_type == ContentType.CODE_BLOCK

    def test_plain_text_tool_result(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(
                role="tool",
                content="File edited successfully.",
                tool_call_id="tc1",
            ),
            MemoSiftMessage(role="user", content="query"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=0))
        assert result[1].content_type == ContentType.TOOL_RESULT_TEXT


# ── Old conversation and assistant reasoning ────────────────────────────────


class TestOldMessages:
    """Old messages before the recent boundary."""

    def test_old_user_message(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="Old question"),
            MemoSiftMessage(role="assistant", content="Old answer"),
            MemoSiftMessage(role="user", content="New question"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=1))
        assert result[1].content_type == ContentType.OLD_CONVERSATION
        assert result[2].content_type == ContentType.ASSISTANT_REASONING

    def test_assistant_reasoning_not_protected(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="user", content="Old"),
            MemoSiftMessage(role="assistant", content="Reasoning"),
            MemoSiftMessage(role="user", content="Current"),
        ]
        result = classify_messages(msgs, MemoSiftConfig(recent_turns=1))
        assert result[2].protected is False


# ── Policy overrides ────────────────────────────────────────────────────────


class TestPolicyOverrides:
    """Config policy overrides should take effect."""

    def test_override_error_trace_to_preserve(self) -> None:
        traceback = (
            "Traceback (most recent call last):\n"
            '  File "a.py", line 1, in <module>\n'
            "    raise ValueError\n"
            '  File "b.py", line 2, in foo\n'
            "    raise ValueError\n"
            "ValueError: bad"
        )
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="tool", content=traceback, tool_call_id="tc1"),
            MemoSiftMessage(role="user", content="query"),
        ]
        config = MemoSiftConfig(
            recent_turns=0,
            policies={ContentType.ERROR_TRACE: CompressionPolicy.PRESERVE},
        )
        result = classify_messages(msgs, config)
        assert result[1].content_type == ContentType.ERROR_TRACE
        assert result[1].policy == CompressionPolicy.PRESERVE


# ── Edge cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_messages(self) -> None:
        result = classify_messages([], MemoSiftConfig())
        assert result == []

    def test_single_system_message(self) -> None:
        msgs = [MemoSiftMessage(role="system", content="System")]
        result = classify_messages(msgs, MemoSiftConfig())
        assert len(result) == 1
        assert result[0].content_type == ContentType.SYSTEM_PROMPT

    def test_original_index_preserved(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="S"),
            MemoSiftMessage(role="user", content="Q"),
        ]
        result = classify_messages(msgs, MemoSiftConfig())
        assert result[0].original_index == 0
        assert result[1].original_index == 1

    def test_no_user_messages(self) -> None:
        """A conversation with no user messages shouldn't crash."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="assistant", content="Hello"),
        ]
        result = classify_messages(msgs, MemoSiftConfig())
        assert result[0].content_type == ContentType.SYSTEM_PROMPT
        # Assistant with no user context — everything is recent (boundary=0).
        assert result[1].content_type == ContentType.RECENT_TURN


# ── Helper function unit tests ──────────────────────────────────────────────


class TestHelpers:
    """Direct tests for classification helper functions."""

    def test_find_nth_user_from_end(self) -> None:
        msgs = [
            MemoSiftMessage(role="user", content="U1"),
            MemoSiftMessage(role="assistant", content="A1"),
            MemoSiftMessage(role="user", content="U2"),
            MemoSiftMessage(role="assistant", content="A2"),
            MemoSiftMessage(role="user", content="U3"),
        ]
        assert _find_nth_user_message_from_end(msgs, 1) == 4
        assert _find_nth_user_message_from_end(msgs, 2) == 2
        assert _find_nth_user_message_from_end(msgs, 3) == 0
        assert _find_nth_user_message_from_end(msgs, 5) == 0

    def test_find_nth_user_zero_returns_len(self) -> None:
        msgs = [MemoSiftMessage(role="user", content="U1")]
        assert _find_nth_user_message_from_end(msgs, 0) == 1

    def test_last_user_index(self) -> None:
        msgs = [
            MemoSiftMessage(role="system", content="S"),
            MemoSiftMessage(role="user", content="U1"),
            MemoSiftMessage(role="assistant", content="A1"),
            MemoSiftMessage(role="user", content="U2"),
        ]
        assert _last_user_index(msgs) == 3

    def test_last_user_index_none(self) -> None:
        msgs = [MemoSiftMessage(role="system", content="S")]
        assert _last_user_index(msgs) == -1

    def test_is_valid_json_object(self) -> None:
        assert _is_valid_json('{"key": "value"}') is True

    def test_is_valid_json_array(self) -> None:
        assert _is_valid_json("[1, 2, 3]") is True

    def test_is_valid_json_bare_string(self) -> None:
        assert _is_valid_json('"hello"') is False

    def test_is_valid_json_bare_number(self) -> None:
        assert _is_valid_json("42") is False

    def test_is_valid_json_empty(self) -> None:
        assert _is_valid_json("") is False

    def test_is_valid_json_invalid(self) -> None:
        assert _is_valid_json("{broken json") is False

    def test_contains_error_trace_python(self) -> None:
        text = (
            "Traceback (most recent call last):\n"
            '  File "a.py", line 1\n'
            '  File "b.py", line 2\n'
            "  raise ValueError\n"
        )
        assert _contains_error_trace(text) is True

    def test_contains_error_trace_below_threshold(self) -> None:
        assert _contains_error_trace("Error: something broke") is False

    def test_contains_code_fenced(self) -> None:
        assert _contains_code("```python\nprint('hi')\n```") is True

    def test_contains_code_tool_name(self) -> None:
        assert _contains_code("some content", "read_file") is True

    def test_contains_code_no_match(self) -> None:
        assert _contains_code("plain text", "search") is False
