# Tests for framework auto-detection.
from __future__ import annotations

import pytest

from memosift.core.types import MemoSiftMessage
from memosift.detect import VALID_FRAMEWORKS, detect_framework


class TestDetectFramework:
    """Test detect_framework() heuristics."""

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            detect_framework([])

    def test_all_none_returns_openai(self):
        assert detect_framework([None, None]) == "openai"

    # ── MemoSiftMessage ──────────────────────────────────────────────

    def test_memosift_messages(self):
        msgs = [
            MemoSiftMessage(role="user", content="hello"),
            MemoSiftMessage(role="assistant", content="hi"),
        ]
        assert detect_framework(msgs) == "memosift"

    # ── OpenAI ───────────────────────────────────────────────────────

    def test_openai_simple(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        assert detect_framework(msgs) == "openai"

    def test_openai_with_tool_calls(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "result", "tool_call_id": "tc1"},
        ]
        assert detect_framework(msgs) == "openai"

    # ── Anthropic ────────────────────────────────────────────────────

    def test_anthropic_content_blocks(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        ]
        assert detect_framework(msgs) == "anthropic"

    def test_anthropic_with_tool_use_blocks(self):
        msgs = [
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "tu1", "name": "read_file", "input": {}},
            ]},
        ]
        assert detect_framework(msgs) == "anthropic"

    def test_anthropic_string_content_falls_to_openai(self):
        """Anthropic messages with string content (valid but simplified) look like OpenAI."""
        msgs = [{"role": "user", "content": "Hello"}]
        assert detect_framework(msgs) == "openai"

    # ── Google ADK ───────────────────────────────────────────────────

    def test_adk_function_calls(self):
        msgs = [
            {"role": "model", "function_calls": [{"name": "search", "args": {"q": "test"}}]},
        ]
        assert detect_framework(msgs) == "adk"

    def test_adk_function_responses(self):
        msgs = [
            {"role": "function", "function_responses": [{"name": "search", "response": {}}]},
        ]
        assert detect_framework(msgs) == "adk"

    def test_adk_parts_with_function_call(self):
        msgs = [
            {"role": "model", "parts": [{"function_call": {"name": "search", "args": {}}}]},
        ]
        assert detect_framework(msgs) == "adk"

    # ── LangChain ────────────────────────────────────────────────────

    def test_langchain_with_additional_kwargs(self):
        class FakeHumanMessage:
            additional_kwargs: dict = {}
            content = "Hello"
            type = "human"

        msgs = [FakeHumanMessage()]
        assert detect_framework(msgs) == "langchain"

    def test_langchain_dict_with_additional_kwargs(self):
        """LangChain messages serialized as dicts."""
        msgs = [{"role": "user", "content": "Hi", "additional_kwargs": {}}]
        assert detect_framework(msgs) == "langchain"

    # ── Claude Agent SDK ─────────────────────────────────────────────

    def test_agent_sdk_typed_objects(self):
        class SystemMessage:
            pass

        class AssistantMessage:
            pass

        msgs = [SystemMessage(), AssistantMessage()]
        assert detect_framework(msgs) == "agent_sdk"

    def test_agent_sdk_not_confused_with_langchain(self):
        """Agent SDK type name but with additional_kwargs → LangChain wins."""
        class SystemMessage:
            additional_kwargs: dict = {}

        msgs = [SystemMessage()]
        assert detect_framework(msgs) == "langchain"

    # ── Priority / Edge Cases ────────────────────────────────────────

    def test_memosift_takes_priority_over_openai(self):
        """MemoSiftMessage has role+content like OpenAI dicts, but should detect as memosift."""
        msgs = [MemoSiftMessage(role="user", content="hello")]
        assert detect_framework(msgs) == "memosift"

    def test_adk_takes_priority_over_anthropic(self):
        """ADK with content blocks should still detect as ADK due to function_calls."""
        msgs = [
            {"role": "model", "content": [{"type": "text", "text": "hi"}], "function_calls": []},
        ]
        assert detect_framework(msgs) == "adk"

    def test_valid_frameworks_constant(self):
        assert VALID_FRAMEWORKS == {"openai", "anthropic", "agent_sdk", "adk", "langchain", "memosift"}
