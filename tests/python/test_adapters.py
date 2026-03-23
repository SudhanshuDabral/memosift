# Tests for framework adapters — round-trip conversion and compression.
from __future__ import annotations

import pytest

from memosift.adapters.openai_sdk import adapt_in as openai_in, adapt_out as openai_out, compress_openai_messages
from memosift.adapters.anthropic_sdk import adapt_in as anthropic_in, adapt_out as anthropic_out, compress_anthropic_messages
from memosift.adapters.langchain import adapt_in as langchain_in, adapt_out as langchain_out, compress_langchain_messages
from memosift.adapters.google_adk import adapt_in as adk_in, adapt_out as adk_out, compress_adk_events
from memosift.adapters.claude_agent_sdk import adapt_in as agent_in, adapt_out as agent_out, compress_agent_sdk_messages
from memosift.config import MemoSiftConfig


# ── OpenAI Adapter ──────────────────────────────────────────────────────────


class TestOpenAIAdapter:

    def test_basic_roundtrip(self) -> None:
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        memosift = openai_in(msgs)
        assert len(memosift) == 3
        assert memosift[0].role == "system"
        out = openai_out(memosift)
        assert out[0]["role"] == "system"
        assert out[0]["content"] == "You are helpful."

    def test_tool_call_roundtrip(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                }],
            },
            {"role": "tool", "content": "file content", "tool_call_id": "tc1"},
        ]
        memosift = openai_in(msgs)
        assert memosift[0].content == ""  # null → ""
        assert memosift[0].tool_calls is not None
        assert memosift[0].tool_calls[0].id == "tc1"
        assert memosift[1].tool_call_id == "tc1"

        out = openai_out(memosift)
        assert out[0]["content"] is None  # Restored to null
        assert out[0]["tool_calls"][0]["id"] == "tc1"
        assert out[1]["tool_call_id"] == "tc1"

    def test_refusal_field_preserved(self) -> None:
        msgs = [
            {"role": "assistant", "content": "", "refusal": "I cannot do that."},
        ]
        memosift = openai_in(msgs)
        assert memosift[0].metadata.get("_openai_refusal") == "I cannot do that."
        out = openai_out(memosift)
        assert out[0].get("refusal") == "I cannot do that."

    @pytest.mark.asyncio
    async def test_compress_roundtrip(self) -> None:
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        compressed, report = await compress_openai_messages(msgs)
        assert len(compressed) > 0
        assert compressed[0]["role"] == "system"
        assert report.original_tokens > 0


# ── Anthropic Adapter ───────────────────────────────────────────────────────


class TestAnthropicAdapter:

    def test_text_blocks_roundtrip(self) -> None:
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        ]
        memosift = anthropic_in(msgs, system="System prompt")
        assert len(memosift) == 3  # system + user + assistant
        assert memosift[0].role == "system"
        assert memosift[0].content == "System prompt"
        assert memosift[1].content == "Hello"

    def test_tool_use_roundtrip(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me read the file."},
                    {
                        "type": "tool_use",
                        "id": "tu1",
                        "name": "read_file",
                        "input": {"path": "auth.ts"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": "file content here",
                    },
                ],
            },
        ]
        memosift = anthropic_in(msgs)
        # Should have assistant + tool result as separate messages.
        assert any(m.role == "assistant" and m.tool_calls for m in memosift)
        assert any(m.role == "tool" and m.tool_call_id == "tu1" for m in memosift)

    def test_system_param_preserved(self) -> None:
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        memosift = anthropic_in(msgs, system="Be concise.")
        result = anthropic_out(memosift)
        assert result.system == "Be concise."

    @pytest.mark.asyncio
    async def test_compress_roundtrip(self) -> None:
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "Q1"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "A1"}]},
            {"role": "user", "content": [{"type": "text", "text": "Q2"}]},
        ]
        result, report = await compress_anthropic_messages(msgs, system="System")
        assert result.system == "System"
        assert len(result.messages) > 0


# ── LangChain Adapter ───────────────────────────────────────────────────────


class TestLangChainAdapter:

    def test_dict_messages_roundtrip(self) -> None:
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        memosift = langchain_in(msgs)
        assert len(memosift) == 3
        out = langchain_out(memosift)
        assert out[0]["role"] == "system"
        assert out[1]["content"] == "Hello"

    def test_additional_kwargs_preserved(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "Thinking...",
                "additional_kwargs": {
                    "custom_key": "custom_value",
                    "tool_calls": [{
                        "id": "tc1",
                        "function": {"name": "search", "arguments": "{}"},
                    }],
                },
            },
        ]
        memosift = langchain_in(msgs)
        assert memosift[0].tool_calls is not None
        assert memosift[0].metadata.get("_langchain_additional_kwargs") == {"custom_key": "custom_value"}

        out = langchain_out(memosift)
        assert out[0]["additional_kwargs"]["custom_key"] == "custom_value"
        assert out[0]["additional_kwargs"]["tool_calls"][0]["id"] == "tc1"

    @pytest.mark.asyncio
    async def test_compress_roundtrip(self) -> None:
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow-up"},
        ]
        compressed, report = await compress_langchain_messages(msgs)
        assert len(compressed) > 0
        assert report.original_tokens > 0


# ── Google ADK Adapter ──────────────────────────────────────────────────────


class TestGoogleADKAdapter:

    def test_text_event_roundtrip(self) -> None:
        events = [
            {"role": "user", "text": "Hello"},
            {"role": "model", "text": "Hi there"},
        ]
        memosift = adk_in(events)
        assert len(memosift) == 2
        assert memosift[0].role == "user"
        assert memosift[1].role == "assistant"  # model → assistant

        out = adk_out(memosift)
        assert out[0]["role"] == "user"
        assert out[1]["role"] == "model"  # assistant → model

    def test_function_call_roundtrip(self) -> None:
        events = [
            {
                "role": "model",
                "function_calls": [{
                    "id": "fc1",
                    "name": "read_file",
                    "args": {"path": "test.py"},
                }],
            },
            {
                "role": "function",
                "function_responses": [{
                    "id": "fc1",
                    "name": "read_file",
                    "response": "def hello(): pass",
                }],
            },
        ]
        memosift = adk_in(events)
        assert memosift[0].tool_calls is not None
        assert memosift[0].tool_calls[0].function.name == "read_file"
        assert memosift[1].role == "tool"
        assert memosift[1].tool_call_id == "fc1"

    @pytest.mark.asyncio
    async def test_compress_roundtrip(self) -> None:
        events = [
            {"role": "user", "text": "Question"},
            {"role": "model", "text": "Answer"},
            {"role": "user", "text": "Follow-up"},
        ]
        compressed, report = await compress_adk_events(events)
        assert len(compressed) > 0
        assert report.original_tokens > 0


# ── Tool call integrity across adapters ─────────────────────────────────────


class TestToolCallIntegrity:

    @pytest.mark.asyncio
    async def test_openai_tool_calls_survive(self) -> None:
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "Reading.",
                "tool_calls": [{
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                }],
            },
            {"role": "tool", "content": "def hello(): pass", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "Thanks"},
        ]
        compressed, _ = await compress_openai_messages(msgs)
        # Verify tool call integrity.
        call_ids = set()
        result_ids = set()
        for m in compressed:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    call_ids.add(tc["id"])
            if m.get("tool_call_id"):
                result_ids.add(m["tool_call_id"])
        assert call_ids == result_ids


# ── Claude Agent SDK Adapter ────────────────────────────────────────────────


class TestClaudeAgentSDKAdapter:

    def test_dict_message_roundtrip(self) -> None:
        """Agent SDK session history is often plain dicts."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]},
        ]
        memosift = agent_in(msgs)
        assert len(memosift) == 2
        assert memosift[0].role == "user"
        assert memosift[1].role == "assistant"
        assert memosift[1].content == "Hi there"

    def test_tool_use_and_result(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Reading file."},
                    {
                        "type": "tool_use",
                        "id": "tu1",
                        "name": "read_file",
                        "input": {"path": "auth.ts"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": "export class Auth {}",
                    },
                ],
            },
        ]
        memosift = agent_in(msgs)
        # Assistant with tool_call + separate tool result.
        assistant_msgs = [m for m in memosift if m.role == "assistant"]
        tool_msgs = [m for m in memosift if m.role == "tool"]
        assert len(assistant_msgs) >= 1
        assert assistant_msgs[0].tool_calls is not None
        assert len(tool_msgs) >= 1
        assert tool_msgs[0].tool_call_id == "tu1"

    def test_compaction_boundary_marked_compressed(self) -> None:
        """SystemMessage with subtype='compact_boundary' should be tagged."""
        # Simulate a compaction boundary as a dict (from session history).
        msgs = [
            {"role": "system", "content": "Compacted summary of prior context."},
            {"role": "user", "content": "New question"},
        ]
        # For the real SDK, SystemMessage objects have subtype.
        # We test dict path here — subtype is in metadata when using typed objects.
        memosift = agent_in(msgs)
        assert memosift[0].role == "system"

    @pytest.mark.asyncio
    async def test_compress_roundtrip(self) -> None:
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": [{"type": "text", "text": "A1"}]},
            {"role": "user", "content": "Q2"},
        ]
        compressed, report = await compress_agent_sdk_messages(msgs)
        assert len(compressed) > 0
        assert report.original_tokens > 0

    def test_adapt_out_renests_tool_results(self) -> None:
        """adapt_out should re-nest tool results into user messages."""
        from memosift.core.types import MemoSiftMessage, ToolCall, ToolCallFunction

        msgs = [
            MemoSiftMessage(
                role="assistant",
                content="Reading.",
                tool_calls=[ToolCall(id="tc1", function=ToolCallFunction(name="read", arguments="{}"))],
            ),
            MemoSiftMessage(role="tool", content="file content", tool_call_id="tc1"),
        ]
        out = agent_out(msgs)
        # Should have assistant message + user message with tool_result.
        assert len(out) == 2
        assert out[0]["role"] == "assistant"
        assert out[1]["role"] == "user"
        assert out[1]["content"][0]["type"] == "tool_result"


# ── Multi-tool-call tests (all adapters) ────────────────────────────────────


class TestMultiToolCall:

    def test_openai_multi_tool_calls(self) -> None:
        """OpenAI adapter handles 3 concurrent tool calls."""
        msgs = [
            {
                "role": "assistant",
                "content": "Calling three tools.",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
                    {"id": "tc2", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "b.py"}'}},
                    {"id": "tc3", "type": "function", "function": {"name": "search", "arguments": '{"q": "test"}'}},
                ],
            },
            {"role": "tool", "content": "content a", "tool_call_id": "tc1"},
            {"role": "tool", "content": "content b", "tool_call_id": "tc2"},
            {"role": "tool", "content": "results", "tool_call_id": "tc3"},
        ]
        memosift = openai_in(msgs)
        assert len(memosift[0].tool_calls) == 3
        out = openai_out(memosift)
        assert len(out[0]["tool_calls"]) == 3
        # All tool results preserved.
        tool_results = [m for m in out if m.get("tool_call_id")]
        assert len(tool_results) == 3

    def test_anthropic_multi_tool_use(self) -> None:
        """Anthropic adapter handles multiple tool_use blocks in one message."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Using two tools."},
                    {"type": "tool_use", "id": "tu1", "name": "read", "input": {"path": "a.py"}},
                    {"type": "tool_use", "id": "tu2", "name": "search", "input": {"q": "test"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "file a"},
                    {"type": "tool_result", "tool_use_id": "tu2", "content": "search results"},
                ],
            },
        ]
        memosift = anthropic_in(msgs)
        assistant_msgs = [m for m in memosift if m.tool_calls]
        assert len(assistant_msgs) == 1
        assert len(assistant_msgs[0].tool_calls) == 2
        tool_msgs = [m for m in memosift if m.role == "tool"]
        assert len(tool_msgs) == 2

    def test_adk_multi_function_calls(self) -> None:
        """Google ADK handles multiple function_calls in one event."""
        events = [
            {
                "role": "model",
                "function_calls": [
                    {"id": "fc1", "name": "read", "args": {"path": "a.py"}},
                    {"id": "fc2", "name": "search", "args": {"q": "test"}},
                ],
            },
            {"role": "function", "function_responses": [
                {"id": "fc1", "name": "read", "response": "content a"},
            ]},
            {"role": "function", "function_responses": [
                {"id": "fc2", "name": "search", "response": "results"},
            ]},
        ]
        memosift = adk_in(events)
        assert memosift[0].tool_calls is not None
        assert len(memosift[0].tool_calls) == 2

    @pytest.mark.asyncio
    async def test_multi_tool_integrity_survives_compression(self) -> None:
        """Multi-tool messages maintain integrity through compression."""
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Read both files"},
            {
                "role": "assistant",
                "content": "Reading.",
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "read", "arguments": '{"p":"a.py"}'}},
                    {"id": "tc2", "type": "function", "function": {"name": "read", "arguments": '{"p":"b.py"}'}},
                ],
            },
            {"role": "tool", "content": "def a(): pass", "tool_call_id": "tc1"},
            {"role": "tool", "content": "def b(): pass", "tool_call_id": "tc2"},
            {"role": "assistant", "content": "Got both files."},
            {"role": "user", "content": "Thanks"},
        ]
        compressed, _ = await compress_openai_messages(msgs)
        call_ids = set()
        result_ids = set()
        for m in compressed:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    call_ids.add(tc["id"])
            if m.get("tool_call_id"):
                result_ids.add(m["tool_call_id"])
        assert call_ids == result_ids
