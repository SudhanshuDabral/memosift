# Tests for the Vercel AI SDK adapter (Python).
from __future__ import annotations

import json

from memosift.adapters.vercel_ai import adapt_in, adapt_out, compress_vercel_messages
from memosift.core.types import MemoSiftMessage


# ── adapt_in ─────────────────────────────────────────────────────────────────


def test_adapt_in_string_content():
    """String content should be converted directly."""
    messages = [{"role": "user", "content": "Hello, world!"}]
    result = adapt_in(messages)
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content == "Hello, world!"
    assert result[0].metadata["_vercel_content_type"] == "string"


def test_adapt_in_text_parts():
    """TextPart[] should be concatenated."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        }
    ]
    result = adapt_in(messages)
    assert len(result) == 1
    assert result[0].content == "Hello\nWorld"
    assert result[0].metadata["_vercel_content_type"] == "parts"


def test_adapt_in_tool_call_parts():
    """ToolCallPart should be converted to ToolCall."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool-call",
                    "toolCallId": "tc1",
                    "toolName": "read_file",
                    "args": {"path": "test.py"},
                },
            ],
        }
    ]
    result = adapt_in(messages)
    assert len(result) == 1
    assert result[0].tool_calls is not None
    assert len(result[0].tool_calls) == 1
    tc = result[0].tool_calls[0]
    assert tc.id == "tc1"
    assert tc.function.name == "read_file"
    assert json.loads(tc.function.arguments) == {"path": "test.py"}


def test_adapt_in_tool_result_parts():
    """ToolResultPart should produce tool role messages."""
    messages = [
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "tc1",
                    "toolName": "read_file",
                    "result": "file content here",
                }
            ],
        }
    ]
    result = adapt_in(messages)
    assert len(result) == 1
    assert result[0].role == "tool"
    assert result[0].content == "file content here"
    assert result[0].tool_call_id == "tc1"
    assert result[0].name == "read_file"


def test_adapt_in_tool_result_json():
    """JSON tool results should be serialized."""
    messages = [
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "tc2",
                    "toolName": "search",
                    "result": {"count": 42, "items": [1, 2, 3]},
                }
            ],
        }
    ]
    result = adapt_in(messages)
    assert result[0].content == json.dumps({"count": 42, "items": [1, 2, 3]})


def test_adapt_in_image_parts_preserved():
    """ImagePart should be skipped in content but preserved in metadata."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image", "image": "base64data", "mimeType": "image/png"},
            ],
        }
    ]
    result = adapt_in(messages)
    assert len(result) == 1
    assert result[0].content == "What is this?"
    preserved = result[0].metadata.get("_vercel_preserved_parts")
    assert preserved is not None
    assert len(preserved) == 1
    assert preserved[0]["type"] == "image"


def test_adapt_in_tool_result_with_error():
    """isError flag should be preserved."""
    messages = [
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "tc3",
                    "toolName": "run_test",
                    "result": "AssertionError: expected 1 got 2",
                    "isError": True,
                }
            ],
        }
    ]
    result = adapt_in(messages)
    assert result[0].metadata.get("_vercel_is_error") is True


# ── adapt_out ────────────────────────────────────────────────────────────────


def test_adapt_out_string_content():
    """Simple string content should produce {role, content: string}."""
    msgs = [
        MemoSiftMessage(
            role="user",
            content="Hello",
            metadata={"_vercel_content_type": "string"},
        )
    ]
    result = adapt_out(msgs)
    assert result[0] == {"role": "user", "content": "Hello"}


def test_adapt_out_tool_call():
    """ToolCall should be converted to ToolCallPart."""
    from memosift.core.types import ToolCall, ToolCallFunction

    msgs = [
        MemoSiftMessage(
            role="assistant",
            content="Checking.",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    function=ToolCallFunction(name="read_file", arguments='{"path": "x.py"}'),
                )
            ],
            metadata={"_vercel_content_type": "parts"},
        )
    ]
    result = adapt_out(msgs)
    assert result[0]["role"] == "assistant"
    parts = result[0]["content"]
    assert isinstance(parts, list)
    text_parts = [p for p in parts if p["type"] == "text"]
    tool_parts = [p for p in parts if p["type"] == "tool-call"]
    assert len(text_parts) == 1
    assert text_parts[0]["text"] == "Checking."
    assert len(tool_parts) == 1
    assert tool_parts[0]["toolCallId"] == "tc1"
    assert tool_parts[0]["toolName"] == "read_file"


def test_adapt_out_tool_result():
    """Tool result should produce {role: "tool", content: [ToolResultPart]}."""
    msgs = [
        MemoSiftMessage(
            role="tool",
            content='{"result": true}',
            tool_call_id="tc1",
            name="check",
            metadata={"_vercel_content_type": "tool-result", "_vercel_tool_name": "check"},
        )
    ]
    result = adapt_out(msgs)
    assert result[0]["role"] == "tool"
    parts = result[0]["content"]
    assert len(parts) == 1
    assert parts[0]["type"] == "tool-result"
    assert parts[0]["toolCallId"] == "tc1"
    assert parts[0]["result"] == {"result": True}


def test_adapt_out_preserves_image_parts():
    """Preserved image parts should appear in the output."""
    msgs = [
        MemoSiftMessage(
            role="user",
            content="Describe this",
            metadata={
                "_vercel_content_type": "parts",
                "_vercel_preserved_parts": [
                    {"type": "image", "image": "base64data", "mimeType": "image/png"}
                ],
            },
        )
    ]
    result = adapt_out(msgs)
    parts = result[0]["content"]
    assert isinstance(parts, list)
    types = [p["type"] for p in parts]
    assert "image" in types
    assert "text" in types


# ── Round-trip ───────────────────────────────────────────────────────────────


def test_round_trip_simple():
    """Simple messages should survive adapt_in → adapt_out losslessly."""
    original = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    internal = adapt_in(original)
    restored = adapt_out(internal)
    for orig, rest in zip(original, restored):
        assert orig["role"] == rest["role"]
        assert orig["content"] == rest["content"]


def test_round_trip_tool_calls():
    """Tool call messages should survive round-trip."""
    original = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool-call",
                    "toolCallId": "tc1",
                    "toolName": "read_file",
                    "args": {"path": "test.py"},
                },
            ],
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "tc1",
                    "toolName": "read_file",
                    "result": "def hello(): pass",
                }
            ],
        },
    ]
    internal = adapt_in(original)
    restored = adapt_out(internal)

    # Assistant message.
    assert restored[0]["role"] == "assistant"
    parts = restored[0]["content"]
    text_parts = [p for p in parts if p["type"] == "text"]
    tool_parts = [p for p in parts if p["type"] == "tool-call"]
    assert text_parts[0]["text"] == "Let me check."
    assert tool_parts[0]["toolCallId"] == "tc1"
    assert tool_parts[0]["toolName"] == "read_file"

    # Tool result.
    assert restored[1]["role"] == "tool"
    result_parts = restored[1]["content"]
    assert result_parts[0]["toolCallId"] == "tc1"
    assert result_parts[0]["result"] == "def hello(): pass"


# ── Compression ──────────────────────────────────────────────────────────────


async def test_compress_vercel_messages():
    """End-to-end compression should work with Vercel format."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What is Python?"},
    ]
    compressed, report = await compress_vercel_messages(messages)
    assert len(compressed) >= 1
    assert report.original_tokens > 0
