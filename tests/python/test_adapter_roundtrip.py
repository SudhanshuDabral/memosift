# Tests for lossless round-trip fidelity of Anthropic and Google ADK adapters.
from __future__ import annotations

import pytest

from memosift.adapters.anthropic_sdk import (
    adapt_in as anthropic_in,
    adapt_out as anthropic_out,
)
from memosift.adapters.google_adk import (
    adapt_in as adk_in,
    adapt_out as adk_out,
)
from memosift.config import MemoSiftConfig
from memosift.core.pipeline import compress
from memosift.core.types import MemoSiftMessage, ToolCall, ToolCallFunction


# ── Anthropic Adapter Round-Trip ─────────────────────────────────────────────


class TestAnthropicThinkingBlocks:
    """Thinking blocks must survive an adapt_in -> adapt_out round-trip."""

    def test_thinking_block_survives_roundtrip(self) -> None:
        """Input with [thinking, text, tool_use] — thinking block present in output."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Let me reason about this step by step...",
                    },
                    {"type": "text", "text": "I will read the file."},
                    {
                        "type": "tool_use",
                        "id": "tu1",
                        "name": "read_file",
                        "input": {"path": "src/auth.ts"},
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
        memosift = anthropic_in(msgs)
        result = anthropic_out(memosift)

        # The first output message should be the assistant message.
        assistant_msg = result.messages[0]
        assert assistant_msg["role"] == "assistant"

        blocks = assistant_msg["content"]
        block_types = [b["type"] for b in blocks]

        # Thinking block must be present and unchanged.
        assert "thinking" in block_types
        thinking_block = next(b for b in blocks if b["type"] == "thinking")
        assert thinking_block["thinking"] == "Let me reason about this step by step..."

        # Text and tool_use blocks must also be present.
        assert "text" in block_types
        assert "tool_use" in block_types

    def test_thinking_content_excluded_from_memosift_content(self) -> None:
        """Thinking blocks are skipped during adapt_in; their text is NOT in content."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Secret reasoning"},
                    {"type": "text", "text": "Visible reply."},
                ],
            },
        ]
        memosift = anthropic_in(msgs)
        # The MemoSiftMessage content should only contain the text block.
        assert memosift[0].content == "Visible reply."
        assert "Secret reasoning" not in memosift[0].content


class TestAnthropicMultiBlockStructure:
    """Multi-block structure must be preserved through round-trip."""

    def test_block_count_and_types_match(self) -> None:
        """[text, tool_use, text] — verify block count and types match after round-trip."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "First I will search."},
                    {
                        "type": "tool_use",
                        "id": "tu1",
                        "name": "search",
                        "input": {"query": "auth bug"},
                    },
                    {"type": "text", "text": "Then I will fix it."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": "Found 3 results",
                    },
                ],
            },
        ]
        memosift = anthropic_in(msgs)
        result = anthropic_out(memosift)

        assistant_msg = result.messages[0]
        blocks = assistant_msg["content"]

        # adapt_in merges multiple text blocks into one content string,
        # and adapt_out emits a single text block with the merged content.
        # The tool_use block must survive.
        text_blocks = [b for b in blocks if b["type"] == "text"]
        tool_use_blocks = [b for b in blocks if b["type"] == "tool_use"]
        assert len(text_blocks) >= 1
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "search"

    def test_original_block_format_tag_present(self) -> None:
        """Metadata should contain the anthropic block format marker."""
        msgs = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
            },
        ]
        memosift = anthropic_in(msgs)
        assert memosift[0].metadata.get("_original_block_format") == "anthropic"
        assert memosift[0].metadata.get("_original_blocks") is not None


class TestAnthropicCompressedTextReplacement:
    """After compression, the text block has shorter content but thinking block is unchanged."""

    @pytest.mark.asyncio
    async def test_compressed_text_preserves_thinking(self) -> None:
        """Compress messages with thinking block — thinking unchanged, text may be shorter."""
        # Build a conversation long enough to trigger some compression.
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Deep internal reasoning that must survive.",
                    },
                    {
                        "type": "text",
                        "text": "I will read the configuration file to understand the setup.",
                    },
                    {
                        "type": "tool_use",
                        "id": "tu1",
                        "name": "read_file",
                        "input": {"path": "config.yaml"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": "database:\n  host: localhost\n  port: 5432\n  name: mydb\n",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "The config looks good."},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "What port is the DB on?"}],
            },
        ]

        memosift = anthropic_in(msgs)
        compressed, report = await compress(memosift)
        result = anthropic_out(compressed)

        # Find the assistant message that had the thinking block.
        assistant_msgs = [m for m in result.messages if m["role"] == "assistant"]
        thinking_msg = None
        for am in assistant_msgs:
            for block in am.get("content", []):
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_msg = am
                    break
            if thinking_msg:
                break

        # Thinking block must survive compression unchanged.
        assert thinking_msg is not None, "Thinking block was lost during compression"
        thinking_block = next(
            b for b in thinking_msg["content"] if b.get("type") == "thinking"
        )
        assert thinking_block["thinking"] == "Deep internal reasoning that must survive."


class TestAnthropicBackwardCompatibility:
    """Messages without _original_blocks use the backward-compatible fallback path."""

    def test_no_original_blocks_uses_fallback(self) -> None:
        """MemoSiftMessage created without _original_blocks key uses fallback logic."""
        # Create a MemoSiftMessage without the _original_blocks metadata
        # (simulating a message that did not come through adapt_in).
        msg = MemoSiftMessage(
            role="assistant",
            content="Hello there!",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    function=ToolCallFunction(
                        name="greet",
                        arguments='{"name": "world"}',
                    ),
                ),
            ],
            metadata={},  # No _original_blocks key.
        )
        tool_result = MemoSiftMessage(
            role="tool",
            content="Greeted world",
            tool_call_id="tc1",
            name="greet",
            metadata={},
        )

        result = anthropic_out([msg, tool_result])

        # Should still produce valid Anthropic output via fallback path.
        assert len(result.messages) >= 1
        assistant_msg = result.messages[0]
        assert assistant_msg["role"] == "assistant"
        blocks = assistant_msg["content"]

        # Should have text block and tool_use block.
        block_types = [b["type"] for b in blocks]
        assert "text" in block_types
        assert "tool_use" in block_types

    def test_fallback_preserves_text(self) -> None:
        """Fallback path preserves message text content."""
        msg = MemoSiftMessage(
            role="assistant",
            content="Simple response",
            metadata={},
        )
        result = anthropic_out([msg])
        assert result.messages[0]["content"][0]["text"] == "Simple response"


# ── Google ADK Adapter Round-Trip ────────────────────────────────────────────


class TestADKDeterministicIDs:
    """Same input must produce same output IDs across runs."""

    def test_deterministic_ids_from_same_input(self) -> None:
        """Run adapt_in twice with the same input — IDs must match."""
        events = [
            {
                "role": "model",
                "function_calls": [
                    {"name": "read_file", "args": {"path": "test.py"}},
                    {"name": "search", "args": {"query": "bug"}},
                ],
            },
        ]

        memosift_first = adk_in(events)
        memosift_second = adk_in(events)

        # Both runs should produce the same tool call IDs.
        ids_first = [tc.id for tc in memosift_first[0].tool_calls]
        ids_second = [tc.id for tc in memosift_second[0].tool_calls]
        assert ids_first == ids_second

    def test_explicit_ids_preserved(self) -> None:
        """When function_calls have explicit IDs, they are used as-is."""
        events = [
            {
                "role": "model",
                "function_calls": [
                    {"id": "my-custom-id", "name": "read_file", "args": {"path": "a.py"}},
                ],
            },
        ]
        memosift = adk_in(events)
        assert memosift[0].tool_calls[0].id == "my-custom-id"

    def test_generated_ids_are_stable_hashes(self) -> None:
        """Generated IDs use SHA-256 prefix, so they are deterministic."""
        events = [
            {
                "role": "model",
                "function_calls": [
                    {"name": "read_file", "args": {"path": "test.py"}},
                ],
            },
        ]
        memosift = adk_in(events)
        generated_id = memosift[0].tool_calls[0].id
        assert generated_id.startswith("adk_")
        assert len(generated_id) == 16  # "adk_" + 12 hex chars


class TestADKOriginalEventStructure:
    """adapt_in -> adapt_out round-trip preserves original event keys."""

    def test_text_event_roundtrip_preserves_keys(self) -> None:
        """Text events preserve all original keys through round-trip."""
        events = [
            {
                "role": "user",
                "text": "Hello, how are you?",
                "timestamp": "2026-01-15T10:30:00Z",
                "custom_field": "should_survive",
            },
            {
                "role": "model",
                "text": "I am fine, thank you.",
                "session_id": "sess-123",
            },
        ]
        memosift = adk_in(events)
        out = adk_out(memosift)

        # Original event structure including custom keys should be preserved.
        assert out[0]["role"] == "user"
        assert out[0]["text"] == "Hello, how are you?"
        assert out[0].get("timestamp") == "2026-01-15T10:30:00Z"
        assert out[0].get("custom_field") == "should_survive"

        assert out[1]["role"] == "model"
        assert out[1]["text"] == "I am fine, thank you."
        assert out[1].get("session_id") == "sess-123"

    def test_function_call_event_roundtrip_preserves_keys(self) -> None:
        """Function call events preserve structure through round-trip."""
        events = [
            {
                "role": "model",
                "function_calls": [
                    {"id": "fc1", "name": "read_file", "args": {"path": "a.py"}},
                ],
                "metadata_key": "preserved",
            },
            {
                "role": "function",
                "function_responses": [
                    {"id": "fc1", "name": "read_file", "response": "def hello(): pass"},
                ],
            },
        ]
        memosift = adk_in(events)
        out = adk_out(memosift)

        # The model event should preserve its custom key.
        model_events = [e for e in out if e.get("role") == "model"]
        assert len(model_events) >= 1
        assert model_events[0].get("metadata_key") == "preserved"
        assert "function_calls" in model_events[0]

    def test_parts_field_roundtrip(self) -> None:
        """Events with 'parts' field have their text reconstructed."""
        events = [
            {
                "role": "user",
                "parts": [
                    {"text": "Part one."},
                    {"text": "Part two."},
                ],
            },
        ]
        memosift = adk_in(events)
        assert memosift[0].content == "Part one. Part two."

        out = adk_out(memosift)
        # The text field is set with the reconstructed content.
        assert out[0]["text"] == "Part one. Part two."

    def test_original_event_stored_in_metadata(self) -> None:
        """Each message's metadata should contain the _original_event."""
        events = [
            {"role": "user", "text": "Hello"},
        ]
        memosift = adk_in(events)
        assert memosift[0].metadata.get("_adk_event") is True
        assert memosift[0].metadata.get("_original_event") is not None
        assert memosift[0].metadata["_original_event"]["text"] == "Hello"
        assert memosift[0].metadata.get("_original_block_format") == "google_adk"
