# Claude Agent SDK adapter — handles stateful sessions, compaction boundaries, hooks.
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from memosift.core.pipeline import compress
from memosift.core.types import (
    AnchorLedger,
    MemoSiftMessage,
    ToolCall,
    ToolCallFunction,
)
from memosift.providers.base import LLMResponse, MemoSiftLLMProvider

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig
    from memosift.report import CompressionReport


class ClaudeAgentLLMProvider:
    """Wraps a Claude Agent SDK client into MemoSiftLLMProvider.

    Uses the underlying Anthropic client for generation and token counting.
    """

    def __init__(self, client: Any, model: str = "claude-sonnet-4-6"):
        self._client = client
        self._model = model

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate text via the underlying Anthropic Messages API."""
        # The Agent SDK wraps an Anthropic client internally.
        anthropic_client = getattr(self._client, "_client", self._client)
        resp = await anthropic_client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(
            text=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    async def count_tokens(self, text: str) -> int:
        """Count tokens using heuristic (Agent SDK doesn't expose tokenizer)."""
        import math

        return math.ceil(len(text) / 3.5)


def adapt_in(messages: list[Any]) -> list[MemoSiftMessage]:
    """Convert Claude Agent SDK message objects to MemoSiftMessage list.

    Handles:
    - ``AssistantMessage`` with ``TextBlock`` and ``ToolUseBlock`` content
    - ``UserMessage`` with ``ToolResultBlock`` content
    - ``SystemMessage`` with ``subtype`` (init, compact_boundary)
    - Plain dicts (from ``get_session_messages()``)
    - Compaction boundaries are tagged as already compressed
    """
    result: list[MemoSiftMessage] = []

    for msg in messages:
        # Handle dict-style messages (from session history).
        if isinstance(msg, dict):
            result.extend(_adapt_dict_message(msg))
            continue

        # Handle typed Agent SDK message objects.
        msg_type = type(msg).__name__
        role = getattr(msg, "role", None)
        content_blocks = getattr(msg, "content", [])

        if msg_type == "SystemMessage":
            subtype = getattr(msg, "subtype", "")
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                content = " ".join(getattr(b, "text", str(b)) for b in content)
            dm = MemoSiftMessage(
                role="system",
                content=content if isinstance(content, str) else str(content),
                metadata={"_agent_sdk_subtype": subtype},
            )
            # Mark compaction boundaries as already compressed.
            if subtype == "compact_boundary":
                dm._memosift_compressed = True
            result.append(dm)
            continue

        if msg_type == "ResultMessage":
            # Terminal message with metrics — skip (not conversation content).
            continue

        # Process content blocks (AssistantMessage, UserMessage).
        # Preserve original blocks for lossless round-trip reconstruction.
        original_blocks = list(content_blocks) if isinstance(content_blocks, list) else None

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tool_results: list[tuple[str, str]] = []  # (tool_use_id, content)

        if isinstance(content_blocks, str):
            text_parts.append(content_blocks)
        elif isinstance(content_blocks, list):
            for block in content_blocks:
                block_type = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else None
                )
                if block_type == "thinking":
                    # Thinking blocks must not be compressed; skip them so they
                    # are preserved verbatim via _original_blocks round-trip.
                    continue
                elif block_type == "text":
                    text_parts.append(
                        getattr(block, "text", "") or block.get("text", "")
                        if isinstance(block, dict)
                        else getattr(block, "text", "")
                    )
                elif block_type == "tool_use":
                    tc_id = getattr(block, "id", "") or block.get("id", "")
                    tc_name = getattr(block, "name", "") or block.get("name", "")
                    tc_input = getattr(block, "input", {}) or block.get("input", {})
                    tool_calls.append(
                        ToolCall(
                            id=tc_id,
                            function=ToolCallFunction(
                                name=tc_name,
                                arguments=(
                                    json.dumps(tc_input)
                                    if isinstance(tc_input, dict)
                                    else str(tc_input)
                                ),
                            ),
                        )
                    )
                elif block_type == "tool_result":
                    tr_id = getattr(block, "tool_use_id", "") or block.get("tool_use_id", "")
                    tr_content = getattr(block, "content", "") or block.get("content", "")
                    if isinstance(tr_content, list):
                        tr_content = " ".join(
                            getattr(b, "text", str(b))
                            if not isinstance(b, dict)
                            else b.get("text", str(b))
                            for b in tr_content
                        )
                    tool_results.append((tr_id, str(tr_content)))
                elif isinstance(block, str):
                    text_parts.append(block)

        combined_text = "\n".join(text_parts) if text_parts else ""
        effective_role = role or ("assistant" if tool_calls else "user")

        result.append(
            MemoSiftMessage(
                role=effective_role,
                content=combined_text,
                tool_calls=tool_calls if tool_calls else None,
                metadata={
                    "_agent_sdk_type": msg_type,
                    "_original_blocks": original_blocks,
                    "_original_block_format": "claude_agent_sdk",
                },
            )
        )

        # Emit tool results as separate tool messages.
        for tool_use_id, tr_content in tool_results:
            result.append(
                MemoSiftMessage(
                    role="tool",
                    content=tr_content,
                    tool_call_id=tool_use_id,
                )
            )

    return result


def adapt_out(messages: list[MemoSiftMessage]) -> list[dict[str, Any]]:
    """Convert MemoSiftMessage list back to Anthropic-compatible format.

    Returns dicts suitable for the Anthropic Messages API, which the
    Agent SDK uses internally. Tool results are re-nested into user messages.
    """
    result: list[dict[str, Any]] = []

    tool_result_map: dict[str, MemoSiftMessage] = {}
    for msg in messages:
        if msg.role == "tool" and msg.tool_call_id:
            tool_result_map[msg.tool_call_id] = msg

    consumed_tool_ids: set[str] = set()

    for msg in messages:
        if msg.role == "tool":
            continue  # Re-nested below.

        blocks: list[dict[str, Any]] = []

        original_blocks = msg.metadata.get("_original_blocks")
        if original_blocks is not None:
            # Lossless round-trip: walk the original blocks and patch in
            # compressed content / surviving tool_calls.
            tc_map: dict[str, ToolCall] = {}
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_map[tc.id] = tc

            text_replaced = False
            for orig_block in original_blocks:
                # Handle both typed objects and dicts.
                btype = getattr(orig_block, "type", None) or (
                    orig_block.get("type") if isinstance(orig_block, dict) else None
                )

                if btype == "text":
                    if not text_replaced:
                        # First text block gets the (possibly compressed) content.
                        blocks.append({"type": "text", "text": msg.content})
                        text_replaced = True
                    # Subsequent text blocks are dropped — their content was
                    # merged into msg.content during adapt_in.

                elif btype == "thinking":
                    # Pass thinking blocks through unchanged.
                    if isinstance(orig_block, dict):
                        blocks.append(dict(orig_block))
                    else:
                        # Convert typed object to dict.
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": getattr(orig_block, "thinking", ""),
                            }
                        )

                elif btype == "tool_use":
                    tc_id = getattr(orig_block, "id", "") or (
                        orig_block.get("id", "") if isinstance(orig_block, dict) else ""
                    )
                    if tc_id in tc_map:
                        tc = tc_map[tc_id]
                        try:
                            input_data = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, ValueError):
                            input_data = {"raw": tc.function.arguments}
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.function.name,
                                "input": input_data,
                            }
                        )

                else:
                    # Unknown / future block types — pass through unchanged.
                    if isinstance(orig_block, dict):
                        blocks.append(dict(orig_block))
                    elif isinstance(orig_block, str):
                        blocks.append(orig_block)
                    else:
                        # Convert typed object to dict representation.
                        blocks.append({"type": btype} if btype else orig_block)

            # If no text block existed in originals but we have content, add it.
            if not text_replaced and msg.content:
                blocks.insert(0, {"type": "text", "text": msg.content})
        else:
            # Fallback: reconstruct from scratch.
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})

            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        input_data = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, ValueError):
                        input_data = {"raw": tc.function.arguments}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": input_data,
                        }
                    )

        result.append({"role": msg.role, "content": blocks})

        # Re-nest tool results after assistant tool_use messages.
        if msg.role == "assistant" and msg.tool_calls:
            tr_blocks: list[dict[str, Any]] = []
            for tc in msg.tool_calls:
                if tc.id in tool_result_map and tc.id not in consumed_tool_ids:
                    tr_msg = tool_result_map[tc.id]
                    tr_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": tr_msg.content,
                        }
                    )
                    consumed_tool_ids.add(tc.id)
            if tr_blocks:
                result.append({"role": "user", "content": tr_blocks})

    return result


async def compress_agent_sdk_messages(
    messages: list[Any],
    *,
    llm: MemoSiftLLMProvider | None = None,
    config: MemoSiftConfig | None = None,
    task: str | None = None,
    ledger: AnchorLedger | None = None,
    client: Any = None,
    model: str = "claude-sonnet-4-6",
) -> tuple[list[dict[str, Any]], CompressionReport]:
    """Compress Claude Agent SDK session messages end-to-end.

    Respects compaction boundaries (``compact_boundary`` markers are
    tagged ``_memosift_compressed=True`` and skip re-compression).

    Args:
        messages: Agent SDK message objects or session history dicts.
        llm: MemoSiftLLMProvider (or auto-wrap ``client``).
        config: Pipeline configuration.
        task: Optional task description.
        ledger: Optional AnchorLedger.
        client: Claude Agent SDK client for auto-wrapping.
        model: Model name for auto-wrapping.

    Returns:
        Tuple of (compressed Anthropic-format messages, compression report).
    """
    provider = llm
    if provider is None and client is not None:
        provider = ClaudeAgentLLMProvider(client, model)

    memosift_msgs = adapt_in(messages)
    compressed, report = await compress(
        memosift_msgs,
        llm=provider,
        config=config,
        task=task,
        ledger=ledger,
    )
    return adapt_out(compressed), report


def _adapt_dict_message(msg: dict[str, Any]) -> list[MemoSiftMessage]:
    """Convert a plain dict message (from session history) to MemoSiftMessage(s).

    Returns a list because a single Anthropic-format user message with
    tool_result blocks expands into multiple MemoSiftMessages (one per result).
    """
    role = msg.get("role", "user")
    content = msg.get("content", "")

    if not isinstance(content, list):
        return [
            MemoSiftMessage(
                role=role,
                content=content if isinstance(content, str) else str(content),
            )
        ]

    # Preserve original blocks for lossless round-trip reconstruction.
    original_blocks = list(content)

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    tool_results: list[tuple[str, str]] = []  # (tool_use_id, content)

    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "thinking":
                # Thinking blocks must not be compressed; skip them so they
                # are preserved verbatim via _original_blocks round-trip.
                continue
            elif block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        function=ToolCallFunction(
                            name=block.get("name", ""),
                            arguments=json.dumps(block.get("input", {})),
                        ),
                    )
                )
            elif block.get("type") == "tool_result":
                tr_content = block.get("content", "")
                if isinstance(tr_content, list):
                    tr_content = " ".join(
                        b.get("text", str(b)) for b in tr_content if isinstance(b, dict)
                    )
                tool_results.append((block.get("tool_use_id", ""), str(tr_content)))

    result: list[MemoSiftMessage] = []
    combined_text = "\n".join(text_parts) if text_parts else ""

    # Main message (assistant with tool_calls, or user with text).
    if combined_text or tool_calls:
        result.append(
            MemoSiftMessage(
                role=role,
                content=combined_text,
                tool_calls=tool_calls if tool_calls else None,
                metadata={
                    "_original_blocks": original_blocks,
                    "_original_block_format": "claude_agent_sdk",
                },
            )
        )

    # Emit tool results as separate tool messages.
    for tool_use_id, tr_content in tool_results:
        result.append(
            MemoSiftMessage(
                role="tool",
                content=tr_content,
                tool_call_id=tool_use_id,
            )
        )

    # If we produced nothing, emit an empty message to preserve the role.
    if not result:
        result.append(MemoSiftMessage(role=role, content=""))

    return result
