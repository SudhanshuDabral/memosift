# Vercel AI SDK adapter — convert between Vercel CoreMessage format and MemoSiftMessage.
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from memosift.core.pipeline import compress
from memosift.core.types import (
    AnchorLedger,
    MemoSiftMessage,
    ToolCall,
    ToolCallFunction,
)

logger = logging.getLogger("memosift.adapters.vercel_ai")

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig
    from memosift.core.context_window import ContextWindowState
    from memosift.providers.base import MemoSiftLLMProvider
    from memosift.report import CompressionReport


def _try_parse_json(value: str) -> Any:
    """Try to parse a string as JSON, return the original string if it fails."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def adapt_in(messages: list[dict[str, Any]]) -> list[MemoSiftMessage]:
    """Convert Vercel AI SDK CoreMessage dicts to MemoSiftMessage list.

    Handles:
    - ``content: string`` → direct content
    - ``TextPart`` → concatenated text content
    - ``ToolCallPart`` → ToolCall with toolCallId → id, toolName → function.name
    - ``ToolResultPart`` → tool role message with tool_call_id
    - ``ImagePart``/``FilePart`` → skipped, stored in metadata for round-trip
    """
    result: list[MemoSiftMessage] = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        # String content — simple case.
        if isinstance(content, str):
            result.append(
                MemoSiftMessage(
                    role=role,
                    content=content,
                    metadata={"_vercel_content_type": "string"},
                )
            )
            continue

        # Array content — process parts.
        if not isinstance(content, list):
            result.append(
                MemoSiftMessage(
                    role=role,
                    content=str(content),
                    metadata={"_vercel_content_type": "unknown"},
                )
            )
            continue

        parts: list[dict[str, Any]] = content

        if role == "tool":
            # Tool messages contain ToolResultPart[].
            for part in parts:
                if part.get("type") == "tool-result":
                    raw_result = part.get("result", "")
                    content_str = (
                        raw_result if isinstance(raw_result, str) else json.dumps(raw_result)
                    )
                    meta: dict[str, Any] = {
                        "_vercel_content_type": "tool-result",
                        "_vercel_tool_name": part.get("toolName"),
                    }
                    if part.get("isError"):
                        meta["_vercel_is_error"] = True
                    result.append(
                        MemoSiftMessage(
                            role="tool",
                            content=content_str,
                            tool_call_id=part.get("toolCallId"),
                            name=part.get("toolName"),
                            metadata=meta,
                        )
                    )
            continue

        # User or assistant message with mixed parts.
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        preserved_parts: list[dict[str, Any]] = []

        for part in parts:
            ptype = part.get("type")
            if ptype == "text":
                text_parts.append(part.get("text", ""))
            elif ptype == "tool-call":
                args = part.get("args", "")
                tool_calls.append(
                    ToolCall(
                        id=part["toolCallId"],
                        function=ToolCallFunction(
                            name=part["toolName"],
                            arguments=args if isinstance(args, str) else json.dumps(args),
                        ),
                    )
                )
            else:
                # Image, file, or unknown parts — preserve for round-trip.
                preserved_parts.append(part)

        joined_content = "\n".join(text_parts)
        meta = {"_vercel_content_type": "parts"}
        if preserved_parts:
            meta["_vercel_preserved_parts"] = preserved_parts

        result.append(
            MemoSiftMessage(
                role=role,
                content=joined_content,
                tool_calls=tool_calls if tool_calls else None,
                metadata=meta,
            )
        )

    return result


def adapt_out(messages: list[MemoSiftMessage]) -> list[dict[str, Any]]:
    """Convert MemoSiftMessage list back to Vercel AI SDK CoreMessage format.

    Restores:
    - String content → ``content: string``
    - ToolCall → ``ToolCallPart`` with toolCallId/toolName
    - Tool messages → ``ToolResultPart[]``
    - Preserved ImagePart/FilePart from metadata
    """
    result: list[dict[str, Any]] = []

    for msg in messages:
        content_type = msg.metadata.get("_vercel_content_type")

        if msg.role == "tool":
            # Reconstruct ToolResultPart.
            tool_result: dict[str, Any] = {
                "type": "tool-result",
                "toolCallId": msg.tool_call_id,
                "toolName": msg.name or msg.metadata.get("_vercel_tool_name", "unknown"),
                "result": _try_parse_json(msg.content),
            }
            if msg.metadata.get("_vercel_is_error"):
                tool_result["isError"] = True
            result.append({"role": "tool", "content": [tool_result]})
            continue

        # String content — simple reconstruction.
        if content_type == "string" or (
            not msg.tool_calls and not msg.metadata.get("_vercel_preserved_parts")
        ):
            result.append({"role": msg.role, "content": msg.content})
            continue

        # Reconstruct parts array.
        parts: list[dict[str, Any]] = []

        # Restore preserved parts (images, files) at the beginning.
        preserved = msg.metadata.get("_vercel_preserved_parts")
        if preserved:
            parts.extend(preserved)

        # Add text content.
        if msg.content:
            parts.append({"type": "text", "text": msg.content})

        # Add tool calls.
        if msg.tool_calls:
            for tc in msg.tool_calls:
                parts.append(
                    {
                        "type": "tool-call",
                        "toolCallId": tc.id,
                        "toolName": tc.function.name,
                        "args": _try_parse_json(tc.function.arguments),
                    }
                )

        result.append({"role": msg.role, "content": parts})

    return result


async def compress_vercel_messages(
    messages: list[dict[str, Any]],
    *,
    llm: MemoSiftLLMProvider | None = None,
    config: MemoSiftConfig | None = None,
    task: str | None = None,
    ledger: AnchorLedger | None = None,
    context_window: ContextWindowState | None = None,
) -> tuple[list[dict[str, Any]], CompressionReport]:
    """Compress Vercel AI SDK CoreMessage dicts end-to-end.

    Converts to MemoSiftMessage, runs the pipeline, converts back.

    Args:
        messages: Vercel AI SDK CoreMessage dicts.
        llm: Optional MemoSiftLLMProvider.
        config: Pipeline configuration.
        task: Optional task description for relevance scoring.
        ledger: Optional AnchorLedger for fact preservation.
        context_window: Context window state for adaptive compression.

    Returns:
        Tuple of (compressed Vercel messages, compression report).
    """
    memosift_msgs = adapt_in(messages)
    compressed, report = await compress(
        memosift_msgs,
        llm=llm,
        config=config,
        task=task,
        ledger=ledger,
        context_window=context_window,
    )
    return adapt_out(compressed), report
