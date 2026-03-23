# Google ADK adapter — handles Events with function_calls/function_responses.
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from memosift.core.pipeline import compress
from memosift.core.types import (
    AnchorLedger,
    MemoSiftMessage,
    ToolCall,
    ToolCallFunction,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig
    from memosift.core.context_window import ContextWindowState
    from memosift.providers.base import MemoSiftLLMProvider
    from memosift.report import CompressionReport


def adapt_in(events: list[dict[str, Any]]) -> list[MemoSiftMessage]:
    """Convert Google ADK events to MemoSiftMessage list.

    ADK uses Events with ``function_calls`` and ``function_responses``
    instead of tool_calls/tool_results.
    """
    result: list[MemoSiftMessage] = []

    for event in events:
        role = event.get("role", "user")

        # Handle function calls (ADK's equivalent of tool_calls).
        function_calls = event.get("function_calls", [])
        if function_calls:
            tool_calls = [
                ToolCall(
                    id=fc.get("id")
                    or f"adk_{
                        hashlib.sha256(json.dumps(fc, sort_keys=True).encode()).hexdigest()[:12]
                    }",
                    function=ToolCallFunction(
                        name=fc.get("name", ""),
                        arguments=json.dumps(fc.get("args", {})),
                    ),
                )
                for i, fc in enumerate(function_calls)
            ]
            text = event.get("text", "")
            result.append(
                MemoSiftMessage(
                    role="assistant",
                    content=text,
                    tool_calls=tool_calls,
                    metadata={
                        "_adk_event": True,
                        "_original_event": dict(event),
                        "_original_block_format": "google_adk",
                    },
                )
            )
            continue

        # Handle function responses (ADK's equivalent of tool results).
        function_responses = event.get("function_responses", [])
        if function_responses:
            for fr in function_responses:
                content = fr.get("response", "")
                if isinstance(content, dict):
                    content = json.dumps(content)
                result.append(
                    MemoSiftMessage(
                        role="tool",
                        content=content,
                        tool_call_id=fr.get("id", ""),
                        name=fr.get("name", ""),
                        metadata={
                            "_adk_event": True,
                            "_original_event": dict(event),
                            "_original_block_format": "google_adk",
                        },
                    )
                )
            continue

        # Regular text event.
        parts = event.get("parts", [])
        text = event.get("text", "")
        if not text and parts:
            text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p)

        result.append(
            MemoSiftMessage(
                role=_adk_role_to_memosift(role),
                content=text,
                metadata={
                    "_adk_event": True,
                    "_original_event": dict(event),
                    "_original_block_format": "google_adk",
                },
            )
        )

    return result


def adapt_out(messages: list[MemoSiftMessage]) -> list[dict[str, Any]]:
    """Convert MemoSiftMessage list back to ADK event format."""
    result: list[dict[str, Any]] = []

    for msg in messages:
        original_event = msg.metadata.get("_original_event")
        if original_event is not None:
            # Lossless round-trip: reconstruct from original event,
            # replacing only the text field with (possibly compressed) content.
            event = dict(original_event)
            if "text" in event or (
                not event.get("function_calls") and not event.get("function_responses")
            ):
                event["text"] = msg.content
            result.append(event)
            continue

        # Fallback: reconstruct from scratch.
        event: dict[str, Any] = {"role": _memosift_role_to_adk(msg.role)}

        if msg.tool_calls:
            event["function_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments) if tc.function.arguments else {},
                }
                for tc in msg.tool_calls
            ]
            if msg.content:
                event["text"] = msg.content
        elif msg.role == "tool" and msg.tool_call_id:
            event["function_responses"] = [
                {
                    "id": msg.tool_call_id,
                    "name": msg.name or "",
                    "response": msg.content,
                }
            ]
            event["role"] = "function"
        else:
            event["text"] = msg.content

        result.append(event)

    return result


async def compress_adk_events(
    events: list[dict[str, Any]],
    *,
    llm: MemoSiftLLMProvider | None = None,
    config: MemoSiftConfig | None = None,
    task: str | None = None,
    ledger: AnchorLedger | None = None,
    context_window: ContextWindowState | None = None,
) -> tuple[list[dict[str, Any]], CompressionReport]:
    """Compress Google ADK events end-to-end."""
    memosift_msgs = adapt_in(events)
    compressed, report = await compress(
        memosift_msgs,
        llm=llm,
        config=config,
        task=task,
        ledger=ledger,
        context_window=context_window,
    )
    return adapt_out(compressed), report


def _adk_role_to_memosift(role: str) -> str:
    """Map ADK roles to MemoSift roles."""
    mapping = {"model": "assistant", "function": "tool"}
    return mapping.get(role, role)


def _memosift_role_to_adk(role: str) -> str:
    """Map MemoSift roles to ADK roles."""
    mapping = {"assistant": "model", "tool": "function"}
    return mapping.get(role, role)
