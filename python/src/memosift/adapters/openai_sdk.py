# OpenAI Agents SDK adapter — convert between OpenAI message format and MemoSiftMessage.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from memosift.core.pipeline import compress
from memosift.core.types import (
    AnchorLedger,
    MemoSiftMessage,
    ToolCall,
    ToolCallFunction,
)
from memosift.providers.base import LLMResponse, MemoSiftLLMProvider

logger = logging.getLogger("memosift.adapters.openai")

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig
    from memosift.core.context_window import ContextWindowState
    from memosift.report import CompressionReport


class OpenAILLMProvider:
    """Wraps the OpenAI AsyncOpenAI client into MemoSiftLLMProvider."""

    def __init__(self, client: Any, model: str = "gpt-4o"):
        self._client = client
        self._model = model

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate text via OpenAI chat completions."""
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
        )

    async def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken if available, else heuristic."""
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self._model)
            return len(enc.encode(text))
        except (ImportError, KeyError) as e:
            import math

            logger.debug("tiktoken unavailable (%s), using heuristic token count.", e)
            return math.ceil(len(text) / 3.5)


def adapt_in(messages: list[dict[str, Any]]) -> list[MemoSiftMessage]:
    """Convert OpenAI-format message dicts to MemoSiftMessage list.

    Handles:
    - ``content: null`` on assistant messages with tool_calls → ``content: ""``
    - ``tool_calls`` list with ``id``, ``function.name``, ``function.arguments``
    - ``tool_call_id`` on tool result messages
    """
    result: list[MemoSiftMessage] = []
    for msg in messages:
        tool_calls = None
        raw_calls = msg.get("tool_calls")
        if raw_calls:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type=tc.get("type", "function"),
                    function=ToolCallFunction(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in raw_calls
            ]
        meta: dict[str, Any] = {"_openai_original_keys": list(msg.keys())}
        if msg.get("refusal"):
            meta["_openai_refusal"] = msg["refusal"]
        if msg.get("annotations"):
            meta["_openai_annotations"] = msg["annotations"]
        result.append(
            MemoSiftMessage(
                role=msg["role"],
                content=msg.get("content") or "",
                name=msg.get("name"),
                tool_call_id=msg.get("tool_call_id"),
                tool_calls=tool_calls,
                metadata=meta,
            )
        )
    return result


def adapt_out(messages: list[MemoSiftMessage]) -> list[dict[str, Any]]:
    """Convert MemoSiftMessage list back to OpenAI-format message dicts.

    Preserves tool_calls structure and handles content=null for tool-call
    assistant messages per OpenAI API requirements.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        d: dict[str, Any] = {"role": msg.role}

        if msg.tool_calls:
            # OpenAI allows content=null for assistant messages with tool_calls.
            d["content"] = msg.content if msg.content else None
            d["tool_calls"] = [tc.to_dict() for tc in msg.tool_calls]
        else:
            d["content"] = msg.content

        if msg.name is not None:
            d["name"] = msg.name
        if msg.tool_call_id is not None:
            d["tool_call_id"] = msg.tool_call_id

        # Restore preserved OpenAI-specific fields.
        if msg.metadata.get("_openai_refusal"):
            d["refusal"] = msg.metadata["_openai_refusal"]
        if msg.metadata.get("_openai_annotations"):
            d["annotations"] = msg.metadata["_openai_annotations"]

        result.append(d)
    return result


async def compress_openai_messages(
    messages: list[dict[str, Any]],
    *,
    llm: MemoSiftLLMProvider | None = None,
    config: MemoSiftConfig | None = None,
    task: str | None = None,
    ledger: AnchorLedger | None = None,
    context_window: ContextWindowState | None = None,
) -> tuple[list[dict[str, Any]], CompressionReport]:
    """Compress OpenAI-format messages end-to-end.

    Converts to MemoSiftMessage, runs the pipeline, converts back.

    Args:
        messages: OpenAI-format message dicts.
        llm: Optional MemoSiftLLMProvider (or use ``OpenAILLMProvider``).
        config: Pipeline configuration.
        task: Optional task description for relevance scoring.
        ledger: Optional AnchorLedger for fact preservation.
        context_window: Context window state for adaptive compression.

    Returns:
        Tuple of (compressed OpenAI messages, compression report).
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
