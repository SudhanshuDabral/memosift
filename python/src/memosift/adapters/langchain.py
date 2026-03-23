# LangChain/LangGraph adapter — handles typed message classes and additional_kwargs.
from __future__ import annotations

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
    from memosift.core.context_window import ContextWindowState
    from memosift.report import CompressionReport


class LangChainLLMProvider:
    """Wraps any LangChain BaseLanguageModel into MemoSiftLLMProvider."""

    def __init__(self, llm: Any):
        self._llm = llm

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate text via LangChain's async invoke.

        Uses ``.bind()`` for model kwargs (not the config dict).
        """
        bound = self._llm.bind(max_tokens=max_tokens, temperature=temperature)
        response = await bound.ainvoke([{"role": "user", "content": prompt}])
        usage = getattr(response, "usage_metadata", None) or {}
        if isinstance(usage, dict):
            in_tokens = usage.get("input_tokens", 0)
            out_tokens = usage.get("output_tokens", 0)
        else:
            in_tokens = getattr(usage, "input_tokens", 0)
            out_tokens = getattr(usage, "output_tokens", 0)
        content = getattr(response, "content", str(response))
        return LLMResponse(text=content, input_tokens=in_tokens, output_tokens=out_tokens)

    async def count_tokens(self, text: str) -> int:
        """Count tokens via LangChain's sync get_num_tokens."""
        try:
            return self._llm.get_num_tokens(text)
        except Exception as e:
            import logging
            import math

            logging.getLogger("memosift.adapters.langchain").debug(
                "LangChain get_num_tokens failed (%s), using heuristic.",
                e,
            )
            return math.ceil(len(text) / 3.5)


def adapt_in(messages: list[Any]) -> list[MemoSiftMessage]:
    """Convert LangChain message objects to MemoSiftMessage list.

    Handles HumanMessage, AIMessage, SystemMessage, ToolMessage and
    their ``additional_kwargs`` / ``response_metadata``.
    """
    result: list[MemoSiftMessage] = []

    for msg in messages:
        # Support both dict-style and LangChain typed messages.
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            additional_kwargs = msg.get("additional_kwargs", {})
            tool_call_id = msg.get("tool_call_id")
            name = msg.get("name")
        else:
            # LangChain typed message classes.
            msg_type = type(msg).__name__
            role = _langchain_type_to_role(msg_type)
            content = getattr(msg, "content", str(msg))
            additional_kwargs = getattr(msg, "additional_kwargs", {})
            tool_call_id = getattr(msg, "tool_call_id", None)
            name = getattr(msg, "name", None)

        # Extract tool_calls from additional_kwargs (LangChain pattern).
        tool_calls = None
        raw_calls = additional_kwargs.get("tool_calls", [])
        if raw_calls:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    function=ToolCallFunction(
                        name=tc.get("function", {}).get("name", tc.get("name", "")),
                        arguments=tc.get("function", {}).get(
                            "arguments",
                            tc.get("args", "{}"),
                        ),
                    ),
                )
                for tc in raw_calls
            ]

        # Preserve LangChain metadata for round-tripping.
        metadata: dict[str, Any] = {"_langchain_type": type(msg).__name__}
        if additional_kwargs:
            non_tc_kwargs = {k: v for k, v in additional_kwargs.items() if k != "tool_calls"}
            if non_tc_kwargs:
                metadata["_langchain_additional_kwargs"] = non_tc_kwargs
        response_metadata = getattr(msg, "response_metadata", None)
        if response_metadata:
            metadata["_langchain_response_metadata"] = response_metadata

        result.append(
            MemoSiftMessage(
                role=role,
                content=content if isinstance(content, str) else str(content),
                name=name,
                tool_call_id=tool_call_id,
                tool_calls=tool_calls,
                metadata=metadata,
            )
        )

    return result


def adapt_out(messages: list[MemoSiftMessage]) -> list[dict[str, Any]]:
    """Convert MemoSiftMessage list back to LangChain-compatible dicts.

    Returns dicts that can be used with LangChain's message constructors.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        d: dict[str, Any] = {
            "role": msg.role,
            "content": msg.content,
        }
        if msg.name is not None:
            d["name"] = msg.name
        if msg.tool_call_id is not None:
            d["tool_call_id"] = msg.tool_call_id

        additional_kwargs: dict[str, Any] = {}
        if msg.tool_calls:
            additional_kwargs["tool_calls"] = [tc.to_dict() for tc in msg.tool_calls]
        # Restore preserved kwargs.
        saved_kwargs = msg.metadata.get("_langchain_additional_kwargs", {})
        additional_kwargs.update(saved_kwargs)
        if additional_kwargs:
            d["additional_kwargs"] = additional_kwargs

        response_metadata = msg.metadata.get("_langchain_response_metadata")
        if response_metadata:
            d["response_metadata"] = response_metadata

        result.append(d)
    return result


async def compress_langchain_messages(
    messages: list[Any],
    *,
    llm: MemoSiftLLMProvider | None = None,
    config: MemoSiftConfig | None = None,
    task: str | None = None,
    ledger: AnchorLedger | None = None,
    context_window: ContextWindowState | None = None,
) -> tuple[list[dict[str, Any]], CompressionReport]:
    """Compress LangChain messages end-to-end.

    Args:
        messages: LangChain message objects or dicts.
        llm: MemoSiftLLMProvider (or use ``LangChainLLMProvider(llm)``).
        config: Pipeline configuration.
        task: Optional task description.
        ledger: Optional AnchorLedger.
        context_window: Context window state for adaptive compression.

    Returns:
        Tuple of (compressed message dicts, compression report).
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


def _langchain_type_to_role(type_name: str) -> str:
    """Map LangChain message class names to roles."""
    mapping = {
        "HumanMessage": "user",
        "AIMessage": "assistant",
        "SystemMessage": "system",
        "ToolMessage": "tool",
        "ChatMessage": "user",
        "FunctionMessage": "tool",
    }
    return mapping.get(type_name, "user")
