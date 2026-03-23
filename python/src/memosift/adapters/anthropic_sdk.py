# Anthropic/Claude SDK adapter — handles content blocks, nested tool results, separate system.
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memosift.core.context_window import ContextWindowState
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


@dataclass
class AnthropicCompressedResult:
    """Result of compressing Anthropic-format messages.

    Anthropic's API takes ``system`` as a separate parameter, so the
    compressed result separates it from the message list.
    """

    system: str
    messages: list[dict[str, Any]]


class AnthropicLLMProvider:
    """Wraps the Anthropic AsyncAnthropic client into MemoSiftLLMProvider."""

    def __init__(self, client: Any, model: str = "claude-sonnet-4-6"):
        self._client = client
        self._model = model

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate text via Anthropic messages API."""
        resp = await self._client.messages.create(
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
        """Count tokens via Anthropic's count_tokens API."""
        try:
            resp = await self._client.messages.count_tokens(
                model=self._model,
                messages=[{"role": "user", "content": text}],
            )
            return resp.input_tokens
        except Exception as e:
            import logging
            import math

            logging.getLogger("memosift.adapters.anthropic").debug(
                "Anthropic count_tokens failed (%s), using heuristic.",
                e,
            )
            return math.ceil(len(text) / 3.5)


def adapt_in(
    messages: list[dict[str, Any]],
    system: str | None = None,
) -> list[MemoSiftMessage]:
    """Convert Anthropic-format messages to MemoSiftMessage list.

    Anthropic differences:
    - ``content`` is an ARRAY of blocks: ``[{"type": "text", "text": "..."}]``
    - Tool use blocks appear in assistant messages as ``{"type": "tool_use", ...}``
    - Tool results are NESTED in user messages as ``{"type": "tool_result", ...}``
    - System prompt is a separate parameter, not in the messages array
    """
    result: list[MemoSiftMessage] = []

    # Add system prompt as a system message.
    if system:
        result.append(MemoSiftMessage(role="system", content=system))

    for msg in messages:
        role = msg["role"]
        content_blocks = msg.get("content", [])

        if isinstance(content_blocks, str):
            result.append(MemoSiftMessage(role=role, content=content_blocks))
            continue

        # Preserve original blocks for lossless round-trip reconstruction.
        original_blocks = list(content_blocks) if isinstance(content_blocks, list) else None

        # Process content blocks.
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tool_results: list[tuple[str, str, str]] = []  # (tool_use_id, content, name)

        for block in content_blocks:
            if isinstance(block, str):
                text_parts.append(block)
            elif block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                import json

                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        function=ToolCallFunction(
                            name=block["name"],
                            arguments=json.dumps(block.get("input", {})),
                        ),
                    )
                )
            elif block.get("type") == "thinking":
                # Thinking blocks must not be compressed; skip them so they
                # are preserved verbatim via _original_blocks round-trip.
                continue
            elif block.get("type") == "tool_result":
                tr_content = block.get("content", "")
                if isinstance(tr_content, list):
                    tr_content = " ".join(
                        b.get("text", "") for b in tr_content if isinstance(b, dict)
                    )
                tr_is_error = block.get("is_error", False)
                tr_cache = block.get("cache_control")
                tool_results.append(
                    (
                        block.get("tool_use_id", ""),
                        tr_content,
                        block.get("name", ""),
                        tr_is_error,
                        tr_cache,
                    )
                )

        # Emit the main message (assistant with text + tool_use, or user with text).
        combined_text = "\n".join(text_parts) if text_parts else ""
        main_msg = MemoSiftMessage(
            role=role,
            content=combined_text,
            tool_calls=tool_calls if tool_calls else None,
            metadata={
                "_anthropic_blocks": True,
                "_original_blocks": original_blocks,
                "_original_block_format": "anthropic",
            },
        )
        result.append(main_msg)

        # Emit tool results as separate tool messages (MemoSift's internal format).
        for tool_use_id, tr_content, tr_name, tr_is_error, tr_cache in tool_results:
            tr_meta: dict[str, Any] = {}
            if tr_is_error:
                tr_meta["_anthropic_is_error"] = True
            if tr_cache:
                tr_meta["_anthropic_cache_control"] = tr_cache
            result.append(
                MemoSiftMessage(
                    role="tool",
                    content=tr_content,
                    tool_call_id=tool_use_id,
                    name=tr_name,
                    metadata=tr_meta,
                )
            )

    return result


def adapt_out(
    messages: list[MemoSiftMessage],
) -> AnthropicCompressedResult:
    """Convert MemoSiftMessage list back to Anthropic format.

    Re-nests tool results inside user messages and separates the system prompt.
    """
    system = ""
    anthropic_msgs: list[dict[str, Any]] = []

    # Collect tool results for re-nesting.
    tool_result_map: dict[str, MemoSiftMessage] = {}
    for msg in messages:
        if msg.role == "tool" and msg.tool_call_id:
            tool_result_map[msg.tool_call_id] = msg

    consumed_tool_ids: set[str] = set()

    for msg in messages:
        if msg.role == "system":
            system = msg.content
            continue

        if msg.role == "tool":
            # Tool results will be nested into user messages.
            continue

        blocks: list[dict[str, Any]] = []

        original_blocks = msg.metadata.get("_original_blocks")
        if original_blocks is not None:
            # Lossless round-trip: walk the original blocks and patch in
            # compressed content / surviving tool_calls.
            import json

            tc_map: dict[str, ToolCall] = {}
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_map[tc.id] = tc

            text_replaced = False
            for orig_block in original_blocks:
                btype = orig_block.get("type") if isinstance(orig_block, dict) else None

                if btype == "text":
                    if not text_replaced:
                        # First text block gets the (possibly compressed) content.
                        blocks.append({"type": "text", "text": msg.content})
                        text_replaced = True
                    # Subsequent text blocks are dropped — their content was
                    # merged into msg.content during adapt_in.

                elif btype == "thinking":
                    # Pass thinking blocks through unchanged.
                    blocks.append(dict(orig_block))

                elif btype == "tool_use":
                    tc_id = orig_block.get("id", "")
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
                    else:
                        blocks.append(orig_block)

            # If no text block existed in originals but we have content, add it.
            if not text_replaced and msg.content:
                blocks.insert(0, {"type": "text", "text": msg.content})
        else:
            # Backward-compatible fallback: reconstruct from scratch.
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})

            if msg.role == "assistant" and msg.tool_calls:
                import json

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

        anthropic_msgs.append({"role": msg.role, "content": blocks})

        # After an assistant message with tool_calls, nest tool results into
        # the next user message (or create one).
        if msg.role == "assistant" and msg.tool_calls:
            tr_blocks: list[dict[str, Any]] = []
            for tc in msg.tool_calls:
                if tc.id in tool_result_map and tc.id not in consumed_tool_ids:
                    tr_msg = tool_result_map[tc.id]
                    tr_blocks.append(_build_tool_result_block(tc.id, tr_msg))
                    consumed_tool_ids.add(tc.id)
            if tr_blocks:
                anthropic_msgs.append({"role": "user", "content": tr_blocks})

    return AnthropicCompressedResult(system=system, messages=anthropic_msgs)


def _build_tool_result_block(tc_id: str, tr_msg: MemoSiftMessage) -> dict[str, Any]:
    """Build a tool_result block, restoring is_error and cache_control."""
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tc_id,
        "content": tr_msg.content,
    }
    if tr_msg.metadata.get("_anthropic_is_error"):
        block["is_error"] = True
    cache_control = tr_msg.metadata.get("_anthropic_cache_control")
    if cache_control:
        block["cache_control"] = cache_control
    return block


async def compress_anthropic_messages(
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
    llm: MemoSiftLLMProvider | None = None,
    config: MemoSiftConfig | None = None,
    task: str | None = None,
    ledger: AnchorLedger | None = None,
    client: Any = None,
    model: str = "claude-sonnet-4-6",
    context_window: ContextWindowState | None = None,
) -> tuple[AnthropicCompressedResult, CompressionReport]:
    """Compress Anthropic-format messages end-to-end.

    Args:
        messages: Anthropic-format message dicts (content is array of blocks).
        system: System prompt (separate parameter per Anthropic API).
        llm: MemoSiftLLMProvider (or auto-wrap ``client`` + ``model``).
        config: Pipeline configuration.
        task: Optional task description.
        ledger: Optional AnchorLedger.
        client: Anthropic AsyncAnthropic client (auto-wraps into provider).
        model: Model name for auto-wrapping.
        context_window: Context window state for adaptive compression.
            When None, auto-resolves from ``model`` using the model registry.

    Returns:
        Tuple of (AnthropicCompressedResult, CompressionReport).
    """
    provider = llm
    if provider is None and client is not None:
        provider = AnthropicLLMProvider(client, model)

    # Auto-resolve context window from model name if not explicitly provided.
    cw = context_window
    if cw is None and model:
        cw = ContextWindowState.from_model(model)

    memosift_msgs = adapt_in(messages, system=system)
    compressed, report = await compress(
        memosift_msgs,
        llm=provider,
        config=config,
        task=task,
        ledger=ledger,
        context_window=cw,
    )
    return adapt_out(compressed), report
