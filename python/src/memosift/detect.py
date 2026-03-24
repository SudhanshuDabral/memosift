# Framework auto-detection — inspect message shape to determine the source SDK.
from __future__ import annotations

from typing import Any


def detect_framework(messages: list[Any]) -> str:
    """Detect which framework produced the given messages.

    Inspects the shape of the first few messages to determine the source SDK.
    Uses duck-typing heuristics — no framework imports required.

    Detection order (first match wins):
    1. MemoSiftMessage instances → "memosift"
    2. Dict with "function_calls" or "function_responses" → "adk"
    3. Object with .additional_kwargs attribute → "langchain"
    4. Object with Agent SDK type names → "agent_sdk"
    5. Dict with "content" as a list of blocks → "anthropic"
    6. Default → "openai" (most permissive format)

    Args:
        messages: List of messages in any supported framework format.

    Returns:
        Framework identifier string.

    Raises:
        ValueError: If messages is empty.
    """
    if not messages:
        raise ValueError("Cannot detect framework from empty message list")

    # Sample first few non-None messages for detection.
    samples = [m for m in messages[:5] if m is not None]
    if not samples:
        return "openai"

    for msg in samples:
        # 1. MemoSiftMessage instances (check by attribute, not import).
        if (
            hasattr(msg, "_memosift_compressed")
            and hasattr(msg, "role")
            and hasattr(msg, "content")
        ):
            return "memosift"

        # 2. Google ADK — uses function_calls/function_responses instead of tool_calls.
        if isinstance(msg, dict):
            if "function_calls" in msg or "function_responses" in msg:
                return "adk"
            # Check nested in "parts" (ADK event structure).
            parts = msg.get("parts", [])
            if isinstance(parts, list):
                for part in parts:
                    if isinstance(part, dict) and (
                        "function_call" in part or "function_response" in part
                    ):
                        return "adk"

        # 3. LangChain — has additional_kwargs attribute or key (LangChain BaseMessage).
        # Check before Agent SDK because LangChain's SystemMessage shares a name
        # with Agent SDK's SystemMessage, but has additional_kwargs.
        if hasattr(msg, "additional_kwargs") and not isinstance(msg, dict):
            return "langchain"
        if isinstance(msg, dict) and "additional_kwargs" in msg:
            return "langchain"

        # 4. Claude Agent SDK — typed message objects.
        type_name = type(msg).__name__
        if type_name in ("SystemMessage", "AssistantMessage", "UserMessage", "ResultMessage"):
            module = type(msg).__module__ or ""
            if "langchain" not in module:
                return "agent_sdk"

        # 5. Content as list of blocks — distinguish Vercel AI SDK from Anthropic.
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list) and len(content) > 0:
                first_block = content[0]
                if isinstance(first_block, dict):
                    block_type = first_block.get("type", "")
                    # 5a. Vercel AI SDK — uses "tool-call" and "tool-result" (hyphenated).
                    if block_type in ("tool-call", "tool-result"):
                        return "vercel_ai"
                    if block_type in ("text", "tool-call", "tool-result") and (
                        "toolCallId" in first_block or "toolName" in first_block
                    ):
                        return "vercel_ai"
                    # 5b. Anthropic — uses "tool_use", "tool_result" (underscored).
                    if block_type in (
                        "text",
                        "tool_use",
                        "tool_result",
                        "thinking",
                        "image",
                    ):
                        return "anthropic"

    # 6. Default — OpenAI format (role + content as string).
    return "openai"


# Valid framework identifiers for validation.
VALID_FRAMEWORKS = frozenset(
    {"openai", "anthropic", "agent_sdk", "adk", "langchain", "memosift", "vercel_ai"}
)
