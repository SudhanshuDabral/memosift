# Layer 1: Content segmentation and classification.
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from memosift.core.types import (
    DEFAULT_POLICIES,
    ClassifiedMessage,
    ContentType,
    MemoSiftMessage,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig
    from memosift.core.state import CompressionState

# Error trace patterns — require at least 3 matching lines in a 20-line window.
_ERROR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Traceback \(most recent"),
    re.compile(r"^\s+at .+\(.+:\d+\)", re.MULTILINE),
    re.compile(r"Error: .+\n\s+at", re.MULTILINE),
    re.compile(r"panic:"),
    re.compile(r"Exception in thread"),
    re.compile(r"^\s+File \".+\", line \d+", re.MULTILINE),
    re.compile(r"raise \w+Error"),
    re.compile(r"throw new \w*Error"),  # JavaScript/TypeScript
    re.compile(r"panic!\("),  # Rust
    re.compile(r"Unhandled\s+(?:promise\s+)?rejection", re.IGNORECASE),  # Node.js
    re.compile(r"^\s+at Object\.\<anonymous\>", re.MULTILINE),  # Node.js stack
]

_MIN_ERROR_LINES = 3

# Code block detection.
_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```")

# Default tool names whose output is likely code.
_CODE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "read_file",
        "cat",
        "view_file",
        "get_file_contents",
        "ReadFileTool",
        "read",
        "Read",
    }
)


def classify_messages(
    messages: list[MemoSiftMessage],
    config: MemoSiftConfig,
    state: CompressionState | None = None,
) -> list[ClassifiedMessage]:
    """Classify each message by content type and assign a compression policy.

    Args:
        messages: The raw conversation messages.
        config: Pipeline configuration (controls ``recent_turns`` and policy overrides).
        state: Optional CompressionState for caching classification results.

    Returns:
        A list of ``ClassifiedMessage`` wrappers, one per input message.
    """
    recent_boundary = _find_nth_user_message_from_end(messages, config.recent_turns)
    last_user = _last_user_index(messages)
    result: list[ClassifiedMessage] = []

    for i, msg in enumerate(messages):
        in_recent_window = i > recent_boundary

        # Rule 0: Previously compressed messages are always PRESERVE.
        if msg._memosift_compressed:
            ctype = ContentType.PREVIOUSLY_COMPRESSED

        # Rule 1: System prompts are always PRESERVE.
        elif msg.role == "system":
            ctype = ContentType.SYSTEM_PROMPT

        # Rule 2: The most recent user message is the current query.
        elif msg.role == "user" and i == last_user:
            ctype = ContentType.USER_QUERY

        # Rule 3: Tool results — sub-classify by content shape.
        # Within the recent window, notable types (ERROR_TRACE, CODE_BLOCK,
        # TOOL_RESULT_JSON) keep their specific classification because their
        # compression engines are safe and beneficial.  Plain TOOL_RESULT_TEXT
        # becomes RECENT_TURN within the recent window.
        elif msg.role == "tool":
            # Try state cache for tool result sub-classification.
            cached = state.get_cached_classification(msg.content) if state else None
            if cached is not None and cached in {
                ContentType.TOOL_RESULT_JSON,
                ContentType.TOOL_RESULT_TEXT,
                ContentType.CODE_BLOCK,
                ContentType.ERROR_TRACE,
            }:
                tool_type = cached
            else:
                tool_type = _classify_tool_result(msg)
                if state is not None:
                    state.cache_classification(msg.content, tool_type)
            if in_recent_window and tool_type == ContentType.TOOL_RESULT_TEXT:
                ctype = ContentType.RECENT_TURN
            else:
                ctype = tool_type

        # Rule 4: Recent turns (non-tool, non-system, non-query).
        elif in_recent_window:
            ctype = ContentType.RECENT_TURN

        # Rule 5: Old assistant messages.
        elif msg.role == "assistant":
            ctype = ContentType.ASSISTANT_REASONING

        # Rule 6: Everything else (old user messages).
        else:
            ctype = ContentType.OLD_CONVERSATION

        policy = config.policies.get(ctype, DEFAULT_POLICIES[ctype])
        protected = ctype in {
            ContentType.SYSTEM_PROMPT,
            ContentType.USER_QUERY,
            ContentType.RECENT_TURN,
            ContentType.PREVIOUSLY_COMPRESSED,
        }
        result.append(
            ClassifiedMessage(
                message=msg,
                content_type=ctype,
                policy=policy,
                original_index=i,
                protected=protected,
            )
        )

    return result


def _find_nth_user_message_from_end(messages: list[MemoSiftMessage], n: int) -> int:
    """Return the index of the Nth user message from the end.

    Everything AFTER this index (strictly greater) is in the recent window.
    The message at the returned index itself is NOT recent — it's the boundary.

    If there are fewer than ``n`` user messages total, returns 0 (everything is recent).
    """
    if n <= 0:
        return len(messages)  # Nothing is recent
    user_count = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            user_count += 1
            if user_count == n:
                return i
    return 0  # Fewer than n user messages — everything is recent


def _last_user_index(messages: list[MemoSiftMessage]) -> int:
    """Return the index of the last user message, or -1 if none."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            return i
    return -1


def _classify_tool_result(msg: MemoSiftMessage) -> ContentType:
    """Sub-classify a tool result message by inspecting its content."""
    content = msg.content

    # Check for error traces first (they can appear in any tool output).
    if _contains_error_trace(content):
        return ContentType.ERROR_TRACE

    # Check for valid JSON (objects or arrays, not bare primitives).
    if _is_valid_json(content):
        return ContentType.TOOL_RESULT_JSON

    # Check for code content.
    if _contains_code(content, msg.name):
        return ContentType.CODE_BLOCK

    return ContentType.TOOL_RESULT_TEXT


def _is_valid_json(text: str) -> bool:
    """Return True if ``text`` parses as a JSON object or array.

    Bare primitives (strings, numbers, booleans) that parse as valid JSON
    are intentionally excluded — they should be classified as TEXT.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # Quick check: must start with { or [
    if stripped[0] not in ("{", "["):
        return False
    try:
        parsed = json.loads(stripped)
        return isinstance(parsed, (dict, list))
    except (json.JSONDecodeError, ValueError):
        return False


def _contains_error_trace(text: str) -> bool:
    """Return True if ``text`` contains enough error-trace markers.

    Requires at least ``_MIN_ERROR_LINES`` matching lines within the text
    to avoid false positives from single "Error:" mentions.
    """
    match_count = 0
    for pattern in _ERROR_PATTERNS:
        match_count += len(pattern.findall(text))
        if match_count >= _MIN_ERROR_LINES:
            return True
    return False


def _contains_code(text: str, tool_name: str | None = None) -> bool:
    """Return True if ``text`` appears to contain code.

    Checks for fenced code blocks or tool-name heuristic.
    """
    if _FENCED_CODE_RE.search(text):
        return True
    return bool(tool_name and tool_name in _CODE_TOOL_NAMES)
