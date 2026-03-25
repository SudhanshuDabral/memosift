# Layer 1.5: Agentic pattern detector — identifies and compresses waste patterns
# specific to AI agent conversations (duplicate tool calls, failed retries,
# large code arguments, thought process bloat, KPI restatement).
from __future__ import annotations

import hashlib
import re
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from memosift.core.types import (
    AnchorLedger,
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    MemoSiftMessage,
    ToolCall,
    ToolCallFunction,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig

# ── Pattern 1: Duplicate tool call detection ─────────────────────────────

# Minimum content length to consider for dedup (skip tiny results).
_MIN_DEDUP_CONTENT = 20


def _tool_call_signature(seg: ClassifiedMessage) -> str | None:
    """Compute a hash signature for a tool call (function name + arguments).

    Returns None if the segment has no tool calls.
    """
    if not seg.message.tool_calls:
        return None
    tc = seg.message.tool_calls[0]
    key = f"{tc.function.name}:{tc.function.arguments}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _tool_result_signature(seg: ClassifiedMessage) -> str | None:
    """Compute a hash of a tool result's content for duplicate detection."""
    if seg.message.role != "tool" or not seg.content:
        return None
    if len(seg.content) < _MIN_DEDUP_CONTENT:
        return None
    return hashlib.sha256(seg.content.encode()).hexdigest()[:16]


# ── Pattern 2: Failed + retried tool calls ───────────────────────────────

_ERROR_INDICATORS = re.compile(
    r'"exitCode"\s*:\s*1|"stderr"\s*:\s*"[^"]{5,}|'
    r"Traceback|Error:|Exception:|FAILED|exitCode.*1",
    re.IGNORECASE,
)


def _is_error_result(seg: ClassifiedMessage) -> bool:
    """Check if a tool result indicates a failure/error."""
    if seg.message.role != "tool" or not seg.content:
        return False
    if seg.content_type == ContentType.ERROR_TRACE:
        return True
    return bool(_ERROR_INDICATORS.search(seg.content))


def _find_tool_call_for_result(
    result_seg: ClassifiedMessage,
    segments: list[ClassifiedMessage],
    result_index: int,
) -> int | None:
    """Find the assistant message with the tool_call matching this result's tool_call_id."""
    if not result_seg.message.tool_call_id:
        return None
    for i in range(result_index - 1, -1, -1):
        seg = segments[i]
        if seg.message.tool_calls:
            for tc in seg.message.tool_calls:
                if tc.id == result_seg.message.tool_call_id:
                    return i
    return None


def _get_tool_name_from_result(
    result_seg: ClassifiedMessage,
    segments: list[ClassifiedMessage],
    result_index: int,
) -> str | None:
    """Get the function name for a tool result by finding its matching tool call."""
    call_idx = _find_tool_call_for_result(result_seg, segments, result_index)
    if call_idx is not None:
        seg = segments[call_idx]
        if seg.message.tool_calls:
            for tc in seg.message.tool_calls:
                if tc.id == result_seg.message.tool_call_id:
                    return tc.function.name
    return None


# ── Pattern 3: Large code arguments ──────────────────────────────────────

_CODE_INDICATORS = re.compile(
    r"\b(?:import |from |def |class |function |const |let |var |async )\b"
)

_LARGE_CODE_THRESHOLD = 4000


# ── Pattern 4: Thought process blocks ───────────────────────────────────

_THOUGHT_PATTERNS = [
    re.compile(r"<thinking>", re.IGNORECASE),
    re.compile(r"<thought>", re.IGNORECASE),
    re.compile(r"\*\*Thought Process\*\*", re.IGNORECASE),
    re.compile(r"\*\*(?:Considering|Planning|Processing|Determining|Figuring)\b"),
    re.compile(r"^```\n\*\*(?:Considering|Planning|Analyzing)", re.MULTILINE),
]

_THOUGHT_MIN_LENGTH = 200


# ── Pattern 5: KPI restatement ───────────────────────────────────────────

_KPI_RESTATEMENT_THRESHOLD = 3  # Minimum anchor facts restated to trigger


# ── Main detector ────────────────────────────────────────────────────────


def detect_agentic_patterns(
    segments: list[ClassifiedMessage],
    config: MemoSiftConfig | None = None,
    ledger: AnchorLedger | None = None,
) -> list[ClassifiedMessage]:
    """Detect and annotate agentic waste patterns in classified segments.

    Runs 5 deterministic pattern detectors:
    1. Duplicate tool calls — collapse identical call+result pairs
    2. Failed + retried tool calls — mark resolved errors as AGGRESSIVE
    3. Large code arguments — compress tool call args > 4KB containing code
    4. Thought process blocks — reclassify as ASSISTANT_REASONING + AGGRESSIVE
    5. KPI restatement — mark messages restating 3+ anchor facts as MODERATE

    This layer runs after L1 classification and anchor extraction, before L2
    deduplication. It does NOT delete messages — it reclassifies and annotates
    them so downstream layers (L2, L3, L6) compress more effectively.

    Args:
        segments: Classified messages from Layer 1.
        config: Pipeline configuration (unused currently, reserved for tuning).
        ledger: Anchor ledger for KPI restatement detection.

    Returns:
        Segments with agentic patterns annotated via policy/content_type changes.
    """
    result = list(segments)

    # Pattern 1: Duplicate tool call results.
    result = _detect_duplicate_tool_results(result)

    # Pattern 2: Failed + retried tool calls.
    result = _detect_failed_retries(result)

    # Pattern 3: Large code arguments in tool calls.
    result = _detect_large_code_args(result)

    # Pattern 4: Thought process blocks.
    result = _detect_thought_blocks(result)

    # Pattern 5: KPI restatement.
    if ledger is not None:
        result = _detect_kpi_restatement(result, ledger)

    return result


# ── Pattern 1 implementation ─────────────────────────────────────────────


def _detect_duplicate_tool_results(
    segments: list[ClassifiedMessage],
) -> list[ClassifiedMessage]:
    """Collapse duplicate tool call + result pairs to back-references.

    When the same tool is called with identical arguments and produces
    identical results, subsequent occurrences are replaced with a
    back-reference to the first occurrence.
    """
    result: list[ClassifiedMessage] = []
    # Map: result content hash -> first occurrence index in result list.
    seen_results: dict[str, int] = {}
    # Map: tool_call_id -> index in segments for the call message.
    call_indices: dict[str, int] = {}

    for i, seg in enumerate(segments):
        # Track tool call messages.
        if seg.message.tool_calls:
            for tc in seg.message.tool_calls:
                call_indices[tc.id] = i

        # Check tool result messages for duplicates.
        if seg.message.role == "tool" and seg.content:
            sig = _tool_result_signature(seg)
            if sig is not None and sig in seen_results:
                # Duplicate result — replace content with back-reference.
                first_idx = seen_results[sig]
                ref_content = (
                    f"[Identical result — same as tool output at message {first_idx}]"
                )
                new_msg = MemoSiftMessage(
                    role=seg.message.role,
                    content=ref_content,
                    name=seg.message.name,
                    tool_call_id=seg.message.tool_call_id,
                    tool_calls=seg.message.tool_calls,
                    metadata=seg.message.metadata,
                )
                result.append(
                    dc_replace(seg, message=new_msg, policy=CompressionPolicy.AGGRESSIVE)
                )
                continue
            elif sig is not None:
                seen_results[sig] = seg.original_index

        result.append(seg)

    return result


# ── Pattern 2 implementation ─────────────────────────────────────────────


def _detect_failed_retries(
    segments: list[ClassifiedMessage],
) -> list[ClassifiedMessage]:
    """Mark failed tool call + result pairs as AGGRESSIVE when a retry succeeded.

    Scans for error tool results, then looks forward for a successful call
    to the same function within 8 messages. If found, the failed pair is
    marked AGGRESSIVE (the error is resolved, carrying it wastes tokens).
    """
    result = list(segments)
    error_indices: list[int] = []

    # Find all error results.
    for i, seg in enumerate(segments):
        if _is_error_result(seg):
            error_indices.append(i)

    # For each error, check if a successful retry exists within 8 messages.
    for err_idx in error_indices:
        err_seg = segments[err_idx]
        err_tool_name = _get_tool_name_from_result(err_seg, segments, err_idx)
        if not err_tool_name:
            continue

        # Look forward for a success with the same tool name.
        found_retry = False
        for j in range(err_idx + 1, min(err_idx + 9, len(segments))):
            seg_j = segments[j]
            if seg_j.message.role == "tool" and not _is_error_result(seg_j):
                retry_name = _get_tool_name_from_result(seg_j, segments, j)
                if retry_name == err_tool_name:
                    found_retry = True
                    break

        if found_retry:
            # Use MODERATE (not AGGRESSIVE) to allow pruning but avoid
            # full observation masking that could lose file paths in errors.
            # Only upgrade to AGGRESSIVE if the error content is very short
            # (< 200 chars — unlikely to contain unique file paths).
            err_content = err_seg.content or ""
            policy = (
                CompressionPolicy.AGGRESSIVE
                if len(err_content) < 200
                else CompressionPolicy.MODERATE
            )
            result[err_idx] = dc_replace(result[err_idx], policy=policy)
            # Also mark the corresponding tool call.
            call_idx = _find_tool_call_for_result(err_seg, segments, err_idx)
            if call_idx is not None:
                result[call_idx] = dc_replace(
                    result[call_idx], policy=policy
                )

    return result


# ── Pattern 3 implementation ─────────────────────────────────────────────


def _detect_large_code_args(
    segments: list[ClassifiedMessage],
) -> list[ClassifiedMessage]:
    """Compress large code arguments in tool calls to signature-only form.

    When a tool call has arguments > 4KB that contain code patterns (import,
    def, class, function), the arguments are truncated to the first 500 chars
    with a truncation marker. The full code is not needed in context — only
    the tool name + key parameters matter for downstream reasoning.
    """
    result: list[ClassifiedMessage] = []

    for seg in segments:
        if not seg.message.tool_calls:
            result.append(seg)
            continue

        modified = False
        new_tool_calls = []
        for tc in seg.message.tool_calls:
            args = tc.function.arguments
            if len(args) > _LARGE_CODE_THRESHOLD and _CODE_INDICATORS.search(args):
                # Truncate to first 500 chars + marker.
                truncated = args[:500] + f"\n... [truncated {len(args) - 500} chars of code]"

                new_tc = ToolCall(
                    id=tc.id,
                    type=tc.type,
                    function=ToolCallFunction(
                        name=tc.function.name,
                        arguments=truncated,
                    ),
                )
                new_tool_calls.append(new_tc)
                modified = True
            else:
                new_tool_calls.append(tc)

        if modified:
            new_msg = MemoSiftMessage(
                role=seg.message.role,
                content=seg.message.content,
                name=seg.message.name,
                tool_call_id=seg.message.tool_call_id,
                tool_calls=new_tool_calls,
                metadata=seg.message.metadata,
            )
            result.append(dc_replace(seg, message=new_msg))
        else:
            result.append(seg)

    return result


# ── Pattern 4 implementation ─────────────────────────────────────────────


def _detect_thought_blocks(
    segments: list[ClassifiedMessage],
) -> list[ClassifiedMessage]:
    """Reclassify thought process blocks as ASSISTANT_REASONING + AGGRESSIVE.

    Thought processes (marked by ``<thinking>``, ``**Thought Process**``, etc.)
    are internal reasoning that the user never sees. They consume tokens without
    adding value in compressed context.
    """
    result: list[ClassifiedMessage] = []

    for seg in segments:
        if seg.message.role != "assistant" or not seg.content:
            result.append(seg)
            continue

        content = seg.content
        if len(content) < _THOUGHT_MIN_LENGTH:
            result.append(seg)
            continue

        is_thought = any(p.search(content) for p in _THOUGHT_PATTERNS)
        if is_thought:
            result.append(
                dc_replace(
                    seg,
                    content_type=ContentType.ASSISTANT_REASONING,
                    policy=CompressionPolicy.AGGRESSIVE,
                )
            )
        else:
            result.append(seg)

    return result


# ── Pattern 5 implementation ─────────────────────────────────────────────


def _detect_kpi_restatement(
    segments: list[ClassifiedMessage],
    ledger: AnchorLedger,
) -> list[ClassifiedMessage]:
    """Mark messages that restate 3+ anchor facts as MODERATE compression.

    When an assistant message restates numerical KPIs already captured in the
    anchor ledger, subsequent restatements can be compressed more aggressively
    without data loss — the facts are preserved in the ledger.

    Only applies to non-recent assistant messages (skips the last 2).
    """
    critical_strings = ledger.get_critical_strings()
    if not critical_strings:
        return segments

    # Find recent boundary — skip last 2 assistant messages.
    assistant_indices = [
        i for i, seg in enumerate(segments)
        if seg.message.role == "assistant" and seg.content
    ]
    if len(assistant_indices) <= 2:
        return segments
    recent_boundary = assistant_indices[-2]

    result: list[ClassifiedMessage] = []
    first_restatement_seen = False

    for i, seg in enumerate(segments):
        if (
            seg.message.role != "assistant"
            or not seg.content
            or i >= recent_boundary
            or seg.policy in {CompressionPolicy.PRESERVE, CompressionPolicy.LIGHT}
        ):
            result.append(seg)
            continue

        # Count how many anchor facts appear in this message.
        fact_count = sum(1 for s in critical_strings if s.lower() in seg.content.lower())

        if fact_count >= _KPI_RESTATEMENT_THRESHOLD:
            if not first_restatement_seen:
                # Keep the first restatement — it's the original synthesis.
                first_restatement_seen = True
                result.append(seg)
            else:
                # Subsequent restatements → MODERATE policy for harder pruning.
                result.append(dc_replace(seg, policy=CompressionPolicy.MODERATE))
        else:
            result.append(seg)

    return result
