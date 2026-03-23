# Layer 6: Token budget enforcement — ensure output fits within constraints.
from __future__ import annotations

import math
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from memosift.core.types import (
    AnchorLedger,
    ClassifiedMessage,
    ContentType,
    DependencyMap,
    MemoSiftMessage,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig
    from memosift.providers.base import MemoSiftLLMProvider

# Per-domain compression caps: content types that should not be aggressively
# compressed beyond a certain ratio. Code and error traces need higher fidelity.
_DOMAIN_MAX_COMPRESSION: dict[ContentType, float] = {
    ContentType.CODE_BLOCK: 4.0,  # Cap at 4x — code needs fidelity
    ContentType.ERROR_TRACE: 3.0,  # Cap at 3x — errors are critical
}


async def enforce_budget(
    segments: list[ClassifiedMessage],
    config: MemoSiftConfig,
    deps: DependencyMap,
    counter: MemoSiftLLMProvider | None = None,
    ledger: AnchorLedger | None = None,
) -> list[ClassifiedMessage]:
    """Enforce token budget by dropping lowest-scored segments.

    If no ``token_budget`` is set, returns segments unchanged.

    Protected segments (SYSTEM_PROMPT, USER_QUERY, RECENT_TURN) are never
    dropped. Dependency map and anchor ledger are consulted before dropping.
    Segments containing ledger facts are truncated instead of fully dropped.
    Per-domain compression caps prevent over-compression of code and error content.

    Args:
        segments: Scored segments from previous layers.
        config: Pipeline configuration (controls ``token_budget``).
        deps: Deduplication dependency map.
        counter: Token counter (LLM provider or heuristic fallback).
        ledger: Optional anchor ledger — segments with ledger facts are
            truncated instead of dropped.

    Returns:
        Budget-compliant segments.
    """
    if config.token_budget is None:
        return segments

    # Estimate tokens for each segment.
    segments = await _estimate_tokens(segments, counter)
    total = sum(seg.estimated_tokens for seg in segments)

    if total <= config.token_budget:
        return segments

    # Pre-compute tool call/result ID sets for O(1) lookup.
    active_call_ids: set[str] = set()
    active_result_ids: set[str] = set()
    for seg in segments:
        if seg.message.tool_calls:
            for tc in seg.message.tool_calls:
                active_call_ids.add(tc.id)
        if seg.message.tool_call_id:
            active_result_ids.add(seg.message.tool_call_id)

    # Sort droppable segments by relevance score (ascending).
    # Tie-break by original_index (ascending) for deterministic ordering.
    # Per-domain caps: skip segments whose content type has a max compression ratio
    # and they've already been compressed beyond that cap.
    indexed = list(enumerate(segments))
    droppable = [
        (i, seg)
        for i, seg in indexed
        if not seg.protected
        and _can_drop(seg, active_call_ids, active_result_ids)
        and not _exceeds_domain_cap(seg)
    ]
    droppable.sort(key=lambda x: (x[1].relevance_score, x[1].original_index))

    dropped_indices: set[int] = set()
    for idx, seg in droppable:
        if total <= config.token_budget:
            break
        # Before dropping, check if any other message references this one
        # via the dependency map.
        if not deps.can_drop(seg.original_index):
            # Expand back-references to this message before dropping.
            _expand_dependents(seg.original_index, segments, deps)

        # If the segment contains anchor ledger facts, truncate instead of
        # fully dropping — preserve the lines with critical facts.
        if ledger and ledger.contains_anchor_fact(seg.content):
            truncated = _head_tail_truncate(seg.content, ledger=ledger)
            new_msg = MemoSiftMessage(
                role=seg.message.role,
                content=truncated,
                name=seg.message.name,
                tool_call_id=seg.message.tool_call_id,
                tool_calls=seg.message.tool_calls,
                metadata=seg.message.metadata,
            )
            old_tokens = seg.estimated_tokens
            new_tokens = _heuristic_count(truncated)
            segments[idx] = dc_replace(seg, message=new_msg, estimated_tokens=new_tokens)
            total -= old_tokens - new_tokens
            continue

        total -= seg.estimated_tokens
        dropped_indices.add(idx)

    result = [seg for i, seg in enumerate(segments) if i not in dropped_indices]

    # If still over budget, truncate the largest non-system segment.
    total = sum(seg.estimated_tokens for seg in result)
    if total > config.token_budget and len(result) > 1:
        result = _truncate_largest(result, total - config.token_budget)

    return result


async def _estimate_tokens(
    segments: list[ClassifiedMessage],
    counter: MemoSiftLLMProvider | None,
) -> list[ClassifiedMessage]:
    """Estimate token count for each segment."""
    result: list[ClassifiedMessage] = []
    for seg in segments:
        if counter:
            tokens = await counter.count_tokens(seg.content)
        else:
            tokens = _heuristic_count(seg.content)
        result.append(dc_replace(seg, estimated_tokens=tokens))
    return result


def _heuristic_count(text: str) -> int:
    """Estimate token count: ~3.5 chars per BPE token."""
    if not text:
        return 0
    return math.ceil(len(text) / 3.5)


def _exceeds_domain_cap(seg: ClassifiedMessage) -> bool:
    """Return True if the segment's content type has a domain cap and the segment
    has already been compressed beyond it (based on original vs current token ratio).

    Uses the _memosiftOriginalTokens annotation if set, otherwise estimates from
    the current content length.
    """
    max_ratio = _DOMAIN_MAX_COMPRESSION.get(seg.content_type)
    if max_ratio is None:
        return False

    # If we know the original token count, check the compression ratio.
    original = seg.message._memosift_original_tokens
    if original is not None and original > 0:
        current = max(_heuristic_count(seg.content), 1)
        ratio = original / current
        return ratio >= max_ratio

    # Without original tokens, protect all domain-capped types from dropping.
    # They can still be truncated but not fully removed.
    return True


def _can_drop(
    seg: ClassifiedMessage,
    active_call_ids: set[str],
    active_result_ids: set[str],
) -> bool:
    """Check if a segment can be dropped without breaking tool call integrity.

    Uses pre-computed ID sets for O(1) lookup instead of scanning all segments.
    """
    # Don't drop tool results whose tool_call still exists.
    if seg.message.tool_call_id and seg.message.tool_call_id in active_call_ids:
        return False

    # Don't drop assistant messages with tool_calls if any matching results exist.
    if seg.message.tool_calls:
        for tc in seg.message.tool_calls:
            if tc.id in active_result_ids:
                return False

    return True


def _expand_dependents(
    original_index: int,
    segments: list[ClassifiedMessage],
    deps: DependencyMap,
) -> None:
    """Expand back-references to a message that's about to be dropped.

    Replaces the back-reference with a truncated version of the original.
    """
    dependent_indices = deps.dependents_of(original_index)
    for dep_idx in dependent_indices:
        for i, seg in enumerate(segments):
            if seg.original_index == dep_idx:
                # Find the original content.
                for orig in segments:
                    if orig.original_index == original_index:
                        # Truncate to ~20% of original.
                        truncated = _head_tail_truncate(orig.content)
                        new_msg = MemoSiftMessage(
                            role=seg.message.role,
                            content=truncated,
                            name=seg.message.name,
                            tool_call_id=seg.message.tool_call_id,
                            tool_calls=seg.message.tool_calls,
                            metadata=seg.message.metadata,
                        )
                        segments[i] = dc_replace(seg, message=new_msg)
                        break
                break
        # Remove the dependency.
        deps.references.pop(dep_idx, None)


def _head_tail_truncate(
    text: str,
    ledger: AnchorLedger | None = None,
) -> str:
    """Keep first/last ~10% of lines plus any lines containing ledger facts."""
    lines = text.split("\n")
    if len(lines) <= 10:
        return text

    keep = max(3, len(lines) // 10)
    head_indices = set(range(keep))
    tail_indices = set(range(len(lines) - keep, len(lines)))
    keep_indices = head_indices | tail_indices

    # Also keep any line that contains a ledger-protected string.
    if ledger:
        protected = ledger.get_protected_strings()
        if protected:
            for i, line in enumerate(lines):
                if i in keep_indices:
                    continue
                line_lower = line.lower()
                if any(s.lower() in line_lower for s in protected):
                    keep_indices.add(i)

    kept_lines: list[str] = []
    last_kept = -1
    for i in sorted(keep_indices):
        if last_kept >= 0 and i > last_kept + 1:
            omitted = i - last_kept - 1
            kept_lines.append(f"[... {omitted} lines omitted ...]")
        kept_lines.append(lines[i])
        last_kept = i

    # Final gap to end.
    if last_kept < len(lines) - 1 and last_kept not in tail_indices:
        omitted = len(lines) - 1 - last_kept
        kept_lines.append(f"[... {omitted} lines omitted ...]")

    return "\n".join(kept_lines)


def _truncate_largest(
    segments: list[ClassifiedMessage],
    overshoot_tokens: int,
) -> list[ClassifiedMessage]:
    """Truncate the largest non-system segment to fit budget."""
    # Find the largest non-system, non-protected segment.
    candidates = [
        (i, seg)
        for i, seg in enumerate(segments)
        if seg.content_type != ContentType.SYSTEM_PROMPT and not seg.protected
    ]
    if not candidates:
        # All segments are protected — truncate the largest non-system one.
        candidates = [
            (i, seg)
            for i, seg in enumerate(segments)
            if seg.content_type != ContentType.SYSTEM_PROMPT
        ]
    if not candidates:
        return segments

    largest_idx, largest = max(candidates, key=lambda x: x[1].estimated_tokens)

    # Estimate how many characters to remove.
    chars_to_remove = int(overshoot_tokens * 3.5)
    content = largest.content
    if chars_to_remove >= len(content):
        # Remove entire content.
        new_content = "[Content removed to fit budget.]"
    else:
        # Truncate from the middle.
        keep = len(content) - chars_to_remove
        half = keep // 2
        new_content = (
            content[:half]
            + f"\n[... {chars_to_remove} characters omitted to fit budget ...]\n"
            + content[-half:]
        )

    new_msg = MemoSiftMessage(
        role=largest.message.role,
        content=new_content,
        name=largest.message.name,
        tool_call_id=largest.message.tool_call_id,
        tool_calls=largest.message.tool_calls,
        metadata=largest.message.metadata,
    )
    segments[largest_idx] = dc_replace(largest, message=new_msg)
    return segments
