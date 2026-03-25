# Layer 3F: Elaboration compression — compress satellite clauses, don't delete them.
from __future__ import annotations

import re
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from memosift.core.types import (
    AnchorLedger,
    ClassifiedMessage,
    CompressionPolicy,
    MemoSiftMessage,
    Shield,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig

# Policies that skip elaboration compression (already protected).
_SKIP_POLICIES = {CompressionPolicy.PRESERVE, CompressionPolicy.LIGHT}

# Satellite/elaboration markers — clauses starting with these are compressible.
_SATELLITE_MARKERS: list[re.Pattern[str]] = [
    re.compile(r"^\s*(?:for example|e\.g\.|such as|in other words)", re.IGNORECASE),
    re.compile(r"^\s*(?:specifically|to clarify|note that|in particular)", re.IGNORECASE),
    re.compile(r"^\s*(?:because|since|as a result|as mentioned)", re.IGNORECASE),
    re.compile(r"^\s*(?:which is why|this means that|namely)", re.IGNORECASE),
]

# Parenthetical pattern — content wrapped in parentheses.
_PARENTHETICAL_RE = re.compile(r"^\s*\(.*\)\s*$")

# Numbered list elaboration — "1. First...", "2. Second..." style items.
_NUMBERED_LIST_RE = re.compile(r"^\s*\d+\.\s+")


def elaborate_compress(
    segments: list[ClassifiedMessage],
    config: MemoSiftConfig,
    ledger: AnchorLedger | None = None,
) -> list[ClassifiedMessage]:
    """Compress elaboration/satellite clauses in eligible segments.

    Critical design: COMPRESS satellites, don't DELETE them. This is validated
    by RST literature — satellites in dialogue often contain irreplaceable
    reasoning rationale. We compress to ~20% of tokens, preserving key concepts.

    Only applies to:
    - Non-recent turns (older messages)
    - COMPRESSIBLE shield level (set by L3G importance scoring)
    - MODERATE or AGGRESSIVE compression policy

    Args:
        segments: Classified messages with shield levels assigned by L3G.
        config: Pipeline configuration (uses recent_turns for age check).
        ledger: Optional anchor ledger — satellite clauses containing anchor
            facts are preserved in full.

    Returns:
        Segments with elaboration clauses compressed.
    """
    # Determine the recent turn boundary.
    recent_boundary = _find_recent_boundary(segments, config.recent_turns)

    result: list[ClassifiedMessage] = []
    for seg in segments:
        # Skip protected/recent/shielded segments.
        if seg.policy in _SKIP_POLICIES or seg.protected:
            result.append(seg)
            continue
        if seg.original_index >= recent_boundary:
            result.append(seg)
            continue
        if seg.shield != Shield.COMPRESSIBLE:
            result.append(seg)
            continue

        compressed = _compress_elaborations(seg.content or "", ledger)
        if compressed != seg.content:
            new_msg = MemoSiftMessage(
                role=seg.message.role,
                content=compressed,
                name=seg.message.name,
                tool_call_id=seg.message.tool_call_id,
                tool_calls=seg.message.tool_calls,
                metadata=seg.message.metadata,
            )
            result.append(dc_replace(seg, message=new_msg))
        else:
            result.append(seg)

    return result


def _find_recent_boundary(
    segments: list[ClassifiedMessage],
    recent_turns: int,
) -> int:
    """Find the original_index boundary for recent turns.

    Counts user messages from the end; returns the index of the Nth-from-last
    user message. Messages at or after this index are "recent".
    """
    user_indices = sorted(seg.original_index for seg in segments if seg.message.role == "user")
    if len(user_indices) <= recent_turns:
        return 0  # All messages are recent.
    return user_indices[-recent_turns]


def _compress_elaborations(
    text: str,
    ledger: AnchorLedger | None,
) -> str:
    """Compress satellite/elaboration clauses within text.

    Splits at sentence boundaries, identifies satellites, and compresses
    them to ~20% of their tokens while keeping nucleus clauses intact.
    """
    clauses = _split_at_sentence_boundaries(text)
    result: list[str] = []

    for clause in clauses:
        is_satellite = _is_satellite_clause(clause)
        is_parenthetical = bool(_PARENTHETICAL_RE.match(clause))
        is_numbered = bool(_NUMBERED_LIST_RE.match(clause))

        if is_satellite or is_parenthetical or is_numbered:
            # Check if clause contains an anchor fact — preserve if so.
            if ledger is not None and ledger.contains_anchor_fact(clause):
                result.append(clause)
            else:
                # Causal clauses with numerical evidence get a higher keep
                # ratio to preserve the link between conclusions and data.
                has_numerics = bool(re.search(r"\d[\d,.]+", clause))
                ratio = 0.6 if has_numerics else 0.2
                compressed = _prune_clause(clause, keep_ratio=ratio)
                result.append(compressed)
        else:
            # Nucleus clause — keep intact.
            result.append(clause)

    return " ".join(result)


def _split_at_sentence_boundaries(text: str) -> list[str]:
    """Split text at sentence boundaries (period, exclamation, newline).

    Preserves empty results from splitting to maintain structure.
    """
    # Split on sentence-ending punctuation followed by whitespace.
    sentences = re.split(r"(?<=[.!])\s+", text)
    return [s for s in sentences if s.strip()]


def _is_satellite_clause(clause: str) -> bool:
    """Return True if the clause is a satellite/elaboration."""
    return any(marker.match(clause) for marker in _SATELLITE_MARKERS)


def _prune_clause(clause: str, keep_ratio: float = 0.2) -> str:
    """Compress a clause by keeping only the most important tokens.

    Keeps the first few words (to maintain the marker/context) and
    enough additional words to meet keep_ratio.
    """
    words = clause.split()
    if len(words) <= 5:
        return clause  # Too short to compress meaningfully.

    keep_count = max(3, int(len(words) * keep_ratio))
    if keep_count >= len(words):
        return clause

    # Keep the first 2 words (marker context) + distribute remaining.
    kept = words[:2]
    remaining_budget = keep_count - 2
    if remaining_budget > 0:
        # Sample evenly from the rest.
        rest = words[2:]
        step = max(1, len(rest) // remaining_budget)
        for j in range(0, len(rest), step):
            kept.append(rest[j])
            if len(kept) >= keep_count:
                break

    return " ".join(kept)
