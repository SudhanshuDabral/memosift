# Engine B: IDF-based token pruning — remove low-information words.
from __future__ import annotations

import re
from dataclasses import replace as dc_replace
from math import log
from typing import TYPE_CHECKING

from memosift.core.types import (
    AnchorLedger,
    ClassifiedMessage,
    CompressionPolicy,
    MemoSiftMessage,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig

# Applies to TOOL_RESULT_TEXT, ASSISTANT_REASONING.
_TARGET_POLICIES = {
    CompressionPolicy.MODERATE,
    CompressionPolicy.AGGRESSIVE,
}

# Protected token patterns — never pruned regardless of IDF score.
_PROTECTED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:[A-Za-z]:)?[\w.\-]+(?:[/\\][\w.\-]+)+(?::\d+)?"),  # File paths
    re.compile(r"\b\d[\d.,:]*\b"),  # Numbers, line numbers, error codes
    re.compile(r"\b[a-z]+(?:[A-Z][a-z]+)+\b"),  # camelCase
    re.compile(r"\b[a-z]+(?:_[a-z]+)+\b"),  # snake_case
    re.compile(r"\b[A-Z][A-Z_]+\b"),  # UPPER_CASE
    re.compile(r"https?://\S+"),  # URLs
    re.compile(r"\S+@\S+\.\S+"),  # Email addresses
    re.compile(r"\b[A-Za-z0-9]{10,}\b"),  # Tracking numbers (e.g., 1Z999AA10123456784)
    re.compile(r"\b(?:ORD|RET|INV|TXN|REF)-[\w\-]+\b"),  # Structured IDs (ORD-2026-78432)
    re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
    ),  # UUIDs
    re.compile(r"\b[A-Z]{2,4}-\d{4,}\b"),  # Order IDs (AB-12345)
    re.compile(r"\b[A-Za-z0-9]{12,}\b"),  # Alphanumeric identifiers ≥12 chars
]


def prune_tokens(
    segments: list[ClassifiedMessage],
    config: MemoSiftConfig,
    ledger: AnchorLedger | None = None,
) -> list[ClassifiedMessage]:
    """Prune low-information tokens from eligible segments using IDF scoring.

    High-IDF tokens (appearing in few messages) are information-dense and kept.
    Low-IDF tokens (appearing in many messages) are noise and pruned.
    Protected tokens (file paths, identifiers, numbers, anchor ledger facts)
    are never pruned.

    Args:
        segments: Classified messages from previous layers.
        config: Pipeline configuration (controls ``token_prune_keep_ratio``).
        ledger: Optional anchor ledger — tokens matching ledger facts are protected.

    Returns:
        Segments with low-information tokens removed.
    """
    # Compute IDF scores across all message contents.
    all_contents = [seg.content for seg in segments]
    idf_scores = _compute_idf_scores(all_contents)
    # Auto-protect tokens that appear in the anchor ledger (Item 1.3).
    # This ensures file paths, error messages, IDs, and other critical tokens
    # extracted into the ledger are never pruned from the source text.
    ledger_lower: frozenset[str] = frozenset()
    if ledger is not None:
        ledger_lower = frozenset(s.lower() for s in ledger.get_protected_strings())

    result: list[ClassifiedMessage] = []
    for seg in segments:
        if seg.policy in _TARGET_POLICIES:
            pruned = _prune_segment(
                seg.content, idf_scores, config.token_prune_keep_ratio, ledger_lower
            )
            if pruned != seg.content:
                new_msg = MemoSiftMessage(
                    role=seg.message.role,
                    content=pruned,
                    name=seg.message.name,
                    tool_call_id=seg.message.tool_call_id,
                    tool_calls=seg.message.tool_calls,
                    metadata=seg.message.metadata,
                )
                result.append(dc_replace(seg, message=new_msg))
            else:
                result.append(seg)
        else:
            result.append(seg)
    return result


def _compute_idf_scores(documents: list[str]) -> dict[str, float]:
    """Compute IDF scores across the message corpus.

    IDF = log((N + 1) / (df + 1)) + 1
    """
    n_docs = len(documents)
    doc_freq: dict[str, int] = {}
    for doc in documents:
        unique_tokens = set(re.findall(r"\b\w+\b", doc.lower()))
        for token in unique_tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    return {token: log((n_docs + 1) / (freq + 1)) + 1 for token, freq in doc_freq.items()}


def _is_protected_token(
    token: str,
    ledger_lower: frozenset[str] = frozenset(),
) -> bool:
    """Return True if token matches any protected pattern or anchor ledger fact."""
    for pattern in _PROTECTED_PATTERNS:
        if pattern.fullmatch(token):
            return True
    # Check if the token matches any ledger critical string (pre-lowered).
    return bool(ledger_lower and token.lower() in ledger_lower)


def _prune_segment(
    text: str,
    idf_scores: dict[str, float],
    keep_ratio: float,
    ledger_lower: frozenset[str] = frozenset(),
) -> str:
    """Prune low-IDF tokens from a text segment.

    Processes line-by-line to preserve structure.
    Protected tokens and tokens with above-threshold IDF are kept.
    """
    lines = text.split("\n")
    result_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result_lines.append(line)
            continue

        # Tokenize preserving whitespace structure.
        tokens = re.findall(r"\S+|\s+", line)
        word_tokens = [(t, idf_scores.get(t.lower().strip(), 1.0)) for t in tokens if t.strip()]

        if not word_tokens:
            result_lines.append(line)
            continue

        # Determine keep threshold based on keep_ratio.
        scores = sorted([s for _, s in word_tokens])
        keep_count = max(1, int(len(scores) * keep_ratio))
        if keep_count >= len(scores):
            result_lines.append(line)
            continue

        threshold = scores[len(scores) - keep_count]

        # Keep tokens above threshold or that are protected.
        kept: list[str] = []
        for token, score in word_tokens:
            if _is_protected_token(token, ledger_lower) or score >= threshold:
                kept.append(token)

        if kept:
            result_lines.append(" ".join(kept))
        # Skip entirely empty lines from pruning.

    return "\n".join(result_lines)
