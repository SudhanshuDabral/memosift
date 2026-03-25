# Layer 3G: Multi-signal importance scoring (BudgetMem-inspired, 6 signals).
from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace as dc_replace
from math import log
from typing import TYPE_CHECKING

from memosift.core.types import (
    AnchorLedger,
    ClassifiedMessage,
    CompressionPolicy,
    Shield,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig

# Policies that skip importance scoring (already protected).
_SKIP_POLICIES = {CompressionPolicy.PRESERVE, CompressionPolicy.LIGHT}

# ── Signal detection patterns ──────────────────────────────────────────────

# Entity patterns: file paths, URLs, identifiers, function/class names.
_ENTITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:[A-Za-z]:)?[\w.\-]+(?:[/\\][\w.\-]+)+(?:\.\w+)?"),  # File paths
    re.compile(r"https?://\S+"),  # URLs
    re.compile(r"\b[a-z]+(?:[A-Z][a-z]+)+\b"),  # camelCase
    re.compile(r"\b[a-z]+(?:_[a-z]+)+\b"),  # snake_case
    re.compile(r"\bclass\s+\w+"),  # class names
    re.compile(r"\bdef\s+\w+"),  # function names
    re.compile(r"\bfunction\s+\w+"),  # JS function names
    re.compile(r"\b[A-Z][A-Z_]{2,}\b"),  # UPPER_CASE constants
]

# Numerical patterns: line numbers, ports, counts, versions, error codes.
_NUMERICAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bline\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bport\s+\d+\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:items?|rows?|records?|files?|bytes?|KB|MB|GB)\b", re.IGNORECASE),
    re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b"),  # Version numbers
    re.compile(r"\b(?:0x[0-9a-f]+|E\d{4,})\b", re.IGNORECASE),  # Error codes
    re.compile(r"\b\d{3,}\b"),  # 3+ digit numbers (ports, line numbers)
]

# Combined patterns for single-pass matching.
_ENTITY_COMBINED = re.compile("|".join(p.pattern for p in _ENTITY_PATTERNS))
_NUMERICAL_COMBINED = re.compile("|".join(p.pattern for p in _NUMERICAL_PATTERNS), re.IGNORECASE)

# Domain metric pattern: numbers likely to be KPIs (ratio units, precision, non-generic).
# Matches: "1,992.32 Mcf/d", "2,573 psig", "126 mg/dL", "12,500 req/s"
# Does NOT match: "line 47", "3 items", "v2.1.0" (those are generic numericals above).
_DOMAIN_METRIC_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s+(?:[A-Za-z]+/[A-Za-z]+|[A-Z][a-z]*[A-Z])",  # ratio units or CamelCase
    re.IGNORECASE,
)

# Discourse markers: questions, conclusions, decisions.
_QUESTION_PATTERN = re.compile(r"\?\s*$", re.MULTILINE)
_CONCLUSION_MARKERS = re.compile(
    r"\b(?:therefore|in conclusion|the result is|this means|in summary|"
    r"to summarize|the key takeaway|ultimately|finally)\b",
    re.IGNORECASE,
)
_DECISION_MARKERS = re.compile(
    r"\b(?:decided to|chose|let's go with|I'll use|we'll use|"
    r"the decision is|going with|selecting)\b",
    re.IGNORECASE,
)

# Instruction detection (graduated).
_ABSOLUTE_INSTRUCTIONS = re.compile(
    r"\b(?:must|never|always|required|mandatory|critical|essential|"
    r"do not|don't ever|absolutely)\b",
    re.IGNORECASE,
)
_IMPERATIVE_INSTRUCTIONS = re.compile(
    r"\b(?:use|run|install|create|add|remove|delete|update|set|configure|"
    r"ensure|make sure|implement|deploy|fix|change)\b",
    re.IGNORECASE,
)
_CONDITIONAL_INSTRUCTIONS = re.compile(
    r"\b(?:if\b.{1,40}\bthen|only when|unless|when\b.{1,40}\bshould|"
    r"in case|provided that)\b",
    re.IGNORECASE,
)
_HEDGED_INSTRUCTIONS = re.compile(
    r"\b(?:maybe|consider|could|might|perhaps|possibly|optionally)\b",
    re.IGNORECASE,
)

# Stop words for TF-IDF.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "this",
        "that",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "you",
        "he",
        "she",
        "they",
        "not",
        "no",
        "so",
        "if",
        "then",
        "just",
        "also",
    }
)


def score_importance(
    segments: list[ClassifiedMessage],
    config: MemoSiftConfig,
    ledger: AnchorLedger | None = None,
    phase_multiplier: float = 1.0,
) -> list[ClassifiedMessage]:
    """Score each segment's importance using 6 signals and assign shield levels.

    Signals (aligned with BudgetMem's validated feature set + instruction detection):
    1. Entity density (file paths, IDs, code names per token)
    2. Numerical density (line numbers, ports, counts per token)
    3. Discourse markers (questions, conclusions, decisions)
    4. Instruction detection (graduated: absolute > imperative > conditional > hedged)
    5. Position weight (recency bias — closer to end = more important)
    6. TF-IDF importance (mean TF-IDF score of segment tokens)

    Shield assignment:
    - importance > 0.7 → PRESERVE
    - importance > 0.3 → MODERATE
    - else → COMPRESSIBLE

    Args:
        segments: Classified messages from previous layers.
        config: Pipeline configuration.
        ledger: Optional anchor ledger (not directly used — entities extracted inline).

    Returns:
        Segments with importance_score and shield populated.
    """
    if not segments:
        return segments

    total_segments = len(segments)

    # Pre-compute TF-IDF scores only for scorable segments (skip PRESERVE/LIGHT).
    scorable_for_idf = [
        seg for seg in segments if seg.policy not in _SKIP_POLICIES and not seg.protected
    ]
    corpus_idf = _compute_corpus_idf(scorable_for_idf) if scorable_for_idf else {}

    result: list[ClassifiedMessage] = []
    for i, seg in enumerate(segments):
        if seg.policy in _SKIP_POLICIES or seg.protected:
            result.append(dc_replace(seg, importance_score=1.0, shield=Shield.PRESERVE))
            continue

        text = seg.content or ""
        token_count = max(len(text.split()), 1)

        # Signal 1: Entity density (weight: 0.15)
        entity_count = len(_ENTITY_COMBINED.findall(text))
        entity_density = min(entity_count / token_count, 1.0)

        # Signal 2a: Generic numerical density (weight: 0.05)
        # Line numbers, ports, version numbers, error codes.
        generic_num_count = len(_NUMERICAL_COMBINED.findall(text))
        generic_numerical = min(generic_num_count / token_count, 1.0)

        # Signal 2b: Domain metric density (weight: 0.10)
        # Numbers with units (ratio patterns like X/Y, comma-separated, non-common words).
        # Uses the contextual metric heuristic — numbers likely to be KPIs.
        domain_metric_count = len(_DOMAIN_METRIC_RE.findall(text))
        domain_numerical = min(domain_metric_count / token_count, 1.0)

        # Signal 3: Discourse markers (weight: 0.15)
        has_question = bool(_QUESTION_PATTERN.search(text))
        has_conclusion = bool(_CONCLUSION_MARKERS.search(text))
        has_decision = bool(_DECISION_MARKERS.search(text))
        discourse_score = 1.0 if (has_question or has_conclusion or has_decision) else 0.0

        # Signal 4: Instruction detection -- graduated (weight: 0.15)
        instruction_strength = _compute_instruction_strength(text, seg.role)

        # Signal 5: Position weight (weight: 0.15)
        distance_from_end = total_segments - 1 - i
        position_weight = 1.0 / (1.0 + distance_from_end * 0.1)

        # Signal 6: TF-IDF importance (weight: 0.10)
        tfidf_importance = _mean_tfidf_score(text, corpus_idf)

        # Signal 7: Anchor fact coverage (weight: 0.10)
        # What fraction of this message's content overlaps with anchor ledger facts.
        anchor_coverage = 0.0
        if ledger is not None:
            critical = ledger.get_critical_strings()
            if critical:
                covered = sum(1 for s in critical if s.lower() in text.lower())
                anchor_coverage = min(covered / max(len(critical), 1), 1.0)

        # Combined importance (7 signals).
        importance = (
            entity_density * 0.15
            + generic_numerical * 0.05
            + domain_numerical * 0.10
            + discourse_score * 0.15
            + instruction_strength * 0.15
            + position_weight * 0.15
            + tfidf_importance * 0.10
            + anchor_coverage * 0.10
        )

        # Absolute override: hard constraints always PRESERVE.
        if instruction_strength >= 0.7:
            importance = max(importance, 0.75)

        # Apply phase multiplier (conversation phase detection).
        importance *= phase_multiplier

        # Assign shield.
        if importance > 0.7:
            shield = Shield.PRESERVE
        elif importance > 0.3:
            shield = Shield.MODERATE
        else:
            shield = Shield.COMPRESSIBLE

        result.append(dc_replace(seg, importance_score=importance, shield=shield))

    return result


def _compute_instruction_strength(text: str, role: str) -> float:
    """Compute graduated instruction strength (0.0-1.0).

    Higher for absolute constraints, lower for hedged suggestions.
    User instructions weighted higher than assistant suggestions.
    """
    strength = 0.0
    if _ABSOLUTE_INSTRUCTIONS.search(text):
        strength = 1.0
    elif _IMPERATIVE_INSTRUCTIONS.search(text):
        strength = 0.7
    elif _CONDITIONAL_INSTRUCTIONS.search(text):
        strength = 0.5
    elif _HEDGED_INSTRUCTIONS.search(text):
        strength = 0.2

    # Speaker weighting: user instructions > assistant suggestions.
    if role == "user":
        strength *= 1.0
    elif role == "assistant":
        strength *= 0.6
    else:
        strength *= 0.8

    return strength


def _compute_corpus_idf(segments: list[ClassifiedMessage]) -> dict[str, float]:
    """Compute IDF scores across the segment corpus."""
    n_docs = len(segments)
    doc_freq: Counter[str] = Counter()
    for seg in segments:
        tokens = set(re.findall(r"\b\w+\b", (seg.content or "").lower()))
        tokens -= _STOP_WORDS
        doc_freq.update(tokens)

    return {token: log((n_docs + 1) / (freq + 1)) + 1 for token, freq in doc_freq.items()}


def _mean_tfidf_score(text: str, corpus_idf: dict[str, float]) -> float:
    """Compute mean TF-IDF score for tokens in text."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [t for t in tokens if t not in _STOP_WORDS]
    if not tokens:
        return 0.0

    tf: Counter[str] = Counter(tokens)
    total = len(tokens)
    scores = [(tf[token] / total) * corpus_idf.get(token, 1.0) for token in set(tokens)]
    return min(sum(scores) / max(len(scores), 1), 1.0)
