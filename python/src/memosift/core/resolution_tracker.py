# Resolution tracker — audit-only detection of question→decision arcs in conversations.
#
# IMPORTANT: This module is READ-ONLY — it detects patterns and reports them
# but does NOT modify shields, scores, or any compression behavior. It exists
# to gather data on whether semantic detection would improve compression quality,
# without risking regressions from false positives.
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memosift.core.types import ClassifiedMessage

# Question detection patterns.
_QUESTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\?\s*$", re.MULTILINE),  # Ends with ?
    re.compile(r"\bshould (?:we|I)\b", re.IGNORECASE),
    re.compile(r"\bwhich (?:one|approach|option)\b", re.IGNORECASE),
    re.compile(r"\bhow (?:to|should|do)\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:about|if)\b", re.IGNORECASE),
]

# Deliberation patterns (comparison, alternatives).
_DELIBERATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\balternatively\b", re.IGNORECASE),
    re.compile(r"\bon (?:one|the other) hand\b", re.IGNORECASE),
    re.compile(r"\bpros and cons\b", re.IGNORECASE),
    re.compile(r"\boption [A-Z]\b", re.IGNORECASE),
    re.compile(r"\bcould (?:also |either )?use\b", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\btradeoff\b", re.IGNORECASE),
]

# Decision/resolution patterns (reuses anchor_extractor's markers).
_RESOLUTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bI'll use\b", re.IGNORECASE),
    re.compile(r"\bLet's go with\b", re.IGNORECASE),
    re.compile(r"\bchoosing\b.{1,60}\bbecause\b", re.IGNORECASE),
    re.compile(r"\bdecided to\b", re.IGNORECASE),
    re.compile(r"\bI'll go with\b", re.IGNORECASE),
    re.compile(r"\bwe(?:'ll| will) use\b", re.IGNORECASE),
    re.compile(r"\bgoing with\b", re.IGNORECASE),
    re.compile(r"\bthe (?:best|right) (?:choice|approach|option) is\b", re.IGNORECASE),
]

# Supersession patterns (later message corrects/updates earlier).
_SUPERSESSION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bactually\b", re.IGNORECASE),
    re.compile(r"\bturns out\b", re.IGNORECASE),
    re.compile(r"\bcorrection\b", re.IGNORECASE),
    re.compile(r"\bnot .{1,30} but\b", re.IGNORECASE),
    re.compile(r"\bupdated?\b.*\bnow\b", re.IGNORECASE),
    re.compile(r"\ball \d+ tests? pass", re.IGNORECASE),  # Status update pattern
]

# Stop words for keyword extraction.
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
        "may",
        "might",
        "can",
        "not",
        "no",
        "if",
        "then",
        "else",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "we",
        "you",
        "they",
        "use",
        "using",
        "used",
        "going",
        "think",
        "want",
        "need",
        "like",
    }
)


@dataclass(frozen=True)
class ResolutionArc:
    """A detected question→deliberation→decision arc. Audit-only — no compression effect."""

    question_index: int
    deliberation_indices: tuple[int, ...]
    resolution_index: int | None
    resolved: bool
    topic_keywords: frozenset[str]


@dataclass(frozen=True)
class SupersessionSignal:
    """A detected supersession (later message corrects/updates earlier). Audit-only."""

    superseded_index: int
    superseding_index: int
    reason: str  # "correction", "status_update", "refinement"
    shared_entities: frozenset[str]


@dataclass
class ResolutionReport:
    """Summary of detected arcs and supersessions. Logged to CompressionReport."""

    arcs: list[ResolutionArc] = field(default_factory=list)
    supersessions: list[SupersessionSignal] = field(default_factory=list)

    @property
    def resolved_count(self) -> int:
        return sum(1 for a in self.arcs if a.resolved)

    @property
    def unresolved_count(self) -> int:
        return sum(1 for a in self.arcs if not a.resolved)

    def to_dict(self) -> dict:
        """Serialize for CompressionReport."""
        return {
            "arcs_detected": len(self.arcs),
            "arcs_resolved": self.resolved_count,
            "arcs_unresolved": self.unresolved_count,
            "supersessions_detected": len(self.supersessions),
            "supersession_reasons": dict(_count_by(s.reason for s in self.supersessions)),
        }


def detect_resolution_arcs(
    segments: list[ClassifiedMessage],
) -> ResolutionReport:
    """Detect question→deliberation→decision arcs in classified messages.

    This is AUDIT-ONLY — it returns a report but does NOT modify any
    segment's shield, relevance score, or compression behavior.

    Args:
        segments: Classified messages from L1.

    Returns:
        ResolutionReport with detected arcs and supersessions.
    """
    report = ResolutionReport()

    # Step 1: Identify message roles (question, deliberation, resolution).
    questions: list[int] = []
    deliberations: list[int] = []
    resolutions: list[int] = []

    for i, seg in enumerate(segments):
        content = seg.content
        if not content or len(content) < 10:
            continue

        role = seg.message.role

        # Questions come from user messages.
        if role == "user" and _matches_any(content, _QUESTION_PATTERNS):
            questions.append(i)

        # Deliberation and resolution come from assistant messages.
        if role == "assistant":
            if _matches_any(content, _RESOLUTION_PATTERNS):
                resolutions.append(i)
            elif _matches_any(content, _DELIBERATION_PATTERNS):
                deliberations.append(i)

    # Step 2: Link questions to resolutions via keyword overlap.
    for q_idx in questions:
        q_keywords = _extract_keywords(segments[q_idx].content)
        if len(q_keywords) < 2:
            continue

        # Find deliberations between this question and the next question.
        next_q = min((qi for qi in questions if qi > q_idx), default=len(segments))
        arc_deliberations = [d for d in deliberations if q_idx < d < next_q]

        # Find the first resolution after this question with keyword overlap.
        arc_resolution: int | None = None
        for r_idx in resolutions:
            if r_idx <= q_idx:
                continue
            r_keywords = _extract_keywords(segments[r_idx].content)
            overlap = q_keywords & r_keywords
            if len(overlap) >= 2:  # Require at least 2 shared non-stop-word tokens.
                arc_resolution = r_idx
                break

        resolved = arc_resolution is not None
        topic = q_keywords
        if arc_resolution is not None:
            topic = topic | _extract_keywords(segments[arc_resolution].content)

        report.arcs.append(
            ResolutionArc(
                question_index=q_idx,
                deliberation_indices=tuple(arc_deliberations),
                resolution_index=arc_resolution,
                resolved=resolved,
                topic_keywords=frozenset(topic),
            )
        )

    # Step 3: Detect supersessions (corrections, status updates).
    for i, seg in enumerate(segments):
        if seg.message.role != "assistant" or len(seg.content) < 20:
            continue
        if not _matches_any(seg.content, _SUPERSESSION_PATTERNS):
            continue

        # Look for an earlier message with shared entities.
        i_entities = _extract_entities(seg.content)
        if not i_entities:
            continue

        for j in range(max(0, i - 20), i):
            if segments[j].message.role not in ("assistant", "tool"):
                continue
            j_entities = _extract_entities(segments[j].content)
            shared = i_entities & j_entities
            if len(shared) >= 1:
                reason = _classify_supersession(seg.content)
                report.supersessions.append(
                    SupersessionSignal(
                        superseded_index=j,
                        superseding_index=i,
                        reason=reason,
                        shared_entities=frozenset(shared),
                    )
                )
                break  # One supersession per message.

    return report


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    """Return True if any pattern matches the text."""
    return any(p.search(text) for p in patterns)


def _extract_keywords(text: str) -> frozenset[str]:
    """Extract non-stop-word tokens from text."""
    tokens = set(re.findall(r"\b[a-zA-Z]\w{2,}\b", text.lower()))
    return frozenset(tokens - _STOP_WORDS)


def _extract_entities(text: str) -> frozenset[str]:
    """Extract likely entity names: file paths, camelCase, function names."""
    entities: set[str] = set()
    # File paths.
    for m in re.finditer(r"[\w./\\]+\.\w{1,8}", text):
        entities.add(m.group(0).lower())
    # camelCase / PascalCase.
    for m in re.finditer(r"\b[a-z]+(?:[A-Z][a-z]+)+\b", text):
        entities.add(m.group(0).lower())
    # Function-like patterns.
    for m in re.finditer(r"\b(\w+)\s*\(", text):
        name = m.group(1).lower()
        if name not in _STOP_WORDS and len(name) > 2:
            entities.add(name)
    return frozenset(entities)


def _classify_supersession(text: str) -> str:
    """Classify the type of supersession."""
    lower = text.lower()
    if any(w in lower for w in ("actually", "turns out", "correction", "not", "but")):
        return "correction"
    if any(w in lower for w in ("all", "pass", "now", "updated")):
        return "status_update"
    return "refinement"


def _count_by(items):
    """Count items by value."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts
