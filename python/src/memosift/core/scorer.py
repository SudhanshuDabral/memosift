# Layer 4: Task-aware relevance scoring — keyword mode + optional LLM mode.
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from memosift.core.types import (
    AnchorLedger,
    ClassifiedMessage,
    ContentType,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig
    from memosift.providers.base import MemoSiftLLMProvider

logger = logging.getLogger("memosift")

# Stop words excluded from keyword extraction.
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
        "being",
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
        "shall",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "not",
        "no",
        "so",
        "if",
        "then",
        "than",
        "just",
        "also",
        "very",
        "too",
        "all",
        "any",
        "some",
        "each",
    }
)

# Content types that are never dropped by scoring.
_PROTECTED_TYPES: frozenset[ContentType] = frozenset(
    {
        ContentType.SYSTEM_PROMPT,
        ContentType.USER_QUERY,
        ContentType.RECENT_TURN,
        ContentType.PREVIOUSLY_COMPRESSED,
    }
)

# Base relevance scores per content type — minimum score for each type.
_CONTENT_TYPE_BASE_SCORES: dict[ContentType, float] = {
    ContentType.SYSTEM_PROMPT: 1.0,
    ContentType.USER_QUERY: 1.0,
    ContentType.RECENT_TURN: 1.0,
    ContentType.PREVIOUSLY_COMPRESSED: 1.0,
    ContentType.ERROR_TRACE: 0.6,
    ContentType.CODE_BLOCK: 0.3,
    ContentType.TOOL_RESULT_TEXT: 0.0,
    ContentType.TOOL_RESULT_JSON: 0.0,
    ContentType.OLD_CONVERSATION: 0.0,
    ContentType.ASSISTANT_REASONING: 0.0,
}

# Boost for segments containing anchor ledger facts.
_ANCHOR_BOOST = 0.3

LLM_SCORE_PROMPT = """Rate the relevance of this SEGMENT to the given TASK on a scale of 0-10.

TASK: {task}

SEGMENT:
{content}

Respond with ONLY a JSON object: {{"score": <0-10>, "reason": "<brief reason>"}}"""


async def score_relevance(
    segments: list[ClassifiedMessage],
    config: MemoSiftConfig,
    task: str | None = None,
    ledger: AnchorLedger | None = None,
) -> list[ClassifiedMessage]:
    """Score each segment's relevance using TF-IDF weighted keywords + anchor boost.

    Three scoring signals are combined:
    1. **TF-IDF keyword score** — Rare, distinctive task keywords (e.g.,
       "authentication") contribute more than common ones (e.g., "file").
    2. **Content-type base score** — ERROR_TRACE always scores at least 0.6,
       CODE_BLOCK at least 0.4, etc.
    3. **Anchor boost** — Segments containing anchor ledger facts (file paths,
       errors, identifiers) get +0.3 boost. This prevents critical segments
       from being dropped when they don't share keywords with the current task.

    Protected segments (SYSTEM_PROMPT, USER_QUERY, RECENT_TURN,
    PREVIOUSLY_COMPRESSED) are always scored 1.0.

    Args:
        segments: Classified messages from previous layers.
        config: Pipeline configuration (controls ``relevance_drop_threshold``).
        task: Optional task description for keyword scoring.
        ledger: Optional anchor ledger for anchor-boosted scoring.

    Returns:
        Segments with ``relevance_score`` populated. Segments below
        ``relevance_drop_threshold`` are removed.
    """
    if not task:
        return [dc_replace(seg, relevance_score=1.0 if seg.protected else 0.5) for seg in segments]

    task_keywords = _extract_keywords(task)
    if not task_keywords:
        return [dc_replace(seg, relevance_score=1.0 if seg.protected else 0.5) for seg in segments]

    # Use critical strings (FILES + ERRORS only) for rescue to avoid
    # broad IDENTIFIER matches that kill compression ratios.
    critical_strings = ledger.get_critical_strings() if ledger else frozenset()

    result: list[ClassifiedMessage] = []
    for seg in segments:
        if seg.content_type in _PROTECTED_TYPES:
            result.append(dc_replace(seg, relevance_score=1.0))
            continue

        # 1. Simple keyword overlap score (same as before).
        content_keywords = _extract_keywords(seg.content)
        if not content_keywords:
            score = 0.0
        else:
            overlap = task_keywords & content_keywords
            score = len(overlap) / len(task_keywords)

        # 2. Anchor rescue — segments containing critical facts get a floor
        # score to prevent dropping. Two tiers:
        #   a) Segments with critical strings (FILES, ERRORS, high-value IDs):
        #      floor at threshold (never dropped by L4).
        #   b) Segments with shield=PRESERVE from importance scoring:
        #      floor at threshold (importance scorer already validated).
        if score < config.relevance_drop_threshold:
            rescued = False

            # Tier a: critical strings rescue.
            if critical_strings:
                text_lower = seg.content.lower()
                if any(s.lower() in text_lower for s in critical_strings):
                    score = max(score, config.relevance_drop_threshold)
                    rescued = True

            # Tier b: importance shield rescue.
            if not rescued and hasattr(seg, "shield"):
                from memosift.core.types import Shield

                if seg.shield == Shield.PRESERVE:
                    score = max(score, config.relevance_drop_threshold)
                    rescued = True

        # 3. Position-dependent compression (Lost in the Middle mitigation).
        # Multiply relevance by a U-shaped position factor: lighter compression
        # at start/end, heavier in the middle — exploits attention curve.
        # Uses original_index for stable positioning, not loop index.
        if len(segments) > 10:
            max_idx = max(s.original_index for s in segments) or 1
            position_factor = _position_factor(seg.original_index, max_idx)
            score *= position_factor

        # Drop below threshold.
        if score < config.relevance_drop_threshold:
            continue

        result.append(dc_replace(seg, relevance_score=score))

    return result


def _position_factor(index: int, total: int) -> float:
    """U-shaped position factor for Lost in the Middle mitigation.

    First 15%: 1.1 (lighter compression — primacy effect).
    Middle 70%: 0.9 (slightly heavier compression — model pays less attention).
    Last 15%: 1.15 (lightest compression — recency matters most).

    Kept close to 1.0 to avoid large swings that tank compression ratio.
    """
    if total <= 0:
        return 1.0
    position_pct = index / total
    if position_pct < 0.15:
        return 1.1
    elif position_pct > 0.85:
        return 1.15
    else:
        return 0.9


def _compute_keyword_idf(
    task_keywords: set[str],
    all_keywords: list[set[str]],
) -> dict[str, float]:
    """Compute IDF weights for task keywords across segment corpus.

    Rare keywords get higher IDF (more distinctive, more important).
    """
    n_docs = len(all_keywords)
    if n_docs == 0:
        return {kw: 1.0 for kw in task_keywords}

    doc_freq: Counter[str] = Counter()
    for kw_set in all_keywords:
        for kw in task_keywords & kw_set:
            doc_freq[kw] += 1

    return {kw: math.log((n_docs + 1) / (doc_freq.get(kw, 0) + 1)) + 1 for kw in task_keywords}


async def score_relevance_llm(
    segments: list[ClassifiedMessage],
    config: MemoSiftConfig,
    task: str,
    llm: MemoSiftLLMProvider,
) -> list[ClassifiedMessage]:
    """Score relevance using the LLM (opt-in mode).

    Sends each non-protected segment to the LLM for scoring on a 0-10 scale.
    Drops segments scoring below 3/10. Falls back to keyword mode on failure.

    Args:
        segments: Classified messages from previous layers.
        config: Pipeline configuration.
        task: Task description for scoring.
        llm: LLM provider for generating scores.

    Returns:
        Segments with LLM-derived relevance scores. Low-scored dropped.
    """
    import asyncio

    # Separate protected (instant) from scorable (need LLM calls).
    protected: list[tuple[int, ClassifiedMessage]] = []
    scorable: list[tuple[int, ClassifiedMessage]] = []
    for i, seg in enumerate(segments):
        if seg.content_type in _PROTECTED_TYPES:
            protected.append((i, dc_replace(seg, relevance_score=1.0)))
        else:
            scorable.append((i, seg))

    async def _score_one(idx: int, seg: ClassifiedMessage) -> tuple[int, ClassifiedMessage | None]:
        try:
            prompt = LLM_SCORE_PROMPT.format(task=task, content=seg.content[:2000])
            response = await llm.generate(prompt, max_tokens=100, temperature=0.0)
            raw = response.text.strip()
            if "```" in raw:
                raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
            json_match = re.search(r"\{[^}]+\}", raw)
            parsed = json.loads(json_match.group()) if json_match else json.loads(raw)
            llm_score = float(parsed.get("score", 5)) / 10.0
            if llm_score < 0.3:
                return (idx, None)  # Drop.
            return (idx, dc_replace(seg, relevance_score=llm_score))
        except Exception as e:
            logger.warning("LLM scoring failed for segment %d: %s", seg.original_index, e)
            content_keywords = _extract_keywords(seg.content)
            task_keywords = _extract_keywords(task)
            if content_keywords and task_keywords:
                overlap = task_keywords & content_keywords
                score = len(overlap) / len(task_keywords)
            else:
                score = 0.5
            if score >= config.relevance_drop_threshold:
                return (idx, dc_replace(seg, relevance_score=score))
            return (idx, None)

    # Fire all LLM calls in parallel.
    scored = await asyncio.gather(*[_score_one(i, seg) for i, seg in scorable])

    # Reassemble in original order.
    all_results: dict[int, ClassifiedMessage] = {}
    for i, seg in protected:
        all_results[i] = seg
    for i, seg in scored:
        if seg is not None:
            all_results[i] = seg

    return [all_results[i] for i in sorted(all_results.keys())]


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text, excluding stop words."""
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    return tokens - _STOP_WORDS
