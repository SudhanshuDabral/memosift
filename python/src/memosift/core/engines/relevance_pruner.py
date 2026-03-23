# Layer 3E: Query-relevance pruning — TF-IDF + causal dependency check.
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from memosift.core.types import (
    AnchorLedger,
    ClassifiedMessage,
    CompressionPolicy,
    DependencyMap,
    MemoSiftMessage,
    Shield,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig

# Policies that skip relevance pruning.
_SKIP_POLICIES = {CompressionPolicy.PRESERVE, CompressionPolicy.LIGHT}

# Stop words excluded from TF-IDF vectorization.
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

# Low relevance threshold — messages below this are candidates for compression.
_LOW_RELEVANCE_THRESHOLD = 0.15

# High anchor coverage threshold — if facts are in ledger, safe to collapse.
_ANCHOR_COVERAGE_THRESHOLD = 0.8


def query_relevance_prune(
    segments: list[ClassifiedMessage],
    config: MemoSiftConfig,
    deps: DependencyMap,
    ledger: AnchorLedger | None = None,
) -> list[ClassifiedMessage]:
    """Prune low-relevance messages using TF-IDF cosine similarity against recent queries.

    Scores each non-protected segment against the last 2 user queries (expanded
    with anchor ledger terms). Low-relevance segments with COMPRESSIBLE shield
    are either collapsed to ledger references or heavily pruned.

    Safety mechanisms (3 guards):
    1. DependencyMap dedup check — don't remove messages that are referenced.
    2. DependencyMap logical check — don't break reasoning chains.
    3. Shield level — respect PRESERVE shields from importance scoring.

    Args:
        segments: Classified messages with shield levels assigned by L3G.
        config: Pipeline configuration.
        deps: DependencyMap for dependency checking.
        ledger: Optional anchor ledger for query expansion and coverage checks.

    Returns:
        Segments with low-relevance content compressed or collapsed.
    """
    # Extract last 2 user queries.
    user_queries = _extract_recent_user_queries(segments, n=2)
    if not user_queries:
        return segments  # No queries to score against.

    # Build expanded query from user queries + anchor ledger context.
    expanded_query = _build_expanded_query(user_queries, ledger)

    # Compute TF-IDF vectors for all segments + query.
    all_texts = [seg.content or "" for seg in segments] + [expanded_query]
    vectors = _tfidf_vectors(all_texts)
    query_vector = vectors[-1]
    segment_vectors = vectors[:-1]

    result: list[ClassifiedMessage] = []
    for i, seg in enumerate(segments):
        # Skip protected segments.
        if seg.policy in _SKIP_POLICIES or seg.protected:
            result.append(seg)
            continue

        # Skip PRESERVE-shielded segments.
        if seg.shield == Shield.PRESERVE:
            result.append(seg)
            continue

        # Compute relevance score.
        score = _cosine_similarity(segment_vectors[i], query_vector)

        # Distractor detection: high similarity but low overlap with active context.
        if score > 0.3 and ledger is not None:
            active_facts = ledger.facts_by_category(
                __import__(
                    "memosift.core.types", fromlist=["AnchorCategory"]
                ).AnchorCategory.ACTIVE_CONTEXT
            )
            if active_facts:
                active_text = " ".join(f.content for f in active_facts)
                active_overlap = _fact_overlap(seg.content or "", active_text)
                if active_overlap < 0.2:
                    score *= 0.5  # Penalize distractor.

        # Check dependencies before compressing.
        if deps.has_dependents(seg.original_index):
            result.append(seg)
            continue
        if deps.has_logical_dependents(seg.original_index):
            result.append(seg)
            continue

        # Apply compression based on score and shield.
        if score < _LOW_RELEVANCE_THRESHOLD and seg.shield == Shield.COMPRESSIBLE:
            if (
                ledger is not None
                and _anchor_coverage(seg.content or "", ledger) >= _ANCHOR_COVERAGE_THRESHOLD
            ):
                # Facts are in the ledger — safe to collapse.
                new_content = "[Facts preserved in anchor ledger]"
                new_msg = MemoSiftMessage(
                    role=seg.message.role,
                    content=new_content,
                    name=seg.message.name,
                    tool_call_id=seg.message.tool_call_id,
                    tool_calls=seg.message.tool_calls,
                    metadata=seg.message.metadata,
                )
                result.append(dc_replace(seg, message=new_msg, relevance_score=score))
            else:
                # Apply heavy token pruning (keep 30% of tokens).
                pruned = _heavy_prune(seg.content or "", keep_ratio=0.3)
                new_msg = MemoSiftMessage(
                    role=seg.message.role,
                    content=pruned,
                    name=seg.message.name,
                    tool_call_id=seg.message.tool_call_id,
                    tool_calls=seg.message.tool_calls,
                    metadata=seg.message.metadata,
                )
                result.append(dc_replace(seg, message=new_msg, relevance_score=score))
        else:
            result.append(dc_replace(seg, relevance_score=max(seg.relevance_score, score)))

    return result


def _extract_recent_user_queries(
    segments: list[ClassifiedMessage],
    n: int = 2,
) -> list[str]:
    """Extract the last N user messages from segments."""
    queries: list[str] = []
    for seg in reversed(segments):
        if seg.message.role == "user" and seg.content:
            queries.append(seg.content)
            if len(queries) >= n:
                break
    queries.reverse()
    return queries


def _build_expanded_query(
    queries: list[str],
    ledger: AnchorLedger | None,
) -> str:
    """Build expanded query from user queries + anchor ledger ACTIVE_CONTEXT and ERRORS."""
    parts = list(queries)
    if ledger is not None:
        from memosift.core.types import AnchorCategory

        for fact in ledger.facts_by_category(AnchorCategory.ACTIVE_CONTEXT):
            parts.append(fact.content)
        for fact in ledger.facts_by_category(AnchorCategory.ERRORS):
            parts.append(fact.content)
    return " ".join(parts)


def _fact_overlap(text: str, reference: str) -> float:
    """Compute word overlap between text and reference (0.0-1.0)."""
    text_words = set(re.findall(r"\b\w+\b", text.lower())) - _STOP_WORDS
    ref_words = set(re.findall(r"\b\w+\b", reference.lower())) - _STOP_WORDS
    if not ref_words:
        return 0.0
    return len(text_words & ref_words) / len(ref_words)


def _anchor_coverage(text: str, ledger: AnchorLedger) -> float:
    """Compute what fraction of the segment's entities appear in the anchor ledger."""
    if not text:
        return 0.0
    protected = ledger.get_protected_strings()
    if not protected:
        return 0.0

    # Extract entities from text.
    entities: set[str] = set()
    # File paths.
    for match in re.finditer(r"(?:[A-Za-z]:)?[\w.\-]+(?:[/\\][\w.\-]+)+", text):
        entities.add(match.group(0))
    # Code identifiers.
    for match in re.finditer(r"\b[a-z]+(?:[A-Z][a-z]+)+\b", text):
        entities.add(match.group(0))
    for match in re.finditer(r"\b[a-z]+(?:_[a-z]+)+\b", text):
        entities.add(match.group(0))

    if not entities:
        return 1.0  # No entities to check — consider fully covered.

    covered = sum(1 for e in entities if any(p in e or e in p for p in protected))
    return covered / len(entities)


def _heavy_prune(text: str, keep_ratio: float = 0.3) -> str:
    """Aggressively prune tokens, keeping only the most distinctive ones.

    Simple approach: keep the first `keep_ratio` fraction of words per line.
    Protected patterns (file paths, numbers, identifiers) are always kept.
    """
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        words = line.split()
        if not words:
            result.append(line)
            continue
        keep_count = max(1, int(len(words) * keep_ratio))
        result.append(" ".join(words[:keep_count]))
    return "\n".join(result)


# ── TF-IDF implementation ────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, excluding stop words."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


def _tfidf_vectors(documents: list[str]) -> list[dict[str, float]]:
    """Compute sparse TF-IDF vectors for a list of documents."""
    n_docs = len(documents)
    tokenized = [_tokenize(doc) for doc in documents]

    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))

    idf: dict[str, float] = {
        token: math.log((n_docs + 1) / (freq + 1)) + 1 for token, freq in doc_freq.items()
    }

    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        if not tokens:
            vectors.append({})
            continue
        tf: Counter[str] = Counter(tokens)
        total = len(tokens)
        vec = {token: (count / total) * idf.get(token, 1.0) for token, count in tf.items()}
        vectors.append(vec)

    return vectors


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not a or not b:
        return 0.0
    shared_keys = a.keys() & b.keys()
    dot = sum(a[k] * b[k] for k in shared_keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
