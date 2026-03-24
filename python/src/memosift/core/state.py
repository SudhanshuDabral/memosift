# CompressionState — cached pipeline artifacts for incremental compression.
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memosift.core.types import ContentType


@dataclass
class CompressionState:
    """Cached pipeline artifacts for incremental compression.

    Stores intermediate results across ``compress()`` calls so subsequent
    invocations can skip redundant work on already-compressed messages.

    Fields:
        idf_vocabulary: IDF scores from Engine B (pruner). Merged across calls
            so the current batch uses fresh scores and historical tokens fill gaps.
            Stale entries are harmless — they correspond to tokens absent from the
            current batch and are never looked up during pruning.
        content_hashes: SHA-256 hashes of message content from previous calls.
            Seeded into L2 (deduplicator) to detect cross-call exact duplicates
            without needing ``CrossWindowState``.
        classification_cache: Mapping of content hash → tool-result sub-type
            (TOOL_RESULT_JSON, CODE_BLOCK, etc.). Avoids re-running JSON parsing,
            regex matching, and error detection on identical content. Position-
            dependent classification (RECENT_TURN) runs after the cache lookup.
        token_cache: Mapping of (length, hash) → estimated token count. Uses a
            two-tier key: length for quick discrimination, hash only on collision.
            Primarily benefits real tokenizers (tiktoken); negligible for heuristic.
        sequence: Incremented on each ``compress()`` call. Exposed for callers
            to track how many compression cycles have run.
        output_hash: SHA-256 hash of the last compressed output. Exposed for
            callers to detect whether compression produced changes.
    """

    idf_vocabulary: dict[str, float] = field(default_factory=dict)
    content_hashes: dict[str, int] = field(default_factory=dict)
    classification_cache: dict[str, ContentType] = field(default_factory=dict)
    token_cache: dict[str, int] = field(default_factory=dict)
    sequence: int = 0
    output_hash: str = ""

    def cache_classification(self, content: str, content_type: ContentType) -> None:
        """Cache the classification result for a message's content."""
        key = _content_hash(content)
        self.classification_cache[key] = content_type

    def get_cached_classification(self, content: str) -> ContentType | None:
        """Retrieve a cached classification, or None if not cached."""
        key = _content_hash(content)
        return self.classification_cache.get(key)

    def cache_token_count(self, content: str, count: int) -> None:
        """Cache the token count for a message's content."""
        key = _token_cache_key(content)
        self.token_cache[key] = count

    def get_cached_token_count(self, content: str) -> int | None:
        """Retrieve a cached token count, or None if not cached."""
        key = _token_cache_key(content)
        return self.token_cache.get(key)

    def record_content_hash(self, content: str, index: int) -> None:
        """Record a content hash for dedup cross-referencing."""
        key = _content_hash(content)
        if key not in self.content_hashes:
            self.content_hashes[key] = index

    def get_content_hashes(self) -> dict[str, int]:
        """Return all recorded content hashes (hash → original index)."""
        return dict(self.content_hashes)

    def has_content(self, content: str) -> bool:
        """Check if content has been seen before (exact match)."""
        return _content_hash(content) in self.content_hashes

    def bump_sequence(self) -> int:
        """Increment and return the new sequence number."""
        self.sequence += 1
        return self.sequence

    def set_output_hash(self, messages_content: list[str]) -> None:
        """Compute and store the output hash from compressed message contents."""
        combined = "\n---\n".join(messages_content)
        self.output_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:32]


def _content_hash(content: str) -> str:
    """Compute a short SHA-256 hash of content for cache keying."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _token_cache_key(content: str) -> str:
    """Two-tier cache key: length for fast discrimination, hash on collision.

    Most messages have unique lengths, so ``len:NNNN`` is enough to
    distinguish them without SHA-256. For the rare length collisions,
    the full hash ensures correctness.
    """
    n = len(content)
    # Short content (<256 chars): length + first 32 chars is unique enough.
    if n < 256:
        return f"L{n}:{content[:32]}"
    # Longer content: length + SHA-256 prefix.
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"L{n}:{h}"
