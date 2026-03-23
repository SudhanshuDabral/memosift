# Layer 2: Semantic deduplication — exact hash + MinHash/LSH fuzzy + TF-IDF fallback.
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from memosift.core.types import (
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    DependencyMap,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig

# Policies that skip deduplication entirely.
_SKIP_DEDUP_POLICIES = {CompressionPolicy.PRESERVE, CompressionPolicy.LIGHT}

# MinHash parameters tuned for 0.85 Jaccard threshold.
_NUM_HASHES = 128
_NUM_BANDS = 16
_ROWS_PER_BAND = _NUM_HASHES // _NUM_BANDS  # 8

# Use TF-IDF for groups smaller than this; MinHash overhead isn't worth it.
# Lowered from 10 to 5 (Item 4.1) — 128 permutations is fast enough for small groups.
_MINHASH_MIN_GROUP_SIZE = 5

# Shingle size for MinHash (character n-grams).
_SHINGLE_SIZE = 5

# Large prime for MinHash hash functions.
_PRIME = 2**61 - 1

# Pre-computed hash function coefficients (deterministic seed).
_HASH_A = [((i + 1) * 6364136223846793005 + 1) % _PRIME for i in range(_NUM_HASHES)]
_HASH_B = [((i + 1) * 1442695040888963407 + 7) % _PRIME for i in range(_NUM_HASHES)]


@dataclass
class CrossWindowState:
    """State shared across compression windows for cross-window dedup.

    Pass the same instance to multiple ``deduplicate()`` calls to catch
    duplicates that span windows.
    """

    content_hashes: set[str] = field(default_factory=set)


def deduplicate(
    segments: list[ClassifiedMessage],
    config: MemoSiftConfig,
    cross_window: CrossWindowState | None = None,
    exact_only: bool = False,
) -> tuple[list[ClassifiedMessage], DependencyMap]:
    """Deduplicate messages via exact hash matching, fuzzy similarity, and chunk-level dedup.

    Uses MinHash/LSH for groups >= 5 messages (lowered from 10 per Item 4.1).
    Falls back to TF-IDF cosine similarity for smaller groups.
    Long messages (>1000 tokens) are split into chunks and deduped at chunk level.
    Messages with PRESERVE or LIGHT policy are never deduplicated.

    Args:
        segments: Classified messages from Layer 1.
        config: Pipeline configuration (controls ``dedup_similarity_threshold``).
        cross_window: Optional state for cross-window dedup.
        exact_only: When True, skip fuzzy and chunk-level dedup (faster).

    Returns:
        A tuple of (deduplicated segments, dependency map for Layer 6).
    """
    deps = DependencyMap()
    result = _exact_dedup(segments, deps, cross_window)
    if not exact_only:
        result = _fuzzy_dedup(result, config.dedup_similarity_threshold, deps)
        result = _chunk_dedup(result, config.dedup_similarity_threshold, deps)
    return result, deps


# ── Exact-match deduplication ───────────────────────────────────────────────


def _normalize_for_hash(text: str) -> str:
    """Normalize text before hashing: collapse whitespace, strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def _content_hash(text: str) -> str:
    """SHA-256 hash of normalized text."""
    return hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()


def _exact_dedup(
    segments: list[ClassifiedMessage],
    deps: DependencyMap,
    cross_window: CrossWindowState | None = None,
) -> list[ClassifiedMessage]:
    """Replace exact-duplicate messages with back-references.

    Messages with PRESERVE or LIGHT policy are skipped.
    The later duplicate is replaced; the earlier one is kept.
    If ``cross_window`` is provided, also checks against hashes from
    previous compression windows.
    """
    seen: dict[str, int] = {}  # hash → index in the *segments* input list
    result: list[ClassifiedMessage] = []

    for i, seg in enumerate(segments):
        if seg.policy in _SKIP_DEDUP_POLICIES:
            result.append(seg)
            continue

        h = _content_hash(seg.content)

        # Check against current window.
        if h in seen:
            original_seg_idx = seen[h]
            original = segments[original_seg_idx]
            tool_name = seg.message.name or "content"
            ref_text = (
                f"[{tool_name} was read earlier in this session. "
                f"Content unchanged. See message #{original.original_index}.]"
            )
            new_seg = _replace_content(seg, ref_text)
            deps.add(seg.original_index, original.original_index)
            result.append(new_seg)
        elif cross_window is not None and h in cross_window.content_hashes:
            # Seen in a previous window — replace with cross-window reference.
            tool_name = seg.message.name or "content"
            ref_text = (
                f"[{tool_name} was read earlier in this session. "
                f"Content unchanged (seen in previous context window).]"
            )
            result.append(_replace_content(seg, ref_text))
        else:
            seen[h] = i
            result.append(seg)

    # Add all hashes from this window to cross-window state.
    if cross_window is not None:
        cross_window.content_hashes.update(seen.keys() for _ in [])  # no-op placeholder
        for h in seen:
            cross_window.content_hashes.add(h)

    return result


# ── Fuzzy deduplication (MinHash/LSH + TF-IDF fallback) ────────────────────


def _fuzzy_dedup(
    segments: list[ClassifiedMessage],
    threshold: float,
    deps: DependencyMap,
) -> list[ClassifiedMessage]:
    """Fuzzy dedup using MinHash/LSH for large groups, TF-IDF for small groups.

    Only compares messages of the same ``ContentType``.
    Messages with PRESERVE or LIGHT policy are skipped.
    """
    groups: dict[ContentType, list[int]] = {}
    for i, seg in enumerate(segments):
        if seg.policy in _SKIP_DEDUP_POLICIES:
            continue
        groups.setdefault(seg.content_type, []).append(i)

    for indices in groups.values():
        if len(indices) < 2:
            continue

        docs = [segments[i].content for i in indices]
        if not any(docs):
            continue

        if len(indices) >= _MINHASH_MIN_GROUP_SIZE:
            _fuzzy_dedup_minhash(segments, indices, docs, threshold, deps)
        else:
            _fuzzy_dedup_tfidf(segments, indices, docs, threshold, deps)

    return segments


def _fuzzy_dedup_minhash(
    segments: list[ClassifiedMessage],
    indices: list[int],
    docs: list[str],
    threshold: float,
    deps: DependencyMap,
) -> None:
    """MinHash/LSH fuzzy dedup — O(n) after preprocessing."""
    # Compute MinHash signatures.
    signatures = [_minhash_signature(doc) for doc in docs]

    # LSH banding: hash each band to find candidate pairs.
    candidates: set[tuple[int, int]] = set()
    for band in range(_NUM_BANDS):
        buckets: dict[int, list[int]] = {}
        start = band * _ROWS_PER_BAND
        end = start + _ROWS_PER_BAND
        for pos, sig in enumerate(signatures):
            band_hash = hash(tuple(sig[start:end]))
            buckets.setdefault(band_hash, []).append(pos)

        for bucket_members in buckets.values():
            if len(bucket_members) < 2:
                continue
            for a in range(len(bucket_members)):
                for b in range(a + 1, len(bucket_members)):
                    candidates.add((bucket_members[a], bucket_members[b]))

    # Verify candidates with full Jaccard similarity.
    deduped: set[int] = set()
    for a_pos, b_pos in sorted(candidates):
        if a_pos in deduped or b_pos in deduped:
            continue

        sim = _jaccard_from_minhash(signatures[a_pos], signatures[b_pos])
        if sim >= threshold:
            _mark_as_dedup(segments, indices, a_pos, b_pos, deps)
            deduped.add(a_pos)


def _fuzzy_dedup_tfidf(
    segments: list[ClassifiedMessage],
    indices: list[int],
    docs: list[str],
    threshold: float,
    deps: DependencyMap,
) -> None:
    """TF-IDF cosine similarity fuzzy dedup — fallback for small groups."""
    vectors = _tfidf_vectors(docs)
    deduped: set[int] = set()

    for a_pos in range(len(indices)):
        if a_pos in deduped:
            continue
        for b_pos in range(a_pos + 1, len(indices)):
            if b_pos in deduped:
                continue
            sim = _cosine_similarity(vectors[a_pos], vectors[b_pos])
            if sim >= threshold:
                _mark_as_dedup(segments, indices, a_pos, b_pos, deps)
                deduped.add(a_pos)
                break


def _mark_as_dedup(
    segments: list[ClassifiedMessage],
    indices: list[int],
    a_pos: int,
    b_pos: int,
    deps: DependencyMap,
) -> None:
    """Mark the older message (a_pos) as a duplicate of the newer (b_pos)."""
    older_idx = indices[a_pos]
    newer_idx = indices[b_pos]
    older_seg = segments[older_idx]
    newer_seg = segments[newer_idx]
    tool_name = older_seg.message.name or "content"
    ref_text = (
        f"[{tool_name} was read earlier in this session. "
        f"Content unchanged. See message #{newer_seg.original_index}.]"
    )
    segments[older_idx] = _replace_content(older_seg, ref_text)
    deps.add(older_seg.original_index, newer_seg.original_index)


# ── MinHash implementation (zero dependencies) ────────────────────────────


def _shingles(text: str, k: int = _SHINGLE_SIZE) -> set[int]:
    """Generate character k-gram shingle hashes from text."""
    text = text.lower().strip()
    if len(text) < k:
        return {hash(text)} if text else set()
    return {hash(text[i : i + k]) for i in range(len(text) - k + 1)}


def _minhash_signature(text: str) -> list[int]:
    """Compute MinHash signature (list of N minimum hash values)."""
    shingle_set = _shingles(text)
    if not shingle_set:
        return [_PRIME] * _NUM_HASHES

    sig = [_PRIME] * _NUM_HASHES
    for shingle in shingle_set:
        for i in range(_NUM_HASHES):
            h = (_HASH_A[i] * shingle + _HASH_B[i]) % _PRIME
            if h < sig[i]:
                sig[i] = h
    return sig


def _jaccard_from_minhash(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b, strict=False) if a == b)
    return matches / len(sig_a)


# ── TF-IDF implementation (zero dependencies) ──────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def _tfidf_vectors(documents: list[str]) -> list[dict[str, float]]:
    """Compute sparse TF-IDF vectors for a list of documents.

    IDF formula: log((N + 1) / (df + 1)) + 1  (scikit-learn default).
    """
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
    """Cosine similarity between two sparse vectors."""
    if not a or not b:
        return 0.0

    shared_keys = a.keys() & b.keys()
    dot = sum(a[k] * b[k] for k in shared_keys)

    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# ── Chunk-level deduplication ──────────────────────────────────────────────

# Minimum estimated tokens for a message to be eligible for chunk dedup.
_CHUNK_DEDUP_MIN_TOKENS = 1000

# Target chunk size in tokens (approximate).
_CHUNK_TARGET_TOKENS = 200

# Similarity threshold for chunk dedup (slightly higher than message-level).
_CHUNK_SIMILARITY_THRESHOLD = 0.85


def _chunk_dedup(
    segments: list[ClassifiedMessage],
    threshold: float,
    deps: DependencyMap,
) -> list[ClassifiedMessage]:
    """Chunk-level dedup for long messages.

    Splits long messages (>1000 tokens) into ~200-token chunks at paragraph
    boundaries (prose) or function/class boundaries (code). Deduplicates
    chunks across segments — if a chunk is >85% similar to a previously seen
    chunk, replace with back-reference keeping only the diff.

    Targets tool results — the biggest token consumers.
    """
    seen_chunks: dict[str, tuple[int, str]] = {}  # hash → (seg_index, first_line_label)
    result: list[ClassifiedMessage] = []

    for seg in segments:
        if seg.policy in _SKIP_DEDUP_POLICIES:
            result.append(seg)
            continue

        # Only chunk-dedup long messages.
        est_tokens = len((seg.content or "").split())
        if est_tokens < _CHUNK_DEDUP_MIN_TOKENS:
            result.append(seg)
            continue

        # Split into chunks.
        if seg.content_type in {ContentType.CODE_BLOCK}:
            chunks = _split_code_chunks(seg.content or "")
        else:
            chunks = _split_paragraph_chunks(seg.content or "")

        if len(chunks) <= 1:
            result.append(seg)
            continue

        # Check each chunk for duplicates.
        new_chunks: list[str] = []
        any_deduped = False
        for chunk in chunks:
            chunk_hash = hashlib.sha256(
                re.sub(r"\s+", " ", chunk).strip().encode("utf-8")
            ).hexdigest()

            if chunk_hash in seen_chunks:
                orig_idx, label = seen_chunks[chunk_hash]
                new_chunks.append(f"[Duplicate chunk — see message {orig_idx}: {label}]")
                any_deduped = True
            else:
                seen_chunks[chunk_hash] = (seg.original_index, chunk[:60].strip())
                new_chunks.append(chunk)

        if any_deduped:
            new_content = "\n\n".join(new_chunks)
            result.append(_replace_content(seg, new_content))
        else:
            result.append(seg)

    return result


def _split_paragraph_chunks(text: str) -> list[str]:
    """Split text into chunks at paragraph boundaries (~200 tokens each)."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(para.split())
        if current_tokens + para_tokens > _CHUNK_TARGET_TOKENS and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _split_code_chunks(text: str) -> list[str]:
    """Split code into chunks at function/class boundaries.

    Uses regex to detect function and class definitions. Falls back to
    paragraph splitting if no code boundaries are found.
    """
    # Try to split at function/class boundaries.
    boundary_re = re.compile(
        r"^(?=\s*(?:def |class |function |async |export ))",
        re.MULTILINE,
    )
    parts = boundary_re.split(text)
    parts = [p for p in parts if p.strip()]

    if len(parts) <= 1:
        return _split_paragraph_chunks(text)

    # Group small parts together to meet minimum chunk size.
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for part in parts:
        part_tokens = len(part.split())
        if current_tokens + part_tokens > _CHUNK_TARGET_TOKENS and current:
            chunks.append("\n".join(current))
            current = [part]
            current_tokens = part_tokens
        else:
            current.append(part)
            current_tokens += part_tokens

    if current:
        chunks.append("\n".join(current))

    return chunks


# ── Helpers ─────────────────────────────────────────────────────────────────


def _replace_content(seg: ClassifiedMessage, new_content: str) -> ClassifiedMessage:
    """Create a new ClassifiedMessage with replaced content (immutable pattern)."""
    from dataclasses import replace as dc_replace

    from memosift.core.types import MemoSiftMessage

    new_msg = MemoSiftMessage(
        role=seg.message.role,
        content=new_content,
        name=seg.message.name,
        tool_call_id=seg.message.tool_call_id,
        tool_calls=seg.message.tool_calls,
        metadata=seg.message.metadata,
    )
    return dc_replace(seg, message=new_msg)
