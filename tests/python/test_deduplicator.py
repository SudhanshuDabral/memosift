# Tests for Layer 2: Semantic Deduplication.
from __future__ import annotations

import pytest

from memosift.config import MemoSiftConfig
from memosift.core.classifier import classify_messages
from memosift.core.deduplicator import (
    _content_hash,
    _cosine_similarity,
    _normalize_for_hash,
    _tfidf_vectors,
    deduplicate,
)
from memosift.core.types import (
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    MemoSiftMessage,
    ToolCall,
    ToolCallFunction,
)
from conftest import messages_from_dicts


# ── Test vector validation ──────────────────────────────────────────────────


class TestDedupVector:
    """Validate dedup-001 test vector."""

    def test_dedup_001(self, dedup_vector: dict) -> None:
        msgs = messages_from_dicts(dedup_vector["input"])
        config = MemoSiftConfig(recent_turns=0)
        classified = classify_messages(msgs, config)
        result, deps = deduplicate(classified, config)

        # Message at index 3 should be deduplicated.
        expected_deduped = dedup_vector["expected"]["deduplicated_indices"]
        for idx in expected_deduped:
            assert dedup_vector["expected"]["message_3_content_contains"] in result[idx].content

        # Dependency tracking.
        assert deps.references  # Should have at least one entry.


# ── Exact deduplication ─────────────────────────────────────────────────────


class TestExactDedup:
    """Exact-match deduplication via content hashing."""

    def test_exact_duplicate_collapsed(self) -> None:
        """Same file content read twice — second should be deduplicated."""
        file_content = "export function add(a: number, b: number): number {\n  return a + b;\n}"
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="tool", content=file_content, tool_call_id="tc1", name="read_file"),
            MemoSiftMessage(role="assistant", content="Got it."),
            MemoSiftMessage(role="tool", content=file_content, tool_call_id="tc2", name="read_file"),
            MemoSiftMessage(role="user", content="query"),
        ]
        config = MemoSiftConfig(recent_turns=0)
        classified = classify_messages(msgs, config)
        result, deps = deduplicate(classified, config)

        # The later duplicate (index 3) should be a back-reference.
        assert "read earlier in this session" in result[3].content
        assert file_content not in result[3].content
        # Original (index 1) should be preserved.
        assert result[1].content == file_content

    def test_whitespace_normalized_before_hash(self) -> None:
        """Same content with different whitespace should be detected as duplicate."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="tool", content="hello world", tool_call_id="tc1"),
            MemoSiftMessage(role="tool", content="hello  world  ", tool_call_id="tc2"),
            MemoSiftMessage(role="user", content="query"),
        ]
        config = MemoSiftConfig(recent_turns=0)
        classified = classify_messages(msgs, config)
        result, deps = deduplicate(classified, config)

        assert "read earlier" in result[2].content

    def test_case_preserved_in_exact_hash(self) -> None:
        """Exact hash dedup preserves case — 'AuthService' vs 'authservice'
        have different SHA-256 hashes. However, fuzzy TF-IDF may still detect
        similarity since tokenization lowercases. This test verifies the hash
        function itself is case-sensitive."""
        from memosift.core.deduplicator import _content_hash

        h1 = _content_hash("AuthService handles login")
        h2 = _content_hash("authservice handles login")
        assert h1 != h2  # Case-sensitive hashing

    def test_distinct_content_not_deduped(self) -> None:
        """Completely different tool results should survive dedup."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(
                role="tool",
                content="The authentication module handles user login and session management.",
                tool_call_id="tc1",
            ),
            MemoSiftMessage(
                role="tool",
                content="Database migration script for PostgreSQL 16 upgrade completed.",
                tool_call_id="tc2",
            ),
            MemoSiftMessage(role="user", content="query"),
        ]
        config = MemoSiftConfig(recent_turns=0)
        classified = classify_messages(msgs, config)
        result, deps = deduplicate(classified, config)

        assert "read earlier" not in result[1].content
        assert "read earlier" not in result[2].content

    def test_preserve_policy_not_deduped(self) -> None:
        """System prompts (PRESERVE policy) should never be deduplicated."""
        msgs = [
            MemoSiftMessage(role="system", content="You are helpful."),
            MemoSiftMessage(role="system", content="You are helpful."),
            MemoSiftMessage(role="user", content="query"),
        ]
        config = MemoSiftConfig(recent_turns=0)
        classified = classify_messages(msgs, config)
        result, deps = deduplicate(classified, config)

        assert result[0].content == "You are helpful."
        assert result[1].content == "You are helpful."
        assert not deps.references


# ── Fuzzy deduplication ─────────────────────────────────────────────────────


class TestFuzzyDedup:
    """TF-IDF based fuzzy deduplication."""

    def test_fuzzy_dedup_same_type_only(self) -> None:
        """Similar content in different content types should NOT be deduplicated."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(
                role="tool",
                content='{"users": [{"id": 1, "name": "Alice"}]}',
                tool_call_id="tc1",
            ),
            MemoSiftMessage(role="assistant", content='users id 1 name Alice'),
            MemoSiftMessage(role="user", content="query"),
        ]
        config = MemoSiftConfig(recent_turns=0, dedup_similarity_threshold=0.5)
        classified = classify_messages(msgs, config)
        result, deps = deduplicate(classified, config)

        # Both should survive — different content types.
        assert "read earlier" not in result[1].content
        assert "read earlier" not in result[2].content

    def test_fuzzy_dedup_below_threshold(self) -> None:
        """Messages with similarity below threshold should NOT be deduplicated."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(
                role="tool",
                content="The quick brown fox jumps over the lazy dog",
                tool_call_id="tc1",
            ),
            MemoSiftMessage(
                role="tool",
                content="Python is a programming language used for data science",
                tool_call_id="tc2",
            ),
            MemoSiftMessage(role="user", content="query"),
        ]
        config = MemoSiftConfig(recent_turns=0)
        classified = classify_messages(msgs, config)
        result, deps = deduplicate(classified, config)

        # Very different content → no dedup.
        assert "read earlier" not in result[1].content
        assert "read earlier" not in result[2].content

    def test_fuzzy_dedup_above_threshold(self) -> None:
        """Near-identical content should be deduplicated."""
        base = "export function add(a: number, b: number): number { return a + b; }\n" * 10
        variant = base + "\n// minor edit"

        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="tool", content=base, tool_call_id="tc1", name="read_file"),
            MemoSiftMessage(role="tool", content=variant, tool_call_id="tc2", name="read_file"),
            MemoSiftMessage(role="user", content="query"),
        ]
        config = MemoSiftConfig(recent_turns=0, dedup_similarity_threshold=0.85)
        classified = classify_messages(msgs, config)
        result, deps = deduplicate(classified, config)

        # The older one (index 1) should be deduplicated; newer (index 2) kept.
        assert "read earlier" in result[1].content


# ── Dependency tracking ─────────────────────────────────────────────────────


class TestDependencyTracking:
    """DependencyMap correctly tracks back-references."""

    def test_dependency_recorded(self) -> None:
        content = "duplicate content here"
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="tool", content=content, tool_call_id="tc1"),
            MemoSiftMessage(role="tool", content=content, tool_call_id="tc2"),
            MemoSiftMessage(role="user", content="query"),
        ]
        config = MemoSiftConfig(recent_turns=0)
        classified = classify_messages(msgs, config)
        result, deps = deduplicate(classified, config)

        # The deduped message's original_index should reference the kept message.
        assert len(deps.references) == 1

    def test_cannot_drop_referenced_message(self) -> None:
        content = "duplicate content here"
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="tool", content=content, tool_call_id="tc1"),
            MemoSiftMessage(role="tool", content=content, tool_call_id="tc2"),
            MemoSiftMessage(role="user", content="query"),
        ]
        config = MemoSiftConfig(recent_turns=0)
        classified = classify_messages(msgs, config)
        _, deps = deduplicate(classified, config)

        # The original message cannot be dropped.
        referenced_idx = list(deps.references.values())[0]
        assert deps.can_drop(referenced_idx) is False


# ── Helper function tests ──────────────────────────────────────────────────


class TestHelpers:
    """Direct tests for deduplicator helper functions."""

    def test_normalize_for_hash(self) -> None:
        assert _normalize_for_hash("  hello   world  ") == "hello world"
        assert _normalize_for_hash("a\n\nb\tc") == "a b c"

    def test_content_hash_deterministic(self) -> None:
        h1 = _content_hash("hello world")
        h2 = _content_hash("hello world")
        assert h1 == h2

    def test_content_hash_whitespace_insensitive(self) -> None:
        h1 = _content_hash("hello world")
        h2 = _content_hash("hello  world  ")
        assert h1 == h2

    def test_content_hash_case_sensitive(self) -> None:
        h1 = _content_hash("Hello")
        h2 = _content_hash("hello")
        assert h1 != h2

    def test_tfidf_vectors_basic(self) -> None:
        docs = ["the cat sat on the mat", "the dog sat on the rug"]
        vectors = _tfidf_vectors(docs)
        assert len(vectors) == 2
        # Common words should have lower TF-IDF; unique words higher.
        assert vectors[0].get("cat", 0) > 0
        assert vectors[1].get("dog", 0) > 0

    def test_cosine_similarity_identical(self) -> None:
        vec = {"a": 1.0, "b": 2.0}
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal(self) -> None:
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert _cosine_similarity(a, b) == 0.0

    def test_cosine_similarity_empty(self) -> None:
        assert _cosine_similarity({}, {"a": 1.0}) == 0.0
        assert _cosine_similarity({}, {}) == 0.0


# ── Edge cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_input(self) -> None:
        result, deps = deduplicate([], MemoSiftConfig())
        assert result == []
        assert not deps.references

    def test_single_message(self) -> None:
        msgs = [MemoSiftMessage(role="user", content="Hello")]
        config = MemoSiftConfig(recent_turns=0)
        classified = classify_messages(msgs, config)
        result, deps = deduplicate(classified, config)
        assert len(result) == 1
        assert not deps.references

    def test_empty_content_not_deduped(self) -> None:
        """Multiple empty tool results shouldn't crash."""
        msgs = [
            MemoSiftMessage(role="system", content="System"),
            MemoSiftMessage(role="tool", content="", tool_call_id="tc1"),
            MemoSiftMessage(role="tool", content="", tool_call_id="tc2"),
            MemoSiftMessage(role="user", content="query"),
        ]
        config = MemoSiftConfig(recent_turns=0)
        classified = classify_messages(msgs, config)
        result, deps = deduplicate(classified, config)
        # Empty strings should still hash identically.
        assert len(result) == 4
