# Tests for Sprint 2 improvements (v0.3): chunk dedup, importance scoring,
# relevance pruning, elaboration compression, reasoning chain tracking.
from __future__ import annotations

import pytest

from memosift.config import MemoSiftConfig
from memosift.core.anchor_extractor import extract_reasoning_chains
from memosift.core.deduplicator import deduplicate
from memosift.core.engines.discourse_compressor import (
    _compress_elaborations,
    _is_satellite_clause,
    _split_at_sentence_boundaries,
    elaborate_compress,
)
from memosift.core.engines.importance import (
    _compute_instruction_strength,
    score_importance,
)
from memosift.core.engines.relevance_pruner import (
    _build_expanded_query,
    _extract_recent_user_queries,
    query_relevance_prune,
)
from memosift.core.types import (
    AnchorCategory,
    AnchorFact,
    AnchorLedger,
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    DependencyMap,
    MemoSiftMessage,
    Shield,
)


def _make_segment(
    content: str,
    role: str = "tool",
    policy: CompressionPolicy = CompressionPolicy.MODERATE,
    content_type: ContentType = ContentType.TOOL_RESULT_TEXT,
    original_index: int = 0,
    tool_call_id: str | None = "tc1",
    protected: bool = False,
    shield: Shield = Shield.MODERATE,
) -> ClassifiedMessage:
    return ClassifiedMessage(
        message=MemoSiftMessage(
            role=role,
            content=content,
            tool_call_id=tool_call_id if role == "tool" else None,
        ),
        content_type=content_type,
        policy=policy,
        original_index=original_index,
        protected=protected,
        shield=shield,
    )


# ── Item 10: Reasoning Chain Tracking ────────────────────────────────────────


class TestReasoningChainTracking:
    def test_dependency_map_logical_edges(self) -> None:
        deps = DependencyMap()
        deps.add_logical(5, 2)
        assert deps.has_logical_dependents(2) is True
        assert deps.has_logical_dependents(5) is False
        assert deps.logical_dependents_of(2) == [5]

    def test_can_drop_checks_logical(self) -> None:
        deps = DependencyMap()
        deps.add_logical(5, 2)
        assert deps.can_drop(2) is False  # Message 5 depends on 2.
        assert deps.can_drop(5) is True  # No one depends on 5.

    def test_reasoning_chain_detected(self) -> None:
        segments = [
            _make_segment(
                "We need to fix the auth bug",
                role="assistant",
                content_type=ContentType.ASSISTANT_REASONING,
                original_index=0,
                tool_call_id=None,
            ),
            _make_segment(
                "Therefore, we should update the middleware",
                role="assistant",
                content_type=ContentType.ASSISTANT_REASONING,
                original_index=2,
                tool_call_id=None,
            ),
        ]
        deps = DependencyMap()
        extract_reasoning_chains(segments, deps)
        # Message 2 should depend on message 0 ("therefore" marker).
        assert 2 in deps.logical_deps
        assert deps.logical_deps[2] == 0

    def test_no_chain_without_markers(self) -> None:
        segments = [
            _make_segment(
                "Here is the code",
                role="assistant",
                content_type=ContentType.ASSISTANT_REASONING,
                original_index=0,
                tool_call_id=None,
            ),
            _make_segment(
                "And here is more code",
                role="assistant",
                content_type=ContentType.ASSISTANT_REASONING,
                original_index=2,
                tool_call_id=None,
            ),
        ]
        deps = DependencyMap()
        extract_reasoning_chains(segments, deps)
        assert len(deps.logical_deps) == 0


# ── Item 7: Multi-Signal Importance Scoring ──────────────────────────────────


class TestImportanceScoring:
    def test_protected_gets_preserve_shield(self) -> None:
        seg = _make_segment(
            "system prompt",
            policy=CompressionPolicy.PRESERVE,
            protected=True,
        )
        result = score_importance([seg], MemoSiftConfig())
        assert result[0].shield == Shield.PRESERVE
        assert result[0].importance_score == 1.0

    def test_entity_dense_content_scores_higher(self) -> None:
        entity_seg = _make_segment(
            "Error in src/auth.ts at https://api.example.com with authService and user_handler",
            policy=CompressionPolicy.AGGRESSIVE,
            original_index=0,
        )
        plain_seg = _make_segment(
            "The quick brown fox jumps over the lazy dog near the river bank today",
            policy=CompressionPolicy.AGGRESSIVE,
            original_index=1,
        )
        result = score_importance([entity_seg, plain_seg], MemoSiftConfig())
        assert result[0].importance_score > result[1].importance_score

    def test_instruction_detection_graduated(self) -> None:
        # Absolute instruction → highest strength.
        assert _compute_instruction_strength("You must never delete this file", "user") >= 0.9
        # Imperative → medium.
        assert 0.5 <= _compute_instruction_strength("Use Redis for caching", "user") < 0.9
        # Hedged → low.
        assert _compute_instruction_strength("Maybe consider using Redis", "user") < 0.3

    def test_user_instructions_weighted_higher(self) -> None:
        user_strength = _compute_instruction_strength("Use Redis for caching", "user")
        assistant_strength = _compute_instruction_strength("Use Redis for caching", "assistant")
        assert user_strength > assistant_strength

    def test_shield_assignment(self) -> None:
        # Create a segment that should be COMPRESSIBLE (plain prose, no entities).
        seg = _make_segment(
            "The quick brown fox jumps over the lazy dog",
            policy=CompressionPolicy.AGGRESSIVE,
        )
        result = score_importance([seg], MemoSiftConfig())
        assert result[0].shield in {Shield.COMPRESSIBLE, Shield.MODERATE}

    def test_empty_segments(self) -> None:
        result = score_importance([], MemoSiftConfig())
        assert result == []


# ── Item 8: TF-IDF Query-Relevance Pruning ──────────────────────────────────


class TestQueryRelevancePruning:
    def test_extract_recent_queries(self) -> None:
        segments = [
            _make_segment("First question", role="user", original_index=0, tool_call_id=None),
            _make_segment("Answer", role="assistant", original_index=1, tool_call_id=None),
            _make_segment("Second question", role="user", original_index=2, tool_call_id=None),
            _make_segment("Answer 2", role="assistant", original_index=3, tool_call_id=None),
            _make_segment("Third question", role="user", original_index=4, tool_call_id=None),
        ]
        queries = _extract_recent_user_queries(segments, n=2)
        assert len(queries) == 2
        assert "Second question" in queries[0]
        assert "Third question" in queries[1]

    def test_query_expansion_with_ledger(self) -> None:
        ledger = AnchorLedger()
        ledger.add(AnchorFact(
            category=AnchorCategory.ACTIVE_CONTEXT,
            content="Current task: fix auth middleware",
            turn=5,
        ))
        ledger.add(AnchorFact(
            category=AnchorCategory.ERRORS,
            content="TypeError: Cannot read 'userId'",
            turn=4,
        ))
        expanded = _build_expanded_query(["fix the bug"], ledger)
        assert "auth middleware" in expanded
        assert "TypeError" in expanded

    def test_protected_segments_not_pruned(self) -> None:
        segments = [
            _make_segment(
                "Irrelevant old content about cooking recipes",
                policy=CompressionPolicy.PRESERVE,
                protected=True,
                original_index=0,
            ),
            _make_segment(
                "Fix the authentication bug",
                role="user",
                policy=CompressionPolicy.PRESERVE,
                protected=True,
                original_index=1,
                tool_call_id=None,
            ),
        ]
        deps = DependencyMap()
        result = query_relevance_prune(segments, MemoSiftConfig(), deps)
        assert len(result) == 2

    def test_dependency_protected_segments_kept(self) -> None:
        """Segments with dependents should not be compressed."""
        segments = [
            _make_segment(
                "Old irrelevant content",
                policy=CompressionPolicy.AGGRESSIVE,
                shield=Shield.COMPRESSIBLE,
                original_index=0,
            ),
            _make_segment(
                "Fix the auth bug",
                role="user",
                policy=CompressionPolicy.PRESERVE,
                protected=True,
                original_index=1,
                tool_call_id=None,
            ),
        ]
        deps = DependencyMap()
        deps.add(5, 0)  # Something depends on message 0.
        result = query_relevance_prune(segments, MemoSiftConfig(), deps)
        # Message 0 should be kept because it has dependents.
        assert any(seg.original_index == 0 for seg in result)


# ── Item 9: Elaboration Compression ──────────────────────────────────────────


class TestElaborationCompression:
    def test_satellite_detection(self) -> None:
        assert _is_satellite_clause("For example, you could use Redis here.") is True
        assert _is_satellite_clause("Because the old implementation was slow.") is True
        assert _is_satellite_clause("The function returns a User object.") is False

    def test_sentence_splitting(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        parts = _split_at_sentence_boundaries(text)
        assert len(parts) == 3

    def test_satellite_compressed_not_deleted(self) -> None:
        """Satellites should be compressed to ~20% tokens, not deleted entirely."""
        text = (
            "I chose bcrypt for password hashing. "
            "For example, bcrypt automatically handles salt generation and "
            "provides configurable cost factors that make brute force attacks "
            "computationally expensive even with modern hardware."
        )
        result = _compress_elaborations(text, ledger=None)
        # The elaboration should still be present, just shorter.
        assert "For example" in result or "bcrypt" in result
        assert len(result) < len(text)

    def test_nucleus_clause_preserved(self) -> None:
        text = "I chose bcrypt for password hashing."
        result = _compress_elaborations(text, ledger=None)
        assert result == text

    def test_anchor_facts_preserved_in_satellite(self) -> None:
        """Satellites containing anchor facts are kept in full."""
        ledger = AnchorLedger()
        ledger.add(AnchorFact(
            category=AnchorCategory.FILES,
            content="src/auth.ts — modified at turn 3",
            turn=3,
        ))
        text = "Specifically, the changes to src/auth.ts include the new JWT validation logic and updated error handling."
        result = _compress_elaborations(text, ledger)
        # Should be preserved because it contains an anchor fact.
        assert "src/auth.ts" in result

    def test_recent_turns_not_compressed(self) -> None:
        """Recent turns should not be elaboration-compressed."""
        segments = [
            _make_segment(
                "User question",
                role="user",
                original_index=0,
                tool_call_id=None,
            ),
            _make_segment(
                "For example, this is a detailed elaboration with many words",
                role="assistant",
                policy=CompressionPolicy.AGGRESSIVE,
                shield=Shield.COMPRESSIBLE,
                content_type=ContentType.ASSISTANT_REASONING,
                original_index=1,
                tool_call_id=None,
            ),
        ]
        # With recent_turns=2, message at index 1 is recent → not compressed.
        result = elaborate_compress(segments, MemoSiftConfig(recent_turns=5))
        assert result[1].content == segments[1].content

    def test_preserve_shield_respected(self) -> None:
        segments = [
            _make_segment(
                "For example, this important content should not be compressed",
                policy=CompressionPolicy.AGGRESSIVE,
                shield=Shield.PRESERVE,
                original_index=0,
            ),
        ]
        result = elaborate_compress(segments, MemoSiftConfig())
        assert result[0].content == segments[0].content


# ── Item 6: Chunk-Level Deduplication ────────────────────────────────────────


class TestChunkLevelDedup:
    def test_long_duplicate_chunks_collapsed(self) -> None:
        """Identical long chunks across messages should be deduped."""
        # Create a long repeated block.
        long_block = "\n\n".join([f"Paragraph {i}: " + "x " * 50 for i in range(10)])
        seg1 = _make_segment(
            long_block,
            policy=CompressionPolicy.AGGRESSIVE,
            content_type=ContentType.TOOL_RESULT_TEXT,
            original_index=0,
        )
        seg2 = _make_segment(
            long_block,
            policy=CompressionPolicy.AGGRESSIVE,
            content_type=ContentType.TOOL_RESULT_TEXT,
            original_index=5,
        )
        # The exact dedup should catch this before chunk dedup.
        config = MemoSiftConfig()
        result, deps = deduplicate([seg1, seg2], config)
        # Second message should be deduplicated.
        assert "earlier in this session" in result[1].content or "Duplicate chunk" in result[1].content

    def test_short_messages_not_chunked(self) -> None:
        """Short messages (<1000 tokens) should not undergo chunk dedup."""
        seg = _make_segment(
            "Short content here",
            policy=CompressionPolicy.AGGRESSIVE,
            original_index=0,
        )
        config = MemoSiftConfig()
        result, deps = deduplicate([seg], config)
        assert result[0].content == "Short content here"

    def test_minhash_threshold_lowered(self) -> None:
        """MinHash should now be used for groups of 5+ (was 10)."""
        from memosift.core.deduplicator import _MINHASH_MIN_GROUP_SIZE
        assert _MINHASH_MIN_GROUP_SIZE == 5


# ── Pipeline Integration ─────────────────────────────────────────────────────


class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_new_layers_execute_in_pipeline(self) -> None:
        """New Sprint 2 layers should execute without errors in the pipeline."""
        from memosift.core.pipeline import compress

        messages = [
            MemoSiftMessage(role="system", content="You are a helpful assistant."),
            MemoSiftMessage(role="user", content="Fix the authentication bug in auth.ts"),
            MemoSiftMessage(
                role="assistant",
                content="I'll use JWT for authentication. For example, JWT tokens are stateless and don't require server-side session storage.",
            ),
            MemoSiftMessage(
                role="tool",
                content="Contents of src/auth.ts:\n" + "function authenticate() {\n  // auth logic\n}\n" * 20,
                tool_call_id="tc1",
                name="read_file",
            ),
            MemoSiftMessage(
                role="assistant",
                content="Therefore, we should update the middleware to validate JWT tokens. Building on that, the error handler needs updating too.",
            ),
            MemoSiftMessage(role="user", content="Now add rate limiting"),
        ]
        ledger = AnchorLedger()
        config = MemoSiftConfig(token_budget=2000)
        compressed, report = await compress(messages, config=config, ledger=ledger)

        # Should produce output without errors.
        assert len(compressed) >= 2  # At least system + something.
        assert report.compression_ratio >= 1.0

        # New layers should appear in the report.
        layer_names = [lr.name for lr in report.layers]
        assert "importance_scorer" in layer_names
        assert "relevance_pruner" in layer_names
        assert "discourse_compressor" in layer_names

        # Anchor ledger should have structured sections.
        assert len(ledger.facts) > 0
        intent_facts = ledger.facts_by_category(AnchorCategory.INTENT)
        assert len(intent_facts) >= 1
