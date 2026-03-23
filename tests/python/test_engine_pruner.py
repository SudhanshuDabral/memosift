# Tests for Engine B: IDF-based Token Pruning.
from __future__ import annotations

import pytest

from memosift.config import MemoSiftConfig
from memosift.core.engines.pruner import (
    _compute_idf_scores,
    _is_protected_token,
    prune_tokens,
)
from memosift.core.types import (
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    MemoSiftMessage,
)


def _make_segment(
    content: str,
    policy: CompressionPolicy = CompressionPolicy.MODERATE,
) -> ClassifiedMessage:
    return ClassifiedMessage(
        message=MemoSiftMessage(role="tool", content=content, tool_call_id="tc1"),
        content_type=ContentType.TOOL_RESULT_TEXT,
        policy=policy,
    )


class TestIdfScoring:
    """Tests for IDF score computation."""

    def test_common_words_low_idf(self) -> None:
        docs = ["the cat is here", "the dog is there", "the bird is flying"]
        scores = _compute_idf_scores(docs)
        # "the" and "is" appear in all 3 docs → low IDF.
        assert scores["the"] < scores["cat"]
        assert scores["is"] < scores["dog"]

    def test_unique_words_high_idf(self) -> None:
        docs = ["the cat is here", "the dog is there"]
        scores = _compute_idf_scores(docs)
        assert scores["cat"] > scores["the"]
        assert scores["dog"] > scores["the"]


class TestProtectedTokens:
    """Protected tokens are never pruned."""

    def test_file_path_protected(self) -> None:
        assert _is_protected_token("src/auth.ts") is True

    def test_camel_case_protected(self) -> None:
        assert _is_protected_token("authService") is True
        assert _is_protected_token("getUserName") is True

    def test_snake_case_protected(self) -> None:
        assert _is_protected_token("auth_service") is True

    def test_upper_case_protected(self) -> None:
        assert _is_protected_token("AUTH_SERVICE") is True

    def test_number_protected(self) -> None:
        assert _is_protected_token("42") is True
        assert _is_protected_token("3.14") is True

    def test_url_protected(self) -> None:
        assert _is_protected_token("https://example.com") is True

    def test_plain_word_not_protected(self) -> None:
        assert _is_protected_token("the") is False
        assert _is_protected_token("hello") is False


class TestPruneTokens:
    """Integration tests for token pruning."""

    def test_preserve_policy_untouched(self) -> None:
        seg = _make_segment("don't touch this at all", CompressionPolicy.PRESERVE)
        result = prune_tokens([seg], MemoSiftConfig())
        assert result[0].content == "don't touch this at all"

    def test_output_is_subset_of_input(self) -> None:
        """Every token in output must exist verbatim in input."""
        content = "The quick brown fox jumps over the lazy dog near the river bank today"
        seg = _make_segment(content, CompressionPolicy.MODERATE)
        result = prune_tokens([seg], MemoSiftConfig(token_prune_keep_ratio=0.5))
        output_tokens = set(result[0].content.split())
        input_tokens = set(content.split())
        assert output_tokens.issubset(input_tokens)

    def test_identifiers_always_kept(self) -> None:
        content = "the authService is configured in src/config.ts for the main_handler"
        seg = _make_segment(content, CompressionPolicy.MODERATE)
        # Even with aggressive pruning, identifiers should survive.
        segs = [seg, _make_segment("the the the is is is")]
        result = prune_tokens(segs, MemoSiftConfig(token_prune_keep_ratio=0.3))
        output = result[0].content
        assert "authService" in output or "src/config.ts" in output or "main_handler" in output

    def test_empty_input(self) -> None:
        result = prune_tokens([], MemoSiftConfig())
        assert result == []

    def test_empty_content(self) -> None:
        seg = _make_segment("", CompressionPolicy.MODERATE)
        result = prune_tokens([seg], MemoSiftConfig())
        assert result[0].content == ""
