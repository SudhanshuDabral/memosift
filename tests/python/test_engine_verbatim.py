# Tests for Engine A: Verbatim Deletion.
from __future__ import annotations

import pytest

from memosift.config import MemoSiftConfig
from memosift.core.engines.verbatim import (
    _collapse_blank_lines,
    _collapse_repetitive_patterns,
    _is_protected_line,
    _remove_low_entropy_lines,
    shannon_entropy,
    verbatim_compress,
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
    content_type: ContentType = ContentType.TOOL_RESULT_TEXT,
) -> ClassifiedMessage:
    """Helper to create a ClassifiedMessage for testing."""
    return ClassifiedMessage(
        message=MemoSiftMessage(role="tool", content=content, tool_call_id="tc1"),
        content_type=content_type,
        policy=policy,
    )


# ── Shannon Entropy ─────────────────────────────────────────────────────────


class TestShannonEntropy:
    """Tests for entropy calculation."""

    def test_empty_string(self) -> None:
        assert shannon_entropy("") == 0.0

    def test_single_char_repeat(self) -> None:
        assert shannon_entropy("aaaaaaa") == 0.0

    def test_separator_line_low_entropy(self) -> None:
        entropy = shannon_entropy("=" * 30)
        assert entropy < 1.0

    def test_english_prose_high_entropy(self) -> None:
        text = "The quick brown fox jumps over the lazy dog near the river."
        entropy = shannon_entropy(text)
        assert entropy > 3.0

    def test_mixed_content(self) -> None:
        # "---" has lower entropy than prose.
        assert shannon_entropy("---") < shannon_entropy("hello world")


# ── Blank Line Collapse ─────────────────────────────────────────────────────


class TestBlankLineCollapse:
    """5 consecutive blank lines → 1."""

    def test_collapse_many_blanks(self) -> None:
        lines = ["line1", "", "", "", "", "", "line2"]
        result = _collapse_blank_lines(lines)
        assert result == ["line1", "", "line2"]

    def test_single_blank_preserved(self) -> None:
        lines = ["line1", "", "line2"]
        result = _collapse_blank_lines(lines)
        assert result == ["line1", "", "line2"]

    def test_no_blanks(self) -> None:
        lines = ["a", "b", "c"]
        result = _collapse_blank_lines(lines)
        assert result == ["a", "b", "c"]


# ── Low-Entropy Line Removal ────────────────────────────────────────────────


class TestLowEntropyRemoval:
    """Remove separator/decorative lines below entropy threshold."""

    def test_separator_lines_removed(self) -> None:
        lines = ["hello world", "=" * 30, "goodbye"]
        result = _remove_low_entropy_lines(lines, 2.0)
        assert "=" * 30 not in result
        assert "hello world" in result

    def test_real_content_preserved(self) -> None:
        lines = ["The function returns a User object or None if not found."]
        result = _remove_low_entropy_lines(lines, 2.0)
        assert len(result) == 1

    def test_file_paths_never_deleted(self) -> None:
        """Lines containing file paths survive all deletion rules."""
        lines = ["src/auth.ts:47"]
        result = _remove_low_entropy_lines(lines, 100.0)  # Impossible threshold
        assert len(result) == 1

    def test_lines_with_numbers_protected(self) -> None:
        lines = ["line 47"]
        result = _remove_low_entropy_lines(lines, 100.0)
        assert len(result) == 1

    def test_blank_lines_pass_through(self) -> None:
        lines = ["", "content", ""]
        result = _remove_low_entropy_lines(lines, 2.0)
        assert result == ["", "content", ""]


# ── Repetitive Pattern Collapse ─────────────────────────────────────────────


class TestRepetitivePatterns:
    """200 lines of '✓ test passed' → first 3 + count annotation."""

    def test_repetitive_pattern_truncated(self) -> None:
        lines = [f"✓ test passed: test_{i:03d}" for i in range(200)]
        result = _collapse_repetitive_patterns(lines)
        # Should have 3 kept + 1 annotation = 4 lines.
        assert len(result) == 4
        assert "similar lines omitted" in result[-1]
        assert "197" in result[-1]

    def test_below_threshold_not_collapsed(self) -> None:
        lines = ["same line", "same line", "same line"]
        result = _collapse_repetitive_patterns(lines)
        assert result == lines  # 3 = threshold, not exceeded.

    def test_mixed_content(self) -> None:
        lines = ["unique line 1", "repeated"] * 3 + ["unique line 2"]
        result = _collapse_repetitive_patterns(lines)
        # No runs exceed threshold, so all preserved.
        assert len(result) == 7


# ── Protected Lines ─────────────────────────────────────────────────────────


class TestProtectedLines:
    """Lines with file paths and numbers are never deleted."""

    def test_file_path_protected(self) -> None:
        assert _is_protected_line("src/auth.ts") is True
        assert _is_protected_line("/usr/local/bin/python") is True

    def test_line_with_number_protected(self) -> None:
        assert _is_protected_line("Error on line 42") is True

    def test_plain_text_not_protected(self) -> None:
        assert _is_protected_line("hello world") is False

    def test_empty_not_protected(self) -> None:
        assert _is_protected_line("") is False
        assert _is_protected_line("   ") is False


# ── Full verbatim_compress ──────────────────────────────────────────────────


class TestVerbatimCompress:
    """Integration tests for the full verbatim compression engine."""

    def test_preserve_policy_untouched(self) -> None:
        seg = _make_segment("don't touch this", CompressionPolicy.PRESERVE)
        result = verbatim_compress([seg], MemoSiftConfig())
        assert result[0].content == "don't touch this"

    def test_light_policy_untouched(self) -> None:
        seg = _make_segment("keep this too", CompressionPolicy.LIGHT)
        result = verbatim_compress([seg], MemoSiftConfig())
        assert result[0].content == "keep this too"

    def test_moderate_policy_compressed(self) -> None:
        content = "line1\n" + "=" * 30 + "\nline2\n\n\n\n\n\nline3"
        seg = _make_segment(content, CompressionPolicy.MODERATE)
        result = verbatim_compress([seg], MemoSiftConfig())
        compressed = result[0].content
        assert "=" * 30 not in compressed
        assert "line1" in compressed
        assert "line2" in compressed
        assert compressed.count("\n\n") <= 1  # Blank lines collapsed

    def test_aggressive_policy_compressed(self) -> None:
        content = "important\n" + "-" * 50 + "\nmore important"
        seg = _make_segment(content, CompressionPolicy.AGGRESSIVE)
        result = verbatim_compress([seg], MemoSiftConfig())
        assert "-" * 50 not in result[0].content

    def test_multiple_segments(self) -> None:
        segs = [
            _make_segment("keep me", CompressionPolicy.PRESERVE),
            _make_segment("clean\n" + "=" * 30 + "\nme", CompressionPolicy.MODERATE),
        ]
        result = verbatim_compress(segs, MemoSiftConfig())
        assert result[0].content == "keep me"
        assert "=" * 30 not in result[1].content

    def test_empty_input(self) -> None:
        result = verbatim_compress([], MemoSiftConfig())
        assert result == []

    def test_truncation_for_very_long_content(self) -> None:
        """Content over 100 lines gets truncated with a marker."""
        lines = [f"line {i}: important data here" for i in range(200)]
        content = "\n".join(lines)
        seg = _make_segment(content, CompressionPolicy.MODERATE)
        result = verbatim_compress([seg], MemoSiftConfig())
        compressed = result[0].content
        assert "lines omitted" in compressed
        # Should be roughly 100 lines + marker.
        assert compressed.count("\n") < 150
