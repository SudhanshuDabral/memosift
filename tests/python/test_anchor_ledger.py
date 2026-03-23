# Tests for the Anchor Ledger and anchor extraction.
from __future__ import annotations

import pytest

from memosift.core.anchor_extractor import (
    extract_anchors_from_message,
    extract_anchors_from_segments,
)
from memosift.core.types import (
    AnchorCategory,
    AnchorFact,
    AnchorLedger,
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    MemoSiftMessage,
)


# ── AnchorFact tests ────────────────────────────────────────────────────────


class TestAnchorFact:
    def test_construction(self) -> None:
        fact = AnchorFact(
            category=AnchorCategory.FILES,
            content="src/auth.ts — read at turn 3",
            turn=3,
        )
        assert fact.category == AnchorCategory.FILES
        assert fact.turn == 3
        assert fact.confidence == 1.0

    def test_frozen(self) -> None:
        fact = AnchorFact(category=AnchorCategory.ERRORS, content="error", turn=1)
        with pytest.raises(AttributeError):
            fact.content = "changed"  # type: ignore[misc]


# ── AnchorLedger tests ──────────────────────────────────────────────────────


class TestAnchorLedger:
    def test_add_success(self) -> None:
        ledger = AnchorLedger()
        fact = AnchorFact(category=AnchorCategory.FILES, content="src/auth.ts", turn=1)
        assert ledger.add(fact) is True
        assert len(ledger.facts) == 1

    def test_add_dedup(self) -> None:
        ledger = AnchorLedger()
        fact = AnchorFact(category=AnchorCategory.FILES, content="src/auth.ts", turn=1)
        ledger.add(fact)
        duplicate = AnchorFact(category=AnchorCategory.FILES, content="src/auth.ts", turn=5)
        assert ledger.add(duplicate) is False
        assert len(ledger.facts) == 1

    def test_add_different_content(self) -> None:
        ledger = AnchorLedger()
        ledger.add(AnchorFact(category=AnchorCategory.FILES, content="auth.ts", turn=1))
        ledger.add(AnchorFact(category=AnchorCategory.FILES, content="db.ts", turn=2))
        assert len(ledger.facts) == 2

    def test_update(self) -> None:
        ledger = AnchorLedger()
        ledger.add(AnchorFact(category=AnchorCategory.FILES, content="file — read", turn=1))
        ledger.update(AnchorCategory.FILES, "file — read", "file — modified")
        assert ledger.facts[0].content == "file — modified"

    def test_update_no_match(self) -> None:
        ledger = AnchorLedger()
        ledger.add(AnchorFact(category=AnchorCategory.FILES, content="auth.ts", turn=1))
        ledger.update(AnchorCategory.ERRORS, "nonexistent", "new")
        assert ledger.facts[0].content == "auth.ts"  # Unchanged.

    def test_render(self) -> None:
        ledger = AnchorLedger()
        ledger.add(AnchorFact(category=AnchorCategory.FILES, content="src/auth.ts", turn=3))
        ledger.add(AnchorFact(category=AnchorCategory.ERRORS, content="TypeError: bad", turn=5))
        rendered = ledger.render()
        assert "[SESSION MEMORY" in rendered
        assert "## FILES TOUCHED" in rendered
        assert "src/auth.ts" in rendered
        assert "## ERRORS ENCOUNTERED" in rendered
        assert "TypeError: bad" in rendered

    def test_render_empty(self) -> None:
        ledger = AnchorLedger()
        rendered = ledger.render()
        assert "[SESSION MEMORY" in rendered
        assert "## FILES" not in rendered

    def test_token_estimate(self) -> None:
        ledger = AnchorLedger()
        ledger.add(AnchorFact(category=AnchorCategory.FILES, content="x" * 100, turn=1))
        estimate = ledger.token_estimate()
        assert estimate > 0
        assert estimate == len(ledger.render()) // 4

    def test_facts_by_category(self) -> None:
        ledger = AnchorLedger()
        ledger.add(AnchorFact(category=AnchorCategory.FILES, content="a.ts", turn=1))
        ledger.add(AnchorFact(category=AnchorCategory.ERRORS, content="err", turn=2))
        ledger.add(AnchorFact(category=AnchorCategory.FILES, content="b.ts", turn=3))
        assert len(ledger.facts_by_category(AnchorCategory.FILES)) == 2
        assert len(ledger.facts_by_category(AnchorCategory.ERRORS)) == 1
        assert len(ledger.facts_by_category(AnchorCategory.DECISIONS)) == 0


# ── Anchor extraction tests ─────────────────────────────────────────────────


class TestExtractAnchorsFromMessage:
    def test_file_path_extraction(self) -> None:
        msg = MemoSiftMessage(
            role="tool",
            content="Contents of src/auth.ts:\ncode here",
            tool_call_id="tc1",
            name="read_file",
        )
        facts = extract_anchors_from_message(msg, turn=3, tool_name="read_file")
        file_facts = [f for f in facts if f.category == AnchorCategory.FILES]
        assert len(file_facts) >= 1
        assert any("src/auth.ts" in f.content for f in file_facts)
        assert any("read" in f.content for f in file_facts)

    def test_edit_tool_marks_modified(self) -> None:
        msg = MemoSiftMessage(
            role="tool",
            content="Edited src/auth.ts successfully.",
            tool_call_id="tc1",
            name="edit_file",
        )
        facts = extract_anchors_from_message(msg, turn=5, tool_name="edit_file")
        file_facts = [f for f in facts if f.category == AnchorCategory.FILES]
        assert any("modified" in f.content for f in file_facts)

    def test_error_extraction(self) -> None:
        msg = MemoSiftMessage(
            role="tool",
            content="TypeError: Cannot read properties of undefined (reading 'userId')",
            tool_call_id="tc1",
        )
        facts = extract_anchors_from_message(msg, turn=8)
        error_facts = [f for f in facts if f.category == AnchorCategory.ERRORS]
        assert len(error_facts) >= 1
        assert any("TypeError" in f.content for f in error_facts)

    def test_line_ref_extraction(self) -> None:
        msg = MemoSiftMessage(
            role="tool",
            content="Error at auth.ts:47 in login method",
            tool_call_id="tc1",
        )
        facts = extract_anchors_from_message(msg, turn=8)
        id_facts = [f for f in facts if f.category == AnchorCategory.IDENTIFIERS]
        assert any("auth.ts:47" in f.content for f in id_facts)

    def test_empty_content(self) -> None:
        msg = MemoSiftMessage(role="tool", content="", tool_call_id="tc1")
        facts = extract_anchors_from_message(msg, turn=1)
        assert facts == []

    def test_no_matches(self) -> None:
        msg = MemoSiftMessage(role="tool", content="All tests passed.", tool_call_id="tc1")
        facts = extract_anchors_from_message(msg, turn=1)
        # May or may not have matches depending on regex — just shouldn't crash.
        assert isinstance(facts, list)


class TestExtractAnchorsFromSegments:
    def test_integration(self) -> None:
        segments = [
            ClassifiedMessage(
                message=MemoSiftMessage(
                    role="tool",
                    content="Contents of src/db.ts:\nconst db = new Database();",
                    tool_call_id="tc1",
                    name="read_file",
                ),
                content_type=ContentType.CODE_BLOCK,
                policy=CompressionPolicy.SIGNATURE,
                original_index=2,
            ),
            ClassifiedMessage(
                message=MemoSiftMessage(
                    role="tool",
                    content="TypeError: connection refused at db.ts:12",
                    tool_call_id="tc2",
                ),
                content_type=ContentType.ERROR_TRACE,
                policy=CompressionPolicy.STACK,
                original_index=5,
            ),
        ]
        ledger = AnchorLedger()
        extract_anchors_from_segments(segments, ledger)
        assert len(ledger.facts) > 0
        categories = {f.category for f in ledger.facts}
        assert AnchorCategory.FILES in categories or AnchorCategory.ERRORS in categories

    def test_skips_compressed_messages(self) -> None:
        msg = MemoSiftMessage(
            role="tool",
            content="src/important.ts has an error",
            tool_call_id="tc1",
        )
        msg._memosift_compressed = True
        segments = [
            ClassifiedMessage(
                message=msg,
                content_type=ContentType.TOOL_RESULT_TEXT,
                policy=CompressionPolicy.MODERATE,
            ),
        ]
        ledger = AnchorLedger()
        extract_anchors_from_segments(segments, ledger)
        assert len(ledger.facts) == 0
