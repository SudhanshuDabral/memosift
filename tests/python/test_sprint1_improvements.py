# Tests for Sprint 1 improvements (v0.3): structured ledger, JSON parsing,
# config tuning, anchor token protection, first-read vs re-read.
from __future__ import annotations

import json

import pytest

from memosift.config import MemoSiftConfig
from memosift.core.anchor_extractor import (
    _extract_decisions_from_text,
    _extract_facts_from_json_value,
    extract_anchors_from_message,
    extract_anchors_from_segments,
)
from memosift.core.engines.pruner import _is_protected_token, prune_tokens
from memosift.core.engines.verbatim import _extract_content_label, verbatim_compress
from memosift.core.types import (
    AnchorCategory,
    AnchorFact,
    AnchorLedger,
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    MemoSiftMessage,
    ToolCall,
    ToolCallFunction,
)


def _make_segment(
    content: str,
    role: str = "tool",
    policy: CompressionPolicy = CompressionPolicy.MODERATE,
    content_type: ContentType = ContentType.TOOL_RESULT_TEXT,
    original_index: int = 0,
    tool_call_id: str | None = "tc1",
    tool_calls: list[ToolCall] | None = None,
) -> ClassifiedMessage:
    """Helper to create a ClassifiedMessage for testing."""
    return ClassifiedMessage(
        message=MemoSiftMessage(
            role=role,
            content=content,
            tool_call_id=tool_call_id if role == "tool" else None,
            tool_calls=tool_calls,
        ),
        content_type=content_type,
        policy=policy,
        original_index=original_index,
    )


# ── Item 1.1: Structured Anchor Ledger (5 sections) ─────────────────────────


class TestStructuredAnchorLedger:
    """Tests for the restructured 5-section anchor ledger."""

    def test_new_categories_exist(self) -> None:
        """INTENT and ACTIVE_CONTEXT categories are available."""
        assert AnchorCategory.INTENT == "INTENT"
        assert AnchorCategory.ACTIVE_CONTEXT == "ACTIVE_CONTEXT"
        assert AnchorCategory.FILES == "FILES"
        assert AnchorCategory.DECISIONS == "DECISIONS"
        assert AnchorCategory.ERRORS == "ERRORS"

    def test_render_structured_sections(self) -> None:
        """Render produces the 5-section structured format."""
        ledger = AnchorLedger()
        ledger.add(AnchorFact(
            category=AnchorCategory.INTENT,
            content="User is debugging auth middleware",
            turn=1,
        ))
        ledger.add(AnchorFact(
            category=AnchorCategory.FILES,
            content="src/auth.ts — read at turn 2",
            turn=2,
        ))
        ledger.add(AnchorFact(
            category=AnchorCategory.DECISIONS,
            content="I'll use JWT over session cookies",
            turn=3,
        ))
        ledger.add(AnchorFact(
            category=AnchorCategory.ERRORS,
            content="TypeError: Cannot read 'userId'",
            turn=4,
        ))
        ledger.add(AnchorFact(
            category=AnchorCategory.ACTIVE_CONTEXT,
            content="Current task: add rate limiting",
            turn=5,
        ))

        rendered = ledger.render()
        assert "## SESSION INTENT" in rendered
        assert "## FILES TOUCHED" in rendered
        assert "## KEY DECISIONS" in rendered
        assert "## ERRORS ENCOUNTERED" in rendered
        assert "## ACTIVE CONTEXT" in rendered
        assert "- User is debugging auth middleware" in rendered
        assert "- src/auth.ts — read at turn 2" in rendered

    def test_render_primary_sections_first(self) -> None:
        """Primary sections appear before supplementary sections."""
        ledger = AnchorLedger()
        ledger.add(AnchorFact(
            category=AnchorCategory.IDENTIFIERS,
            content="Tool used: read_file",
            turn=1,
        ))
        ledger.add(AnchorFact(
            category=AnchorCategory.INTENT,
            content="Debugging issue",
            turn=1,
        ))
        rendered = ledger.render()
        intent_pos = rendered.find("## SESSION INTENT")
        identifiers_pos = rendered.find("## IDENTIFIERS")
        assert intent_pos < identifiers_pos

    def test_render_empty_ledger(self) -> None:
        ledger = AnchorLedger()
        rendered = ledger.render()
        assert "[SESSION MEMORY" in rendered
        assert "## SESSION INTENT" not in rendered

    def test_extract_intent_from_first_user_message(self) -> None:
        """INTENT is extracted from the first user message."""
        segments = [
            _make_segment(
                "Fix the authentication bug in the login endpoint",
                role="user",
                content_type=ContentType.USER_QUERY,
                policy=CompressionPolicy.PRESERVE,
                original_index=0,
                tool_call_id=None,
            ),
            _make_segment(
                "What about the database?",
                role="user",
                content_type=ContentType.USER_QUERY,
                policy=CompressionPolicy.PRESERVE,
                original_index=3,
                tool_call_id=None,
            ),
        ]
        ledger = AnchorLedger()
        extract_anchors_from_segments(segments, ledger)
        intent_facts = ledger.facts_by_category(AnchorCategory.INTENT)
        assert len(intent_facts) == 1
        assert "Fix the authentication bug" in intent_facts[0].content

    def test_extract_active_context(self) -> None:
        """ACTIVE_CONTEXT is extracted from last user + last assistant messages."""
        segments = [
            _make_segment(
                "First question",
                role="user",
                content_type=ContentType.USER_QUERY,
                policy=CompressionPolicy.PRESERVE,
                original_index=0,
                tool_call_id=None,
            ),
            _make_segment(
                "Here is my analysis of the code",
                role="assistant",
                content_type=ContentType.ASSISTANT_REASONING,
                policy=CompressionPolicy.AGGRESSIVE,
                original_index=1,
                tool_call_id=None,
            ),
            _make_segment(
                "Now add rate limiting to the login endpoint",
                role="user",
                content_type=ContentType.USER_QUERY,
                policy=CompressionPolicy.PRESERVE,
                original_index=4,
                tool_call_id=None,
            ),
            _make_segment(
                "I'll implement rate limiting using Redis",
                role="assistant",
                content_type=ContentType.ASSISTANT_REASONING,
                policy=CompressionPolicy.AGGRESSIVE,
                original_index=5,
                tool_call_id=None,
            ),
        ]
        ledger = AnchorLedger()
        extract_anchors_from_segments(segments, ledger)
        active_facts = ledger.facts_by_category(AnchorCategory.ACTIVE_CONTEXT)
        assert len(active_facts) >= 1
        contents = " ".join(f.content for f in active_facts)
        assert "rate limiting" in contents


class TestDecisionExtraction:
    """Tests for decision extraction with hedging filter."""

    def test_clear_decision_extracted(self) -> None:
        facts = _extract_decisions_from_text("I'll use JWT for authentication.", turn=3)
        assert len(facts) == 1
        assert facts[0].category == AnchorCategory.DECISIONS
        assert "JWT" in facts[0].content

    def test_hedged_suggestion_filtered(self) -> None:
        """Sentences with hedging language are NOT extracted as decisions."""
        facts = _extract_decisions_from_text(
            "Maybe I'll use Redis? Or perhaps Memcached could work.",
            turn=3,
        )
        assert len(facts) == 0

    def test_question_filtered(self) -> None:
        facts = _extract_decisions_from_text(
            "Should I'll use Redis for this?",
            turn=3,
        )
        assert len(facts) == 0

    def test_multiple_decisions(self) -> None:
        text = (
            "I'll use bcrypt for password hashing. "
            "Let's go with PostgreSQL for the database."
        )
        facts = _extract_decisions_from_text(text, turn=5)
        assert len(facts) == 2

    def test_decided_to_pattern(self) -> None:
        facts = _extract_decisions_from_text(
            "We decided to use TypeScript for the frontend.",
            turn=4,
        )
        assert len(facts) == 1


# ── Item 1.2: Parse Tool Call Arguments as JSON ─────────────────────────────


class TestJsonArgumentParsing:
    """Tests for recursive JSON argument fact extraction."""

    def test_extract_file_path_from_json(self) -> None:
        facts = _extract_facts_from_json_value(
            {"path": "src/auth.ts", "content": "new code"},
            key=None,
            turn=3,
            tool_name="edit_file",
        )
        file_facts = [f for f in facts if f.category == AnchorCategory.FILES]
        assert len(file_facts) >= 1
        assert any("src/auth.ts" in f.content for f in file_facts)
        assert any("modified" in f.content for f in file_facts)

    def test_extract_url_from_json(self) -> None:
        facts = _extract_facts_from_json_value(
            {"url": "https://api.example.com/v2/users"},
            key=None,
            turn=2,
            tool_name=None,
        )
        url_facts = [f for f in facts if "URL" in f.content]
        assert len(url_facts) >= 1

    def test_extract_uuid_from_json(self) -> None:
        facts = _extract_facts_from_json_value(
            {"id": "550e8400-e29b-41d4-a716-446655440000"},
            key=None,
            turn=1,
            tool_name=None,
        )
        uuid_facts = [f for f in facts if "UUID" in f.content]
        assert len(uuid_facts) == 1

    def test_extract_order_id_from_json(self) -> None:
        facts = _extract_facts_from_json_value(
            {"order": "ORD-20261234"},
            key=None,
            turn=1,
            tool_name=None,
        )
        id_facts = [f for f in facts if "ID:" in f.content]
        assert len(id_facts) == 1
        assert "ORD-20261234" in id_facts[0].content

    def test_recursive_extraction(self) -> None:
        """Nested JSON structures are walked recursively."""
        data = {
            "files": [
                {"path": "src/db.ts", "changes": 5},
                {"path": "src/api.ts", "changes": 3},
            ],
        }
        facts = _extract_facts_from_json_value(data, key=None, turn=2, tool_name=None)
        file_facts = [f for f in facts if f.category == AnchorCategory.FILES]
        assert len(file_facts) >= 2

    def test_tool_call_json_parsing_in_segments(self) -> None:
        """Tool call arguments are parsed as JSON in extract_anchors_from_segments."""
        tool_calls = [
            ToolCall(
                id="call_1",
                function=ToolCallFunction(
                    name="edit_file",
                    arguments=json.dumps({"path": "src/middleware/auth.ts", "content": "..."}),
                ),
            ),
        ]
        segments = [
            _make_segment(
                "Editing the file now",
                role="assistant",
                content_type=ContentType.ASSISTANT_REASONING,
                policy=CompressionPolicy.AGGRESSIVE,
                original_index=1,
                tool_call_id=None,
                tool_calls=tool_calls,
            ),
        ]
        ledger = AnchorLedger()
        extract_anchors_from_segments(segments, ledger)
        file_facts = ledger.facts_by_category(AnchorCategory.FILES)
        assert any("src/middleware/auth.ts" in f.content for f in file_facts)
        assert any("modified" in f.content for f in file_facts)

    def test_invalid_json_falls_back_to_regex(self) -> None:
        """Non-JSON arguments fall back to regex extraction."""
        tool_calls = [
            ToolCall(
                id="call_1",
                function=ToolCallFunction(
                    name="read_file",
                    arguments="not valid json src/config.py",
                ),
            ),
        ]
        segments = [
            _make_segment(
                "Reading file",
                role="assistant",
                content_type=ContentType.ASSISTANT_REASONING,
                policy=CompressionPolicy.AGGRESSIVE,
                original_index=1,
                tool_call_id=None,
                tool_calls=tool_calls,
            ),
        ]
        ledger = AnchorLedger()
        extract_anchors_from_segments(segments, ledger)
        file_facts = ledger.facts_by_category(AnchorCategory.FILES)
        assert any("src/config.py" in f.content for f in file_facts)


# ── Item 3.1: Tune Default Preset Config Values ─────────────────────────────


class TestTunedDefaults:
    """Tests for the updated default config values."""

    def test_default_recent_turns(self) -> None:
        config = MemoSiftConfig()
        assert config.recent_turns == 2

    def test_default_entropy_threshold(self) -> None:
        config = MemoSiftConfig()
        assert config.entropy_threshold == 1.8

    def test_default_token_prune_keep_ratio(self) -> None:
        config = MemoSiftConfig()
        assert config.token_prune_keep_ratio == 0.5

    def test_default_dedup_similarity_threshold(self) -> None:
        config = MemoSiftConfig()
        assert config.dedup_similarity_threshold == 0.80

    def test_coding_preset_stays_conservative(self) -> None:
        config = MemoSiftConfig.preset("coding")
        assert config.recent_turns == 3
        assert config.token_prune_keep_ratio == 0.7
        assert config.dedup_similarity_threshold == 0.90
        assert config.entropy_threshold == 2.5

    def test_general_preset_matches_new_defaults(self) -> None:
        config = MemoSiftConfig.preset("general")
        assert config.recent_turns == 2
        assert config.entropy_threshold == 1.8
        assert config.token_prune_keep_ratio == 0.5
        assert config.dedup_similarity_threshold == 0.80


# ── Item 1.3: Auto-Protect Anchored Tokens in Pruner ────────────────────────


class TestAnchorTokenProtection:
    """Tests for anchor ledger token auto-protection in the pruner."""

    def test_uuid_pattern_protected(self) -> None:
        assert _is_protected_token("550e8400-e29b-41d4-a716-446655440000") is True

    def test_order_id_pattern_protected(self) -> None:
        assert _is_protected_token("ORD-20261234") is True

    def test_long_alphanum_protected(self) -> None:
        assert _is_protected_token("abc123def456ghi") is True

    def test_ledger_tokens_protected_during_pruning(self) -> None:
        """Tokens from the anchor ledger survive pruning."""
        ledger = AnchorLedger()
        ledger.add(AnchorFact(
            category=AnchorCategory.FILES,
            content="src/middleware/auth.ts — modified at turn 3",
            turn=3,
        ))
        ledger.add(AnchorFact(
            category=AnchorCategory.ERRORS,
            content="TypeError: Cannot read properties of undefined",
            turn=4,
        ))

        # Create a segment with ledger-protected tokens mixed with filler.
        content = (
            "the the the is is is was was the the "
            "src/middleware/auth.ts has a TypeError in the handler"
        )
        seg = _make_segment(content, CompressionPolicy.MODERATE)
        # Add extra segments to build IDF corpus.
        filler = _make_segment("the the the is is is was was", CompressionPolicy.MODERATE)
        result = prune_tokens([seg, filler], MemoSiftConfig(token_prune_keep_ratio=0.3), ledger=ledger)
        output = result[0].content
        # Protected ledger tokens should survive.
        assert "src/middleware/auth.ts" in output or "auth.ts" in output


# ── Item 2.2: First-Read vs Re-Read Dedup ───────────────────────────────────


class TestFirstReadVsReRead:
    """Tests for first-read vs re-read detection in verbatim engine."""

    def test_first_read_preserved(self) -> None:
        """First read of content is preserved normally."""
        content = "x" * 300  # > 200 chars to trigger re-read check.
        seg = _make_segment(
            content,
            content_type=ContentType.TOOL_RESULT_TEXT,
            original_index=5,
        )
        seen: dict[str, int] = {}
        result = verbatim_compress([seg], MemoSiftConfig(), seen_content_hashes=seen)
        # Should not be collapsed (first read).
        assert "[Previously read:" not in result[0].content
        # Hash should be recorded.
        assert len(seen) == 1

    def test_re_read_collapsed(self) -> None:
        """Second read of identical content is collapsed to back-reference."""
        content = "Contents of src/auth.ts:\n" + "x" * 300
        seg1 = _make_segment(
            content,
            content_type=ContentType.TOOL_RESULT_TEXT,
            original_index=2,
        )
        seg2 = _make_segment(
            content,
            content_type=ContentType.TOOL_RESULT_TEXT,
            original_index=8,
        )
        seen: dict[str, int] = {}
        result = verbatim_compress([seg1, seg2], MemoSiftConfig(), seen_content_hashes=seen)
        # First should be preserved.
        assert "[Previously read:" not in result[0].content
        # Second should be collapsed.
        assert "[Previously read:" in result[1].content
        assert "see message 2" in result[1].content

    def test_short_content_not_tracked(self) -> None:
        """Short content (<= 200 chars) is not tracked for re-read."""
        content = "short content"
        seg1 = _make_segment(content, content_type=ContentType.TOOL_RESULT_TEXT)
        seg2 = _make_segment(content, content_type=ContentType.TOOL_RESULT_TEXT)
        seen: dict[str, int] = {}
        result = verbatim_compress([seg1, seg2], MemoSiftConfig(), seen_content_hashes=seen)
        # Neither should be collapsed (too short).
        assert "[Previously read:" not in result[0].content
        assert "[Previously read:" not in result[1].content

    def test_different_content_not_collapsed(self) -> None:
        """Different content is not collapsed."""
        seg1 = _make_segment(
            "a" * 300,
            content_type=ContentType.TOOL_RESULT_TEXT,
            original_index=1,
        )
        seg2 = _make_segment(
            "b" * 300,
            content_type=ContentType.TOOL_RESULT_TEXT,
            original_index=3,
        )
        seen: dict[str, int] = {}
        result = verbatim_compress([seg1, seg2], MemoSiftConfig(), seen_content_hashes=seen)
        assert "[Previously read:" not in result[0].content
        assert "[Previously read:" not in result[1].content

    def test_preserve_policy_not_tracked(self) -> None:
        """PRESERVE policy segments are not tracked for re-read."""
        content = "x" * 300
        seg = _make_segment(
            content,
            policy=CompressionPolicy.PRESERVE,
            content_type=ContentType.TOOL_RESULT_TEXT,
        )
        seen: dict[str, int] = {}
        result = verbatim_compress([seg], MemoSiftConfig(), seen_content_hashes=seen)
        assert len(seen) == 0

    def test_extract_content_label_file_path(self) -> None:
        label = _extract_content_label("Contents of /src/auth.ts:\ncode here\nmore code")
        assert "auth.ts" in label

    def test_extract_content_label_fallback(self) -> None:
        label = _extract_content_label("Some tool output without file paths\nmore lines")
        assert "Some tool output" in label

    def test_extract_content_label_long_first_line(self) -> None:
        label = _extract_content_label("a" * 100 + "\nmore")
        assert len(label) <= 60
        assert label.endswith("...")
