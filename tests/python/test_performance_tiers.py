# Tests for performance tiering — auto-detection, config overrides, and layer routing.
from __future__ import annotations

import pytest

from memosift.config import MemoSiftConfig
from memosift.core.pipeline import _resolve_tier, compress
from memosift.core.types import MemoSiftMessage, ToolCall, ToolCallFunction


# ── Tier Auto-Detection ──────────────────────────────────────────────────────


class TestTierAutoDetection:
    """Verify _resolve_tier returns correct tier for various message counts."""

    def test_small_session_returns_full(self) -> None:
        """30 messages -> full tier (all layers)."""
        config = MemoSiftConfig()
        assert _resolve_tier(config, 30) == "full"

    def test_medium_session_returns_standard(self) -> None:
        """100 messages -> standard tier."""
        config = MemoSiftConfig()
        assert _resolve_tier(config, 100) == "standard"

    def test_large_session_returns_fast(self) -> None:
        """200 messages -> fast tier."""
        config = MemoSiftConfig()
        assert _resolve_tier(config, 200) == "fast"

    def test_very_large_session_returns_ultra_fast(self) -> None:
        """400 messages -> ultra_fast tier."""
        config = MemoSiftConfig()
        assert _resolve_tier(config, 400) == "ultra_fast"

    def test_boundary_50_returns_full(self) -> None:
        """Exactly 50 messages (boundary) -> full tier."""
        config = MemoSiftConfig()
        assert _resolve_tier(config, 50) == "full"

    def test_boundary_51_returns_standard(self) -> None:
        """51 messages (just above boundary) -> standard tier."""
        config = MemoSiftConfig()
        assert _resolve_tier(config, 51) == "standard"

    def test_boundary_150_returns_standard(self) -> None:
        """150 messages -> standard tier."""
        config = MemoSiftConfig()
        assert _resolve_tier(config, 150) == "standard"

    def test_boundary_151_returns_fast(self) -> None:
        """151 messages -> fast tier."""
        config = MemoSiftConfig()
        assert _resolve_tier(config, 151) == "fast"

    def test_boundary_300_returns_fast(self) -> None:
        """300 messages -> fast tier."""
        config = MemoSiftConfig()
        assert _resolve_tier(config, 300) == "fast"

    def test_boundary_301_returns_ultra_fast(self) -> None:
        """301 messages -> ultra_fast tier."""
        config = MemoSiftConfig()
        assert _resolve_tier(config, 301) == "ultra_fast"


# ── Tier Config Override ─────────────────────────────────────────────────────


class TestTierConfigOverride:
    """Verify config.performance_tier overrides auto-detection."""

    def test_override_to_full(self) -> None:
        """Config override to 'full' overrides auto-detection regardless of count."""
        config = MemoSiftConfig(performance_tier="full")
        # Even with 500 messages (would auto-detect as ultra_fast), override wins.
        assert _resolve_tier(config, 500) == "full"

    def test_override_to_ultra_fast(self) -> None:
        """Config override to 'ultra_fast' overrides auto-detection even for small sessions."""
        config = MemoSiftConfig(performance_tier="ultra_fast")
        assert _resolve_tier(config, 10) == "ultra_fast"

    def test_override_to_standard(self) -> None:
        """Config override to 'standard' is respected."""
        config = MemoSiftConfig(performance_tier="standard")
        assert _resolve_tier(config, 400) == "standard"

    def test_override_to_fast(self) -> None:
        """Config override to 'fast' is respected."""
        config = MemoSiftConfig(performance_tier="fast")
        assert _resolve_tier(config, 5) == "fast"

    def test_none_falls_back_to_auto(self) -> None:
        """None performance_tier falls back to auto-detection."""
        config = MemoSiftConfig(performance_tier=None)
        assert _resolve_tier(config, 30) == "full"
        assert _resolve_tier(config, 200) == "fast"


# ── Invalid Tier Rejected ────────────────────────────────────────────────────


class TestInvalidTierRejected:
    """Verify MemoSiftConfig rejects invalid performance_tier values."""

    def test_invalid_tier_raises_value_error(self) -> None:
        """Providing an unrecognized tier string raises ValueError."""
        with pytest.raises(ValueError, match="performance_tier"):
            MemoSiftConfig(performance_tier="invalid")

    def test_typo_tier_raises_value_error(self) -> None:
        """Common typos are also rejected."""
        with pytest.raises(ValueError, match="performance_tier"):
            MemoSiftConfig(performance_tier="ultrafast")  # Missing underscore.

    def test_empty_string_raises_value_error(self) -> None:
        """Empty string is not a valid tier."""
        with pytest.raises(ValueError, match="performance_tier"):
            MemoSiftConfig(performance_tier="")


# ── Full Tier Runs All Layers ────────────────────────────────────────────────


def _build_conversation(n_turns: int) -> list[MemoSiftMessage]:
    """Build a conversation with n_turns user-assistant exchanges."""
    messages = [MemoSiftMessage(role="system", content="You are a helpful assistant.")]
    for i in range(n_turns):
        messages.append(MemoSiftMessage(role="user", content=f"Question {i}: What is {i * 7}?"))
        messages.append(
            MemoSiftMessage(
                role="assistant",
                content=f"The answer to question {i} is {i * 7}. Let me explain the calculation step by step.",
            )
        )
    # Add a final user message so the last assistant message is not "recent".
    messages.append(MemoSiftMessage(role="user", content="Thank you for all the answers."))
    return messages


class TestFullTierRunsAllLayers:
    """Compress with performance_tier='full' runs importance_scorer."""

    @pytest.mark.asyncio
    async def test_full_tier_includes_importance_scorer(self) -> None:
        """Full tier report.layers should include importance_scorer."""
        config = MemoSiftConfig(performance_tier="full")
        messages = _build_conversation(5)

        compressed, report = await compress(messages, config=config)

        layer_names = [layer.name for layer in report.layers]
        assert "importance_scorer" in layer_names
        assert report.performance_tier == "full"


# ── Ultra-Fast Tier Skips Layers ─────────────────────────────────────────────


class TestUltraFastSkipsLayers:
    """Compress with performance_tier='ultra_fast' and enough messages skips heavy layers."""

    @pytest.mark.asyncio
    async def test_ultra_fast_skips_importance_and_relevance(self) -> None:
        """Ultra-fast tier should NOT run importance_scorer or relevance_pruner."""
        config = MemoSiftConfig(performance_tier="ultra_fast")
        # Build enough messages to ensure the pipeline has work to do.
        messages = _build_conversation(30)

        compressed, report = await compress(messages, config=config)

        layer_names = [layer.name for layer in report.layers]
        assert "importance_scorer" not in layer_names
        assert "relevance_pruner" not in layer_names
        assert report.performance_tier == "ultra_fast"

    @pytest.mark.asyncio
    async def test_ultra_fast_still_runs_core_layers(self) -> None:
        """Ultra-fast tier still runs classifier, deduplicator, verbatim, structural."""
        config = MemoSiftConfig(performance_tier="ultra_fast")
        messages = _build_conversation(30)

        compressed, report = await compress(messages, config=config)

        layer_names = [layer.name for layer in report.layers]
        assert "classifier" in layer_names
        assert "deduplicator" in layer_names
        assert "engine_verbatim" in layer_names
        assert "engine_structural" in layer_names


# ── Pre-Bucketing Maintains Order ────────────────────────────────────────────


class TestPreBucketingMaintainsOrder:
    """Compressed messages must maintain their original relative order."""

    @pytest.mark.asyncio
    async def test_output_order_matches_original_indices(self) -> None:
        """Output messages should preserve the original message ordering."""
        messages = [
            MemoSiftMessage(role="system", content="You are a helpful assistant."),
            MemoSiftMessage(role="user", content="First question about Python."),
            MemoSiftMessage(
                role="assistant",
                content="Python is a great language. Here is a detailed explanation of its features and usage patterns.",
            ),
            MemoSiftMessage(role="user", content="Second question about JavaScript."),
            MemoSiftMessage(
                role="assistant",
                content="JavaScript runs in the browser. Here is a comprehensive overview.",
            ),
            MemoSiftMessage(role="user", content="Third question about Rust."),
            MemoSiftMessage(
                role="assistant",
                content="Rust is a systems programming language. It provides memory safety without garbage collection.",
            ),
            MemoSiftMessage(role="user", content="What was my first question?"),
        ]
        config = MemoSiftConfig(performance_tier="full")
        compressed, report = await compress(messages, config=config)

        # System message must be first.
        assert compressed[0].role == "system"

        # The role sequence must maintain the original alternation pattern.
        # user/assistant messages must not be reordered relative to each other.
        roles = [m.role for m in compressed]
        # System always first.
        assert roles[0] == "system"

        # User messages should retain their relative order.
        # Content may be compressed, so check distinguishing substrings.
        user_contents = [m.content for m in compressed if m.role == "user"]
        keywords = ["Python", "JavaScript", "Rust", "first question"]
        found_indices = []
        for kw in keywords:
            for i, content in enumerate(user_contents):
                if kw.lower() in content.lower():
                    found_indices.append(i)
                    break

        # Each keyword that was found should be in strictly ascending index order.
        assert found_indices == sorted(found_indices), (
            f"User messages are out of order: keywords found at indices {found_indices}"
        )


# ── Tool Integrity Across All Tiers ──────────────────────────────────────────


def _build_tool_conversation() -> list[MemoSiftMessage]:
    """Build a conversation with tool calls for integrity testing."""
    return [
        MemoSiftMessage(role="system", content="You are a coding assistant."),
        MemoSiftMessage(role="user", content="Read the auth module and the config file."),
        MemoSiftMessage(
            role="assistant",
            content="I will read both files for you.",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    function=ToolCallFunction(
                        name="read_file",
                        arguments='{"path": "src/auth.ts"}',
                    ),
                ),
                ToolCall(
                    id="tc2",
                    function=ToolCallFunction(
                        name="read_file",
                        arguments='{"path": "config.yaml"}',
                    ),
                ),
            ],
        ),
        MemoSiftMessage(
            role="tool",
            content="export class AuthService {\n  async login(user: string, pass: string) {\n    return true;\n  }\n}",
            tool_call_id="tc1",
            name="read_file",
        ),
        MemoSiftMessage(
            role="tool",
            content="database:\n  host: localhost\n  port: 5432",
            tool_call_id="tc2",
            name="read_file",
        ),
        MemoSiftMessage(
            role="assistant",
            content="I have read both files. The auth module has a login method and the config uses PostgreSQL on port 5432.",
        ),
        MemoSiftMessage(role="user", content="Thanks, now explain the auth flow."),
    ]


class TestToolIntegrityAllTiers:
    """Tool call integrity must hold for every performance tier."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tier", ["full", "standard", "fast", "ultra_fast"])
    async def test_tool_call_integrity(self, tier: str) -> None:
        """For each tier, compress messages with tool calls and verify integrity."""
        config = MemoSiftConfig(performance_tier=tier)
        messages = _build_tool_conversation()

        compressed, report = await compress(messages, config=config)

        # Collect all tool_call IDs and tool_result IDs from compressed output.
        call_ids: set[str] = set()
        result_ids: set[str] = set()
        for msg in compressed:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    call_ids.add(tc.id)
            if msg.tool_call_id:
                result_ids.add(msg.tool_call_id)

        # Every tool_call must have a matching result and vice versa.
        assert call_ids == result_ids, (
            f"Tool integrity violated in tier '{tier}': "
            f"calls={call_ids}, results={result_ids}"
        )
