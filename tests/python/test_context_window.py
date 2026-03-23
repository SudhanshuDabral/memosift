# Tests for Layer 0: Context-aware adaptive compression.
from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memosift.config import MemoSiftConfig
from memosift.core.context_window import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_OUTPUT_RESERVE,
    MODEL_CONTEXT_WINDOWS,
    MODEL_OUTPUT_LIMITS,
    AdaptiveOverrides,
    ContextWindowState,
    Pressure,
    compute_adaptive_thresholds,
    estimate_tokens_heuristic,
    lookup_context_window,
    lookup_output_limit,
    resolve_context_window,
)


# ── Pressure Enum ─────────────────────────────────────────────────────────


class TestPressure:
    def test_all_values_exist(self):
        assert Pressure.NONE == "NONE"
        assert Pressure.LOW == "LOW"
        assert Pressure.MEDIUM == "MEDIUM"
        assert Pressure.HIGH == "HIGH"
        assert Pressure.CRITICAL == "CRITICAL"

    def test_five_levels(self):
        assert len(Pressure) == 5


# ── ContextWindowState ────────────────────────────────────────────────────


class TestContextWindowState:
    def test_defaults(self):
        state = ContextWindowState()
        assert state.model is None
        assert state.context_window_tokens == DEFAULT_CONTEXT_WINDOW
        assert state.current_usage_tokens == 0
        assert state.output_reserve_tokens == DEFAULT_OUTPUT_RESERVE

    def test_effective_capacity(self):
        state = ContextWindowState(context_window_tokens=200_000, output_reserve_tokens=8_192)
        assert state.effective_capacity == 200_000 - 8_192

    def test_effective_capacity_clamped_to_zero(self):
        state = ContextWindowState(context_window_tokens=1_000, output_reserve_tokens=5_000)
        assert state.effective_capacity == 0

    def test_available_tokens(self):
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=150_000,
            output_reserve_tokens=8_192,
        )
        assert state.available_tokens == 200_000 - 8_192 - 150_000

    def test_available_tokens_clamped(self):
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=250_000,
            output_reserve_tokens=8_192,
        )
        assert state.available_tokens == 0

    def test_usage_ratio_empty(self):
        state = ContextWindowState(context_window_tokens=200_000, current_usage_tokens=0)
        assert state.usage_ratio == pytest.approx(0.0)

    def test_usage_ratio_half(self):
        cap = 200_000 - DEFAULT_OUTPUT_RESERVE
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=cap // 2,
        )
        assert state.usage_ratio == pytest.approx(0.5, abs=0.01)

    def test_usage_ratio_full(self):
        cap = 200_000 - DEFAULT_OUTPUT_RESERVE
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=cap,
        )
        assert state.usage_ratio == pytest.approx(1.0)

    def test_usage_ratio_overflow_clamped(self):
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=300_000,
        )
        assert state.usage_ratio == 1.0

    def test_remaining_ratio_complement(self):
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=100_000,
        )
        assert state.usage_ratio + state.remaining_ratio == pytest.approx(1.0)

    def test_pressure_none_mostly_empty(self):
        # >60% remaining → NONE
        cap = 200_000 - DEFAULT_OUTPUT_RESERVE
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=int(cap * 0.30),  # 30% used, 70% remaining
        )
        assert state.pressure == Pressure.NONE

    def test_pressure_low(self):
        # 40-60% remaining → LOW
        cap = 200_000 - DEFAULT_OUTPUT_RESERVE
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=int(cap * 0.50),  # 50% used, 50% remaining
        )
        assert state.pressure == Pressure.LOW

    def test_pressure_medium(self):
        # 25-40% remaining → MEDIUM
        cap = 200_000 - DEFAULT_OUTPUT_RESERVE
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=int(cap * 0.67),  # 67% used, 33% remaining
        )
        assert state.pressure == Pressure.MEDIUM

    def test_pressure_high(self):
        # 10-25% remaining → HIGH
        cap = 200_000 - DEFAULT_OUTPUT_RESERVE
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=int(cap * 0.82),  # 82% used, 18% remaining
        )
        assert state.pressure == Pressure.HIGH

    def test_pressure_critical(self):
        # <10% remaining → CRITICAL
        cap = 200_000 - DEFAULT_OUTPUT_RESERVE
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=int(cap * 0.95),  # 95% used, 5% remaining
        )
        assert state.pressure == Pressure.CRITICAL

    def test_pressure_boundary_60_percent(self):
        # Exactly at the boundary: remaining_ratio must be strictly > 0.60 for NONE.
        # Use a context window where 40% usage is exact.
        state = ContextWindowState(
            context_window_tokens=100_000,
            current_usage_tokens=40_000,
            output_reserve_tokens=0,  # Eliminate reserve for exact math
        )
        # 40% used, 60% remaining, remaining_ratio = 0.60 exactly, NOT > 0.60
        assert state.remaining_ratio == pytest.approx(0.60)
        assert state.pressure == Pressure.LOW

    def test_frozen(self):
        state = ContextWindowState()
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.current_usage_tokens = 100  # type: ignore[misc]

    def test_from_model_known(self):
        state = ContextWindowState.from_model("claude-sonnet-4-6", current_usage_tokens=50_000)
        assert state.context_window_tokens == 1_000_000
        assert state.current_usage_tokens == 50_000
        assert state.model == "claude-sonnet-4-6"
        assert state.output_reserve_tokens == 64_000  # From MODEL_OUTPUT_LIMITS

    def test_from_model_unknown_falls_back(self):
        state = ContextWindowState.from_model("some-unknown-model-v2")
        assert state.context_window_tokens == DEFAULT_CONTEXT_WINDOW
        assert state.output_reserve_tokens == DEFAULT_OUTPUT_RESERVE

    def test_from_model_output_reserve_override(self):
        state = ContextWindowState.from_model(
            "gpt-4o", output_reserve_tokens=4_096
        )
        assert state.output_reserve_tokens == 4_096

    def test_1m_model_low_usage_no_pressure(self):
        """A 1M-token model with 50K usage should have NONE pressure."""
        state = ContextWindowState.from_model("claude-opus-4-6", current_usage_tokens=50_000)
        assert state.pressure == Pressure.NONE

    def test_200k_model_high_usage_critical(self):
        """A 200K-token model with 180K usage should have CRITICAL pressure."""
        state = ContextWindowState.from_model("claude-haiku-4-5", current_usage_tokens=180_000)
        assert state.pressure == Pressure.CRITICAL

    def test_model_switch_pressure_change(self):
        """Switching from 1M to 200K model at same usage changes pressure."""
        usage = 150_000
        state_1m = ContextWindowState.from_model("claude-sonnet-4-6", current_usage_tokens=usage)
        state_200k = ContextWindowState.from_model("claude-haiku-4-5", current_usage_tokens=usage)
        # 1M model: 150K/~936K effective = ~16% used → >60% remaining → NONE
        assert state_1m.pressure == Pressure.NONE
        # 200K model: 150K/~136K effective = >100% used → CRITICAL
        assert state_200k.pressure == Pressure.CRITICAL


# ── Model Registry Lookup ─────────────────────────────────────────────────


class TestModelRegistryLookup:
    def test_exact_match(self):
        assert lookup_context_window("gpt-4o") == 128_000

    def test_prefix_match(self):
        assert lookup_context_window("gpt-4o-2024-08-06") == 128_000

    def test_longer_prefix_wins(self):
        # "gpt-4o-mini" should match "gpt-4o-mini" (128K) not "gpt-4o" (128K)
        # Both are same in this case, but test the mechanism
        assert lookup_context_window("gpt-4o-mini-2024-07-18") == 128_000
        # "gpt-4.1-mini" should match "gpt-4.1-mini" not "gpt-4.1"
        assert lookup_context_window("gpt-4.1-mini-2025-04-14") == 1_047_576

    def test_case_insensitive(self):
        assert lookup_context_window("GPT-4o") == 128_000
        assert lookup_context_window("Claude-Sonnet-4-6") == 1_000_000

    def test_unknown_model(self):
        assert lookup_context_window("totally-unknown-model") is None

    def test_anthropic_models(self):
        assert lookup_context_window("claude-opus-4-6") == 1_000_000
        assert lookup_context_window("claude-haiku-4-5") == 200_000
        assert lookup_context_window("claude-sonnet-4-5") == 200_000

    def test_google_models(self):
        assert lookup_context_window("gemini-2.5-pro") == 1_048_576
        assert lookup_context_window("gemini-2.5-flash") == 1_048_576

    def test_output_limit_lookup(self):
        assert lookup_output_limit("claude-opus-4-6") == 128_000
        assert lookup_output_limit("gpt-4o") == 16_384
        assert lookup_output_limit("unknown-model") is None


# ── Adaptive Threshold Computation ────────────────────────────────────────


class TestComputeAdaptiveThresholds:
    def test_none_pressure_skips_compression(self):
        state = ContextWindowState.from_model("claude-opus-4-6", current_usage_tokens=50_000)
        config = MemoSiftConfig()
        result = compute_adaptive_thresholds(state, config, total_user_turns=10)
        assert result.pressure == Pressure.NONE
        assert result.skip_compression is True
        assert result.engine_gates == frozenset()

    def test_low_pressure_dedup_and_verbatim_only(self):
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=90_000,
            output_reserve_tokens=8_192,
        )
        assert state.pressure == Pressure.LOW
        config = MemoSiftConfig()
        result = compute_adaptive_thresholds(state, config, total_user_turns=20)
        assert result.skip_compression is False
        assert "dedup" in result.engine_gates
        assert "verbatim" in result.engine_gates
        assert "pruner" not in result.engine_gates
        assert "summarizer" not in result.engine_gates

    def test_medium_pressure_engines(self):
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=130_000,
            output_reserve_tokens=8_192,
        )
        assert state.pressure == Pressure.MEDIUM
        config = MemoSiftConfig()
        result = compute_adaptive_thresholds(state, config, total_user_turns=20)
        assert "pruner" in result.engine_gates
        assert "structural" in result.engine_gates
        assert "discourse" in result.engine_gates
        assert "importance" not in result.engine_gates

    def test_high_pressure_all_deterministic(self):
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=160_000,
            output_reserve_tokens=8_192,
        )
        assert state.pressure == Pressure.HIGH
        config = MemoSiftConfig()
        result = compute_adaptive_thresholds(state, config, total_user_turns=20)
        assert "importance" in result.engine_gates
        assert "relevance_pruner" in result.engine_gates
        assert result.enable_observation_masking is True
        assert result.effective_config.performance_tier == "full"

    def test_critical_pressure_includes_summarizer(self):
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=185_000,
            output_reserve_tokens=8_192,
        )
        assert state.pressure == Pressure.CRITICAL
        config = MemoSiftConfig()
        result = compute_adaptive_thresholds(state, config, total_user_turns=20)
        assert "summarizer" in result.engine_gates
        assert result.enable_observation_masking is True

    def test_recent_turns_percentage_based(self):
        """At HIGH pressure (8% ratio), 100 user turns → 8 recent turns."""
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=160_000,
            output_reserve_tokens=8_192,
        )
        config = MemoSiftConfig(recent_turns=10)
        result = compute_adaptive_thresholds(state, config, total_user_turns=100)
        assert result.effective_config.recent_turns == 8

    def test_recent_turns_capped_at_config(self):
        """Adaptive recent turns never exceeds config.recent_turns."""
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=90_000,  # LOW pressure → 20%
            output_reserve_tokens=8_192,
        )
        config = MemoSiftConfig(recent_turns=2)
        result = compute_adaptive_thresholds(state, config, total_user_turns=100)
        # 20% of 100 = 20, but capped at config.recent_turns = 2
        assert result.effective_config.recent_turns == 2

    def test_recent_turns_minimum_one(self):
        """At CRITICAL pressure with few turns, still protect at least 1."""
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=185_000,
            output_reserve_tokens=8_192,
        )
        config = MemoSiftConfig(recent_turns=5)
        result = compute_adaptive_thresholds(state, config, total_user_turns=3)
        # 5% of 3 = 0.15, rounds to 0, but minimum is 1
        assert result.effective_config.recent_turns >= 1

    def test_auto_budget_from_available(self):
        """Budget is derived from available tokens at the pressure's ratio."""
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=160_000,
            output_reserve_tokens=8_192,
        )
        assert state.pressure == Pressure.HIGH  # ratio = 0.50
        config = MemoSiftConfig()
        result = compute_adaptive_thresholds(state, config, total_user_turns=20)
        expected = int(state.available_tokens * 0.50)
        assert result.effective_config.token_budget == expected

    def test_explicit_budget_respected(self):
        """User-set token_budget is respected (tighter of the two wins)."""
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=100_000,
            output_reserve_tokens=8_192,
        )
        config = MemoSiftConfig(token_budget=5_000)
        result = compute_adaptive_thresholds(state, config, total_user_turns=20)
        # User set 5K, adaptive might compute more — user's tighter value wins
        assert result.effective_config.token_budget <= 5_000

    def test_prune_ratio_scaled(self):
        """Prune ratio is multiplied by pressure multiplier."""
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=185_000,
            output_reserve_tokens=8_192,
        )
        assert state.pressure == Pressure.CRITICAL
        config = MemoSiftConfig(token_prune_keep_ratio=0.5)
        result = compute_adaptive_thresholds(state, config, total_user_turns=20)
        # CRITICAL multiplier = 0.50, so 0.5 * 0.50 = 0.25
        assert result.effective_config.token_prune_keep_ratio == pytest.approx(0.25)

    def test_prune_ratio_floor(self):
        """Prune ratio never goes below 0.1."""
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=185_000,
            output_reserve_tokens=8_192,
        )
        config = MemoSiftConfig(token_prune_keep_ratio=0.1)
        result = compute_adaptive_thresholds(state, config, total_user_turns=20)
        assert result.effective_config.token_prune_keep_ratio >= 0.1

    def test_original_config_unchanged(self):
        """compute_adaptive_thresholds never mutates the input config."""
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=160_000,
            output_reserve_tokens=8_192,
        )
        config = MemoSiftConfig(recent_turns=5, token_prune_keep_ratio=0.5)
        original_recent = config.recent_turns
        original_prune = config.token_prune_keep_ratio
        compute_adaptive_thresholds(state, config, total_user_turns=50)
        assert config.recent_turns == original_recent
        assert config.token_prune_keep_ratio == original_prune

    def test_zero_user_turns_uses_config_recent(self):
        """When total_user_turns=0, falls back to config.recent_turns."""
        state = ContextWindowState(
            context_window_tokens=200_000,
            current_usage_tokens=130_000,
            output_reserve_tokens=8_192,
        )
        config = MemoSiftConfig(recent_turns=3)
        result = compute_adaptive_thresholds(state, config, total_user_turns=0)
        assert result.effective_config.recent_turns == 3


# ── resolve_context_window ────────────────────────────────────────────────


class TestResolveContextWindow:
    def test_explicit_wins(self):
        explicit = ContextWindowState(model="test", context_window_tokens=500_000)
        result = resolve_context_window(explicit, "claude-sonnet-4-6", 10_000)
        assert result is explicit

    def test_model_name_resolution(self):
        result = resolve_context_window(None, "gpt-4o", 50_000)
        assert result is not None
        assert result.context_window_tokens == 128_000
        assert result.current_usage_tokens == 50_000

    def test_nothing_returns_none(self):
        result = resolve_context_window(None, None, 0)
        assert result is None

    def test_unknown_model_uses_default(self):
        result = resolve_context_window(None, "unknown-model", 10_000)
        assert result is not None
        assert result.context_window_tokens == DEFAULT_CONTEXT_WINDOW


# ── estimate_tokens_heuristic ─────────────────────────────────────────────


class TestEstimateTokensHeuristic:
    def test_empty(self):
        assert estimate_tokens_heuristic([]) == 0
        assert estimate_tokens_heuristic([""]) == 0

    def test_approximation(self):
        # ~3.5 chars/token, so 350 chars ≈ 100 tokens
        text = "a" * 350
        result = estimate_tokens_heuristic([text])
        assert 90 <= result <= 110

    def test_multiple_messages(self):
        msgs = ["hello world", "how are you doing today"]
        result = estimate_tokens_heuristic(msgs)
        assert result > 0


# ── Hypothesis Property Tests ─────────────────────────────────────────────


class TestProperties:
    @given(
        window=st.integers(min_value=1000, max_value=2_000_000),
        usage=st.integers(min_value=0, max_value=2_000_000),
        reserve=st.integers(min_value=0, max_value=500_000),
    )
    @settings(max_examples=200)
    def test_pressure_is_valid_enum(self, window, usage, reserve):
        state = ContextWindowState(
            context_window_tokens=window,
            current_usage_tokens=usage,
            output_reserve_tokens=reserve,
        )
        assert state.pressure in list(Pressure)

    @given(
        window=st.integers(min_value=1000, max_value=2_000_000),
        usage=st.integers(min_value=0, max_value=2_000_000),
        reserve=st.integers(min_value=0, max_value=500_000),
    )
    @settings(max_examples=200)
    def test_remaining_plus_usage_is_one(self, window, usage, reserve):
        state = ContextWindowState(
            context_window_tokens=window,
            current_usage_tokens=usage,
            output_reserve_tokens=reserve,
        )
        assert state.usage_ratio + state.remaining_ratio == pytest.approx(1.0)

    @given(
        usage_pct=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_higher_usage_means_higher_or_equal_pressure(self, usage_pct):
        window = 200_000
        reserve = 8_192
        cap = window - reserve
        usage_a = int(cap * usage_pct)
        usage_b = int(cap * min(1.0, usage_pct + 0.15))
        state_a = ContextWindowState(
            context_window_tokens=window,
            current_usage_tokens=usage_a,
            output_reserve_tokens=reserve,
        )
        state_b = ContextWindowState(
            context_window_tokens=window,
            current_usage_tokens=usage_b,
            output_reserve_tokens=reserve,
        )
        pressure_order = [Pressure.NONE, Pressure.LOW, Pressure.MEDIUM, Pressure.HIGH, Pressure.CRITICAL]
        assert pressure_order.index(state_b.pressure) >= pressure_order.index(state_a.pressure)

    @given(
        window=st.integers(min_value=10_000, max_value=2_000_000),
        usage=st.integers(min_value=0, max_value=2_000_000),
        turns=st.integers(min_value=1, max_value=500),
    )
    @settings(max_examples=100)
    def test_adaptive_thresholds_return_valid_config(self, window, usage, turns):
        state = ContextWindowState(context_window_tokens=window, current_usage_tokens=usage)
        config = MemoSiftConfig()
        result = compute_adaptive_thresholds(state, config, total_user_turns=turns)
        # Should not raise; effective_config should be valid
        assert result.effective_config.recent_turns >= 0
        if result.effective_config.token_budget is not None:
            assert result.effective_config.token_budget >= 100
        assert 0.1 <= result.effective_config.token_prune_keep_ratio <= 1.0
