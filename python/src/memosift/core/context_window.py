# Layer 0: Context-aware adaptive compression — dynamic thresholds based on model context window.
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig

logger = logging.getLogger("memosift")


class Pressure(StrEnum):
    """Context window pressure level — drives adaptive compression thresholds.

    Derived from the remaining ratio of the model's context window.
    Higher pressure triggers progressively more aggressive compression.
    """

    NONE = "NONE"  # >60% remaining — no compression needed
    LOW = "LOW"  # 40-60% remaining — light (dedup + verbatim only)
    MEDIUM = "MEDIUM"  # 25-40% remaining — standard pipeline
    HIGH = "HIGH"  # 10-25% remaining — aggressive
    CRITICAL = "CRITICAL"  # <10% remaining — maximum compression


# ── Model context window registry ─────────────────────────────────────────
# Maps model name prefixes to context window sizes (input tokens).
# Lookup does longest-prefix matching so "gpt-4o-2024-08-06" matches "gpt-4o".
# Models not in this registry require explicit ContextWindowState.

MODEL_CONTEXT_WINDOWS: MappingProxyType[str, int] = MappingProxyType(
    {
        # OpenAI
        "gpt-4o-mini": 128_000,
        "gpt-4o": 128_000,
        "gpt-4.1-nano": 1_047_576,
        "gpt-4.1-mini": 1_047_576,
        "gpt-4.1": 1_047_576,
        "o3-mini": 200_000,
        "o3": 200_000,
        "o4-mini": 200_000,
        # Anthropic
        "claude-opus-4-6": 1_000_000,
        "claude-sonnet-4-6": 1_000_000,
        "claude-opus-4-5": 200_000,
        "claude-sonnet-4-5": 200_000,
        "claude-sonnet-4": 200_000,
        "claude-haiku-4-5": 200_000,
        # Google
        "gemini-2.5-pro": 1_048_576,
        "gemini-2.5-flash-lite": 1_048_576,
        "gemini-2.5-flash": 1_048_576,
    }
)

# Max output tokens per model (used for output_reserve_tokens estimation).
MODEL_OUTPUT_LIMITS: MappingProxyType[str, int] = MappingProxyType(
    {
        "gpt-4o-mini": 16_384,
        "gpt-4o": 16_384,
        "gpt-4.1-nano": 32_768,
        "gpt-4.1-mini": 32_768,
        "gpt-4.1": 32_768,
        "o3-mini": 100_000,
        "o3": 100_000,
        "o4-mini": 100_000,
        "claude-opus-4-6": 128_000,
        "claude-sonnet-4-6": 64_000,
        "claude-opus-4-5": 64_000,
        "claude-sonnet-4-5": 64_000,
        "claude-sonnet-4": 64_000,
        "claude-haiku-4-5": 64_000,
        "gemini-2.5-pro": 65_535,
        "gemini-2.5-flash-lite": 65_535,
        "gemini-2.5-flash": 65_535,
    }
)

# Default context window assumed when model is unknown.
DEFAULT_CONTEXT_WINDOW = 200_000

# Default output reserve when model is unknown.
DEFAULT_OUTPUT_RESERVE = 8_192


def lookup_context_window(model: str) -> int | None:
    """Look up context window size for a model by longest-prefix match.

    Args:
        model: Model identifier (e.g., "gpt-4o-2024-08-06", "claude-sonnet-4-6").

    Returns:
        Context window in tokens, or None if no matching prefix found.
    """
    model_lower = model.lower()
    # Sort by key length descending so longer (more specific) prefixes match first.
    for prefix in sorted(MODEL_CONTEXT_WINDOWS.keys(), key=len, reverse=True):
        if model_lower.startswith(prefix.lower()):
            return MODEL_CONTEXT_WINDOWS[prefix]
    return None


def lookup_output_limit(model: str) -> int | None:
    """Look up max output tokens for a model by longest-prefix match."""
    model_lower = model.lower()
    for prefix in sorted(MODEL_OUTPUT_LIMITS.keys(), key=len, reverse=True):
        if model_lower.startswith(prefix.lower()):
            return MODEL_OUTPUT_LIMITS[prefix]
    return None


@dataclass(frozen=True)
class ContextWindowState:
    """Snapshot of model context window capacity and current utilization.

    This immutable object drives the adaptive compression system. Create a new
    instance when the context changes (e.g., after each turn, or on model switch).

    The system uses this to compute ``Pressure`` and derive dynamic thresholds
    for recent-turn protection, auto-budget, pruning aggressiveness, and engine
    selection — replacing fixed thresholds that don't adapt to the model or
    conversation state.
    """

    model: str | None = None
    """Model identifier (e.g., "claude-sonnet-4-6", "gpt-4o")."""

    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW
    """Total context window size in tokens."""

    current_usage_tokens: int = 0
    """Tokens currently consumed in the context window (input so far)."""

    output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE
    """Tokens reserved for model output (not available for input context)."""

    @property
    def effective_capacity(self) -> int:
        """Usable capacity: total window minus output reserve."""
        return max(0, self.context_window_tokens - self.output_reserve_tokens)

    @property
    def available_tokens(self) -> int:
        """Tokens remaining for input context."""
        return max(0, self.effective_capacity - self.current_usage_tokens)

    @property
    def usage_ratio(self) -> float:
        """Fraction of effective capacity consumed (0.0–1.0)."""
        cap = self.effective_capacity
        if cap <= 0:
            return 1.0
        return min(1.0, max(0.0, self.current_usage_tokens / cap))

    @property
    def remaining_ratio(self) -> float:
        """Fraction of effective capacity still available (0.0–1.0)."""
        return 1.0 - self.usage_ratio

    @property
    def pressure(self) -> Pressure:
        """Context window pressure level derived from remaining capacity."""
        r = self.remaining_ratio
        if r > 0.60:
            return Pressure.NONE
        if r > 0.40:
            return Pressure.LOW
        if r > 0.25:
            return Pressure.MEDIUM
        if r > 0.10:
            return Pressure.HIGH
        return Pressure.CRITICAL

    @classmethod
    def from_model(
        cls,
        model: str,
        current_usage_tokens: int = 0,
        *,
        output_reserve_tokens: int | None = None,
    ) -> ContextWindowState:
        """Create from a model name, using the registry for context window size.

        Falls back to DEFAULT_CONTEXT_WINDOW if the model is unknown.

        Args:
            model: Model identifier.
            current_usage_tokens: Tokens consumed so far.
            output_reserve_tokens: Override output reserve. When None, uses
                the model's known output limit or DEFAULT_OUTPUT_RESERVE.
        """
        window = lookup_context_window(model) or DEFAULT_CONTEXT_WINDOW
        reserve = output_reserve_tokens
        if reserve is None:
            reserve = lookup_output_limit(model) or DEFAULT_OUTPUT_RESERVE
        return cls(
            model=model,
            context_window_tokens=window,
            current_usage_tokens=current_usage_tokens,
            output_reserve_tokens=reserve,
        )


# ── Adaptive threshold computation ────────────────────────────────────────
# Maps Pressure → percentage of user turns to protect as "recent".
_RECENT_TURN_RATIOS: dict[Pressure, float] = {
    Pressure.NONE: 0.30,  # Generous — room is cheap
    Pressure.LOW: 0.20,
    Pressure.MEDIUM: 0.12,
    Pressure.HIGH: 0.08,
    Pressure.CRITICAL: 0.05,
}

# Maps Pressure → fraction of available tokens to use as auto-budget.
_BUDGET_RATIOS: dict[Pressure, float | None] = {
    Pressure.NONE: None,  # No budget needed
    Pressure.LOW: 0.90,  # Gentle trim
    Pressure.MEDIUM: 0.70,
    Pressure.HIGH: 0.50,
    Pressure.CRITICAL: 0.30,
}

# Maps Pressure → multiplier applied to token_prune_keep_ratio.
_PRUNE_MULTIPLIERS: dict[Pressure, float] = {
    Pressure.NONE: 1.0,
    Pressure.LOW: 1.0,
    Pressure.MEDIUM: 0.85,
    Pressure.HIGH: 0.70,
    Pressure.CRITICAL: 0.50,
}

# Maps Pressure → multiplier applied to entropy_threshold.
_ENTROPY_MULTIPLIERS: dict[Pressure, float] = {
    Pressure.NONE: 1.0,
    Pressure.LOW: 1.0,
    Pressure.MEDIUM: 0.90,
    Pressure.HIGH: 0.75,
    Pressure.CRITICAL: 0.55,
}


@dataclass(frozen=True)
class AdaptiveOverrides:
    """Computed adaptive thresholds from Layer 0 context assessment.

    Produced by ``compute_adaptive_thresholds()``. The pipeline uses these
    to gate engine selection and override fixed config values.
    """

    pressure: Pressure
    """Current context window pressure level."""

    effective_config: MemoSiftConfig
    """Config with adaptive thresholds applied (new instance, original unchanged)."""

    skip_compression: bool
    """True when pressure is NONE — skip the entire pipeline."""

    enable_observation_masking: bool
    """True when pressure is HIGH or CRITICAL — enable observation masking regardless
    of tool result count."""

    engine_gates: frozenset[str]
    """Which engine groups are active at this pressure level.

    Possible values: "dedup", "verbatim", "pruner", "structural", "importance",
    "relevance_pruner", "discourse", "summarizer".
    """

    context_window: ContextWindowState
    """The context window state used to compute these overrides."""

    overrides: dict[str, tuple[object, object]] = dataclasses.field(default_factory=dict)
    """Fields that were overridden by adaptive compression.

    Maps field name to ``(original_value, effective_value)`` for every config
    field that was changed. Empty when ``skip_compression`` is True.
    """


# Engine sets per pressure level.
_ENGINES_NONE: frozenset[str] = frozenset()
_ENGINES_LOW: frozenset[str] = frozenset({"dedup", "verbatim"})
_ENGINES_MEDIUM: frozenset[str] = frozenset(
    {"dedup", "verbatim", "pruner", "structural", "discourse"}
)
_ENGINES_HIGH: frozenset[str] = frozenset(
    {"dedup", "verbatim", "pruner", "structural", "importance", "relevance_pruner", "discourse"}
)
_ENGINES_CRITICAL: frozenset[str] = frozenset(
    {
        "dedup",
        "verbatim",
        "pruner",
        "structural",
        "importance",
        "relevance_pruner",
        "discourse",
        "summarizer",
    }
)

_PRESSURE_ENGINES: dict[Pressure, frozenset[str]] = {
    Pressure.NONE: _ENGINES_NONE,
    Pressure.LOW: _ENGINES_LOW,
    Pressure.MEDIUM: _ENGINES_MEDIUM,
    Pressure.HIGH: _ENGINES_HIGH,
    Pressure.CRITICAL: _ENGINES_CRITICAL,
}


def compute_adaptive_thresholds(
    state: ContextWindowState,
    config: MemoSiftConfig,
    *,
    total_user_turns: int = 0,
) -> AdaptiveOverrides:
    """Compute adaptive compression thresholds from context window state.

    This is the core of Layer 0. It takes a snapshot of the model's context
    window utilization and produces a modified config with thresholds tuned
    to the current pressure level.

    The function NEVER mutates the input config — it returns a new one via
    ``dataclasses.replace()``.

    Rules:
    - ``recent_turns`` becomes a percentage of total user turns, capped at the
      original config value (which acts as a maximum).
    - ``token_budget`` is auto-derived from available capacity. An explicit
      user-set budget is respected as a lower bound (we never loosen it).
    - ``token_prune_keep_ratio`` and ``entropy_threshold`` are scaled by
      pressure multipliers.
    - Engine selection gates which L3 engines run.

    Args:
        state: Current context window state.
        config: The user's original pipeline config.
        total_user_turns: Number of user messages in the conversation.
            Used for percentage-based recent turn protection.

    Returns:
        AdaptiveOverrides with the computed thresholds and modified config.
    """
    pressure = state.pressure

    # ── Skip compression at NONE pressure ──
    if pressure == Pressure.NONE:
        return AdaptiveOverrides(
            pressure=pressure,
            effective_config=config,
            skip_compression=True,
            enable_observation_masking=False,
            engine_gates=_ENGINES_NONE,
            context_window=state,
        )

    # ── Recent turn protection (percentage-based) ──
    ratio = _RECENT_TURN_RATIOS[pressure]
    if total_user_turns > 0:
        adaptive_recent = max(1, round(total_user_turns * ratio))
    else:
        adaptive_recent = config.recent_turns
    # Cap at the user's configured maximum (never protect MORE than config says).
    effective_recent = min(adaptive_recent, config.recent_turns)

    # ── Auto-budget from available capacity ──
    budget_ratio = _BUDGET_RATIOS[pressure]
    if budget_ratio is not None:
        auto_budget = max(100, int(state.available_tokens * budget_ratio))
        if config.token_budget is not None:
            # User set explicit budget — use the tighter (lower) of the two.
            effective_budget = min(config.token_budget, auto_budget)
        else:
            effective_budget = auto_budget
    else:
        effective_budget = config.token_budget  # None (no limit)

    # ── Pruning and entropy thresholds ──
    prune_mult = _PRUNE_MULTIPLIERS[pressure]
    effective_prune = max(0.1, config.token_prune_keep_ratio * prune_mult)

    entropy_mult = _ENTROPY_MULTIPLIERS[pressure]
    effective_entropy = max(0.3, config.entropy_threshold * entropy_mult)

    # ── Performance tier override ──
    # At HIGH/CRITICAL, force "full" tier to ensure all engines can run.
    effective_tier = config.performance_tier
    if pressure in (Pressure.HIGH, Pressure.CRITICAL):
        effective_tier = "full"

    # ── Auto-enable summarization at CRITICAL pressure ──
    # When the context window is nearly exhausted, every compression engine
    # should be available. Engine D (summarization) is normally opt-in, but
    # at CRITICAL pressure the system should actuate itself — if an LLM is
    # available at runtime, it will fire. If no LLM is provided, the pipeline
    # gate (`llm is not None`) still prevents execution.
    effective_summarization = config.enable_summarization
    if pressure == Pressure.CRITICAL:
        effective_summarization = True

    # ── Build new config ──
    effective_config = dataclasses.replace(
        config,
        recent_turns=effective_recent,
        token_budget=effective_budget,
        token_prune_keep_ratio=effective_prune,
        entropy_threshold=effective_entropy,
        performance_tier=effective_tier,
        enable_summarization=effective_summarization,
    )

    # ── Engine gating ──
    engines = _PRESSURE_ENGINES[pressure]

    # ── Observation masking at HIGH+ pressure ──
    enable_obs = pressure in (Pressure.HIGH, Pressure.CRITICAL)

    # ── Track which fields were overridden ──
    overrides: dict[str, tuple[object, object]] = {}
    if effective_recent != config.recent_turns:
        overrides["recent_turns"] = (config.recent_turns, effective_recent)
    if effective_budget != config.token_budget:
        overrides["token_budget"] = (config.token_budget, effective_budget)
    if effective_prune != config.token_prune_keep_ratio:
        overrides["token_prune_keep_ratio"] = (config.token_prune_keep_ratio, effective_prune)
    if effective_entropy != config.entropy_threshold:
        overrides["entropy_threshold"] = (config.entropy_threshold, effective_entropy)
    if effective_tier != config.performance_tier:
        overrides["performance_tier"] = (config.performance_tier, effective_tier)
    if effective_summarization != config.enable_summarization:
        overrides["enable_summarization"] = (config.enable_summarization, effective_summarization)

    logger.info(
        "L0 adaptive: pressure=%s, recent_turns=%d (was %d), budget=%s, "
        "prune_ratio=%.2f, engines=%d active, overrides=%d",
        pressure.value,
        effective_recent,
        config.recent_turns,
        effective_budget,
        effective_prune,
        len(engines),
        len(overrides),
    )

    return AdaptiveOverrides(
        pressure=pressure,
        effective_config=effective_config,
        skip_compression=False,
        enable_observation_masking=enable_obs,
        engine_gates=engines,
        context_window=state,
        overrides=overrides,
    )


def resolve_context_window(
    explicit: ContextWindowState | None,
    model_name: str | None,
    messages_token_estimate: int = 0,
) -> ContextWindowState | None:
    """Resolve a ContextWindowState from available information.

    Priority:
    1. ``explicit`` — user-provided state (always wins).
    2. ``model_name`` — look up in registry + estimate usage from messages.
    3. Returns None if neither is available (no adaptive behavior).

    Args:
        explicit: User-provided ContextWindowState, if any.
        model_name: Model identifier for registry lookup.
        messages_token_estimate: Estimated total tokens in current messages.

    Returns:
        ContextWindowState or None if no model info is available.
    """
    if explicit is not None:
        return explicit

    if model_name:
        return ContextWindowState.from_model(
            model_name,
            current_usage_tokens=messages_token_estimate,
        )

    return None


def estimate_tokens_heuristic(messages_content: list[str]) -> int:
    """Fast heuristic token estimation: ~3.5 chars per token.

    This is intentionally approximate — exact counting requires the model's
    tokenizer. Used only for auto-resolution when no better estimate is available.
    """
    total_chars = sum(len(c) for c in messages_content if c)
    return max(0, total_chars * 10 // 35)  # Equivalent to / 3.5, integer math.
