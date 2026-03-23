# Pipeline configuration with sensible defaults and domain presets.
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from memosift.core.types import CompressionPolicy, ContentType

# ── Model-aware token budget defaults ───────────────────────────────────────

MODEL_BUDGET_DEFAULTS: MappingProxyType[str, int] = MappingProxyType(
    {
        "gpt-4o": 80_000,
        "gpt-4.1": 600_000,
        "claude-sonnet-4-6": 120_000,
        "claude-opus-4-6": 120_000,
        "gemini-2.5-pro": 600_000,
    }
)

# ── Token pricing (USD per 1K tokens, input) ───────────────────────────────

MODEL_PRICING: MappingProxyType[str, float] = MappingProxyType(
    {
        "gpt-4o": 0.0025,
        "gpt-4.1": 0.002,
        "claude-sonnet-4-6": 0.003,
        "claude-opus-4-6": 0.015,
        "gemini-2.5-pro": 0.00125,
        "default": 0.003,
    }
)


@dataclass
class MemoSiftConfig:
    """Configuration for the MemoSift compression pipeline.

    All fields have defaults matching the JSON schema in ``spec/memosift-config.schema.json``.
    Override only what you need — the defaults are tuned for safe, deterministic-only compression.
    """

    # ── Pipeline control ──
    recent_turns: int = 2
    """Number of recent conversational turns to protect from compression."""

    token_budget: int | None = None
    """Maximum output tokens. ``None`` = no limit."""

    enable_summarization: bool = False
    """Enable Engine D (abstractive summarization). Requires an LLM provider."""

    llm_relevance_scoring: bool = False
    """Use LLM for relevance scoring in Layer 4. Requires an LLM provider."""

    reorder_segments: bool = False
    """Enable position optimization (Layer 5). CAUTION: can break API constraints."""

    # ── Engine tuning ──
    dedup_similarity_threshold: float = 0.80
    """Cosine similarity threshold for fuzzy deduplication (0.0–1.0)."""

    entropy_threshold: float = 1.8
    """Minimum Shannon entropy (bits/char) to keep a line in verbatim deletion."""

    token_prune_keep_ratio: float = 0.5
    """Fraction of tokens to retain in IDF-based pruning (0.1–1.0)."""

    json_array_threshold: int = 5
    """Max items in a JSON array before truncation."""

    code_keep_signatures: bool = True
    """Keep function/class signatures in code compression."""

    relevance_drop_threshold: float = 0.05
    """Minimum keyword overlap ratio to survive relevance scoring (0.0–1.0)."""

    # ── Per-type policy overrides ──
    policies: dict[ContentType, CompressionPolicy] = field(default_factory=dict)
    """Per-content-type policy overrides. Missing keys fall back to DEFAULT_POLICIES."""

    # ── Invocation thresholds ──
    soft_compression_pct: float = 0.60
    """Context size threshold (% of budget) for soft compression (dedup only)."""

    full_compression_pct: float = 0.75
    """Context size threshold (% of budget) for full pipeline compression."""

    aggressive_compression_pct: float = 0.90
    """Context size threshold (% of budget) for aggressive compression."""

    # ── Short message coalescence ──
    coalesce_short_messages: bool = True
    """Merge consecutive short assistant messages into one combined message."""

    coalesce_char_threshold: int = 100
    """Maximum character length for a message to be considered 'short'."""

    # ── Anchor Ledger ──
    enable_anchor_ledger: bool = True
    """Extract and maintain anchor facts during compression."""

    anchor_ledger_max_tokens: int = 5000
    """Maximum token budget for the anchor ledger."""

    # ── Cost tracking ──
    cost_per_1k_tokens: float = 0.003
    """Cost per 1K input tokens (USD). Used for ROI tracking in CompressionReport."""

    model_name: str | None = None
    """Model name for auto-detecting token budget and pricing."""

    deterministic_seed: int | None = 42
    """Seed for deterministic compression. When set, seeds all randomness
    (MinHash permutations, tie-breaking). Set to ``None`` to disable."""

    # ── Performance tuning ──
    performance_tier: str | None = None
    """Performance tier override. ``None`` = auto-detect from message count.
    Values: ``"full"`` (all layers), ``"standard"`` (skip L3G),
    ``"fast"`` (skip L3G+L3E), ``"ultra_fast"`` (minimal layers only)."""

    pre_bucket_bypass: bool = True
    """Route SYSTEM_PROMPT, USER_QUERY, RECENT_TURN, PREVIOUSLY_COMPRESSED
    segments directly past compression layers. Reduces N for all engines."""

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> MemoSiftConfig:
        """Create a config from a named domain preset.

        Available presets:
        - ``"coding"`` — Conservative compression for coding agents. Never loses
          file paths, line numbers, error messages. Keeps code signatures.
        - ``"research"`` — Moderate compression for research/analysis agents.
          Aggressive JSON truncation, preserves citations and URLs.
        - ``"support"`` — Aggressive compression for customer support agents.
          Keeps recent conversation, summarizes old context heavily.
        - ``"data"`` — Balanced compression for data analysis agents.
          Preserves numeric values, column names, query results.
        - ``"general"`` — Default balanced compression for any agent type.

        Args:
            name: Preset name.
            **overrides: Additional field overrides applied on top of the preset.

        Returns:
            A MemoSiftConfig with preset values + any overrides.
        """
        presets: dict[str, dict[str, Any]] = {
            "coding": {
                "recent_turns": 3,
                "entropy_threshold": 2.5,
                "token_prune_keep_ratio": 0.7,
                "code_keep_signatures": True,
                "dedup_similarity_threshold": 0.90,
                "relevance_drop_threshold": 0.03,
                "json_array_threshold": 3,
                "enable_anchor_ledger": True,
                "policies": {
                    ContentType.ERROR_TRACE: CompressionPolicy.PRESERVE,
                    ContentType.CODE_BLOCK: CompressionPolicy.SIGNATURE,
                },
            },
            "research": {
                "recent_turns": 2,
                "entropy_threshold": 1.8,
                "token_prune_keep_ratio": 0.5,
                "code_keep_signatures": False,
                "json_array_threshold": 3,
                "dedup_similarity_threshold": 0.80,
                "relevance_drop_threshold": 0.08,
                "enable_anchor_ledger": True,
            },
            "support": {
                "recent_turns": 5,
                "entropy_threshold": 1.5,
                "token_prune_keep_ratio": 0.4,
                "code_keep_signatures": False,
                "enable_summarization": True,
                "dedup_similarity_threshold": 0.75,
                "relevance_drop_threshold": 0.10,
                "json_array_threshold": 2,
                "enable_anchor_ledger": True,
            },
            "data": {
                "recent_turns": 3,
                "entropy_threshold": 2.0,
                "token_prune_keep_ratio": 0.6,
                "code_keep_signatures": False,
                "json_array_threshold": 10,
                "dedup_similarity_threshold": 0.85,
                "relevance_drop_threshold": 0.05,
                "enable_anchor_ledger": True,
            },
            "general": {
                "recent_turns": 2,
                "entropy_threshold": 1.8,
                "token_prune_keep_ratio": 0.5,
                "dedup_similarity_threshold": 0.80,
                "relevance_drop_threshold": 0.05,
                "enable_anchor_ledger": True,
            },
        }
        if name not in presets:
            available = ", ".join(sorted(presets.keys()))
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")

        values = {**presets[name], **overrides}
        return cls(**values)

    def __post_init__(self) -> None:
        """Validate configuration values and apply model-aware defaults."""
        # Auto-detect budget and pricing from model name.
        if self.model_name:
            if self.token_budget is None and self.model_name in MODEL_BUDGET_DEFAULTS:
                self.token_budget = MODEL_BUDGET_DEFAULTS[self.model_name]
            if self.cost_per_1k_tokens == 0.003 and self.model_name in MODEL_PRICING:
                self.cost_per_1k_tokens = MODEL_PRICING[self.model_name]

        if self.recent_turns < 0:
            raise ValueError(f"recent_turns must be >= 0, got {self.recent_turns}")
        if self.token_budget is not None and self.token_budget < 100:
            raise ValueError(f"token_budget must be >= 100 or None, got {self.token_budget}")
        if not 0.0 <= self.dedup_similarity_threshold <= 1.0:
            raise ValueError(
                f"dedup_similarity_threshold must be 0.0–1.0, got {self.dedup_similarity_threshold}"
            )
        if self.entropy_threshold < 0.0:
            raise ValueError(f"entropy_threshold must be >= 0.0, got {self.entropy_threshold}")
        if not 0.1 <= self.token_prune_keep_ratio <= 1.0:
            raise ValueError(
                f"token_prune_keep_ratio must be 0.1–1.0, got {self.token_prune_keep_ratio}"
            )
        if self.json_array_threshold < 1:
            raise ValueError(f"json_array_threshold must be >= 1, got {self.json_array_threshold}")
        if not 0.0 <= self.relevance_drop_threshold <= 1.0:
            raise ValueError(
                f"relevance_drop_threshold must be 0.0–1.0, got {self.relevance_drop_threshold}"
            )
        if not 0.0 <= self.soft_compression_pct <= 1.0:
            raise ValueError(
                f"soft_compression_pct must be 0.0–1.0, got {self.soft_compression_pct}"
            )
        if not 0.0 <= self.full_compression_pct <= 1.0:
            raise ValueError(
                f"full_compression_pct must be 0.0–1.0, got {self.full_compression_pct}"
            )
        if not 0.0 <= self.aggressive_compression_pct <= 1.0:
            raise ValueError(
                f"aggressive_compression_pct must be 0.0–1.0, got {self.aggressive_compression_pct}"
            )
        if self.soft_compression_pct >= self.full_compression_pct:
            raise ValueError("soft_compression_pct must be < full_compression_pct")
        if self.full_compression_pct >= self.aggressive_compression_pct:
            raise ValueError("full_compression_pct must be < aggressive_compression_pct")
        if self.anchor_ledger_max_tokens < 100:
            raise ValueError(
                f"anchor_ledger_max_tokens must be >= 100, got {self.anchor_ledger_max_tokens}"
            )
        valid_tiers = {"full", "standard", "fast", "ultra_fast"}
        if self.performance_tier is not None and self.performance_tier not in valid_tiers:
            raise ValueError(
                f"performance_tier must be one of {valid_tiers} or None, "
                f"got '{self.performance_tier}'"
            )
