# Compression report — observability and metrics for every compress() call.
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Decision:
    """A single compression decision recorded during pipeline execution."""

    layer: str
    action: str  # "deduplicated", "deleted", "summarized", "pruned", "reordered", "dropped"
    message_index: int
    original_tokens: int
    result_tokens: int
    reason: str


@dataclass
class LayerReport:
    """Metrics for a single pipeline layer's execution."""

    name: str
    input_tokens: int
    output_tokens: int
    tokens_removed: int
    latency_ms: float
    llm_calls_made: int = 0
    llm_tokens_consumed: int = 0


@dataclass
class CompressionReport:
    """Full report returned alongside compressed messages from ``compress()``.

    Provides per-layer breakdowns, individual decisions, and aggregate metrics.
    """

    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 1.0
    tokens_saved: int = 0
    estimated_cost_saved: float = 0.0
    total_latency_ms: float = 0.0
    layers: list[LayerReport] = field(default_factory=list)
    segment_counts: dict[str, int] = field(default_factory=dict)
    decisions: list[Decision] = field(default_factory=list)
    performance_tier: str = "full"
    adaptive_overrides: dict[str, tuple[object, object]] | None = None
    """Fields overridden by Layer 0 adaptive compression.

    Maps field name to ``(original_value, effective_value)``. ``None`` when
    Layer 0 is not active (no context_window provided). Empty dict when L0
    ran but no fields were changed.
    """

    def add_layer(
        self,
        name: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        *,
        llm_calls_made: int = 0,
        llm_tokens_consumed: int = 0,
    ) -> None:
        """Record a successful layer execution."""
        self.layers.append(
            LayerReport(
                name=name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tokens_removed=input_tokens - output_tokens,
                latency_ms=latency_ms,
                llm_calls_made=llm_calls_made,
                llm_tokens_consumed=llm_tokens_consumed,
            )
        )
        self.total_latency_ms += latency_ms

    def add_layer_failure(self, name: str, error: str, latency_ms: float) -> None:
        """Record a layer that failed and was skipped."""
        self.layers.append(
            LayerReport(
                name=name,
                input_tokens=0,
                output_tokens=0,
                tokens_removed=0,
                latency_ms=latency_ms,
            )
        )
        self.decisions.append(
            Decision(
                layer=name,
                action="skipped",
                message_index=-1,
                original_tokens=0,
                result_tokens=0,
                reason=f"Layer failed: {error}",
            )
        )
        self.total_latency_ms += latency_ms

    def add_decision(
        self,
        layer: str,
        action: str,
        message_index: int,
        original_tokens: int,
        result_tokens: int,
        reason: str,
    ) -> None:
        """Record an individual compression decision."""
        self.decisions.append(
            Decision(
                layer=layer,
                action=action,
                message_index=message_index,
                original_tokens=original_tokens,
                result_tokens=result_tokens,
                reason=reason,
            )
        )

    def finalize(
        self,
        original_tokens: int,
        compressed_tokens: int,
        cost_per_1k_tokens: float = 0.003,
    ) -> None:
        """Set aggregate metrics after pipeline completes.

        Args:
            original_tokens: Token count before compression.
            compressed_tokens: Token count after compression.
            cost_per_1k_tokens: USD cost per 1K input tokens for ROI calculation.
        """
        self.original_tokens = original_tokens
        self.compressed_tokens = compressed_tokens
        self.tokens_saved = original_tokens - compressed_tokens
        if compressed_tokens == 0:
            self.compression_ratio = float("inf") if original_tokens > 0 else 1.0
        else:
            self.compression_ratio = original_tokens / compressed_tokens
        self.estimated_cost_saved = (self.tokens_saved / 1000) * cost_per_1k_tokens
