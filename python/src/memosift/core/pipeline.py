# Pipeline orchestrator — runs the 6-layer compression pipeline with Three-Zone Model.
from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from memosift.config import MemoSiftConfig
from memosift.core.anchor_extractor import (
    extract_anchors_from_segments,
    extract_anchors_llm,
    extract_reasoning_chains,
)
from memosift.core.budget import enforce_budget
from memosift.core.classifier import classify_messages
from memosift.core.coalescer import coalesce_short_messages
from memosift.core.context_window import (
    AdaptiveOverrides,
    ContextWindowState,
    compute_adaptive_thresholds,
    estimate_tokens_heuristic,
    resolve_context_window,
)
from memosift.core.deduplicator import CrossWindowState, deduplicate
from memosift.core.engines.discourse_compressor import elaborate_compress
from memosift.core.engines.importance import score_importance
from memosift.core.engines.pruner import prune_tokens
from memosift.core.engines.relevance_pruner import query_relevance_prune
from memosift.core.engines.structural import structural_compress
from memosift.core.engines.summarizer import summarize_segments
from memosift.core.engines.verbatim import verbatim_compress
from memosift.core.positioner import optimize_position
from memosift.core.scorer import score_relevance, score_relevance_llm
from memosift.core.types import (
    AnchorLedger,
    ClassifiedMessage,
    ContentType,
    DependencyMap,
    MemoSiftMessage,
)
from memosift.providers.heuristic import HeuristicTokenCounter
from memosift.report import CompressionReport

if TYPE_CHECKING:
    from memosift.providers.base import MemoSiftLLMProvider

logger = logging.getLogger("memosift")


@dataclass
class CompressionCache:
    """Stores original content for messages collapsed during compression.

    Enables selective re-expansion when the agent needs more context
    (e.g., a re-read request). Keyed by message original_index.
    """

    _originals: dict[int, str] = field(default_factory=dict)

    def store(self, original_index: int, content: str) -> None:
        """Store original content before collapse."""
        self._originals[original_index] = content

    def expand(self, original_index: int) -> str | None:
        """Retrieve original content for a collapsed message. Returns None if not stored."""
        return self._originals.get(original_index)

    def has(self, original_index: int) -> bool:
        """Return True if original content is stored for this index."""
        return original_index in self._originals

    @property
    def size(self) -> int:
        """Number of stored originals."""
        return len(self._originals)


def _resolve_tier(config: MemoSiftConfig, message_count: int) -> str:
    """Resolve the performance tier from config or auto-detect from message count."""
    if config.performance_tier is not None:
        return config.performance_tier
    if message_count <= 50:
        return "full"
    if message_count <= 150:
        return "standard"
    if message_count <= 300:
        return "fast"
    return "ultra_fast"


_BYPASS_TYPES = {
    ContentType.SYSTEM_PROMPT,
    ContentType.USER_QUERY,
    ContentType.RECENT_TURN,
    ContentType.PREVIOUSLY_COMPRESSED,
}


def _pre_bucket(
    segments: list[ClassifiedMessage],
) -> tuple[list[ClassifiedMessage], list[ClassifiedMessage]]:
    """Split segments into bypass (skip compression) and compress (run through engines) buckets."""
    bypass = [s for s in segments if s.content_type in _BYPASS_TYPES or s.protected]
    compress = [s for s in segments if s.content_type not in _BYPASS_TYPES and not s.protected]
    return bypass, compress


async def compress(
    messages: list[MemoSiftMessage],
    *,
    llm: MemoSiftLLMProvider | None = None,
    config: MemoSiftConfig | None = None,
    task: str | None = None,
    ledger: AnchorLedger | None = None,
    cross_window: CrossWindowState | None = None,
    cache: CompressionCache | None = None,
    context_window: ContextWindowState | None = None,
) -> tuple[list[MemoSiftMessage], CompressionReport]:
    """Compress a list of messages through the 6-layer pipeline.

    Uses the Three-Zone Memory Model to prevent re-compression:
    - Zone 1 (system prompts): pass through untouched
    - Zone 2 (previously compressed, ``_memosift_compressed=True``): pass through untouched
    - Zone 3 (new raw messages): compressed by the pipeline

    Args:
        messages: The raw conversation messages.
        llm: Optional LLM provider for summarization and scoring.
            When ``None``, operates in deterministic-only mode.
        config: Pipeline configuration. Uses defaults when ``None``.
        task: Optional task description for relevance scoring.
        ledger: Optional AnchorLedger for fact extraction across compression cycles.
            When provided and ``config.enable_anchor_ledger`` is True, critical facts
            are extracted from Zone 3 messages before compression.
        cross_window: Dedup state for long-running agents (cross-window dedup).
        cache: Storage for original content of collapsed messages (for re-expansion).
        context_window: Context window state for adaptive compression (Layer 0).
            When provided, dynamically adjusts thresholds based on model context
            window utilization. Overrides ``config.context_window`` if both are set.

    Returns:
        A tuple of (compressed_messages, compression_report).
    """
    if config is None:
        config = MemoSiftConfig()

    report = CompressionReport()
    counter: MemoSiftLLMProvider = llm or HeuristicTokenCounter()

    # ── Layer 0: Context-Aware Adaptive Thresholds ──
    # Resolve context window state from: explicit param > config field > model_name.
    cw_state = resolve_context_window(
        context_window or config.context_window,
        config.model_name,
        estimate_tokens_heuristic([m.content for m in messages]),
    )
    # Ensure current_usage reflects actual message tokens. When the adapter
    # auto-resolves from model name, current_usage_tokens defaults to 0.
    # The messages themselves ARE the context — their size IS the usage.
    if cw_state is not None:
        _est_msg_tokens = estimate_tokens_heuristic([m.content for m in messages])
        if _est_msg_tokens > cw_state.current_usage_tokens:
            cw_state = dataclasses.replace(cw_state, current_usage_tokens=_est_msg_tokens)

    adaptive: AdaptiveOverrides | None = None
    if cw_state is not None:
        _user_turn_count = sum(1 for m in messages if m.role == "user")
        adaptive = compute_adaptive_thresholds(cw_state, config, total_user_turns=_user_turn_count)
        config = adaptive.effective_config
        report.adaptive_overrides = adaptive.overrides
        report.add_decision(
            layer="L0_adaptive",
            action="computed",
            message_index=-1,
            original_tokens=0,
            result_tokens=0,
            reason=(
                f"pressure={adaptive.pressure.value}, "
                f"model={cw_state.model}, "
                f"remaining={cw_state.remaining_ratio:.0%}, "
                f"recent_turns={config.recent_turns}, "
                f"budget={config.token_budget}, "
                f"overrides={adaptive.overrides}"
            ),
        )

        # Short-circuit: at NONE pressure, no compression needed.
        if adaptive.skip_compression:
            original_tokens = estimate_tokens_heuristic([m.content for m in messages])
            report.finalize(original_tokens, original_tokens, config.cost_per_1k_tokens)
            return list(messages), report

    # ── Three-Zone Partitioning ──
    zone1, zone2, zone3 = _partition_zones(messages)

    # Estimate original token count (all zones).
    original_tokens = 0
    for m in messages:
        original_tokens += await counter.count_tokens(m.content)

    # If no raw messages to compress, return as-is.
    if not zone3:
        report.finalize(original_tokens, original_tokens, config.cost_per_1k_tokens)
        return list(messages), report

    # ── Layer 1: Classify (Zone 3 only) ──
    segments = await _run_layer(
        "classifier",
        lambda segs: _async_wrap(classify_messages(zone3, config)),
        [],
        report,
    )
    if segments is None:
        segments = [
            ClassifiedMessage(
                message=m,
                content_type=_default_type(m),
                policy=_default_policy(m),
                original_index=messages.index(m) if m in messages else i,
            )
            for i, m in enumerate(zone3)
        ]

    # Fix original_index to reference position in full message list.
    _fix_original_indices(segments, messages, zone3)

    # ── Anchor Extraction (before compression, after classification) ──
    # Always run regex extraction first (fast, deterministic, captures file
    # paths, errors, line refs, identifiers).  When an LLM is provided,
    # ALSO run LLM extraction to catch implicit decisions, conclusions, and
    # causal relationships that regex misses.  The two are additive — the
    # ledger deduplicates by content hash.
    if ledger is not None and config.enable_anchor_ledger:
        extract_anchors_from_segments(segments, ledger)
        if llm is not None:
            await extract_anchors_llm(segments, ledger, llm)

    # ── Reasoning Chain Tracking (after anchor extraction) ──
    # Detect logical dependencies ("therefore", "so we can", etc.) and
    # record them in the DependencyMap for safety during aggressive pruning.
    deps = DependencyMap()
    extract_reasoning_chains(segments, deps)

    # Record segment counts.
    for seg in segments:
        key = seg.content_type.value
        report.segment_counts[key] = report.segment_counts.get(key, 0) + 1

    # ── Tier resolution ──
    tier = _resolve_tier(config, len(segments))
    report.performance_tier = tier

    # ── Pre-bucketing: route bypass-eligible segments past compression ──
    bypass_segments: list[ClassifiedMessage] = []
    if config.pre_bucket_bypass:
        bypass_segments, segments = _pre_bucket(segments)

    # ── Layer 2: Deduplicate ──
    # deps already initialized above with reasoning chain edges.
    _dedup_exact_only = tier in ("fast", "ultra_fast")
    result = await _run_layer(
        "deduplicator",
        lambda segs: _async_wrap(
            _unpack_dedup(segs, config, cross_window=cross_window, exact_only=_dedup_exact_only)
        ),
        segments,
        report,
    )
    if result is not None:
        segments, deps = result

    # ── Layer 2.5: Coalesce short messages ──
    if config.coalesce_short_messages and tier != "ultra_fast":
        segments = (
            await _run_layer(
                "coalescer",
                lambda segs: _async_wrap(coalesce_short_messages(segs, config)),
                segments,
                report,
            )
            or segments
        )

    # ── Layer 3: Compress (per-type engines) ──
    # Pass ledger to engines so they protect content containing anchor facts.
    # Track content hashes for first-read vs re-read detection (Item 2.2).
    seen_content_hashes: dict[str, int] = {}
    # Enable observation masking only for conversations with enough tool results
    # that old ones are truly stale (>= 10 tool results in the compress bucket).
    _tool_result_count = sum(
        1
        for s in segments
        if s.content_type in {ContentType.TOOL_RESULT_TEXT, ContentType.TOOL_RESULT_JSON}
    )
    _enable_obs_masking = _tool_result_count >= 10 or (
        adaptive is not None and adaptive.enable_observation_masking
    )

    # Adaptive engine gating: when Layer 0 is active, only run engines in the gate set.
    # When adaptive is None, all engines run as before (backward compatible).
    _gates = adaptive.engine_gates if adaptive is not None else None

    if _gates is None or "verbatim" in _gates:
        segments = (
            await _run_layer(
                "engine_verbatim",
                lambda segs: _async_wrap(
                    verbatim_compress(
                        segs,
                        config,
                        ledger=ledger,
                        seen_content_hashes=seen_content_hashes,
                        enable_observation_masking=_enable_obs_masking,
                    )
                ),
                segments,
                report,
            )
            or segments
        )

    if (_gates is None or "pruner" in _gates) and tier != "ultra_fast":
        segments = (
            await _run_layer(
                "engine_pruner",
                lambda segs: _async_wrap(prune_tokens(segs, config, ledger=ledger)),
                segments,
                report,
            )
            or segments
        )

    if _gates is None or "structural" in _gates:
        segments = (
            await _run_layer(
                "engine_structural",
                lambda segs: _async_wrap(structural_compress(segs, config, ledger=ledger)),
                segments,
                report,
            )
            or segments
        )

    # ── Conversation Phase Detection ──
    from memosift.core.phase_detector import PHASE_KEEP_MULTIPLIERS, detect_phase

    phase = detect_phase(segments)
    phase_mult = PHASE_KEEP_MULTIPLIERS.get(phase, 1.0)

    # ── Layer 3G: Importance Scoring (scoring only, no deletion) ──
    if (_gates is None or "importance" in _gates) and tier not in ("fast", "ultra_fast"):
        segments = (
            await _run_layer(
                "importance_scorer",
                lambda segs: _async_wrap(
                    score_importance(
                        segs,
                        config,
                        ledger=ledger,
                        phase_multiplier=phase_mult,
                    )
                ),
                segments,
                report,
            )
            or segments
        )

    # ── Layer 3E: Query-Relevance Pruning (uses shields from L3G) ──
    if (_gates is None or "relevance_pruner" in _gates) and tier not in ("fast", "ultra_fast"):
        segments = (
            await _run_layer(
                "relevance_pruner",
                lambda segs: _async_wrap(query_relevance_prune(segs, config, deps, ledger=ledger)),
                segments,
                report,
            )
            or segments
        )

    # ── Layer 3F: Elaboration Compression (uses shields from L3G) ──
    if (_gates is None or "discourse" in _gates) and tier != "ultra_fast":
        segments = (
            await _run_layer(
                "discourse_compressor",
                lambda segs: _async_wrap(elaborate_compress(segs, config, ledger=ledger)),
                segments,
                report,
            )
            or segments
        )

    # Engine D: Summarization (LLM-dependent, opt-in).
    _summarizer_gated = _gates is None or "summarizer" in _gates
    if config.enable_summarization and llm is not None and _summarizer_gated:
        segments = (
            await _run_layer(
                "engine_summarizer",
                lambda segs: summarize_segments(segs, config, llm),
                segments,
                report,
            )
            or segments
        )

    # ── Merge bypass segments back (budget and scorer need the full set) ──
    # Merge by original_index to maintain message ordering.
    if bypass_segments:
        segments = sorted(bypass_segments + segments, key=lambda s: s.original_index)

    # ── Layer 4: Score relevance ──
    if config.llm_relevance_scoring and llm is not None and task:
        segments = (
            await _run_layer(
                "scorer_llm",
                lambda segs: score_relevance_llm(segs, config, task, llm),
                segments,
                report,
            )
            or segments
        )
    else:
        segments = (
            await _run_layer(
                "scorer",
                lambda segs: score_relevance(segs, config, task, ledger=ledger),
                segments,
                report,
            )
            or segments
        )

    # ── Layer 5: Position optimization ──
    segments = (
        await _run_layer(
            "positioner",
            lambda segs: _async_wrap(optimize_position(segs, config)),
            segments,
            report,
        )
        or segments
    )

    # ── Layer 6: Budget enforcement ──
    # Compute effective budget: total budget minus zone1 + zone2 tokens.
    budget_config = config
    if config.token_budget is not None:
        zone1_tokens = 0
        for m in zone1:
            zone1_tokens += await counter.count_tokens(m.content)
        zone2_tokens = 0
        for m in zone2:
            zone2_tokens += await counter.count_tokens(m.content)
        effective_budget = max(100, config.token_budget - zone1_tokens - zone2_tokens)
        budget_config = dataclasses.replace(config, token_budget=effective_budget)

    segments = (
        await _run_layer(
            "budget",
            lambda segs: enforce_budget(segs, budget_config, deps, counter, ledger=ledger),
            segments,
            report,
        )
        or segments
    )

    # ── Validate and enforce tool call integrity ──
    segments = _enforce_tool_call_integrity(segments, report)

    # ── Tag and reassemble ──
    compressed_zone3 = _to_messages(segments)
    for msg in compressed_zone3:
        msg._memosift_compressed = True

    compressed = _reassemble_zones(zone1, zone2, compressed_zone3)

    # Estimate compressed token count.
    compressed_tokens = 0
    for m in compressed:
        compressed_tokens += await counter.count_tokens(m.content)

    report.finalize(original_tokens, compressed_tokens, config.cost_per_1k_tokens)
    return compressed, report


# ── Three-Zone helpers ──────────────────────────────────────────────────────


def _partition_zones(
    messages: list[MemoSiftMessage],
) -> tuple[list[MemoSiftMessage], list[MemoSiftMessage], list[MemoSiftMessage]]:
    """Partition messages into three zones.

    Zone 1: System prompts (``role="system"``).
    Zone 2: Previously compressed (``_memosift_compressed=True``, non-system).
    Zone 3: Raw/new messages (everything else — this gets compressed).
    """
    zone1: list[MemoSiftMessage] = []
    zone2: list[MemoSiftMessage] = []
    zone3: list[MemoSiftMessage] = []

    for msg in messages:
        if msg.role == "system":
            zone1.append(msg)
        elif msg._memosift_compressed:
            zone2.append(msg)
        else:
            zone3.append(msg)

    return zone1, zone2, zone3


def _reassemble_zones(
    zone1: list[MemoSiftMessage],
    zone2: list[MemoSiftMessage],
    compressed_zone3: list[MemoSiftMessage],
) -> list[MemoSiftMessage]:
    """Reassemble zones: system prompts + frozen compressed + newly compressed."""
    return list(zone1) + list(zone2) + list(compressed_zone3)


def _fix_original_indices(
    segments: list[ClassifiedMessage],
    all_messages: list[MemoSiftMessage],
    zone3: list[MemoSiftMessage],
) -> None:
    """Fix original_index on classified segments to reference full message list.

    The classifier assigns indices 0..len(zone3)-1, but DependencyMap and
    other layers need indices relative to the full input message list.

    Uses content+role matching instead of object identity for robustness.
    """
    # Build mapping: zone3 position → full list position.
    # Walk both lists in order, matching by position (zone3 is a subsequence of all_messages).
    zone3_to_full: dict[int, int] = {}
    z3_idx = 0
    for full_idx in range(len(all_messages)):
        if z3_idx >= len(zone3):
            break
        if all_messages[full_idx] is zone3[z3_idx]:
            zone3_to_full[z3_idx] = full_idx
            z3_idx += 1
        elif (
            all_messages[full_idx].role == zone3[z3_idx].role
            and all_messages[full_idx].content == zone3[z3_idx].content
            and not all_messages[full_idx]._memosift_compressed
            and all_messages[full_idx].role != "system"
        ):
            # Fallback: match by content if identity fails (e.g., after copy).
            zone3_to_full[z3_idx] = full_idx
            z3_idx += 1

    for seg in segments:
        if seg.original_index in zone3_to_full:
            seg.original_index = zone3_to_full[seg.original_index]


# ── Layer execution ─────────────────────────────────────────────────────────


async def _run_layer(
    name: str,
    layer_fn,
    segments,
    report: CompressionReport,
):
    """Run a single pipeline layer with error recovery."""
    start = time.perf_counter_ns()
    try:
        result = await layer_fn(segments)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

        if isinstance(result, tuple):
            segs = result[0]
        elif isinstance(result, list):
            segs = result
        else:
            segs = segments

        in_tokens = sum(s.estimated_tokens for s in segments) if segments else 0
        out_tokens = sum(s.estimated_tokens for s in segs) if segs else 0
        report.add_layer(name, in_tokens, out_tokens, elapsed_ms)
        return result
    except Exception as e:
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        logger.warning("Layer '%s' failed: %s. Skipping.", name, e)
        report.add_layer_failure(name, str(e), elapsed_ms)
        return None


async def _async_wrap(value):
    """Wrap a synchronous value as an awaitable."""
    return value


def _unpack_dedup(segments, config, cross_window=None, exact_only=False):
    """Wrapper to match the layer_fn signature for dedup."""
    return deduplicate(segments, config, cross_window=cross_window, exact_only=exact_only)


def _default_type(msg: MemoSiftMessage):
    """Fallback classification when classifier fails."""
    from memosift.core.types import ContentType

    if msg.role == "system":
        return ContentType.SYSTEM_PROMPT
    return ContentType.OLD_CONVERSATION


def _default_policy(msg: MemoSiftMessage):
    """Fallback policy when classifier fails."""
    from memosift.core.types import CompressionPolicy

    if msg.role == "system":
        return CompressionPolicy.PRESERVE
    return CompressionPolicy.MODERATE


def validate_tool_call_integrity(
    segments: list[ClassifiedMessage],
) -> list[str]:
    """Return a list of orphaned tool_call_ids."""
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for seg in segments:
        if seg.message.tool_calls:
            for tc in seg.message.tool_calls:
                call_ids.add(tc.id)
        if seg.message.tool_call_id:
            result_ids.add(seg.message.tool_call_id)
    orphaned_calls = call_ids - result_ids
    orphaned_results = result_ids - call_ids
    return [f"call without result: {cid}" for cid in orphaned_calls] + [
        f"result without call: {rid}" for rid in orphaned_results
    ]


def _enforce_tool_call_integrity(
    segments: list[ClassifiedMessage],
    report: CompressionReport,
) -> list[ClassifiedMessage]:
    """Enforce the tool call integrity invariant by removing orphans.

    If an assistant message with tool_calls has no matching tool results,
    remove the tool_calls from that message. If a tool result has no
    matching tool_call, remove the tool result message entirely.
    This is a hard invariant — orphans cause API errors downstream.
    """
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for seg in segments:
        if seg.message.tool_calls:
            for tc in seg.message.tool_calls:
                call_ids.add(tc.id)
        if seg.message.tool_call_id:
            result_ids.add(seg.message.tool_call_id)

    orphaned_calls = call_ids - result_ids
    orphaned_results = result_ids - call_ids

    if not orphaned_calls and not orphaned_results:
        return segments

    result: list[ClassifiedMessage] = []
    for seg in segments:
        # Remove orphaned tool result messages.
        if seg.message.tool_call_id and seg.message.tool_call_id in orphaned_results:
            report.add_decision(
                layer="pipeline",
                action="dropped",
                message_index=seg.original_index,
                original_tokens=seg.estimated_tokens,
                result_tokens=0,
                reason=f"Orphaned tool result (no matching call): {seg.message.tool_call_id}",
            )
            continue

        # Strip orphaned tool_calls from assistant messages.
        if seg.message.tool_calls and any(tc.id in orphaned_calls for tc in seg.message.tool_calls):
            remaining = [tc for tc in seg.message.tool_calls if tc.id not in orphaned_calls]
            seg.message.tool_calls = remaining if remaining else None
            for cid in orphaned_calls:
                report.add_decision(
                    layer="pipeline",
                    action="stripped",
                    message_index=seg.original_index,
                    original_tokens=0,
                    result_tokens=0,
                    reason=f"Orphaned tool call (no matching result): {cid}",
                )

        result.append(seg)

    return result


def _to_messages(segments: list[ClassifiedMessage]) -> list[MemoSiftMessage]:
    """Convert classified segments back to MemoSiftMessage, adding annotations."""
    result: list[MemoSiftMessage] = []
    for seg in segments:
        msg = seg.message
        msg._memosift_content_type = seg.content_type.value
        result.append(msg)
    return result
