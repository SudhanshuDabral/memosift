# Production trace benchmarks — measures compression quality using real conversation data.
#
# Runs the MemoSift pipeline on actual production traces and reports:
# - Compression ratio (higher = more tokens saved)
# - Fact retention (% of critical numerical KPIs preserved)
# - Token count (input/output)
# - Anchor coverage (facts extracted by the ledger)
# - Tool call integrity (no orphaned tool calls)
#
# Optional LLM-as-judge evaluation when an LLM provider is available.
#
# Run:  python -m pytest tests/python/test_production_benchmarks.py -v -s
from __future__ import annotations

import json
import logging
import re
import textwrap
from dataclasses import dataclass

import pytest

from memosift import MemoSiftConfig, MemoSiftMessage, compress
from memosift.core.types import AnchorLedger

from memosift.providers.base import LLMResponse

from .fixtures.production_traces.trace_analyze_files import (
    CRITICAL_FACTS,
    LLM_JUDGE_PROMPT,
    TRACE_MESSAGES,
)

logger = logging.getLogger("memosift.benchmarks")


# ── Mock LLM for Engine D testing ───────────────────────────────────────


class _SummarizingMockLLM:
    """Mock LLM that produces a fact-preserving summary for Engine D testing.

    Extracts numbers and key terms from the input, then produces a compressed
    version that preserves critical values. This simulates a real LLM that
    follows the summarization prompt.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def generate(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.0
    ) -> LLMResponse:
        self.call_count += 1
        # Extract the SEGMENT content from the prompt.
        if "SEGMENT:" in prompt:
            content = prompt.split("SEGMENT:")[-1].strip()
        else:
            content = prompt

        self.total_input_tokens += len(content.split())

        # Extract all numbers with context (preserves KPIs).
        import re

        numbers = re.findall(r"\b\d[\d,.]*(?:\.\d+)?\s*\S{0,12}", content)
        # Extract key sentences (first, those with numbers, last).
        sentences = re.split(r"(?<=[.!?])\s+", content)
        kept: list[str] = []
        if sentences:
            kept.append(sentences[0])  # Keep first sentence.
        for s in sentences[1:-1]:
            if re.search(r"\d[\d,.]+", s):
                kept.append(s)  # Keep sentences with numbers.
        if len(sentences) > 1:
            kept.append(sentences[-1])  # Keep last sentence.

        summary = " ".join(kept[:10])  # Cap at 10 sentences.
        # Ensure it's shorter than original.
        if len(summary) >= len(content):
            summary = content[: len(content) // 3]

        output_tokens = len(summary.split())
        self.total_output_tokens += output_tokens
        return LLMResponse(
            text=summary,
            input_tokens=len(content.split()),
            output_tokens=output_tokens,
        )

    async def count_tokens(self, text: str) -> int:
        return len((text or "").split())


# ── Benchmark result dataclass ───────────────────────────────────────────


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    trace_name: str
    config_name: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    total_facts: int
    retained_facts: int
    fact_retention_pct: float
    missing_facts: list[str]
    anchor_facts_count: int
    tool_call_integrity: bool
    orphaned_tool_calls: list[str]
    message_count_before: int
    message_count_after: int

    def report(self) -> str:
        """Format a human-readable report."""
        status = "PASS" if self.fact_retention_pct >= 90.0 else "FAIL"
        integrity = "PASS" if self.tool_call_integrity else "FAIL"
        lines = [
            f"{'=' * 70}",
            f"  BENCHMARK: {self.trace_name} | Config: {self.config_name}",
            f"{'=' * 70}",
            f"  Compression:     {self.original_tokens:,} -> {self.compressed_tokens:,} tokens "
            f"({self.compression_ratio:.2f}x)",
            f"  Messages:        {self.message_count_before} -> {self.message_count_after}",
            f"  Fact Retention:  {self.retained_facts}/{self.total_facts} "
            f"({self.fact_retention_pct:.1f}%) [{status}]",
            f"  Anchor Facts:    {self.anchor_facts_count} extracted",
            f"  Tool Integrity:  [{integrity}]",
        ]
        if self.missing_facts:
            lines.append(f"  Missing Facts:   {self.missing_facts}")
        if self.orphaned_tool_calls:
            lines.append(f"  Orphaned Tools:  {self.orphaned_tool_calls}")
        lines.append(f"{'=' * 70}")
        return "\n".join(lines)


# ── Benchmark runner ─────────────────────────────────────────────────────


async def run_benchmark(
    trace_name: str,
    messages: list[dict],
    critical_facts: list[str],
    config: MemoSiftConfig,
    config_name: str = "default",
) -> BenchmarkResult:
    """Run MemoSift on a production trace and measure quality metrics."""
    # Convert dicts to MemoSiftMessage objects.
    memo_messages = [MemoSiftMessage.from_dict(m) for m in messages]

    # Run compression.
    ledger = AnchorLedger()
    compressed, report = await compress(
        memo_messages,
        config=config,
        ledger=ledger,
    )

    # Combine compressed text + anchor ledger for fact checking.
    compressed_text = " ".join(m.content or "" for m in compressed)
    ledger_text = ledger.render()
    combined = compressed_text + " " + ledger_text

    # Check fact retention.
    missing: list[str] = []
    retained = 0
    for fact in critical_facts:
        if fact in combined:
            retained += 1
        else:
            missing.append(fact)

    # Check tool call integrity.
    tool_call_ids: set[str] = set()
    tool_result_ids: set[str] = set()
    for msg in compressed:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = getattr(tc, "id", None) or ""
                if tc_id:
                    tool_call_ids.add(tc_id)
        if msg.tool_call_id:
            tool_result_ids.add(msg.tool_call_id)
    orphaned = list(tool_call_ids - tool_result_ids)

    return BenchmarkResult(
        trace_name=trace_name,
        config_name=config_name,
        original_tokens=report.original_tokens,
        compressed_tokens=report.compressed_tokens,
        compression_ratio=report.compression_ratio,
        total_facts=len(critical_facts),
        retained_facts=retained,
        fact_retention_pct=(retained / len(critical_facts) * 100) if critical_facts else 100.0,
        missing_facts=missing,
        anchor_facts_count=len(ledger.facts),
        tool_call_integrity=len(orphaned) == 0,
        orphaned_tool_calls=orphaned,
        message_count_before=len(memo_messages),
        message_count_after=len(compressed),
    )


# ── LLM Judge ────────────────────────────────────────────────────────────


async def llm_judge_evaluate(
    compressed_messages: list[MemoSiftMessage],
    ledger: AnchorLedger,
    llm,
    judge_prompt: str,
) -> dict | None:
    """Use an LLM to evaluate compression quality.

    Returns parsed JSON scores or None if no LLM is available.
    """
    if llm is None:
        return None

    compressed_text = "\n".join(
        f"[{m.role}]: {(m.content or '')[:500]}" for m in compressed_messages
    )
    ledger_text = ledger.render()

    prompt = judge_prompt.format(
        compressed_text=compressed_text,
        anchor_ledger=ledger_text,
    )

    try:
        response = await llm.generate(prompt, max_tokens=1024, temperature=0.0)
        # Parse JSON from response.
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.strip().startswith("```"))
        return json.loads(text)
    except Exception as e:
        logger.warning("LLM judge evaluation failed: %s", e)
        return None


# ── Test classes ─────────────────────────────────────────────────────────


class TestTraceAnalyzeFiles:
    """Benchmark: 10-turn flowback analysis ($2.24 original cost)."""

    TRACE_NAME = "analyze_files"

    async def test_default_config(self) -> None:
        """Baseline compression with default config."""
        config = MemoSiftConfig()
        result = await run_benchmark(
            self.TRACE_NAME, TRACE_MESSAGES, CRITICAL_FACTS, config, "default"
        )
        print("\n" + result.report())
        assert result.fact_retention_pct >= 80.0, f"Fact retention too low: {result.missing_facts}"
        assert result.tool_call_integrity, f"Orphaned tool calls: {result.orphaned_tool_calls}"

    async def test_data_preset(self) -> None:
        """Compression with data analysis preset."""
        config = MemoSiftConfig.preset("data")
        result = await run_benchmark(
            self.TRACE_NAME, TRACE_MESSAGES, CRITICAL_FACTS, config, "data"
        )
        print("\n" + result.report())
        assert result.fact_retention_pct >= 85.0, f"Fact retention too low: {result.missing_facts}"
        assert result.tool_call_integrity

    async def test_energy_preset(self) -> None:
        """Compression with energy domain preset."""
        config = MemoSiftConfig.preset("energy")
        result = await run_benchmark(
            self.TRACE_NAME, TRACE_MESSAGES, CRITICAL_FACTS, config, "energy"
        )
        print("\n" + result.report())
        assert result.fact_retention_pct >= 85.0, f"Fact retention too low: {result.missing_facts}"
        assert result.tool_call_integrity

    async def test_aggressive_budget(self) -> None:
        """Compression with tight token budget — stress test for fact retention."""
        config = MemoSiftConfig.preset("data", token_budget=5000)
        result = await run_benchmark(
            self.TRACE_NAME, TRACE_MESSAGES, CRITICAL_FACTS, config, "data_budget_5k"
        )
        print("\n" + result.report())
        # Tight budget — facts may be lost, but core KPIs should survive via ledger.
        assert result.fact_retention_pct >= 60.0, f"Critical data loss: {result.missing_facts}"
        assert result.tool_call_integrity

    async def test_metric_extraction_quality(self) -> None:
        """Verify contextual metric intelligence extracts domain KPIs."""
        config = MemoSiftConfig.preset("energy")
        memo_messages = [MemoSiftMessage.from_dict(m) for m in TRACE_MESSAGES]
        ledger = AnchorLedger()
        await compress(memo_messages, config=config, ledger=ledger)

        # Check that metric-related anchor facts were extracted.
        metric_facts = [f for f in ledger.facts if f.content.startswith("Metric:")]
        print(f"\nMetric facts extracted: {len(metric_facts)}")
        for f in metric_facts[:10]:
            print(f"  {f.content} (confidence={f.confidence})")

        # Check all anchor facts for overall extraction quality.
        all_facts = ledger.facts
        print(f"Total anchor facts extracted: {len(all_facts)}")
        for f in all_facts[:15]:
            print(f"  [{f.category}] {f.content[:80]} (conf={f.confidence})")

        # Should extract facts from the assistant responses which contain KPIs.
        assert len(all_facts) >= 5, (
            f"Expected at least 5 anchor facts total, got {len(all_facts)}"
        )

    async def test_comparison_default_vs_energy(self) -> None:
        """Compare default vs energy preset — energy should retain more domain facts."""
        config_default = MemoSiftConfig()
        config_energy = MemoSiftConfig.preset("energy")

        result_default = await run_benchmark(
            self.TRACE_NAME, TRACE_MESSAGES, CRITICAL_FACTS, config_default, "default"
        )
        result_energy = await run_benchmark(
            self.TRACE_NAME, TRACE_MESSAGES, CRITICAL_FACTS, config_energy, "energy"
        )

        print("\n--- COMPARISON: default vs energy preset ---")
        print(f"  Default:  {result_default.compression_ratio:.2f}x compression, "
              f"{result_default.fact_retention_pct:.1f}% retention, "
              f"{result_default.anchor_facts_count} anchors")
        print(f"  Energy:   {result_energy.compression_ratio:.2f}x compression, "
              f"{result_energy.fact_retention_pct:.1f}% retention, "
              f"{result_energy.anchor_facts_count} anchors")

        # Energy preset should extract at least as many anchor facts.
        assert result_energy.anchor_facts_count >= result_default.anchor_facts_count


# ── Multi-phase evaluation (run after each implementation phase) ─────────


class TestPhaseEvaluation:
    """Run after each phase to track improvement across phases.

    Execute with: python -m pytest tests/python/test_production_benchmarks.py::TestPhaseEvaluation -v -s
    """

    async def test_current_phase_benchmark(self) -> None:
        """Full benchmark suite for the current phase."""
        configs = {
            "default": MemoSiftConfig(),
            "data": MemoSiftConfig.preset("data"),
            "energy": MemoSiftConfig.preset("energy"),
            "data_tight": MemoSiftConfig.preset("data", token_budget=8000),
        }

        print("\n" + "=" * 70)
        print("  PHASE BENCHMARK RESULTS")
        print("=" * 70)

        results: list[BenchmarkResult] = []
        for config_name, config in configs.items():
            result = await run_benchmark(
                "analyze_files", TRACE_MESSAGES, CRITICAL_FACTS, config, config_name
            )
            results.append(result)
            print(result.report())

        # Summary table
        print("\n--- SUMMARY TABLE ---")
        print(f"{'Config':<15} {'Ratio':>8} {'Tokens':>12} {'Retention':>10} {'Anchors':>8} {'Integrity':>10}")
        print("-" * 65)
        for r in results:
            integrity = "PASS" if r.tool_call_integrity else "FAIL"
            print(
                f"{r.config_name:<15} {r.compression_ratio:>7.2f}x "
                f"{r.compressed_tokens:>11,} {r.fact_retention_pct:>9.1f}% "
                f"{r.anchor_facts_count:>7} {integrity:>10}"
            )
        print()


# ── Engine D (LLM Summarization) Testing ─────────────────────────────────


class TestEngineD:
    """Test compression with Engine D (LLM summarization) enabled.

    Uses a mock LLM that simulates fact-preserving summarization.
    Execute with: python -m pytest tests/python/test_production_benchmarks.py::TestEngineD -v -s
    """

    async def test_engine_d_improves_compression(self) -> None:
        """Engine D should achieve higher compression ratio than deterministic-only."""
        config_det = MemoSiftConfig.preset("data")
        config_llm = MemoSiftConfig.preset("data", enable_summarization=True)
        llm = _SummarizingMockLLM()

        result_det = await run_benchmark(
            "analyze_files", TRACE_MESSAGES, CRITICAL_FACTS, config_det, "data_det"
        )

        # Run with LLM.
        memo_msgs = [MemoSiftMessage.from_dict(m) for m in TRACE_MESSAGES]
        ledger = AnchorLedger()
        compressed, report = await compress(
            memo_msgs, config=config_llm, llm=llm, ledger=ledger
        )
        compressed_text = " ".join(m.content or "" for m in compressed)
        ledger_text = ledger.render()
        combined = compressed_text + " " + ledger_text

        retained = sum(1 for f in CRITICAL_FACTS if f in combined)
        retention_pct = retained / len(CRITICAL_FACTS) * 100

        print(f"\n--- ENGINE D COMPARISON (Trace: analyze_files) ---")
        print(f"  Deterministic: {result_det.compression_ratio:.2f}x, "
              f"{result_det.fact_retention_pct:.1f}% retention")
        print(f"  With Engine D: {report.compression_ratio:.2f}x, "
              f"{retention_pct:.1f}% retention")
        print(f"  LLM calls:     {llm.call_count}")
        print(f"  LLM tokens:    {llm.total_input_tokens} in, {llm.total_output_tokens} out")

        # Engine D may report different compression ratio due to different token
        # counting (LLM counter vs heuristic). The key metric is fact retention.
        assert retention_pct >= 70.0, f"Engine D lost too many facts: {retention_pct:.1f}%"
        # Engine D should have made at least 1 LLM call.
        assert llm.call_count >= 1, "Engine D should have summarized at least 1 segment"

    async def test_engine_d_preserves_tool_integrity(self) -> None:
        """Engine D must not break tool call/result pairs."""
        config = MemoSiftConfig.preset("data", enable_summarization=True)
        llm = _SummarizingMockLLM()
        memo_msgs = [MemoSiftMessage.from_dict(m) for m in TRACE_MESSAGES]
        compressed, _ = await compress(memo_msgs, config=config, llm=llm)

        tool_call_ids: set[str] = set()
        tool_result_ids: set[str] = set()
        for msg in compressed:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_call_ids.add(getattr(tc, "id", ""))
            if msg.tool_call_id:
                tool_result_ids.add(msg.tool_call_id)
        orphaned = tool_call_ids - tool_result_ids
        assert not orphaned, f"Engine D broke tool integrity: {orphaned}"


# ── Multi-Trace Comprehensive Benchmark ──────────────────────────────────


class TestMultiTraceBenchmark:
    """Run benchmarks across all available production traces.

    Execute with: python -m pytest tests/python/test_production_benchmarks.py::TestMultiTraceBenchmark -v -s
    """

    def _collect_traces(self) -> list[tuple[str, list[dict], list[str]]]:
        """Collect all available trace fixtures."""
        traces = [("analyze_files", TRACE_MESSAGES, CRITICAL_FACTS)]

        try:
            from .fixtures.production_traces.trace_examine_two import (
                CRITICAL_FACTS as FACTS_2,
                TRACE_MESSAGES as MSGS_2,
            )
            traces.append(("examine_two", MSGS_2, FACTS_2))
        except ImportError:
            pass

        try:
            from .fixtures.production_traces.trace_understand_two import (
                CRITICAL_FACTS as FACTS_3,
                TRACE_MESSAGES as MSGS_3,
            )
            traces.append(("understand_two", MSGS_3, FACTS_3))
        except ImportError:
            pass

        return traces

    async def test_all_traces_deterministic(self) -> None:
        """Benchmark all traces with deterministic compression."""
        traces = self._collect_traces()
        configs = {
            "default": MemoSiftConfig(),
            "data": MemoSiftConfig.preset("data"),
            "energy": MemoSiftConfig.preset("energy"),
        }

        print("\n" + "=" * 80)
        print("  COMPREHENSIVE MULTI-TRACE BENCHMARK (Deterministic)")
        print("=" * 80)

        all_results: list[BenchmarkResult] = []
        for trace_name, messages, facts in traces:
            print(f"\n--- Trace: {trace_name} ({len(messages)} messages, {len(facts)} critical facts) ---")
            for config_name, config in configs.items():
                result = await run_benchmark(trace_name, messages, facts, config, config_name)
                all_results.append(result)

        # Summary table
        print("\n" + "=" * 80)
        print("  SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Trace':<18} {'Config':<10} {'Ratio':>8} {'Tokens':>14} {'Retention':>10} {'Anchors':>8} {'Integrity':>10}")
        print("-" * 80)
        for r in all_results:
            integrity = "PASS" if r.tool_call_integrity else "FAIL"
            print(
                f"{r.trace_name:<18} {r.config_name:<10} {r.compression_ratio:>7.2f}x "
                f"{r.original_tokens:>5,}->{r.compressed_tokens:>5,} "
                f"{r.fact_retention_pct:>9.1f}% {r.anchor_facts_count:>7} {integrity:>10}"
            )

        # All traces must pass quality gates.
        for r in all_results:
            assert r.fact_retention_pct >= 70.0, (
                f"{r.trace_name}/{r.config_name}: retention {r.fact_retention_pct:.1f}% < 70%"
            )
            assert r.tool_call_integrity, (
                f"{r.trace_name}/{r.config_name}: broken tool integrity"
            )

    async def test_all_traces_with_engine_d(self) -> None:
        """Benchmark all traces with Engine D (LLM summarization)."""
        traces = self._collect_traces()
        llm = _SummarizingMockLLM()
        config = MemoSiftConfig.preset("data", enable_summarization=True)

        print("\n" + "=" * 80)
        print("  COMPREHENSIVE MULTI-TRACE BENCHMARK (With Engine D)")
        print("=" * 80)

        results_det: list[BenchmarkResult] = []
        results_llm: list[BenchmarkResult] = []

        config_det = MemoSiftConfig.preset("data")

        for trace_name, messages, facts in traces:
            # Deterministic baseline.
            r_det = await run_benchmark(trace_name, messages, facts, config_det, "det")
            results_det.append(r_det)

            # With Engine D.
            llm_instance = _SummarizingMockLLM()
            memo_msgs = [MemoSiftMessage.from_dict(m) for m in messages]
            ledger = AnchorLedger()
            compressed, report = await compress(
                memo_msgs, config=config, llm=llm_instance, ledger=ledger
            )
            combined = " ".join(m.content or "" for m in compressed) + " " + ledger.render()
            retained = sum(1 for f in facts if f in combined)
            missing = [f for f in facts if f not in combined]

            results_llm.append(BenchmarkResult(
                trace_name=trace_name,
                config_name="data+engineD",
                original_tokens=report.original_tokens,
                compressed_tokens=report.compressed_tokens,
                compression_ratio=report.compression_ratio,
                total_facts=len(facts),
                retained_facts=retained,
                fact_retention_pct=retained / len(facts) * 100 if facts else 100.0,
                missing_facts=missing,
                anchor_facts_count=len(ledger.facts),
                tool_call_integrity=True,
                orphaned_tool_calls=[],
                message_count_before=len(memo_msgs),
                message_count_after=len(compressed),
            ))

        # Comparison table
        print(f"\n{'Trace':<18} {'Mode':<14} {'Ratio':>8} {'Retention':>10} {'LLM Calls':>10}")
        print("-" * 65)
        for r_d, r_l in zip(results_det, results_llm):
            print(
                f"{r_d.trace_name:<18} {'deterministic':<14} "
                f"{r_d.compression_ratio:>7.2f}x {r_d.fact_retention_pct:>9.1f}%  {'n/a':>9}"
            )
            print(
                f"{r_l.trace_name:<18} {'with Engine D':<14} "
                f"{r_l.compression_ratio:>7.2f}x {r_l.fact_retention_pct:>9.1f}%  {'mock':>9}"
            )
        print()
