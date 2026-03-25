# Tests for contextual metric intelligence — domain-agnostic heuristic detection.
from __future__ import annotations

import pytest

from memosift.core.anchor_extractor import _extract_contextual_metrics


# ── Domain-agnostic detection (Layer A: heuristic, no config needed) ──────


class TestHeuristicMetricDetection:
    """Metrics should be detected by context signals, not hardcoded unit lists."""

    # Energy domain — ratio units catch these via Signal 1.
    def test_energy_gas_rate(self) -> None:
        results = _extract_contextual_metrics("Gas rate averaged 1,992.32 Mcf/d")
        values = [r[0] for r in results]
        assert any("1,992.32 Mcf/d" in v for v in values)

    def test_energy_oil_rate(self) -> None:
        results = _extract_contextual_metrics("Oil rate peaked at 1,730.4 STB/d")
        values = [r[0] for r in results]
        assert any("1,730.4 STB/d" in v for v in values)

    def test_energy_gor(self) -> None:
        results = _extract_contextual_metrics("GOR ratio of 1,528 Scf/STB")
        values = [r[0] for r in results]
        assert any("1,528 Scf/STB" in v for v in values)

    def test_energy_pressure(self) -> None:
        results = _extract_contextual_metrics("WHP was 2,573 psig at test start")
        values = [r[0] for r in results]
        assert any("2,573 psig" in v for v in values)

    def test_energy_water_gas_ratio(self) -> None:
        results = _extract_contextual_metrics("Total WGR of 802.21 STB/MMcf")
        values = [r[0] for r in results]
        assert any("802.21 STB/MMcf" in v for v in values)

    # Medical domain — never hardcoded, caught by ratio and non-common signals.
    def test_medical_glucose(self) -> None:
        results = _extract_contextual_metrics("Blood glucose was 126 mg/dL")
        values = [r[0] for r in results]
        assert any("126 mg/dL" in v for v in values)

    def test_medical_heart_rate(self) -> None:
        results = _extract_contextual_metrics("Heart rate steady at 72 BPM")
        values = [r[0] for r in results]
        assert any("72 BPM" in v for v in values)

    # Tech/SRE domain — caught by ratio units and non-common words.
    def test_tech_throughput(self) -> None:
        results = _extract_contextual_metrics("Throughput reached 12,500 req/s")
        values = [r[0] for r in results]
        assert any("12,500 req/s" in v for v in values)

    def test_tech_disk(self) -> None:
        results = _extract_contextual_metrics("Disk usage at 892 GB")
        values = [r[0] for r in results]
        assert any("892 GB" in v for v in values)

    def test_tech_latency(self) -> None:
        results = _extract_contextual_metrics("API latency p99 at 245 ms")
        values = [r[0] for r in results]
        assert any("245 ms" in v for v in values)

    # Financial domain — caught by non-common word signal.
    def test_financial_bps(self) -> None:
        results = _extract_contextual_metrics("Spread widened by 25 bps")
        values = [r[0] for r in results]
        assert any("25 bps" in v for v in values)

    # Comparison context signal.
    def test_comparison_context(self) -> None:
        results = _extract_contextual_metrics("South rate 2,261 vs North rate 1,992 Mcf/d")
        values = [r[0] for r in results]
        # Both numbers in comparison context should be extracted.
        assert len(results) >= 2

    # Markdown table signal — table context boosts score for numbers with units.
    def test_table_cell_metric(self) -> None:
        text = "| Gas Rate | 1,992.32 Mcf/d | 2,261.28 Mcf/d | +13.5% |"
        results = _extract_contextual_metrics(text)
        assert len(results) >= 2  # Both Mcf/d values detected


class TestNegativeCases:
    """Common English after numbers should NOT be extracted as metrics."""

    def test_common_noun_items(self) -> None:
        results = _extract_contextual_metrics("has 5 items in the list")
        assert len(results) == 0

    def test_common_noun_things(self) -> None:
        results = _extract_contextual_metrics("the 3 main reasons")
        assert len(results) == 0

    def test_common_options(self) -> None:
        results = _extract_contextual_metrics("about 10 different options")
        assert len(results) == 0

    def test_common_attempts(self) -> None:
        results = _extract_contextual_metrics("took 2 attempts")
        assert len(results) == 0

    def test_common_days(self) -> None:
        results = _extract_contextual_metrics("in 3 days we will finish")
        assert len(results) == 0

    def test_common_people(self) -> None:
        results = _extract_contextual_metrics("saw 15 people at the event")
        assert len(results) == 0


# ── Config pattern override (Layer B) ────────────────────────────────────


class TestConfigPatternOverride:
    """Domain-specific patterns from config should always extract."""

    def test_config_pattern_forces_extraction(self) -> None:
        # Without config patterns, "42 FOO" wouldn't be extracted
        # (FOO is not common, but the number is small and plain)
        results = _extract_contextual_metrics(
            "reading of 42 FOOBAR", config_patterns=["FOOBAR"]
        )
        values = [r[0] for r in results]
        assert any("42 FOOBAR" in v for v in values)
        # Confidence should be 0.9 for config-matched patterns.
        assert results[0][1] == 0.9

    def test_energy_preset_patterns(self) -> None:
        energy_patterns = ["Mcf/d", "psig", "bbl/d", "STB"]
        results = _extract_contextual_metrics(
            "rate was 500 bbl/d", config_patterns=energy_patterns
        )
        values = [r[0] for r in results]
        assert any("500 bbl/d" in v for v in values)

    def test_config_patterns_case_insensitive(self) -> None:
        results = _extract_contextual_metrics(
            "pressure at 2500 PSIG", config_patterns=["psig"]
        )
        values = [r[0] for r in results]
        assert any("2500 PSIG" in v for v in values)


# ── Integration with anchor extractor ────────────────────────────────────


class TestMetricAnchorIntegration:
    """Contextual metrics should appear as anchor facts."""

    def test_metrics_extracted_as_anchor_facts(self) -> None:
        from memosift.core.anchor_extractor import extract_anchors_from_message
        from memosift.core.types import MemoSiftMessage

        msg = MemoSiftMessage(
            role="assistant",
            content="Average gas rate was 1,992.32 Mcf/d for Roper North.",
        )
        facts = extract_anchors_from_message(msg, turn=1)
        metric_facts = [f for f in facts if f.content.startswith("Metric:")]
        assert len(metric_facts) >= 1
        assert any("1,992.32 Mcf/d" in f.content for f in metric_facts)

    def test_config_patterns_passed_through(self) -> None:
        from memosift.core.anchor_extractor import extract_anchors_from_message
        from memosift.core.types import MemoSiftMessage

        msg = MemoSiftMessage(
            role="assistant",
            content="Cumulative EUR estimate is 450 BOE for this well.",
        )
        facts = extract_anchors_from_message(
            msg, turn=1, metric_patterns=["BOE"]
        )
        metric_facts = [f for f in facts if "450 BOE" in f.content]
        assert len(metric_facts) >= 1

    def test_no_duplicate_with_percentage(self) -> None:
        """Contextual metrics should not duplicate existing percentage extraction."""
        from memosift.core.anchor_extractor import extract_anchors_from_message
        from memosift.core.types import MemoSiftMessage

        msg = MemoSiftMessage(
            role="assistant",
            content="Increased by 13.5% year over year.",
        )
        facts = extract_anchors_from_message(msg, turn=1)
        # 13.5% should appear exactly once via percent pattern, not duplicated.
        metric_facts = [f for f in facts if "13.5%" in f.content]
        assert len(metric_facts) == 1
