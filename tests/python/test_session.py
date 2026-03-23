# Tests for MemoSiftSession.
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from memosift.core.context_window import Pressure
from memosift.core.types import AnchorCategory, MemoSiftMessage
from memosift.report import CompressionReport
from memosift.session import MemoSiftSession


# ── Helpers ───────────────────────────────────────────────────────────────────

def openai_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, what is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Tell me more about its features."},
        {"role": "assistant", "content": "Python supports multiple paradigms."},
        {"role": "user", "content": "What about error handling?"},
    ]


def anthropic_messages() -> list[dict]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
        {"role": "user", "content": [{"type": "text", "text": "What is Python?"}]},
    ]


def memosift_messages() -> list[MemoSiftMessage]:
    return [
        MemoSiftMessage(role="system", content="You are helpful."),
        MemoSiftMessage(role="user", content="Hello"),
        MemoSiftMessage(role="assistant", content="Hi!"),
        MemoSiftMessage(role="user", content="What is 2+2?"),
    ]


# ── Constructor ───────────────────────────────────────────────────────────────

class TestConstructor:
    def test_default_session(self):
        session = MemoSiftSession()
        assert session.last_report is None
        assert session.facts == []
        assert session.system is None

    def test_preset_and_model(self):
        session = MemoSiftSession("coding", model="claude-sonnet-4-6")
        assert session._preset == "coding"
        assert session._model == "claude-sonnet-4-6"

    def test_config_overrides(self):
        session = MemoSiftSession(token_budget=50_000, recent_turns=3)
        assert session._config.token_budget == 50_000
        assert session._config.recent_turns == 3

    def test_invalid_config_override_raises(self):
        with pytest.raises(ValueError, match="Unknown config fields"):
            MemoSiftSession(nonexistent_field=True)

    def test_invalid_framework_raises(self):
        with pytest.raises(ValueError, match="Unknown framework"):
            MemoSiftSession(framework="invalid_sdk")

    def test_explicit_framework(self):
        session = MemoSiftSession(framework="anthropic")
        assert session._framework == "anthropic"
        assert session._framework_detected is True


# ── Compress ──────────────────────────────────────────────────────────────────

class TestCompress:
    async def test_openai_auto_detect(self):
        session = MemoSiftSession()
        msgs = openai_messages()
        compressed, report = await session.compress(msgs)
        assert isinstance(compressed, list)
        assert isinstance(report, CompressionReport)
        assert report.original_tokens > 0
        assert session._framework == "openai"

    async def test_anthropic_auto_detect(self):
        session = MemoSiftSession()
        msgs = anthropic_messages()
        compressed, report = await session.compress(msgs, system="You are helpful.")
        assert isinstance(compressed, list)
        assert session._framework == "anthropic"
        assert session.system is not None

    async def test_memosift_messages_passthrough(self):
        session = MemoSiftSession()
        msgs = memosift_messages()
        compressed, report = await session.compress(msgs)
        assert isinstance(compressed, list)
        assert all(isinstance(m, MemoSiftMessage) for m in compressed)
        assert session._framework == "memosift"

    async def test_explicit_framework_skips_detection(self):
        session = MemoSiftSession(framework="openai")
        msgs = openai_messages()
        compressed, _ = await session.compress(msgs)
        assert isinstance(compressed, list)
        assert session._framework == "openai"

    async def test_framework_cached_after_first_call(self):
        session = MemoSiftSession()
        await session.compress(openai_messages())
        assert session._framework == "openai"
        # Second call should reuse cached detection.
        await session.compress(openai_messages())
        assert session._framework_detected is True

    async def test_with_model_and_usage_tokens(self):
        session = MemoSiftSession("coding", model="claude-sonnet-4-6")
        msgs = openai_messages()
        compressed, report = await session.compress(msgs, usage_tokens=150_000)
        assert isinstance(report, CompressionReport)

    async def test_with_task(self):
        session = MemoSiftSession()
        msgs = openai_messages()
        compressed, report = await session.compress(msgs, task="explain Python features")
        assert isinstance(report, CompressionReport)


# ── State Persistence ─────────────────────────────────────────────────────────

class TestStatePersistence:
    async def test_ledger_accumulates_across_calls(self):
        session = MemoSiftSession("coding", model="claude-sonnet-4-6")
        msgs1 = openai_messages()
        await session.compress(msgs1)
        facts_after_first = len(session.facts)

        msgs2 = [
            {"role": "user", "content": "Fix the bug in auth.py on line 42"},
            {"role": "assistant", "content": "I found the error: TypeError at line 42."},
            {"role": "user", "content": "Great, now what about tests?"},
        ]
        await session.compress(msgs2)
        # Ledger should accumulate — may have more facts.
        assert len(session.facts) >= facts_after_first

    async def test_expand_returns_none_for_unknown_index(self):
        session = MemoSiftSession()
        assert session.expand(999) is None

    async def test_last_report_updates(self):
        session = MemoSiftSession()
        assert session.last_report is None
        await session.compress(openai_messages())
        assert session.last_report is not None
        first_report = session.last_report
        await session.compress(openai_messages())
        assert session.last_report is not first_report


# ── Check Pressure ────────────────────────────────────────────────────────────

class TestCheckPressure:
    def test_no_model_returns_none_pressure(self):
        session = MemoSiftSession()
        assert session.check_pressure() == Pressure.NONE

    def test_low_usage_returns_none(self):
        session = MemoSiftSession(model="claude-sonnet-4-6")
        assert session.check_pressure(usage_tokens=10_000) == Pressure.NONE

    def test_high_usage_returns_high_or_critical(self):
        session = MemoSiftSession(model="claude-haiku-4-5")
        result = session.check_pressure(usage_tokens=175_000)
        assert result in (Pressure.HIGH, Pressure.CRITICAL)


# ── Reconfigure ───────────────────────────────────────────────────────────────

class TestReconfigure:
    async def test_reconfigure_preserves_ledger(self):
        session = MemoSiftSession("coding", model="claude-sonnet-4-6")
        await session.compress(openai_messages())
        facts_before = len(session.facts)

        session.reconfigure("research", token_budget=10_000)
        assert session._config.token_budget == 10_000
        assert session._preset == "research"
        # Ledger should be preserved.
        assert len(session.facts) == facts_before

    async def test_reconfigure_changes_config(self):
        session = MemoSiftSession()
        session.reconfigure(recent_turns=10)
        assert session._config.recent_turns == 10

    def test_reconfigure_invalid_field_raises(self):
        session = MemoSiftSession()
        with pytest.raises(ValueError, match="Unknown config fields"):
            session.reconfigure(bad_field=True)

    async def test_reconfigure_preserves_cache(self):
        session = MemoSiftSession()
        await session.compress(openai_messages())
        cache_size = session._cache.size
        session.reconfigure("support")
        assert session._cache.size == cache_size


# ── Save / Load ───────────────────────────────────────────────────────────────

class TestSaveLoad:
    async def test_save_and_load_round_trip(self):
        session = MemoSiftSession("coding", model="claude-sonnet-4-6", framework="openai")
        await session.compress(openai_messages())
        facts_before = len(session.facts)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            session.save_state(path)

            # Verify JSON structure.
            data = json.loads(Path(path).read_text())
            assert "ledger" in data
            assert "cross_window_hashes" in data
            assert data["framework"] == "openai"
            assert data["model"] == "claude-sonnet-4-6"
            assert data["config_preset"] == "coding"

            # Load into new session.
            restored = MemoSiftSession.load_state(path)
            assert len(restored.facts) == facts_before
            assert restored._framework == "openai"
            assert restored._model == "claude-sonnet-4-6"
        finally:
            Path(path).unlink(missing_ok=True)

    async def test_expand_after_load_returns_none(self):
        """Cache is not persisted — expand should return None after load."""
        session = MemoSiftSession(framework="openai")
        await session.compress(openai_messages())

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            session.save_state(path)
            restored = MemoSiftSession.load_state(path)
            # Cache was not serialized.
            assert restored.expand(0) is None
            assert restored._cache.size == 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_with_overrides(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({
                "ledger": {"facts": []},
                "cross_window_hashes": [],
                "framework": "openai",
                "model": "gpt-4o",
                "config_preset": "general",
            }, f)
            path = f.name

        try:
            restored = MemoSiftSession.load_state(
                path, preset="coding", model="claude-sonnet-4-6", token_budget=50_000
            )
            assert restored._model == "claude-sonnet-4-6"
            assert restored._config.token_budget == 50_000
        finally:
            Path(path).unlink(missing_ok=True)
