# MemoSiftSession — stateful compression session, the recommended entry point.
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memosift.config import MemoSiftConfig
from memosift.core.context_window import ContextWindowState, Pressure, estimate_tokens_heuristic
from memosift.core.pipeline import CompressionCache, compress
from memosift.core.types import AnchorFact, AnchorLedger, MemoSiftMessage
from memosift.detect import VALID_FRAMEWORKS, detect_framework

if TYPE_CHECKING:
    from memosift.report import CompressionReport

logger = logging.getLogger("memosift")

# Config fields that can be overridden via **kwargs.
_CONFIG_FIELDS = frozenset(f.name for f in dataclasses.fields(MemoSiftConfig))


class MemoSiftSession:
    """Stateful compression session — owns ledger, dedup state, and cache.

    Collapses the 7 objects + 51 knobs of the raw ``compress()`` API into
    a single constructor + a single ``compress()`` call.

    **Before** (raw API)::

        config = MemoSiftConfig.preset("coding")
        ledger = AnchorLedger()
        cross_window = CrossWindowState()
        cache = CompressionCache()
        ctx = ContextWindowState.from_model("claude-sonnet-4-6", current_usage_tokens=150_000)
        compressed, report = await compress(messages, config=config, ledger=ledger,
            cross_window=cross_window, cache=cache, context_window=ctx)

    **After** (session)::

        session = MemoSiftSession("coding", model="claude-sonnet-4-6")
        compressed, report = await session.compress(messages, usage_tokens=150_000)

    The session persists ``AnchorLedger``, ``CrossWindowState``, and
    ``CompressionCache`` across multiple ``compress()`` calls. Config is
    immutable per construction but can be swapped via ``reconfigure()``.
    """

    def __init__(
        self,
        preset: str = "general",
        *,
        model: str | None = None,
        llm: Any = None,
        framework: str | None = None,
        **config_overrides: Any,
    ) -> None:
        """Create a new compression session.

        Args:
            preset: Config preset name ("coding", "research", "support", "data", "general").
            model: Model name for context window lookup + adaptive compression.
            llm: Optional ``MemoSiftLLMProvider`` for summarization/scoring.
            framework: Explicit framework ("openai", "anthropic", "agent_sdk", "adk",
                "langchain"). Auto-detected from messages if omitted.
            **config_overrides: Any ``MemoSiftConfig`` field to override on the preset
                (e.g., ``token_budget=50_000``, ``recent_turns=3``).

        Raises:
            ValueError: If an unknown config field is passed or framework is invalid.
        """
        # Validate config overrides.
        unknown = set(config_overrides) - _CONFIG_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown config fields: {sorted(unknown)}. Valid fields: {sorted(_CONFIG_FIELDS)}"
            )

        # Validate framework.
        if framework is not None and framework not in VALID_FRAMEWORKS:
            raise ValueError(f"Unknown framework {framework!r}. Valid: {sorted(VALID_FRAMEWORKS)}")

        self._model = model
        self._llm = llm
        self._framework: str | None = framework
        self._framework_detected = framework is not None
        self._preset = preset

        # Build config from preset + overrides.
        base = MemoSiftConfig.preset(preset) if preset != "general" else MemoSiftConfig()
        if config_overrides:
            self._config = dataclasses.replace(base, **config_overrides)
        else:
            self._config = base

        # Persistent state across compress() calls.
        self._ledger = AnchorLedger()
        from memosift.core.deduplicator import CrossWindowState

        self._cross_window = CrossWindowState()
        self._cache = CompressionCache()

        # Per-call state.
        self._last_report: CompressionReport | None = None
        self._system: str | None = None

    async def compress(
        self,
        messages: list[Any],
        *,
        task: str | None = None,
        usage_tokens: int | None = None,
        system: str | None = None,
    ) -> tuple[Any, CompressionReport]:
        """Compress messages through the pipeline.

        Accepts framework-native messages (auto-detected or per ``framework``
        param on constructor). Returns compressed messages in the same format.

        Args:
            messages: Framework-native messages.
            task: Task description for relevance scoring.
            usage_tokens: Current token usage for adaptive Layer 0.
                Creates a ``ContextWindowState`` from ``model`` + this value.
            system: Anthropic system prompt (separate from messages).

        Returns:
            Tuple of (compressed_messages, CompressionReport).
        """
        # ── Framework detection (cached after first call) ──
        if not self._framework_detected:
            self._framework = detect_framework(messages)
            self._framework_detected = True

        # ── Adapt in ──
        internal = self._adapt_in(messages, system)

        # ── Build context window state ──
        context_window: ContextWindowState | None = None
        if self._model is not None:
            tokens = (
                usage_tokens
                if usage_tokens is not None
                else (estimate_tokens_heuristic([m.content for m in internal]))
            )
            context_window = ContextWindowState.from_model(self._model, tokens)

        # ── Compress ──
        compressed_internal, report = await compress(
            internal,
            llm=self._llm,
            config=self._config,
            task=task,
            ledger=self._ledger,
            cross_window=self._cross_window,
            cache=self._cache,
            context_window=context_window,
        )

        self._last_report = report

        # ── Adapt out ──
        result = self._adapt_out(compressed_internal, system)
        return result, report

    def _adapt_in(self, messages: list[Any], system: str | None) -> list[MemoSiftMessage]:
        """Convert framework-native messages to MemoSiftMessage list."""
        fw = self._framework

        if fw == "memosift":
            return list(messages)

        if fw == "openai":
            from memosift.adapters.openai_sdk import adapt_in

            return adapt_in(messages)

        if fw == "anthropic":
            from memosift.adapters.anthropic_sdk import adapt_in

            return adapt_in(messages, system)

        if fw == "agent_sdk":
            from memosift.adapters.claude_agent_sdk import adapt_in

            return adapt_in(messages)

        if fw == "adk":
            from memosift.adapters.google_adk import adapt_in

            return adapt_in(messages)

        if fw == "langchain":
            from memosift.adapters.langchain import adapt_in

            return adapt_in(messages)

        # Fallback — treat as OpenAI.
        from memosift.adapters.openai_sdk import adapt_in

        return adapt_in(messages)

    def _adapt_out(self, messages: list[MemoSiftMessage], system: str | None) -> Any:
        """Convert MemoSiftMessage list back to framework-native format."""
        fw = self._framework

        if fw == "memosift":
            return messages

        if fw == "openai":
            from memosift.adapters.openai_sdk import adapt_out

            return adapt_out(messages)

        if fw == "anthropic":
            from memosift.adapters.anthropic_sdk import adapt_out

            result = adapt_out(messages)
            self._system = result.system
            return result.messages

        if fw == "agent_sdk":
            from memosift.adapters.claude_agent_sdk import adapt_out

            return adapt_out(messages)

        if fw == "adk":
            from memosift.adapters.google_adk import adapt_out

            return adapt_out(messages)

        if fw == "langchain":
            from memosift.adapters.langchain import adapt_out

            return adapt_out(messages)

        from memosift.adapters.openai_sdk import adapt_out

        return adapt_out(messages)

    def check_pressure(self, usage_tokens: int | None = None) -> Pressure:
        """Check current context window pressure without compressing.

        Args:
            usage_tokens: Override token usage. If None, returns NONE when
                no model is configured.
        """
        if self._model is None:
            return Pressure.NONE
        tokens = usage_tokens if usage_tokens is not None else 0
        state = ContextWindowState.from_model(self._model, tokens)
        return state.pressure

    @property
    def model(self) -> str | None:
        """The model name this session was created with."""
        return self._model

    @property
    def preset(self) -> str:
        """The current config preset name."""
        return self._preset

    @property
    def framework(self) -> str | None:
        """The detected or configured framework."""
        return self._framework

    def set_framework(self, framework: str) -> None:
        """Set the framework explicitly (skips auto-detection)."""
        if framework not in VALID_FRAMEWORKS:
            raise ValueError(f"Unknown framework {framework!r}. Valid: {sorted(VALID_FRAMEWORKS)}")
        self._framework = framework
        self._framework_detected = True

    @property
    def ledger(self) -> AnchorLedger:
        """The session's anchor ledger (accumulates facts across compress calls)."""
        return self._ledger

    @property
    def facts(self) -> list[AnchorFact]:
        """Shortcut: all extracted anchor facts."""
        return self._ledger.facts

    @property
    def last_report(self) -> CompressionReport | None:
        """Compression report from the most recent compress() call."""
        return self._last_report

    @property
    def system(self) -> str | None:
        """Anthropic system prompt from the most recent compression.

        None for non-Anthropic frameworks.
        """
        return self._system

    def expand(self, original_index: int) -> str | None:
        """Re-expand a previously compressed message.

        Original content is only available within the same session lifecycle.
        Cache is not persisted across ``save_state()``/``load_state()``.

        Args:
            original_index: Message index from the compression report decisions.

        Returns:
            Original content string, or None if not cached.
        """
        return self._cache.expand(original_index)

    def reconfigure(self, preset: str | None = None, **config_overrides: Any) -> None:
        """Change config while preserving session state (ledger, dedup, cache).

        Args:
            preset: New preset name. If None, keeps the current preset.
            **config_overrides: Config fields to override.

        Raises:
            ValueError: If an unknown config field is passed.
        """
        unknown = set(config_overrides) - _CONFIG_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown config fields: {sorted(unknown)}. Valid fields: {sorted(_CONFIG_FIELDS)}"
            )

        if preset is not None:
            self._preset = preset
            base = MemoSiftConfig.preset(preset) if preset != "general" else MemoSiftConfig()
        else:
            base = self._config

        if config_overrides:
            self._config = dataclasses.replace(base, **config_overrides)
        else:
            self._config = base

    def save_state(self, path: str) -> None:
        """Persist session state (ledger + dedup hashes) to a JSON file.

        The ``CompressionCache`` is NOT serialized — original content is only
        available within the same session lifecycle. This is intentional:
        caching originals could balloon the state file.

        Args:
            path: File path to write to.
        """
        state = {
            "version": 1,
            "ledger": {
                "facts": [
                    {
                        "category": (
                            f.category.value if hasattr(f.category, "value") else f.category
                        ),
                        "content": f.content,
                        "turn": f.turn,
                        "confidence": f.confidence,
                    }
                    for f in self._ledger.facts
                ],
            },
            "cross_window_hashes": sorted(self._cross_window.content_hashes),
            "framework": self._framework,
            "model": self._model,
            "config_preset": self._preset,
        }
        Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load_state(
        cls,
        path: str,
        preset: str = "general",
        *,
        model: str | None = None,
        llm: Any = None,
        **config_overrides: Any,
    ) -> MemoSiftSession:
        """Restore a session from saved state.

        Loads the anchor ledger and cross-window dedup hashes. The
        ``CompressionCache`` is NOT restored (see ``save_state`` docstring).

        Args:
            path: File path to read from.
            preset: Config preset (overrides saved preset if provided).
            model: Model name (overrides saved model if provided).
            llm: Optional LLM provider.
            **config_overrides: Config field overrides.

        Returns:
            A new MemoSiftSession with restored state.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        effective_preset = preset if preset != "general" else data.get("config_preset", "general")
        effective_model = model or data.get("model")
        effective_framework = data.get("framework")

        session = cls(
            effective_preset,
            model=effective_model,
            llm=llm,
            framework=effective_framework,
            **config_overrides,
        )

        # Restore ledger facts (defensive: skip malformed entries).
        from memosift.core.types import AnchorCategory, AnchorFact

        for fact_data in data.get("ledger", {}).get("facts", []):
            try:
                fact = AnchorFact(
                    category=AnchorCategory(fact_data["category"]),
                    content=fact_data["content"],
                    turn=fact_data.get("turn", 0),
                    confidence=fact_data.get("confidence", 1.0),
                )
                session._ledger.add(fact)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping malformed fact in state file: %s", exc)

        # Restore cross-window hashes.
        for h in data.get("cross_window_hashes", []):
            session._cross_window.content_hashes.add(h)

        return session
