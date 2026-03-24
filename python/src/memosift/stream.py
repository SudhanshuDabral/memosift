# MemoSiftStream — real-time compression stream, process messages as they arrive.
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memosift.core.context_window import Pressure, estimate_tokens_heuristic
from memosift.session import MemoSiftSession

if TYPE_CHECKING:
    from memosift.core.types import AnchorFact


@dataclass(frozen=True)
class StreamEvent:
    """Result of pushing a message to the stream.

    Attributes:
        action: Either "buffered" (no compression triggered) or "compressed".
        compressed: Whether compression was triggered on this push.
        tokens_saved: Tokens saved by compression (0 if not compressed).
        pressure: The current context pressure level.
    """

    action: str
    compressed: bool
    tokens_saved: int = 0
    pressure: Pressure = Pressure.NONE


class MemoSiftStream:
    """Real-time compression stream — process messages as they arrive.

    Wraps a ``MemoSiftSession`` with incremental mode enabled.
    Messages are buffered until context pressure warrants compression,
    at which point compression runs only on new messages (Zone 3).

    Usage::

        stream = MemoSiftStream("coding", model="claude-sonnet-4-6")

        for message in incoming_messages:
            event = await stream.push(message)
            if event.compressed:
                print(f"Saved {event.tokens_saved} tokens")

        # Get current compressed state
        compressed = stream.messages
        facts = stream.facts
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
        """Create a new compression stream.

        Args:
            preset: Config preset name.
            model: Model name for context window lookup + adaptive compression.
            llm: Optional ``MemoSiftLLMProvider``.
            framework: Explicit framework. Auto-detected if omitted.
            **config_overrides: Config field overrides.
        """
        self._session = MemoSiftSession(
            preset,
            model=model,
            llm=llm,
            framework=framework,
            incremental=True,
            **config_overrides,
        )
        self._messages: list[Any] = []

    async def push(self, message: Any) -> StreamEvent:
        """Push a new message and get a compression decision.

        The message is appended to the internal buffer. Compression is triggered
        only when context pressure exceeds NONE (i.e., the model's context window
        is under meaningful load).

        Args:
            message: A framework-native message (dict, object, etc.).

        Returns:
            A ``StreamEvent`` indicating whether compression was triggered.
        """
        self._messages.append(message)

        # Check if compression is needed based on pressure.
        pressure = self._session.check_pressure(
            usage_tokens=self._estimate_tokens()
        )
        if pressure == Pressure.NONE:
            return StreamEvent(action="buffered", compressed=False, pressure=pressure)

        # Compress.
        compressed, report = await self._session.compress(self._messages)
        self._messages = list(compressed)
        return StreamEvent(
            action="compressed",
            compressed=True,
            tokens_saved=report.tokens_saved,
            pressure=pressure,
        )

    async def flush(self) -> StreamEvent:
        """Force compression regardless of pressure.

        Useful at the end of a conversation or before a long pause.

        Returns:
            A ``StreamEvent`` with compression results.
        """
        if not self._messages:
            return StreamEvent(action="buffered", compressed=False)

        compressed, report = await self._session.compress(self._messages)
        self._messages = list(compressed)
        pressure = self._session.check_pressure(
            usage_tokens=self._estimate_tokens()
        )
        return StreamEvent(
            action="compressed",
            compressed=True,
            tokens_saved=report.tokens_saved,
            pressure=pressure,
        )

    @property
    def messages(self) -> list[Any]:
        """Current message state (may include compressed messages)."""
        return list(self._messages)

    @property
    def facts(self) -> list[AnchorFact]:
        """Accumulated anchor facts."""
        return self._session.facts

    @property
    def session(self) -> MemoSiftSession:
        """The underlying session (for advanced configuration)."""
        return self._session

    @property
    def message_count(self) -> int:
        """Number of messages in the buffer."""
        return len(self._messages)

    def _estimate_tokens(self) -> int:
        """Estimate total token count of buffered messages."""
        contents: list[str] = []
        for msg in self._messages:
            if hasattr(msg, "content"):
                contents.append(str(msg.content))
            elif isinstance(msg, dict):
                c = msg.get("content", "")
                contents.append(str(c) if not isinstance(c, list) else str(c))
            else:
                contents.append(str(msg))
        return estimate_tokens_heuristic(contents)
