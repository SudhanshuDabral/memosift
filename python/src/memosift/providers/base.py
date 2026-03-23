# MemoSiftLLMProvider protocol — the dependency-injection contract for LLM access.
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    """Response from a MemoSiftLLMProvider.generate() call."""

    text: str
    input_tokens: int
    output_tokens: int


class MemoSiftLLMProvider(Protocol):
    """The contract that any LLM must satisfy to power MemoSift's LLM-dependent layers.

    Intentionally minimal — a single ``generate()`` method plus ``count_tokens()``.
    Framework adapters implement this by wrapping the host LLM.

    When no provider is supplied, MemoSift operates in deterministic-only mode:
    Layers 3D (summarization) and Layer 4 (LLM scoring) are automatically disabled.
    """

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate text from the LLM.

        Args:
            prompt: The prompt to send.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            LLMResponse with the generated text and token usage.
        """
        ...

    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in ``text``.

        Implementations should use the model's native tokenizer when available,
        falling back to a heuristic (``len(text) / 3.5``) otherwise.
        """
        ...
