# Heuristic token counter — fallback when no LLM provider is available.
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memosift.providers.base import LLMResponse


class HeuristicTokenCounter:
    """Fallback token counter using character-length heuristic.

    Average ~3.5 characters per BPE token for English text.
    Accuracy is ±15%, sufficient for budget enforcement.

    This class satisfies the ``MemoSiftLLMProvider`` protocol's ``count_tokens``
    method but raises on ``generate()`` since it has no LLM backing.
    """

    chars_per_token: float

    def __init__(self, chars_per_token: float = 3.5) -> None:
        self.chars_per_token = chars_per_token

    async def count_tokens(self, text: str) -> int:
        """Estimate token count using character-length heuristic."""
        if not text:
            return 0
        return math.ceil(len(text) / self.chars_per_token)

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Not supported — HeuristicTokenCounter has no LLM backing.

        Raises:
            NotImplementedError: Always. Use a real LLM provider for generation.
        """
        raise NotImplementedError(
            "HeuristicTokenCounter does not support text generation. "
            "Provide a real MemoSiftLLMProvider to use LLM-dependent features "
            "(summarization, LLM relevance scoring)."
        )
