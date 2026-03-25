# Failure-driven compression feedback — learns from compression failures.
#
# When the agent discovers a fact was lost during compression, the caller
# reports it via CompressionFeedback.report_missing(). The feedback store
# accumulates "do not compress" patterns that are applied as Shield overrides
# in future compression cycles.
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CompressionFeedback:
    """Accumulates feedback about compression failures for future cycles.

    When the agent needs a fact that was compressed away, call
    ``report_missing(fact_text)`` to teach the system to protect similar
    content in future cycles.

    The ``protection_patterns`` property returns a set of strings that should
    be treated as critical by the importance scorer and budget enforcer.
    """

    _missing_facts: list[str] = field(default_factory=list)
    _protection_strings: set[str] = field(default_factory=set)

    def report_missing(self, fact_text: str) -> None:
        """Report a fact that was needed but lost during compression.

        Extracts key tokens and numbers from the fact text and adds them
        to the protection set for future cycles.
        """
        self._missing_facts.append(fact_text)

        # Extract tokens worth protecting.
        # Numbers with context.
        for match in re.finditer(r"\b\d[\d,.]*(?:\.\d+)?\s*\S{0,12}", fact_text):
            self._protection_strings.add(match.group(0).strip())
        # Capitalized entity names.
        for match in re.finditer(r"\b[A-Z][A-Za-z]{2,20}(?:\s+[A-Z][A-Za-z]{2,20})*\b", fact_text):
            self._protection_strings.add(match.group(0))

    @property
    def protection_patterns(self) -> frozenset[str]:
        """Return the set of strings that should be protected in future cycles."""
        return frozenset(self._protection_strings)

    @property
    def missing_count(self) -> int:
        """Number of missing facts reported."""
        return len(self._missing_facts)

    def clear(self) -> None:
        """Reset all feedback."""
        self._missing_facts.clear()
        self._protection_strings.clear()
