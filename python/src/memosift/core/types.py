# Core type definitions for the MemoSift compression pipeline.
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class ContentType(StrEnum):
    """Content type assigned to each message by Layer 1 (Classifier)."""

    SYSTEM_PROMPT = "SYSTEM_PROMPT"
    USER_QUERY = "USER_QUERY"
    RECENT_TURN = "RECENT_TURN"
    TOOL_RESULT_JSON = "TOOL_RESULT_JSON"
    TOOL_RESULT_TEXT = "TOOL_RESULT_TEXT"
    CODE_BLOCK = "CODE_BLOCK"
    ERROR_TRACE = "ERROR_TRACE"
    ASSISTANT_REASONING = "ASSISTANT_REASONING"
    OLD_CONVERSATION = "OLD_CONVERSATION"
    PREVIOUSLY_COMPRESSED = "PREVIOUSLY_COMPRESSED"


class CompressionPolicy(StrEnum):
    """Compression policy that controls how aggressively a message is compressed."""

    PRESERVE = "PRESERVE"  # Never compress (system prompt, current query)
    LIGHT = "LIGHT"  # Dedup only, no deletion (recent turns)
    MODERATE = "MODERATE"  # Verbatim deletion of noise lines
    STRUCTURAL = "STRUCTURAL"  # Schema-aware / signature-only compression
    STACK = "STACK"  # Keep first + last frames, collapse middle
    AGGRESSIVE = "AGGRESSIVE"  # Summarize or heavy deletion
    SIGNATURE = "SIGNATURE"  # Keep function/class signatures, collapse bodies


class Shield(StrEnum):
    """Importance shield level assigned by L3G importance scoring.

    Controls how aggressively Phase 3 layers can compress a segment.
    """

    PRESERVE = "PRESERVE"  # High importance — do not compress further
    MODERATE = "MODERATE"  # Medium importance — limited compression allowed
    COMPRESSIBLE = "COMPRESSIBLE"  # Low importance — aggressive compression OK


# Default policy for each content type.
DEFAULT_POLICIES: dict[ContentType, CompressionPolicy] = {
    ContentType.SYSTEM_PROMPT: CompressionPolicy.PRESERVE,
    ContentType.USER_QUERY: CompressionPolicy.PRESERVE,
    ContentType.RECENT_TURN: CompressionPolicy.LIGHT,
    ContentType.TOOL_RESULT_JSON: CompressionPolicy.STRUCTURAL,
    ContentType.TOOL_RESULT_TEXT: CompressionPolicy.MODERATE,
    ContentType.CODE_BLOCK: CompressionPolicy.SIGNATURE,
    ContentType.ERROR_TRACE: CompressionPolicy.STACK,
    ContentType.ASSISTANT_REASONING: CompressionPolicy.AGGRESSIVE,
    ContentType.OLD_CONVERSATION: CompressionPolicy.AGGRESSIVE,
    ContentType.PREVIOUSLY_COMPRESSED: CompressionPolicy.PRESERVE,
}


@dataclass
class ToolCall:
    """A single tool invocation within an assistant message."""

    id: str
    function: ToolCallFunction
    type: str = "function"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict matching the JSON schema."""
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        """Deserialize from a plain dict."""
        return cls(
            id=data["id"],
            type=data.get("type", "function"),
            function=ToolCallFunction.from_dict(data["function"]),
        )


@dataclass
class ToolCallFunction:
    """The function portion of a tool call."""

    name: str
    arguments: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to a plain dict."""
        return {"name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCallFunction:
        """Deserialize from a plain dict."""
        return cls(name=data["name"], arguments=data["arguments"])


@dataclass
class MemoSiftMessage:
    """Universal message representation used by the compression pipeline.

    Framework adapters convert native messages to/from this format.
    The ``metadata`` dict preserves framework-specific data for lossless round-tripping.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str

    # Optional fields preserved through compression.
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Set by MemoSift during compression (output annotations).
    _memosift_content_type: str | None = field(default=None, repr=False)
    _memosift_compressed: bool = field(default=False, repr=False)
    _memosift_original_tokens: int | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict matching the JSON schema."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoSiftMessage:
        """Deserialize from a plain dict."""
        tool_calls = None
        if data.get("tool_calls") is not None:
            tool_calls = [ToolCall.from_dict(tc) for tc in data["tool_calls"]]
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=tool_calls,
            metadata=data.get("metadata", {}),
        )


@dataclass
class ClassifiedMessage:
    """A MemoSiftMessage annotated with classification and pipeline state.

    Created by Layer 1 (Classifier) and enriched by subsequent layers.
    Every layer from L2 onward accepts and returns ``list[ClassifiedMessage]``.
    """

    message: MemoSiftMessage
    content_type: ContentType
    policy: CompressionPolicy
    original_index: int = 0

    # Populated by Layer 4 (Scorer). Default 0.5 = neutral.
    relevance_score: float = 0.5

    # Populated by token counting (heuristic or LLM provider).
    estimated_tokens: int = 0

    # True for segments that must never be dropped (SYSTEM_PROMPT, USER_QUERY, RECENT_TURN).
    protected: bool = False

    # Populated by L3G (Importance Scoring). Default 0.5 = neutral.
    importance_score: float = 0.5

    # Shield level assigned by L3G. Default MODERATE = standard compression.
    shield: Shield = Shield.MODERATE

    @property
    def role(self) -> str:
        """Shortcut to the underlying message role."""
        return self.message.role

    @property
    def content(self) -> str:
        """Shortcut to the underlying message content. Returns '' for None."""
        return self.message.content or ""

    @content.setter
    def content(self, value: str) -> None:
        """Allow layers to modify content through the wrapper."""
        self.message.content = value


@dataclass
class DependencyMap:
    """Tracks dedup back-references AND logical reasoning dependencies.

    Two edge types:
    - **Dedup references**: message B is a dedup copy of A → dropping A breaks B.
    - **Logical dependencies**: message B builds on reasoning from A ("therefore",
      "so we can", "building on that") → dropping A breaks reasoning chain.

    Both edge types are checked by Layer 6 before dropping any message.
    """

    references: dict[int, int] = field(default_factory=dict)
    logical_deps: dict[int, int] = field(default_factory=dict)

    def add(self, deduped_index: int, original_index: int) -> None:
        """Record that ``deduped_index`` references ``original_index`` (dedup edge)."""
        self.references[deduped_index] = original_index

    def add_logical(self, dependent_index: int, dependency_index: int) -> None:
        """Record that ``dependent_index`` builds on reasoning from ``dependency_index``."""
        self.logical_deps[dependent_index] = dependency_index

    def can_drop(self, index: int) -> bool:
        """Return True if no surviving message depends on this index (dedup or logical)."""
        if index in self.references.values():
            return False
        return index not in self.logical_deps.values()

    def has_dependents(self, index: int) -> bool:
        """Return True if any message references this index (dedup edge)."""
        return index in self.references.values()

    def has_logical_dependents(self, index: int) -> bool:
        """Return True if any message builds on reasoning from this index."""
        return index in self.logical_deps.values()

    def dependents_of(self, index: int) -> list[int]:
        """Return indices of messages that reference ``index`` (dedup edges)."""
        return [k for k, v in self.references.items() if v == index]

    def logical_dependents_of(self, index: int) -> list[int]:
        """Return indices of messages that build on reasoning from ``index``."""
        return [k for k, v in self.logical_deps.items() if v == index]


# ── Anchor Ledger types ─────────────────────────────────────────────────────


class AnchorCategory(StrEnum):
    """Category of a fact in the Anchor Ledger.

    The five primary sections (INTENT, FILES, DECISIONS, ERRORS, ACTIVE_CONTEXT)
    form the structured anchor ledger used as a compression vehicle.
    IDENTIFIERS, OUTCOMES, and OPEN_ITEMS provide supplementary fact storage.
    """

    INTENT = "INTENT"
    FILES = "FILES"
    DECISIONS = "DECISIONS"
    ERRORS = "ERRORS"
    ACTIVE_CONTEXT = "ACTIVE_CONTEXT"
    IDENTIFIERS = "IDENTIFIERS"
    OUTCOMES = "OUTCOMES"
    OPEN_ITEMS = "OPEN_ITEMS"
    PARAMETERS = "PARAMETERS"
    CONSTRAINTS = "CONSTRAINTS"
    ASSUMPTIONS = "ASSUMPTIONS"
    DATA_SCHEMA = "DATA_SCHEMA"
    RELATIONSHIPS = "RELATIONSHIPS"


# Section headers for the structured anchor ledger render().
_LEDGER_SECTION_HEADERS: dict[AnchorCategory, str] = {
    AnchorCategory.INTENT: "## SESSION INTENT",
    AnchorCategory.FILES: "## FILES TOUCHED",
    AnchorCategory.DECISIONS: "## KEY DECISIONS",
    AnchorCategory.ERRORS: "## ERRORS ENCOUNTERED",
    AnchorCategory.ACTIVE_CONTEXT: "## ACTIVE CONTEXT",
    AnchorCategory.IDENTIFIERS: "## IDENTIFIERS",
    AnchorCategory.OUTCOMES: "## OUTCOMES",
    AnchorCategory.OPEN_ITEMS: "## OPEN ITEMS",
    AnchorCategory.PARAMETERS: "## PARAMETERS",
    AnchorCategory.CONSTRAINTS: "## CONSTRAINTS",
    AnchorCategory.ASSUMPTIONS: "## ASSUMPTIONS",
    AnchorCategory.DATA_SCHEMA: "## DATA SCHEMA",
    AnchorCategory.RELATIONSHIPS: "## RELATIONSHIPS",
}

# Primary sections displayed first in the structured ledger.
_LEDGER_PRIMARY_SECTIONS: tuple[AnchorCategory, ...] = (
    AnchorCategory.INTENT,
    AnchorCategory.FILES,
    AnchorCategory.DECISIONS,
    AnchorCategory.ERRORS,
    AnchorCategory.ACTIVE_CONTEXT,
)


@dataclass(frozen=True)
class AnchorFact:
    """A single critical fact extracted from the conversation.

    Immutable once created. Stored in the ``AnchorLedger``.
    """

    category: AnchorCategory
    content: str
    turn: int
    confidence: float = 1.0


@dataclass
class AnchorLedger:
    """Append-only ledger of critical facts extracted before compression.

    Never compressed. Prepended to every LLM call as session memory.
    The ledger is a mutable accumulator — the caller passes it into
    ``compress()`` and expects facts to be added across cycles.
    """

    facts: list[AnchorFact] = field(default_factory=list)
    _seen_hashes: set[str] = field(default_factory=set)

    def add(self, fact: AnchorFact) -> bool:
        """Add a fact if it's not a duplicate. Returns False if duplicate."""
        content_hash = hashlib.sha256(fact.content.encode("utf-8")).hexdigest()[:32]
        if content_hash in self._seen_hashes:
            return False
        self._seen_hashes.add(content_hash)
        self.facts.append(fact)
        self._invalidate_cache()
        return True

    def update(self, category: AnchorCategory, old_content: str, new_content: str) -> None:
        """Update a fact's content (e.g., file 'read' → 'modified').

        Replaces the first matching fact with an updated copy.
        """
        for i, fact in enumerate(self.facts):
            if fact.category == category and fact.content == old_content:
                self.facts[i] = AnchorFact(
                    category=fact.category,
                    content=new_content,
                    turn=fact.turn,
                    confidence=fact.confidence,
                )
                # Update the hash set.
                old_hash = hashlib.sha256(old_content.encode("utf-8")).hexdigest()[:32]
                new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()[:32]
                self._seen_hashes.discard(old_hash)
                self._seen_hashes.add(new_hash)
                self._invalidate_cache()
                return

    def get_protected_strings(self) -> frozenset[str]:
        """Return the set of core identifiers from all facts.

        Extracts the key substring from each fact's content:
        - FILES: the file path before " —" (e.g., "src/auth.ts" from "src/auth.ts — read at turn 2")
        - ERRORS: the error string before " —" or full content
        - IDENTIFIERS: the identifier after "Tool used: ", "Code entity: ", "Reference: ", etc.
        - DECISIONS/TASK/OUTCOMES/OPEN_ITEMS: full content before " —"

        Cached internally. Invalidated when ``add()`` or ``update()`` is called.
        """
        if hasattr(self, "_protected_cache") and self._protected_cache is not None:
            return self._protected_cache

        strings: set[str] = set()
        for fact in self.facts:
            core = fact.content.split(" \u2014 ")[0].split(" — ")[0].strip()

            if fact.category == AnchorCategory.IDENTIFIERS:
                # Extract the value after the label.
                for prefix in ("Tool used: ", "Code entity: ", "Reference: "):
                    if core.startswith(prefix):
                        core = core[len(prefix) :]
                        break

            # Skip very short strings (< 3 chars) — too likely to false-match.
            if len(core) >= 3:
                strings.add(core)

            # For FILES, also add just the filename (last path component).
            if fact.category == AnchorCategory.FILES and ("/" in core or "\\" in core):
                filename = core.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
                if len(filename) >= 3:
                    strings.add(filename)

        result = frozenset(strings)
        self._protected_cache: frozenset[str] | None = result
        return result

    # Prefixes for high-value identifier facts that should be included in critical strings.
    _HIGH_VALUE_PREFIXES: tuple[str, ...] = (
        "Tracking:",
        "Date:",
        "Statute:",
        "Amount:",
        "Metric:",
        "ID:",
        "UUID:",
        "URL:",
    )

    def get_critical_strings(self) -> frozenset[str]:
        """Return protected strings from FILES, ERRORS, and high-value IDENTIFIERS.

        Strict filtering to avoid broad matches that kill compression:
        - FILES: Only paths containing a directory separator AND a file extension.
        - ERRORS: Only error messages >= 10 chars.
        - IDENTIFIERS: Only high-value facts (Tracking, Date, Statute, Amount,
          Metric, ID, UUID, URL) — not Code entities or tool names which are too broad.
        """
        strings: set[str] = set()
        for fact in self.facts:
            core = fact.content.split(" \u2014 ")[0].split(" — ")[0].strip()

            if fact.category == AnchorCategory.FILES:
                if "/" not in core and "\\" not in core:
                    continue
                parts = core.replace("\\", "/").rstrip("/").rsplit("/", 1)
                if len(parts) < 2:
                    continue
                filename = parts[-1]
                if "." not in filename or len(filename) < 3:
                    continue
                name_part = filename.rsplit(".", 1)[0]
                if name_part.replace(".", "").replace("-", "").replace("_", "").isdigit():
                    continue
                if len(core) >= 8:
                    strings.add(core)
                if len(filename) >= 5:
                    strings.add(filename)

            elif fact.category == AnchorCategory.ERRORS:
                if len(core) >= 10:
                    strings.add(core)

            elif fact.category == AnchorCategory.IDENTIFIERS:
                # Only include high-value identifiers, not broad ones.
                for prefix in self._HIGH_VALUE_PREFIXES:
                    if core.startswith(prefix):
                        value = core[len(prefix) :].strip()
                        if len(value) >= 3:
                            strings.add(value)
                        break

        return frozenset(strings)

    def contains_anchor_fact(self, text: str) -> bool:
        """Return True if any critical fact (FILES or ERRORS) appears in ``text``.

        Uses ``get_critical_strings()`` (not the broader ``get_protected_strings()``)
        to avoid false positives from short identifier matches.
        Case-insensitive matching. Used by budget layer to gate segment drops.
        """
        if not self.facts:
            return False
        critical = self.get_critical_strings()
        if not critical:
            return False
        text_lower = text.lower()
        return any(s.lower() in text_lower for s in critical)

    def _invalidate_cache(self) -> None:
        """Invalidate the protected strings cache."""
        self._protected_cache = None

    def render(self) -> str:
        """Render the ledger as a structured formatted string to prepend to context.

        Produces a 5-section structured ledger (INTENT, FILES, DECISIONS, ERRORS,
        ACTIVE_CONTEXT) followed by supplementary sections if they contain facts.
        """
        lines: list[str] = ["[SESSION MEMORY — preserved across compressions]", ""]

        # Render primary sections first, then supplementary.
        all_categories = list(_LEDGER_PRIMARY_SECTIONS) + [
            c for c in AnchorCategory if c not in _LEDGER_PRIMARY_SECTIONS
        ]
        for category in all_categories:
            category_facts = [f for f in self.facts if f.category == category]
            if category_facts:
                header = _LEDGER_SECTION_HEADERS.get(category, f"## {category.value}")
                lines.append(header)
                for fact in category_facts:
                    lines.append(f"- {fact.content}")
                lines.append("")
        return "\n".join(lines)

    def working_memory_summary(self) -> str:
        """Generate a concise working memory summary (Warm tier).

        Combines OUTCOMES, DECISIONS, PARAMETERS, CONSTRAINTS, and RELATIONSHIPS
        into a compact narrative suitable for prepending to the context window.
        This is the deterministic fallback for tiered memory -- no LLM needed.

        Returns:
            Formatted working memory string, or empty string if no relevant facts.
        """
        warm_categories = (
            AnchorCategory.OUTCOMES,
            AnchorCategory.DECISIONS,
            AnchorCategory.PARAMETERS,
            AnchorCategory.CONSTRAINTS,
            AnchorCategory.RELATIONSHIPS,
        )
        lines: list[str] = []
        for category in warm_categories:
            category_facts = [f for f in self.facts if f.category == category]
            if category_facts:
                header = _LEDGER_SECTION_HEADERS.get(category, f"## {category.value}")
                lines.append(header)
                for fact in category_facts[:10]:  # Cap per category to prevent bloat.
                    lines.append(f"- {fact.content}")

        if not lines:
            return ""

        return "[WORKING MEMORY]\n" + "\n".join(lines)

    def token_estimate(self) -> int:
        """Estimate token count of the rendered ledger."""
        return len(self.render()) // 4

    def facts_by_category(self, category: AnchorCategory) -> list[AnchorFact]:
        """Return all facts in a specific category."""
        return [f for f in self.facts if f.category == category]

    def save(self, path: str) -> None:
        """Persist the ledger to a JSON file."""
        import json
        from pathlib import Path

        data = {
            "facts": [
                {
                    "category": f.category.value,
                    "content": f.content,
                    "turn": f.turn,
                    "confidence": f.confidence,
                }
                for f in self.facts
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> AnchorLedger:
        """Load a ledger from a JSON file. Returns empty ledger if file doesn't exist."""
        import json
        from pathlib import Path

        file = Path(path)
        if not file.exists():
            return cls()

        data = json.loads(file.read_text(encoding="utf-8"))
        ledger = cls()
        for item in data.get("facts", []):
            fact = AnchorFact(
                category=AnchorCategory(item["category"]),
                content=item["content"],
                turn=item["turn"],
                confidence=item.get("confidence", 1.0),
            )
            ledger.add(fact)
        return ledger
