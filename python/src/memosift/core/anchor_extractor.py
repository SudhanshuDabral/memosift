# Anchor fact extraction — capture critical facts from messages before compression.
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from memosift.core.types import (
    AnchorCategory,
    AnchorFact,
    AnchorLedger,
    ClassifiedMessage,
    ContentType,
    DependencyMap,
    MemoSiftMessage,
)

if TYPE_CHECKING:
    from memosift.providers.base import MemoSiftLLMProvider

logger = logging.getLogger("memosift")

# File path pattern: matches src/auth.ts, ./config/db.json, C:\Users\file.py, etc.
FILE_PATH_PATTERN = re.compile(
    r"(?:^|\s|[\"'])"
    r"((?:[a-zA-Z]:)?"
    r"(?:[./\\])?"
    r"(?:[\w.\-]+[/\\])*"
    r"[\w.\-]+\.\w{1,10})"
    r"(?::\d+)?",
)

# Error type + message pattern.
ERROR_PATTERN = re.compile(
    r"((?:TypeError|ReferenceError|SyntaxError|ValueError|KeyError|"
    r"AttributeError|ImportError|RuntimeError|Exception|Error|FAIL):"
    r"\s*.{10,200})",
    re.MULTILINE,
)

# Line reference pattern: file.ts:47, auth.py:12.
LINE_REF_PATTERN = re.compile(r"([\w./\\]+:\d+)")

# URL pattern — capture URLs that may be lost during JSON truncation.
URL_PATTERN = re.compile(r"(https?://\S+)")

# Code entity patterns — capture class/function/method names.
_CODE_ENTITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bclass\s+(\w+)"),  # class AuthService
    re.compile(r"\bdef\s+(\w+)"),  # def authenticate
    re.compile(r"\bfunction\s+(\w+)"),  # function authenticate
    re.compile(r"\basync\s+(\w+)\s*\("),  # async authenticate(
    re.compile(r"(?:get|set)\s+(\w+)\s*\("),  # get userId(
]

# Tool names that indicate file modification.
_EDIT_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "edit_file",
        "write_file",
        "create_file",
        "patch_file",
        "Edit",
        "Write",
        "edit",
        "write",
    }
)

# Decision marker patterns — assistant messages containing these indicate decisions.
_DECISION_MARKERS: list[re.Pattern[str]] = [
    re.compile(r"\bI'll use\b", re.IGNORECASE),
    re.compile(r"\bLet's go with\b", re.IGNORECASE),
    re.compile(r"\bchoosing\b.{1,60}\bbecause\b", re.IGNORECASE),
    re.compile(r"\bchose\b", re.IGNORECASE),
    re.compile(r"\bdecided to\b", re.IGNORECASE),
    re.compile(r"\bI'll go with\b", re.IGNORECASE),
    re.compile(r"\bwe(?:'ll| will) use\b", re.IGNORECASE),
]

# Hedging language that disqualifies a sentence as a decision.
_HEDGING_PATTERN = re.compile(
    r"\b(?:maybe|perhaps|could consider|might want to|possibly|not sure)\b|\?",
    re.IGNORECASE,
)

# UUID pattern for JSON argument extraction.
_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)

# Alphanumeric identifier pattern (≥12 chars).
_LONG_ALPHANUM_PATTERN = re.compile(r"\b[A-Za-z0-9]{12,}\b")

# Order ID pattern.
_ORDER_ID_PATTERN = re.compile(r"\b[A-Z]{2,4}-\d{4,}\b")

# Date pattern — ISO dates and common date formats.
_DATE_PATTERN = re.compile(
    r"\b(\d{4}-\d{2}-\d{2})\b"  # 2026-03-15
    r"|\b((?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4})\b",  # March 15, 2026
    re.IGNORECASE,
)

# Tracking number pattern — alphanumeric sequences typical of shipping carriers.
_TRACKING_PATTERN = re.compile(r"\b([A-Z0-9]{15,30})\b")

# Legal statute/section patterns.
_STATUTE_PATTERN = re.compile(
    r"\b(\d+\s+U\.S\.C\.\s*(?:§|Section)\s*\d+(?:\([a-z]\))?)\b"
    r"|\b(Section\s+\d+(?:\.\d+)*(?:\([a-z]\))?)\b"
    r"|\b(Article\s+\d+)\b",
    re.IGNORECASE,
)

# Monetary amounts.
_MONEY_PATTERN = re.compile(r"\$[\d,]+(?:\.\d{2})?")

# Percentage values.
_PERCENT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%")

# Proper noun / person name pattern — capitalized words after lowercase context.
_PROPER_NOUN_PATTERN = re.compile(r"(?<=[a-z]\s)([A-Z][a-z]{2,15})\b")

# Domain-specific term pattern — medical/scientific terms (long, specific words).
_DOMAIN_TERM_PATTERN = re.compile(
    r"\b([a-z]{8,}(?:ide|ine|ase|ose|ate|ism|itis|emia|opathy|amine|mycin|cillin|prazole|sartan|statin))\b",
    re.IGNORECASE,
)

# Reasoning chain markers — indicate logical dependency on prior messages.
_REASONING_CHAIN_PATTERN = re.compile(
    r"\b(?:therefore|so we can|which means|building on that|given that|"
    r"as established|as a result|consequently|based on this|"
    r"following from|this confirms|thus|hence)\b",
    re.IGNORECASE,
)


def extract_anchors_from_message(
    msg: MemoSiftMessage,
    turn: int,
    tool_name: str | None = None,
) -> list[AnchorFact]:
    """Extract anchor facts from a single message.

    Runs lightweight regex extractors for file paths, error messages,
    and line references. The ``tool_name`` parameter determines whether
    a file was "read" or "modified".

    Args:
        msg: The message to extract from.
        turn: The turn number (counted by user messages from session start).
        tool_name: The tool name from the message, if any.

    Returns:
        A list of extracted AnchorFacts.
    """
    if not msg.content:
        return []

    facts: list[AnchorFact] = []

    is_edit = tool_name is not None and tool_name in _EDIT_TOOL_NAMES
    action = "modified" if is_edit else "read"

    # Extract file paths.
    for match in FILE_PATH_PATTERN.finditer(msg.content):
        path = match.group(1)
        facts.append(
            AnchorFact(
                category=AnchorCategory.FILES,
                content=f"{path} — {action} at turn {turn}",
                turn=turn,
            )
        )

    # Extract error messages.
    for match in ERROR_PATTERN.finditer(msg.content):
        facts.append(
            AnchorFact(
                category=AnchorCategory.ERRORS,
                content=match.group(1).strip(),
                turn=turn,
            )
        )

    # Extract line references (file:line).
    for ref in LINE_REF_PATTERN.findall(msg.content):
        facts.append(
            AnchorFact(
                category=AnchorCategory.IDENTIFIERS,
                content=f"Reference: {ref}",
                turn=turn,
                confidence=0.8,
            )
        )

    # Extract URLs (may be lost during JSON array truncation).
    for match in URL_PATTERN.finditer(msg.content):
        url = match.group(1).rstrip(".,;)\"'")
        facts.append(
            AnchorFact(
                category=AnchorCategory.IDENTIFIERS,
                content=f"URL: {url}",
                turn=turn,
                confidence=0.7,
            )
        )

    # Extract code entity names (class, function, method names).
    for pattern in _CODE_ENTITY_PATTERNS:
        for match in pattern.finditer(msg.content):
            name = match.group(1)
            if len(name) > 2:  # Skip tiny names like "db".
                facts.append(
                    AnchorFact(
                        category=AnchorCategory.IDENTIFIERS,
                        content=f"Code entity: {name}",
                        turn=turn,
                        confidence=0.7,
                    )
                )

    # Extract dates.
    for match in _DATE_PATTERN.finditer(msg.content):
        date_str = match.group(1) or match.group(2)
        if date_str:
            facts.append(
                AnchorFact(
                    category=AnchorCategory.IDENTIFIERS,
                    content=f"Date: {date_str}",
                    turn=turn,
                    confidence=0.8,
                )
            )

    # Extract tracking numbers (15-30 char alphanumeric).
    for match in _TRACKING_PATTERN.finditer(msg.content):
        tracking = match.group(1)
        # Filter out common false positives (all letters or all digits).
        if any(c.isdigit() for c in tracking) and any(c.isalpha() for c in tracking):
            facts.append(
                AnchorFact(
                    category=AnchorCategory.IDENTIFIERS,
                    content=f"Tracking: {tracking}",
                    turn=turn,
                    confidence=0.8,
                )
            )

    # Extract legal statutes/sections.
    for match in _STATUTE_PATTERN.finditer(msg.content):
        statute = match.group(1) or match.group(2) or match.group(3)
        if statute:
            facts.append(
                AnchorFact(
                    category=AnchorCategory.IDENTIFIERS,
                    content=f"Statute: {statute}",
                    turn=turn,
                    confidence=0.85,
                )
            )

    # Extract monetary amounts.
    for match in _MONEY_PATTERN.finditer(msg.content):
        facts.append(
            AnchorFact(
                category=AnchorCategory.IDENTIFIERS,
                content=f"Amount: {match.group(0)}",
                turn=turn,
                confidence=0.8,
            )
        )

    # Extract percentages.
    for match in _PERCENT_PATTERN.finditer(msg.content):
        facts.append(
            AnchorFact(
                category=AnchorCategory.IDENTIFIERS,
                content=f"Metric: {match.group(0)}",
                turn=turn,
                confidence=0.7,
            )
        )

    # Extract domain-specific terms (medical, scientific).
    for match in _DOMAIN_TERM_PATTERN.finditer(msg.content):
        term = match.group(1)
        if len(term) >= 8:
            facts.append(
                AnchorFact(
                    category=AnchorCategory.IDENTIFIERS,
                    content=f"Term: {term}",
                    turn=turn,
                    confidence=0.7,
                )
            )

    return facts


def _extract_decisions_from_text(text: str, turn: int) -> list[AnchorFact]:
    """Extract decision facts from assistant text.

    A sentence is a decision if it matches a decision marker AND does NOT
    contain hedging language or a question mark in the same sentence.
    """
    facts: list[AnchorFact] = []
    sentences = re.split(r"(?<=[.!])\s+", text)
    for sentence in sentences:
        if _HEDGING_PATTERN.search(sentence):
            continue
        for marker in _DECISION_MARKERS:
            if marker.search(sentence):
                # Trim to a reasonable length.
                content = sentence.strip()[:200]
                facts.append(
                    AnchorFact(
                        category=AnchorCategory.DECISIONS,
                        content=content,
                        turn=turn,
                        confidence=0.85,
                    )
                )
                break  # One decision per sentence.
    return facts


def _extract_facts_from_json_value(
    value: object,
    key: str | None,
    turn: int,
    tool_name: str | None,
) -> list[AnchorFact]:
    """Recursively walk a parsed JSON value and extract facts.

    Extracts file paths, URLs, UUIDs, order IDs, and long alphanumeric identifiers.
    Records the JSON key as context for better fact descriptions.
    """
    facts: list[AnchorFact] = []

    if isinstance(value, str):
        # Check for file paths.
        for match in FILE_PATH_PATTERN.finditer(value):
            path = match.group(1)
            action = "modified" if tool_name and tool_name in _EDIT_TOOL_NAMES else "referenced"
            context = f" (key={key})" if key else ""
            facts.append(
                AnchorFact(
                    category=AnchorCategory.FILES,
                    content=f"{path} — {action} at turn {turn}{context}",
                    turn=turn,
                )
            )
        # Check for URLs.
        for match in URL_PATTERN.finditer(value):
            url = match.group(1).rstrip(".,;)\"'")
            facts.append(
                AnchorFact(
                    category=AnchorCategory.IDENTIFIERS,
                    content=f"URL: {url}",
                    turn=turn,
                    confidence=0.7,
                )
            )
        # Check for UUIDs.
        for match in _UUID_PATTERN.finditer(value):
            facts.append(
                AnchorFact(
                    category=AnchorCategory.IDENTIFIERS,
                    content=f"UUID: {match.group(0)}",
                    turn=turn,
                    confidence=0.8,
                )
            )
        # Check for order IDs.
        for match in _ORDER_ID_PATTERN.finditer(value):
            facts.append(
                AnchorFact(
                    category=AnchorCategory.IDENTIFIERS,
                    content=f"ID: {match.group(0)}",
                    turn=turn,
                    confidence=0.8,
                )
            )

    elif isinstance(value, dict):
        for k, v in value.items():
            facts.extend(_extract_facts_from_json_value(v, k, turn, tool_name))

    elif isinstance(value, list):
        for item in value:
            facts.extend(_extract_facts_from_json_value(item, key, turn, tool_name))

    return facts


def extract_anchors_from_segments(
    segments: list[ClassifiedMessage],
    ledger: AnchorLedger,
) -> None:
    """Extract anchor facts from classified segments and add to the ledger.

    Produces a structured 5-section ledger:
    - INTENT: extracted from the first user message (session goal)
    - FILES: file paths from tool calls and content, with read/modified action
    - DECISIONS: assistant messages with decision markers (filtered for hedging)
    - ERRORS: segments classified as ERROR_TRACE
    - ACTIVE_CONTEXT: last user message + last assistant message

    Also processes tool call arguments as parsed JSON for nested fact extraction.

    Args:
        segments: Classified messages to extract from.
        ledger: The anchor ledger to add facts to (mutated in place).
    """
    extractable_types = {
        ContentType.TOOL_RESULT_TEXT,
        ContentType.TOOL_RESULT_JSON,
        ContentType.ERROR_TRACE,
        ContentType.CODE_BLOCK,
        ContentType.OLD_CONVERSATION,
        ContentType.ASSISTANT_REASONING,
    }

    # Pre-count user messages to compute turn numbers.
    user_indices = sorted(seg.original_index for seg in segments if seg.message.role == "user")

    def _turn_for_index(idx: int) -> int:
        """Count how many user messages appear at or before this index."""
        count = 0
        for ui in user_indices:
            if ui <= idx:
                count += 1
            else:
                break
        return max(count, 1)

    # ── Extract INTENT from the first user message ──
    first_user = None
    for seg in segments:
        if seg.message.role == "user" and not seg.message._memosift_compressed:
            first_user = seg
            break

    if first_user is not None:
        intent_text = first_user.content.strip()[:300]
        if intent_text:
            ledger.add(
                AnchorFact(
                    category=AnchorCategory.INTENT,
                    content=intent_text,
                    turn=1,
                    confidence=0.9,
                )
            )

    # ── Extract ACTIVE_CONTEXT from last user + last assistant messages ──
    last_user = None
    last_assistant = None
    for seg in reversed(segments):
        if seg.message._memosift_compressed:
            continue
        if seg.message.role == "user" and last_user is None:
            last_user = seg
        elif seg.message.role == "assistant" and last_assistant is None:
            last_assistant = seg
        if last_user is not None and last_assistant is not None:
            break

    if last_user is not None:
        turn = _turn_for_index(last_user.original_index)
        context_text = last_user.content.strip()[:300]
        if context_text:
            ledger.add(
                AnchorFact(
                    category=AnchorCategory.ACTIVE_CONTEXT,
                    content=f"Current task: {context_text}",
                    turn=turn,
                    confidence=0.9,
                )
            )

    if last_assistant is not None:
        turn = _turn_for_index(last_assistant.original_index)
        context_text = last_assistant.content.strip()[:300]
        if context_text:
            ledger.add(
                AnchorFact(
                    category=AnchorCategory.ACTIVE_CONTEXT,
                    content=f"Last response: {context_text}",
                    turn=turn,
                    confidence=0.8,
                )
            )

    # ── Extract facts from all segments ──
    for seg in segments:
        if seg.message._memosift_compressed:
            continue

        turn = _turn_for_index(seg.original_index)

        # Extract DECISIONS from assistant messages.
        if seg.message.role == "assistant" and seg.content:
            decision_facts = _extract_decisions_from_text(seg.content, turn)
            for fact in decision_facts:
                ledger.add(fact)

        # Extract from tool_call arguments — parse as JSON (Item 1.2).
        if seg.message.tool_calls:
            for tc in seg.message.tool_calls:
                # Capture tool name as identifier.
                if tc.function.name:
                    ledger.add(
                        AnchorFact(
                            category=AnchorCategory.IDENTIFIERS,
                            content=f"Tool used: {tc.function.name}",
                            turn=turn,
                            confidence=0.9,
                        )
                    )

                # Parse arguments as JSON for structured extraction.
                args_str = tc.function.arguments
                try:
                    parsed_args = json.loads(args_str)
                    json_facts = _extract_facts_from_json_value(
                        parsed_args,
                        None,
                        turn,
                        tc.function.name,
                    )
                    for fact in json_facts:
                        ledger.add(fact)
                except (json.JSONDecodeError, ValueError):
                    # Fallback: extract file paths from raw string.
                    for match in FILE_PATH_PATTERN.finditer(args_str):
                        path = match.group(1)
                        action = (
                            "modified" if tc.function.name in _EDIT_TOOL_NAMES else "referenced"
                        )
                        ledger.add(
                            AnchorFact(
                                category=AnchorCategory.FILES,
                                content=f"{path} — {action} at turn {turn}",
                                turn=turn,
                            )
                        )

        if seg.content_type not in extractable_types:
            continue

        facts = extract_anchors_from_message(
            seg.message,
            turn=turn,
            tool_name=seg.message.name,
        )
        for fact in facts:
            ledger.add(fact)


def extract_reasoning_chains(
    segments: list[ClassifiedMessage],
    deps: DependencyMap,
) -> None:
    """Detect logical reasoning dependencies across messages.

    Scans for reasoning chain markers ("therefore", "so we can", "building on
    that", etc.) and links the current message back to the most recent prior
    assistant message. This creates logical edges in the DependencyMap so that
    L3E and L6 don't break causal chains.

    Args:
        segments: Classified messages (after anchor extraction).
        deps: DependencyMap to add logical edges to (mutated in place).
    """
    # Build list of assistant message indices in order.
    assistant_indices: list[int] = []
    for seg in segments:
        if seg.message.role == "assistant" and not seg.message._memosift_compressed:
            assistant_indices.append(seg.original_index)

    # For each message with a reasoning chain marker, link it to the most
    # recent prior assistant message.
    for seg in segments:
        if seg.message._memosift_compressed or not seg.content:
            continue
        if not _REASONING_CHAIN_PATTERN.search(seg.content):
            continue

        # Find the most recent assistant message before this one.
        prior = None
        for ai in reversed(assistant_indices):
            if ai < seg.original_index:
                prior = ai
                break

        if prior is not None and prior != seg.original_index:
            deps.add_logical(seg.original_index, prior)


# ── LLM-Powered Anchor Extraction ───────────────────────────────────────────

_ANCHOR_EXTRACTION_PROMPT = """\
Extract ALL critical facts from this conversation that must survive compression.
Return valid JSON with these sections:

{{"files": ["exact/file/path — action (read/modified/created)"],
 "decisions": ["what was decided and why, verbatim quotes where possible"],
 "errors": ["error type: message — resolution status"],
 "identifiers": ["specific IDs, URLs, keys, version numbers"],
 "conclusions": ["key technical findings, 1 sentence each"],
 "open_items": ["unresolved questions or pending tasks"]}}

Rules:
- Include ONLY facts explicitly stated in the conversation text below
- File paths must be exact (copy from source)
- Error messages must be verbatim (copy from source)
- For decisions, include the reasoning
- Do NOT infer or generate facts not present in the text

CONVERSATION:
{conversation}"""

_CATEGORY_MAP: dict[str, AnchorCategory] = {
    "files": AnchorCategory.FILES,
    "decisions": AnchorCategory.DECISIONS,
    "errors": AnchorCategory.ERRORS,
    "identifiers": AnchorCategory.IDENTIFIERS,
    "conclusions": AnchorCategory.OUTCOMES,
    "open_items": AnchorCategory.OPEN_ITEMS,
}


async def extract_anchors_llm(
    segments: list[ClassifiedMessage],
    ledger: AnchorLedger,
    llm: MemoSiftLLMProvider,
) -> None:
    """Extract anchor facts using an LLM instead of regex.

    Falls back to the regex extractor on any failure.
    """
    try:
        condensed = _build_condensed_view(segments)
        if not condensed:
            extract_anchors_from_segments(segments, ledger)
            return

        prompt = _ANCHOR_EXTRACTION_PROMPT.format(conversation=condensed)
        response = await llm.generate(prompt, max_tokens=2048, temperature=0.0)
        full_text = "\n".join(seg.content for seg in segments if seg.content)
        facts = _parse_anchor_response(response.text, full_text)

        for fact in facts:
            ledger.add(fact)

        # Positional anchors (INTENT, ACTIVE_CONTEXT) are already extracted
        # by the regex extractor which runs before this function.

    except Exception as e:
        logger.warning("LLM anchor extraction failed (%s), falling back to regex.", e)
        extract_anchors_from_segments(segments, ledger)


def _build_condensed_view(segments: list[ClassifiedMessage]) -> str:
    """Build a condensed conversation view for the LLM prompt."""
    parts: list[str] = []
    for seg in segments:
        if seg.message._memosift_compressed:
            continue
        role = seg.message.role.upper()
        content = seg.content[:500] if seg.content else ""
        tool_label = f" ({seg.message.name})" if seg.message.name else ""
        parts.append(f"[{role}{tool_label}]: {content}")
        if seg.message.tool_calls:
            for tc in seg.message.tool_calls:
                parts.append(f"  -> tool_call: {tc.function.name}({tc.function.arguments})")
    return "\n\n".join(parts)


def _parse_anchor_response(response_text: str, full_source_text: str) -> list[AnchorFact]:
    """Parse LLM JSON response into validated AnchorFact objects."""
    json_text = response_text.strip()
    if json_text.startswith("```"):
        lines = json_text.split("\n")
        json_text = "\n".join(line for line in lines if not line.strip().startswith("```"))

    try:
        data = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(data, dict):
        return []

    source_lower = full_source_text.lower()
    facts: list[AnchorFact] = []

    for section, category in _CATEGORY_MAP.items():
        items = data.get(section, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, str) or len(item) < 3:
                continue
            if not _validate_fact_against_source(item, category, source_lower):
                continue
            facts.append(AnchorFact(category=category, content=item, turn=0, confidence=0.9))

    return facts


def _validate_fact_against_source(
    fact_content: str, category: AnchorCategory, source_text_lower: str,
) -> bool:
    """Validate that an LLM-extracted fact is grounded in the source text."""
    if category == AnchorCategory.FILES:
        path = fact_content.split(" —")[0].split(" --")[0].strip()
        return path.lower() in source_text_lower
    if category == AnchorCategory.ERRORS:
        error_type = fact_content.split(":")[0].strip()
        return error_type.lower() in source_text_lower
    words = fact_content.split()
    if len(words) >= 3:
        for i in range(len(words) - 2):
            chunk = " ".join(words[i : i + 3]).lower()
            if len(chunk) >= 8 and chunk in source_text_lower:
                return True
    for word in words:
        if len(word) >= 8 and word.lower() in source_text_lower:
            return True
    return False


def _extract_positional_anchors(segments: list[ClassifiedMessage], ledger: AnchorLedger) -> None:
    """Extract INTENT and ACTIVE_CONTEXT anchors using positional logic."""
    first_user = None
    for seg in segments:
        if seg.message.role == "user" and not seg.message._memosift_compressed:
            first_user = seg
            break
    if first_user is not None:
        intent_text = first_user.content.strip()[:300]
        if intent_text:
            ledger.add(AnchorFact(category=AnchorCategory.INTENT, content=intent_text, turn=1, confidence=0.9))

    last_user = None
    last_assistant = None
    for seg in reversed(segments):
        if seg.message._memosift_compressed:
            continue
        if seg.message.role == "user" and last_user is None:
            last_user = seg
        elif seg.message.role == "assistant" and last_assistant is None:
            last_assistant = seg
        if last_user is not None and last_assistant is not None:
            break
    if last_user is not None:
        ctx = last_user.content.strip()[:300]
        if ctx:
            ledger.add(AnchorFact(category=AnchorCategory.ACTIVE_CONTEXT, content=f"Current task: {ctx}", turn=0, confidence=0.9))
    if last_assistant is not None:
        ctx = last_assistant.content.strip()[:300]
        if ctx:
            ledger.add(AnchorFact(category=AnchorCategory.ACTIVE_CONTEXT, content=f"Last response: {ctx}", turn=0, confidence=0.8))
