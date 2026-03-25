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

# ALL-CAPS entity names — well names, operator names, project codes.
# Matches multi-word: "WHITLEY-DUBOSE UNIT 1H", "CRESCENT ENERGY"
_ALLCAPS_ENTITY_PATTERN = re.compile(
    r"\b([A-Z][A-Z\s\-]{3,}(?:\d+[A-Z]?\b)?)\b"
)

# Single ALL-CAPS words (3+ chars) that are likely entity names, not common abbrevs.
# Catches: "EOG", "FESCO", "INEOS" but skips: "THE", "AND", "FOR"
_SINGLE_CAPS_ENTITY_RE = re.compile(r"\b([A-Z]{3,15})\b")
_COMMON_ABBREVIATIONS: frozenset[str] = frozenset({
    "THE", "AND", "FOR", "NOT", "BUT", "ARE", "WAS", "HAS", "HAD", "CAN",
    "ALL", "ANY", "FEW", "NEW", "OLD", "USE", "GET", "SET", "RUN", "ADD",
    "API", "URL", "SQL", "CSS", "HTML", "JSON", "HTTP", "HTTPS", "REST",
    "PDF", "CSV", "XML", "SDK", "CLI", "GUI", "IDE", "GIT", "NPM", "PIP",
    "AWS", "GCP", "CPU", "GPU", "RAM", "SSD", "HDD", "USB", "LAN", "WAN",
    "TRUE", "FALSE", "NULL", "NONE", "PASS", "FAIL", "TODO", "NOTE",
    "SYSTEM", "USER", "TOOL", "ERROR", "TRACE", "DEBUG", "INFO", "WARN",
})

# Domain-specific term pattern — medical/scientific terms (long, specific words).
_DOMAIN_TERM_PATTERN = re.compile(
    r"\b([a-z]{8,}(?:ide|ine|ase|ose|ate|ism|itis|emia|opathy|amine|mycin|cillin|prazole|sartan|statin))\b",
    re.IGNORECASE,
)

# ── Contextual Metric Intelligence ─────────────────────────────────────────
#
# Domain-agnostic heuristic detection of significant numerical metrics.
# Uses 6 contextual signals instead of hardcoded unit lists.

# Number followed by one or more non-whitespace "context" words.
_NUMBER_IN_CONTEXT = re.compile(
    r"(?<!\w)"
    r"(?P<number>\d[\d,]*(?:\.\d+)?)"
    r"\s+"
    r"(?P<context>[A-Za-z°µ/][A-Za-z0-9°µ/]{0,15}(?:\s+[A-Za-z]{1,8})?)",
)

# Comparison phrases — numbers appearing in these contexts are likely KPIs.
_COMPARISON_CONTEXT = re.compile(
    r"(?:\bvs\.?\b|\bversus\b|\bcompared to\b|\bfrom\b.{1,20}\bto\b"
    r"|\bincreased by\b|\bdecreased by\b|\bdropped\b|\brose\b|\bfell\b"
    r"|\bhigher\b|\blower\b|\badvantage\b|\bdelta\b)",
    re.IGNORECASE,
)

# Markdown table cell — numbers in tables are almost always KPIs.
_TABLE_CELL = re.compile(r"\|[^|]*$|^[^|]*\|", re.MULTILINE)

# Expanded common English words — if the word after a number IS in this set,
# it's likely NOT a unit. ~200 words covering common nouns, adjectives, verbs.
# Intentionally excludes any word that could be a measurement unit.
_COMMON_WORDS: frozenset[str] = frozenset({
    # Articles, prepositions, conjunctions
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over", "up", "down", "out", "off",
    # Pronouns & determiners
    "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall", "can",
    "this", "that", "it", "its", "i", "me", "my", "we", "you", "he", "she",
    "they", "them", "their", "our", "your", "his", "her", "who", "which", "what",
    # Common verbs
    "get", "got", "go", "went", "come", "came", "make", "made", "take", "took",
    "see", "saw", "know", "knew", "think", "thought", "give", "gave", "find",
    "found", "tell", "told", "ask", "asked", "use", "used", "try", "tried",
    "need", "needed", "want", "wanted", "run", "ran", "set", "put", "keep",
    "kept", "let", "begin", "began", "show", "showed", "call", "called",
    # Common adjectives
    "new", "old", "good", "bad", "great", "small", "large", "big", "long",
    "short", "high", "low", "right", "left", "last", "first", "next", "early",
    "late", "full", "empty", "same", "different", "other", "more", "less",
    "most", "least", "many", "few", "much", "each", "every", "all", "both",
    "such", "own", "only", "just", "also", "very", "too", "quite", "really",
    # Common nouns (things you count, NOT units)
    "things", "items", "people", "days", "times", "ways", "years", "months",
    "weeks", "hours", "minutes", "seconds", "points", "steps", "parts", "types",
    "cases", "lines", "words", "pages", "rows", "columns", "fields", "files",
    "records", "entries", "users", "results", "values", "options", "changes",
    "issues", "errors", "tests", "tasks", "turns", "calls", "attempts",
    "reasons", "examples", "instances", "versions", "levels", "stages",
    # Misc high-frequency
    "not", "no", "so", "if", "then", "than", "when", "where", "how", "why",
    "here", "there", "now", "still", "already", "yet", "even", "well",
    "about", "like", "some", "any", "per", "total", "main",
    "available", "possible", "specific", "similar", "additional",
})


def _extract_contextual_metrics(
    text: str,
    config_patterns: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Extract numbers that are likely domain metrics using contextual signals.

    Uses 6 heuristic signals instead of hardcoded unit lists:
    1. Ratio-unit pattern (X/Y: Mcf/d, mg/dL, req/s)
    2. High precision (comma-separated or 2+ decimal places)
    3. Non-common context word (word after number not in common English)
    4. Comparison context (vs, compared to, increased by)
    5. Markdown table cell
    6. JSON metric key context

    Args:
        text: The text to extract metrics from.
        config_patterns: Optional domain-specific unit patterns from config.
            Numbers followed by these are always extracted (confidence 0.9).

    Returns:
        List of (matched_text, confidence_score) tuples for metrics scoring >= 0.5.
    """
    metrics: list[tuple[str, float]] = []
    seen: set[str] = set()
    config_set = frozenset(p.lower() for p in (config_patterns or []))

    for match in _NUMBER_IN_CONTEXT.finditer(text):
        number = match.group("number")
        context = match.group("context").strip()
        context_first = context.split()[0] if context else ""
        matched_text = f"{number} {context_first}".strip()

        if matched_text in seen:
            continue

        # Check config patterns first — always extract if matched.
        if config_set and context_first.lower() in config_set:
            seen.add(matched_text)
            metrics.append((matched_text, 0.9))
            continue
        # Also check multi-word config patterns (e.g., "Scf/STB").
        if config_set and context.lower() in config_set:
            seen.add(f"{number} {context}")
            metrics.append((f"{number} {context}", 0.9))
            continue

        score = 0.0

        # Signal 1: Ratio-unit pattern (X/Y) — strongest single signal.
        if "/" in context_first and re.match(r"[A-Za-z]+/[A-Za-z]+", context_first):
            score += 0.9

        # Signal 2: High precision — comma-separated or 2+ decimal places.
        if "," in number:
            score += 0.4
        elif "." in number and len(number.split(".")[-1]) >= 2:
            score += 0.3

        # Signal 3: Context word is NOT common English — likely a unit.
        if context_first and context_first.lower() not in _COMMON_WORDS:
            score += 0.5

        # Signal 4: Number appears near a comparison phrase.
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 80)
        window = text[start:end]
        if _COMPARISON_CONTEXT.search(window):
            score += 0.3

        # Signal 5: Number appears in a markdown table cell.
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end]
        if "|" in line:
            score += 0.3

        if score >= 0.5:
            seen.add(matched_text)
            metrics.append((matched_text, min(score, 1.0)))

    return metrics


# ── Working Memory Extraction Patterns ─────────────────────────────────

# Parameters: numbers after threshold/limit/cap/min/max/target keywords.
_PARAMETER_PATTERN = re.compile(
    r"\b(?:threshold|limit|cap|minimum|maximum|min|max|target|budget|ceiling|floor|"
    r"cutoff|baseline|benchmark|setpoint)\s*(?:of|=|:|\bis\b)?\s*"
    r"(\d[\d,.]*(?:\.\d+)?\s*\S{0,15})",
    re.IGNORECASE,
)

# Constraints: explicit rules the agent must follow.
_CONSTRAINT_PATTERN = re.compile(
    r"((?:must not|do not|don't|should not|shouldn't|cannot|can't|never|"
    r"only if|exclude|excluding|except|avoid|restrict)\s+.{10,120}?[.!;\n])",
    re.IGNORECASE,
)

# Assumptions: implicit conditions the agent has adopted.
_ASSUMPTION_PATTERN = re.compile(
    r"((?:assum(?:e|ing|ed)|by default|unless (?:specified|otherwise|stated)|"
    r"we(?:'re| are) treating|for (?:the purposes|simplicity)|"
    r"taken as|treated as)\s+.{10,120}?[.!;\n])",
    re.IGNORECASE,
)

# Entity co-occurrence: extract (Subject, verb/relation, Object) triples.
_ENTITY_COOCCURRENCE_PATTERNS = [
    # "X has/shows/produces Y" patterns.
    re.compile(
        r"\b([A-Z][A-Za-z\s]{2,30}?)\s+(?:has|shows?|produces?|delivers?|maintains?|"
        r"outperforms?|exceeds?|averages?)\s+(?:a\s+)?(.{5,60}?)(?:\.|,|;|\n)",
        re.MULTILINE,
    ),
    # "X is/was Y" patterns.
    re.compile(
        r"\b([A-Z][A-Za-z\s]{2,30}?)\s+(?:is|was|are|were)\s+(?:the\s+)?(.{5,40}?)(?:\.|,|;|\n)",
        re.MULTILINE,
    ),
]

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
    metric_patterns: list[str] | None = None,
) -> list[AnchorFact]:
    """Extract anchor facts from a single message.

    Runs lightweight regex extractors for file paths, error messages,
    line references, and contextual metric intelligence. The ``tool_name``
    parameter determines whether a file was "read" or "modified".

    Args:
        msg: The message to extract from.
        turn: The turn number (counted by user messages from session start).
        tool_name: The tool name from the message, if any.
        metric_patterns: Optional domain-specific unit patterns from config.

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

    # Extract contextual metrics — domain-agnostic heuristic detection.
    existing_metric_values = {f.content for f in facts if f.content.startswith("Metric:")}
    for matched_text, confidence in _extract_contextual_metrics(msg.content, metric_patterns):
        fact_content = f"Metric: {matched_text}"
        if fact_content not in existing_metric_values:
            facts.append(
                AnchorFact(
                    category=AnchorCategory.IDENTIFIERS,
                    content=fact_content,
                    turn=turn,
                    confidence=confidence,
                )
            )
            existing_metric_values.add(fact_content)

    # Extract ALL-CAPS entity names (well names, operators, project codes).
    seen_caps: set[str] = set()
    for match in _ALLCAPS_ENTITY_PATTERN.finditer(msg.content):
        entity = match.group(1).strip()
        # Filter: must have 2+ words or be at least 5 chars, skip pure whitespace.
        words = entity.split()
        has_multiple_words = len(words) >= 2
        has_digits = len(entity) >= 5 and any(c.isdigit() for c in entity)
        if (has_multiple_words or has_digits) and entity not in seen_caps and len(entity) <= 50:
                seen_caps.add(entity)
                facts.append(
                    AnchorFact(
                        category=AnchorCategory.IDENTIFIERS,
                        content=f"Entity: {entity}",
                        turn=turn,
                        confidence=0.7,
                    )
                )

    # Extract single ALL-CAPS words (3+ chars) that are likely entity names.
    # Catches operator codes like "EOG", "INEOS", company abbrevs like "FESCO".
    for match in _SINGLE_CAPS_ENTITY_RE.finditer(msg.content):
        word = match.group(1)
        if word not in _COMMON_ABBREVIATIONS and word not in seen_caps:
            seen_caps.add(word)
            facts.append(
                AnchorFact(
                    category=AnchorCategory.IDENTIFIERS,
                    content=f"Entity: {word}",
                    turn=turn,
                    confidence=0.6,
                )
            )

    # Extract large comma-separated numbers (likely production/financial KPIs).
    # Catches: 95,467 / 72,193 / 48,201 — values in data tables.
    for match in re.finditer(r"\b(\d{1,3}(?:,\d{3})+)\b", msg.content):
        value = match.group(1)
        fact_content = f"Metric: {value}"
        if fact_content not in existing_metric_values:
            facts.append(
                AnchorFact(
                    category=AnchorCategory.IDENTIFIERS,
                    content=fact_content,
                    turn=turn,
                    confidence=0.75,
                )
            )
            existing_metric_values.add(fact_content)

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

    # Extract parameters (thresholds, limits, targets).
    for match in _PARAMETER_PATTERN.finditer(msg.content):
        value = match.group(1).strip().rstrip(".,;")
        if value and len(value) >= 2:
            facts.append(
                AnchorFact(
                    category=AnchorCategory.PARAMETERS,
                    content=f"Parameter: {match.group(0).strip()[:150]}",
                    turn=turn,
                    confidence=0.8,
                )
            )

    # Extract constraints (explicit rules).
    for match in _CONSTRAINT_PATTERN.finditer(msg.content):
        constraint = match.group(1).strip()[:200]
        if len(constraint) >= 15:
            facts.append(
                AnchorFact(
                    category=AnchorCategory.CONSTRAINTS,
                    content=constraint,
                    turn=turn,
                    confidence=0.8,
                )
            )

    # Extract assumptions.
    for match in _ASSUMPTION_PATTERN.finditer(msg.content):
        assumption = match.group(1).strip()[:200]
        if len(assumption) >= 15:
            facts.append(
                AnchorFact(
                    category=AnchorCategory.ASSUMPTIONS,
                    content=assumption,
                    turn=turn,
                    confidence=0.75,
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
        # Extract DATA_SCHEMA for top-level dicts with many keys.
        if key is None and len(value) >= 3:
            schema_keys = list(value.keys())[:12]
            facts.append(
                AnchorFact(
                    category=AnchorCategory.DATA_SCHEMA,
                    content=f"Schema: {{{', '.join(schema_keys)}}}",
                    turn=turn,
                    confidence=0.7,
                )
            )
        for k, v in value.items():
            facts.extend(_extract_facts_from_json_value(v, k, turn, tool_name))

    elif isinstance(value, list):
        # Extract DATA_SCHEMA for arrays of dicts (table-like data).
        if (
            len(value) >= 3
            and all(isinstance(item, dict) for item in value[:3])
        ):
            sample_keys = list(value[0].keys())[:10]
            facts.append(
                AnchorFact(
                    category=AnchorCategory.DATA_SCHEMA,
                    content=f"Table[{len(value)} rows]: {{{', '.join(sample_keys)}}}",
                    turn=turn,
                    confidence=0.7,
                )
            )
        for item in value:
            facts.extend(_extract_facts_from_json_value(item, key, turn, tool_name))

    return facts


def extract_anchors_from_segments(
    segments: list[ClassifiedMessage],
    ledger: AnchorLedger,
    metric_patterns: list[str] | None = None,
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
            metric_patterns=metric_patterns,
        )
        for fact in facts:
            ledger.add(fact)

    # ── Extract entity relationships (Phase 4.1) ──
    # Scan assistant messages for entity co-occurrence patterns.
    for seg in segments:
        if seg.message.role != "assistant" or not seg.content:
            continue
        if seg.message._memosift_compressed:
            continue
        turn = _turn_for_index(seg.original_index)
        for pattern in _ENTITY_COOCCURRENCE_PATTERNS:
            for match in pattern.finditer(seg.content):
                subject = match.group(1).strip()
                predicate = match.group(2).strip()
                if len(subject) >= 3 and len(predicate) >= 5:
                    content = f"{subject} -> {predicate}"[:150]
                    ledger.add(
                        AnchorFact(
                            category=AnchorCategory.RELATIONSHIPS,
                            content=content,
                            turn=turn,
                            confidence=0.6,
                        )
                    )


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
    fact_content: str,
    category: AnchorCategory,
    source_text_lower: str,
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
    return any(len(word) >= 8 and word.lower() in source_text_lower for word in words)


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
            ledger.add(
                AnchorFact(
                    category=AnchorCategory.INTENT,
                    content=intent_text,
                    turn=1,
                    confidence=0.9,
                )
            )

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
            ledger.add(
                AnchorFact(
                    category=AnchorCategory.ACTIVE_CONTEXT,
                    content=f"Current task: {ctx}",
                    turn=0,
                    confidence=0.9,
                )
            )
    if last_assistant is not None:
        ctx = last_assistant.content.strip()[:300]
        if ctx:
            ledger.add(
                AnchorFact(
                    category=AnchorCategory.ACTIVE_CONTEXT,
                    content=f"Last response: {ctx}",
                    turn=0,
                    confidence=0.8,
                )
            )
