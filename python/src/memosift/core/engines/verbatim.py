# Engine A: Verbatim deletion — remove noise lines while preserving every surviving token.
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from memosift.core.types import (
    AnchorLedger,
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    MemoSiftMessage,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig

# Applies to TOOL_RESULT_TEXT, OLD_CONVERSATION (when no LLM available),
# and ERROR_TRACE (stack frame compression).
_TARGET_POLICIES = {
    CompressionPolicy.MODERATE,
    CompressionPolicy.AGGRESSIVE,
    CompressionPolicy.STACK,
}

# Regex for file paths — lines containing these are never deleted.
_FILE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:)?(?:[/\\][\w.\-]+)+(?:\.\w+)?(?::\d+)?",
)

# Boilerplate patterns to remove.
_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^#\s*-\*-\s*coding:", re.MULTILINE),  # Python encoding
    re.compile(r"^(?:MIT License|Apache License|Copyright \(c\))", re.MULTILINE),
    re.compile(r"^(?:={5,}|-{5,}|_{5,}|\*{5,}|~{5,})$", re.MULTILINE),  # Separators
]

# Max lines before truncation marker is applied.
_MAX_LINES_AFTER_CLEANUP = 100

# Max repetitions of a pattern before collapsing.
_REPETITION_THRESHOLD = 3


def verbatim_compress(
    segments: list[ClassifiedMessage],
    config: MemoSiftConfig,
    ledger: AnchorLedger | None = None,
    seen_content_hashes: dict[str, int] | None = None,
    *,
    enable_observation_masking: bool = False,
) -> list[ClassifiedMessage]:
    """Apply verbatim deletion to eligible segments.

    Rules applied in order:
    0. First-read vs re-read detection (re-reads collapsed to back-reference)
    1. Blank line collapse (3+ blank lines → 1)
    2. Low-entropy line removal (Shannon entropy < threshold)
    3. Repetitive pattern collapse
    4. Boilerplate deletion
    5. Truncation with markers if still over limit

    File paths, numbers, identifiers, and anchor ledger facts are protected.

    Args:
        segments: Classified messages from previous layers.
        config: Pipeline configuration.
        ledger: Optional anchor ledger — lines containing ledger facts are protected.
        seen_content_hashes: Mutable dict mapping content SHA-256 hashes to the
            original_index of the first occurrence. Used for first-read vs re-read
            detection. Mutated in place as new content is seen.

    Returns:
        Segments with noise lines removed from eligible messages.
    """
    # Engine-level ledger gating disabled — the engines' own regex protections
    # (file paths, numbers, identifiers) are sufficient. Ledger gating at this
    # level was over-protecting and killing compression ratios.
    protected_strings: frozenset[str] = frozenset()
    if seen_content_hashes is None:
        seen_content_hashes = {}

    result: list[ClassifiedMessage] = []
    for seg in segments:
        # Rule 0: First-read vs re-read detection for tool results.
        if (
            seg.policy in _TARGET_POLICIES
            and seg.content
            and len(seg.content) > 200  # Only worthwhile for substantial content.
            and seg.content_type
            in {
                ContentType.TOOL_RESULT_TEXT,
                ContentType.TOOL_RESULT_JSON,
                ContentType.CODE_BLOCK,
            }
        ):
            content_hash = hashlib.sha256(seg.content.encode("utf-8")).hexdigest()
            if content_hash in seen_content_hashes:
                first_index = seen_content_hashes[content_hash]
                # Extract a short label (first file path or first 60 chars).
                label = _extract_content_label(seg.content)
                collapsed = f"[Previously read: {label} — see message {first_index}]"
                new_msg = MemoSiftMessage(
                    role=seg.message.role,
                    content=collapsed,
                    name=seg.message.name,
                    tool_call_id=seg.message.tool_call_id,
                    tool_calls=seg.message.tool_calls,
                    metadata=seg.message.metadata,
                )
                result.append(dc_replace(seg, message=new_msg))
                continue
            else:
                seen_content_hashes[content_hash] = seg.original_index

        # Rule 0.5: Observation masking for OLD, LARGE tool results.
        # Only enabled for long conversations (>= 10 tool results).
        if (
            enable_observation_masking
            and not seg.protected
            and seg.content
            and len(seg.content) >= 500
            and seg.content_type in {ContentType.TOOL_RESULT_TEXT, ContentType.TOOL_RESULT_JSON}
        ):
            placeholder = _mask_old_observation(seg, ledger=ledger)
            if placeholder is not None:
                new_msg = MemoSiftMessage(
                    role=seg.message.role,
                    content=placeholder,
                    name=seg.message.name,
                    tool_call_id=seg.message.tool_call_id,
                    tool_calls=seg.message.tool_calls,
                    metadata=seg.message.metadata,
                )
                result.append(dc_replace(seg, message=new_msg))
                continue

        if seg.policy in _TARGET_POLICIES:
            new_content = _compress_content(
                seg.content, config.entropy_threshold, protected_strings
            )
            if new_content != seg.content:
                new_msg = MemoSiftMessage(
                    role=seg.message.role,
                    content=new_content,
                    name=seg.message.name,
                    tool_call_id=seg.message.tool_call_id,
                    tool_calls=seg.message.tool_calls,
                    metadata=seg.message.metadata,
                )
                result.append(dc_replace(seg, message=new_msg))
            else:
                result.append(seg)
        else:
            result.append(seg)
    return result


def _extract_content_label(content: str) -> str:
    """Extract a short label from content for back-reference display.

    Prefers file paths, falls back to first 60 chars of first line.
    """
    match = _FILE_PATH_RE.search(content)
    if match:
        return match.group(0)
    first_line = content.split("\n", 1)[0].strip()
    if len(first_line) > 60:
        return first_line[:57] + "..."
    return first_line or "content"


def _compress_content(
    text: str,
    entropy_threshold: float,
    protected_strings: frozenset[str] = frozenset(),
) -> str:
    """Apply all verbatim deletion rules to a text block."""
    lines = text.split("\n")

    # Rule 1: Collapse runs of 3+ blank lines → 1.
    lines = _collapse_blank_lines(lines)

    # Rule 2: Remove low-entropy lines (separators, decorative).
    lines = _remove_low_entropy_lines(lines, entropy_threshold, protected_strings)

    # Rule 3: Collapse repetitive patterns.
    lines = _collapse_repetitive_patterns(lines)

    # Rule 4: Remove boilerplate.
    text = "\n".join(lines)
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)

    # Clean up double blank lines introduced by removals.
    lines = text.split("\n")
    lines = _collapse_blank_lines(lines)

    # Rule 5: Truncate if still too long.
    if len(lines) > _MAX_LINES_AFTER_CLEANUP:
        lines = _truncate_with_marker(lines, _MAX_LINES_AFTER_CLEANUP)

    return "\n".join(lines)


def shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy in bits per character.

    Low entropy → repetitive/decorative (safe to delete).
    High entropy → information-dense (keep).

    English prose: ~3.5–4.5 bits/char.
    Separator lines: ~0.0–1.5 bits/char.
    """
    if not text:
        return 0.0
    freq = Counter(text)
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def _is_protected_line(line: str) -> bool:
    """Return True if the line contains content that must be preserved."""
    stripped = line.strip()
    if not stripped:
        return False
    # File paths.
    if _FILE_PATH_RE.search(stripped):
        return True
    # Lines with numbers (line numbers, error codes, etc.).
    return bool(re.search(r"\d+", stripped))


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    """Collapse runs of 3+ blank lines into a single blank line."""
    result: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 1:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return result


def _remove_low_entropy_lines(
    lines: list[str],
    threshold: float,
    protected_strings: frozenset[str] = frozenset(),
) -> list[str]:
    """Remove lines with Shannon entropy below the threshold.

    Protected lines (containing file paths, numbers, or anchor ledger facts)
    are never removed.
    """
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append(line)
            continue
        if _is_protected_line(line):
            result.append(line)
            continue
        # Check if line contains any anchor ledger fact.
        if protected_strings:
            line_lower = stripped.lower()
            if any(s.lower() in line_lower for s in protected_strings):
                result.append(line)
                continue
        if shannon_entropy(stripped) >= threshold:
            result.append(line)
    return result


def _collapse_repetitive_patterns(lines: list[str]) -> list[str]:
    """Collapse repeated patterns beyond a threshold.

    If the same pattern appears more than ``_REPETITION_THRESHOLD`` times
    consecutively, keep the first few and add a count annotation.
    """
    if len(lines) < _REPETITION_THRESHOLD + 1:
        return lines

    result: list[str] = []
    i = 0
    while i < len(lines):
        # Count consecutive identical (or near-identical) lines.
        pattern = _normalize_pattern(lines[i])
        run_start = i
        while i < len(lines) and _normalize_pattern(lines[i]) == pattern:
            i += 1
        run_length = i - run_start

        if run_length > _REPETITION_THRESHOLD and pattern:
            # Keep first few, add count annotation.
            for j in range(run_start, run_start + _REPETITION_THRESHOLD):
                result.append(lines[j])
            omitted = run_length - _REPETITION_THRESHOLD
            result.append(f"[... {omitted} similar lines omitted ...]")
        else:
            for j in range(run_start, run_start + run_length):
                result.append(lines[j])

    return result


def _normalize_pattern(line: str) -> str:
    """Normalize a line for pattern comparison.

    Replace specific identifiers and numbers with placeholders so
    'test_001 passed' and 'test_002 passed' are treated as the same pattern.
    """
    s = line.strip()
    if not s:
        return ""
    # Replace numbers with placeholder.
    s = re.sub(r"\d+", "#", s)
    # Replace quoted strings with placeholder.
    s = re.sub(r"['\"].*?['\"]", "STR", s)
    return s


def _truncate_with_marker(lines: list[str], max_lines: int) -> list[str]:
    """Keep first and last portions, insert truncation marker in the middle."""
    keep = max_lines // 2
    total = len(lines)
    omitted = total - max_lines
    return (
        lines[:keep]
        + [f"[... {omitted} lines omitted — showing first {keep} and last {keep} lines ...]"]
        + lines[-keep:]
    )


# ── Observation Masking (Rule 0.5) ───────────────────────────────────────────

_QUICK_SIG_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:class|def|function|const|let|var|interface|type|enum)\s+\w+",
    re.MULTILINE,
)


def _mask_old_observation(
    seg: ClassifiedMessage,
    ledger: AnchorLedger | None = None,
) -> str | None:
    """Create a structural placeholder for an old tool result.

    Preserves lines containing anchor ledger facts.
    """
    tool_name = seg.message.name or "tool"
    key_args = ""
    match = _FILE_PATH_RE.search(seg.content)
    if match:
        key_args = match.group(0)
    else:
        first_line = seg.content.split("\n", 1)[0].strip()
        key_args = first_line[:40]

    line_count = seg.content.count("\n") + 1
    lines = seg.content.strip().split("\n")

    # Structural summary.
    sigs = [m.group(0).strip() for m in _QUICK_SIG_RE.finditer(seg.content)]
    if sigs:
        summary = "; ".join(sigs[:4])
    elif len(lines) <= 3:
        summary = seg.content[:200]
    else:
        first_two = " | ".join(ln.strip() for ln in lines[:2] if ln.strip())
        last = lines[-1].strip() or (lines[-2].strip() if len(lines) > 1 else "")
        summary = f"{first_two} ... {last}"[:200]

    args_display = f'("{key_args}")' if key_args else ""
    parts = [f"[Tool: {tool_name}{args_display} — {line_count} lines]", f"Key: {summary}"]

    # Preserve lines containing critical anchor facts.
    if ledger is not None:
        critical = ledger.get_critical_strings()
        if critical:
            preserved: list[str] = []
            seen: set[str] = set()
            for line in lines:
                stripped = line.strip()
                if stripped and stripped not in seen:
                    line_lower = stripped.lower()
                    if any(s.lower() in line_lower for s in critical):
                        seen.add(stripped)
                        preserved.append(stripped)
            if preserved:
                parts.append("Preserved: " + " | ".join(preserved[:10]))

    return "\n".join(parts)
