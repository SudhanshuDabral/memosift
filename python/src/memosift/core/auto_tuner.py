# Level 2 Content Detection — auto-tunes config based on incoming message analysis.
#
# Replaces static presets with data-driven configuration. Scans message content
# to detect conversation type (code-heavy, data-heavy, tool-heavy, etc.) and
# sets optimal compression parameters accordingly.
#
# Design principles:
# - Runs ONCE per session (or per explicit retune), NOT per compress() call
# - Layer 0 (pressure-based) still runs per-call ON TOP of auto-tuned config
# - Explicit user overrides are never touched ("parameter locking")
# - All decisions logged to CompressionReport for transparency
from __future__ import annotations

import re
from dataclasses import dataclass, fields
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from memosift.core.types import CompressionPolicy, ContentType, MemoSiftMessage

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig

# ── Content signal patterns ──────────────────────────────────────────────

_CODE_RE = re.compile(
    r"\b(?:def |class |function |import |from |const |let |var |async |await |"
    r"return |if \(|else \{|for \(|while \(|switch |try \{|catch )\b"
)
_ERROR_RE = re.compile(
    r"\b(?:Error|Exception|Traceback|FAIL|TypeError|KeyError|ValueError|"
    r"SyntaxError|ReferenceError|AttributeError|RuntimeError)\b"
)
_JSON_START_RE = re.compile(r"^\s*[\[{]", re.MULTILINE)
_FILE_PATH_RE = re.compile(
    r"(?:[a-zA-Z]:)?(?:[./\\])?(?:[\w.\-]+[/\\])+[\w.\-]+\.\w{1,10}"
)
_NUMERIC_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_TABLE_RE = re.compile(r"\|[^|]+\|[^|]+\|")
_UNIT_RATIO_RE = re.compile(r"\b\d[\d,.]*\s+[A-Za-z]+/[A-Za-z]+\b")

# Domain hint patterns — detected from content to auto-add metric_patterns.
_DOMAIN_HINTS: dict[str, list[str]] = {
    "energy": [
        "Mcf/d", "bbl/d", "STB/d", "psig", "psia", "bbl", "STB",
        "Mcf", "MMcf", "BOE", "BOPD", "Scf/STB", "STB/MMcf",
        "GOR", "WOR", "GLR", "WGR", "EUR", "API",
    ],
    "financial": ["bps", "AUM", "NAV", "EPS", "EBITDA", "YoY", "QoQ"],
    "tech": ["QPS", "RPS", "p99", "p95", "p50", "SLA"],
    "medical": ["mg/dL", "mmHg", "BPM", "IU/L", "mEq/L"],
}


# ── Content profile ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContentProfile:
    """Quantified profile of conversation content for auto-tuning decisions."""

    total_messages: int
    total_chars: int
    user_turns: int
    tool_result_count: int
    tool_call_count: int
    code_density: float        # fraction of messages containing code patterns
    error_density: float       # fraction of messages containing error patterns
    json_density: float        # fraction of messages containing JSON structures
    numeric_density: float     # fraction of messages with 3+ numeric values
    file_path_density: float   # fraction of messages containing file paths
    table_density: float       # fraction of messages containing markdown tables
    unit_ratio_density: float  # fraction of messages with X/Y unit ratios
    avg_message_length: int    # average message length in chars
    duplicate_ratio: float     # estimated duplicate content ratio
    detected_domains: tuple[str, ...] = ()  # auto-detected domain hints


def profile_messages(messages: list[MemoSiftMessage]) -> ContentProfile:
    """Analyze message content to build a quantified profile.

    Single-pass analysis, fast (~1ms for 500 messages). Samples first 2000
    chars of each message for pattern matching.
    """
    total = len(messages)
    if total == 0:
        return ContentProfile(
            total_messages=0, total_chars=0, user_turns=0,
            tool_result_count=0, tool_call_count=0,
            code_density=0, error_density=0, json_density=0,
            numeric_density=0, file_path_density=0, table_density=0,
            unit_ratio_density=0, avg_message_length=0, duplicate_ratio=0,
        )

    user_turns = 0
    tool_results = 0
    tool_calls = 0
    code_count = 0
    error_count = 0
    json_count = 0
    numeric_count = 0
    file_path_count = 0
    table_count = 0
    unit_ratio_count = 0
    total_chars = 0
    seen_hashes: set[int] = set()
    duplicates = 0

    for msg in messages:
        content = msg.content or ""
        total_chars += len(content)

        if msg.role == "user":
            user_turns += 1
        if msg.role == "tool":
            tool_results += 1
        if msg.tool_calls:
            tool_calls += len(msg.tool_calls)

        if not content:
            continue

        # Quick duplicate detection via hash of first 200 chars.
        h = hash(content[:200])
        if h in seen_hashes:
            duplicates += 1
        else:
            seen_hashes.add(h)

        # Pattern detection on first 2000 chars for speed.
        sample = content[:2000]
        if _CODE_RE.search(sample):
            code_count += 1
        if _ERROR_RE.search(sample):
            error_count += 1
        if _JSON_START_RE.search(sample):
            json_count += 1
        if len(_NUMERIC_RE.findall(sample)) >= 3:
            numeric_count += 1
        if _FILE_PATH_RE.search(sample):
            file_path_count += 1
        if _TABLE_RE.search(sample):
            table_count += 1
        if _UNIT_RATIO_RE.search(sample):
            unit_ratio_count += 1

    non_empty = max(
        total - sum(1 for m in messages if not (m.content or "")), 1
    )

    # Detect domains from content of first 30 messages.
    detected_domains: list[str] = []
    sample_text = " ".join((m.content or "")[:500] for m in messages[:30]).lower()
    for domain, patterns in _DOMAIN_HINTS.items():
        if any(p.lower() in sample_text for p in patterns):
            detected_domains.append(domain)

    return ContentProfile(
        total_messages=total,
        total_chars=total_chars,
        user_turns=user_turns,
        tool_result_count=tool_results,
        tool_call_count=tool_calls,
        code_density=code_count / non_empty,
        error_density=error_count / non_empty,
        json_density=json_count / non_empty,
        numeric_density=numeric_count / non_empty,
        file_path_density=file_path_count / non_empty,
        table_density=table_count / non_empty,
        unit_ratio_density=unit_ratio_count / non_empty,
        avg_message_length=total_chars // max(total, 1),
        duplicate_ratio=duplicates / max(total, 1),
        detected_domains=tuple(detected_domains),
    )


# ── Auto-tuner ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AutoTuneResult:
    """Result of auto-tuning with full transparency."""

    profile: ContentProfile
    tuned_params: dict[str, object]   # param_name -> chosen value
    reasons: dict[str, str]           # param_name -> human-readable reason
    locked_params: frozenset[str]     # params the caller explicitly set (not touched)
    detected_style: str               # "code", "data", "mixed", "conversation"


def auto_tune(
    config: MemoSiftConfig,
    messages: list[MemoSiftMessage],
    locked_params: frozenset[str] = frozenset(),
) -> tuple[MemoSiftConfig, AutoTuneResult]:
    """Analyze messages and adapt config parameters based on content.

    Parameter locking: any param name in ``locked_params`` is never changed.
    Explicit user overrides (values that differ from MemoSiftConfig defaults)
    are also treated as locked — the auto-tuner respects intentional choices.

    Args:
        config: The base config (may have user overrides).
        messages: The incoming messages to analyze.
        locked_params: Parameter names the caller explicitly wants preserved.

    Returns:
        Tuple of (tuned_config, result_with_transparency).
    """
    profile = profile_messages(messages)
    if profile.total_messages == 0:
        return config, AutoTuneResult(
            profile=profile, tuned_params={}, reasons={},
            locked_params=locked_params, detected_style="empty",
        )

    # Detect which params the user explicitly set (differ from defaults).
    defaults = _get_defaults()
    user_locked = set(locked_params)
    for f in fields(config):
        if f.name in ("policies", "metric_patterns", "context_window"):
            continue  # These are handled specially below.
        current = getattr(config, f.name)
        default = defaults.get(f.name)
        if current != default and f.name not in user_locked:
            user_locked.add(f.name)

    tuned: dict[str, object] = {}
    reasons: dict[str, str] = {}

    def _set(name: str, value: object, reason: str) -> None:
        if name not in user_locked:
            tuned[name] = value
            reasons[name] = reason

    # ── Detect conversation style ──
    is_code = profile.code_density > 0.25
    is_data = profile.numeric_density > 0.35 or profile.table_density > 0.15
    is_error_heavy = profile.error_density > 0.15
    is_tool_heavy = profile.tool_result_count > 10
    is_json_heavy = profile.json_density > 0.4
    is_duplicate_heavy = profile.duplicate_ratio > 0.08
    is_long = profile.total_messages > 100
    has_unit_ratios = profile.unit_ratio_density > 0.05

    if is_code and is_error_heavy:
        style = "code_debug"
    elif is_code:
        style = "code"
    elif is_data:
        style = "data"
    elif is_tool_heavy and is_json_heavy:
        style = "tool_heavy"
    else:
        style = "mixed"

    # ── recent_turns ──
    # Always 2 — anchor ledger provides the safety net for the 3rd+ turn.
    _set("recent_turns", 2, f"{profile.user_turns} user turns, ledger as safety net")

    # ── entropy_threshold ──
    # Code has naturally high entropy — use higher threshold to preserve it.
    # Data/tables have lower entropy — use lower threshold.
    if is_code:
        _set("entropy_threshold", 2.3,
             f"code-heavy ({profile.code_density:.0%} code density), preserve structure")
    elif is_data or has_unit_ratios:
        _set("entropy_threshold", 1.9,
             f"data-heavy ({profile.numeric_density:.0%} numeric), remove boilerplate")
    elif is_json_heavy:
        _set("entropy_threshold", 2.0,
             f"JSON-heavy ({profile.json_density:.0%}), moderate filtering")
    else:
        _set("entropy_threshold", 2.1, "balanced default")

    # ── token_prune_keep_ratio ──
    # More file paths/errors = keep more (these are critical identifiers).
    # More JSON/tables = can prune more (protected patterns catch numbers).
    if profile.file_path_density > 0.3 and is_error_heavy:
        _set("token_prune_keep_ratio", 0.6,
             f"file-path + error heavy ({profile.file_path_density:.0%} paths, "
             f"{profile.error_density:.0%} errors), preserve identifiers")
    elif is_data or is_json_heavy:
        _set("token_prune_keep_ratio", 0.5,
             "data/JSON-heavy, numbers protected by anchor extractor")
    else:
        _set("token_prune_keep_ratio", 0.55, "balanced default")

    # ── dedup_similarity_threshold ──
    # Many duplicates detected: lower threshold to catch more.
    # Tool-heavy with repeated outputs: lower threshold.
    if is_duplicate_heavy:
        _set("dedup_similarity_threshold", 0.80,
             f"high duplicate ratio ({profile.duplicate_ratio:.0%}), aggressive dedup")
    elif is_tool_heavy:
        _set("dedup_similarity_threshold", 0.82,
             f"tool-heavy ({profile.tool_result_count} results), moderate dedup")
    elif is_code:
        _set("dedup_similarity_threshold", 0.87,
             "code sessions have natural similarity, moderate threshold")
    else:
        _set("dedup_similarity_threshold", 0.85, "balanced default")

    # ── relevance_drop_threshold ──
    # Long conversations accumulate irrelevant old context — drop more.
    # Short conversations: keep more, everything might be relevant.
    if is_long:
        _set("relevance_drop_threshold", 0.06,
             f"long session ({profile.total_messages} msgs), drop old irrelevant segments")
    elif is_tool_heavy:
        _set("relevance_drop_threshold", 0.05,
             "tool-heavy, moderate relevance filtering")
    else:
        _set("relevance_drop_threshold", 0.04,
             "short/medium session, preserve more context")

    # ── json_array_threshold ──
    # Data-heavy: keep more items (numbers matter).
    # Code: small JSON configs, truncate earlier.
    if is_data:
        _set("json_array_threshold", 8,
             f"data-heavy ({profile.numeric_density:.0%} numeric), preserve more array items")
    elif is_json_heavy:
        _set("json_array_threshold", 6,
             f"JSON-heavy ({profile.json_density:.0%}), moderate truncation")
    else:
        _set("json_array_threshold", 5, "balanced default")

    # ── code_keep_signatures ──
    _set("code_keep_signatures", is_code or profile.code_density > 0.1,
         f"code density {profile.code_density:.0%}")

    # ── enable_resolution_compression ──
    # Enable for longer conversations where deliberation accumulates.
    if profile.user_turns >= 5:
        _set("enable_resolution_compression", True,
             f"{profile.user_turns} user turns, compress resolved deliberation")

    # ── enable_anchor_ledger ──
    _set("enable_anchor_ledger", True, "always enabled for safety net")

    # ── ERROR_TRACE policy ──
    policies: dict[ContentType, CompressionPolicy] = dict(config.policies)
    if "policies" not in user_locked:
        if is_error_heavy and is_long:
            # Many errors in a long session — most are resolved, use STACK.
            policies[ContentType.ERROR_TRACE] = CompressionPolicy.STACK
            reasons["policy:ERROR_TRACE"] = (
                f"error-heavy ({profile.error_density:.0%}) + long session, "
                "anchor ledger preserves error type+message"
            )
        elif is_code and not is_long:
            # Code debugging in progress — preserve full traces.
            policies[ContentType.ERROR_TRACE] = CompressionPolicy.PRESERVE
            reasons["policy:ERROR_TRACE"] = "active code debugging, preserve full traces"
        else:
            policies[ContentType.ERROR_TRACE] = CompressionPolicy.STACK
            reasons["policy:ERROR_TRACE"] = "balanced default"

    if is_code or profile.code_density > 0.1:
        policies[ContentType.CODE_BLOCK] = CompressionPolicy.SIGNATURE
        reasons["policy:CODE_BLOCK"] = f"code present ({profile.code_density:.0%}), keep signatures"

    tuned["policies"] = policies

    # ── metric_patterns ──
    # Auto-detect domain from content and add patterns.
    metric_patterns = list(config.metric_patterns)
    for domain in profile.detected_domains:
        hints = _DOMAIN_HINTS.get(domain, [])
        for p in hints:
            if p not in metric_patterns:
                metric_patterns.append(p)
    if metric_patterns != list(config.metric_patterns):
        tuned["metric_patterns"] = metric_patterns
        reasons["metric_patterns"] = f"auto-detected domains: {', '.join(profile.detected_domains)}"

    # ── Apply tuned parameters ──
    replace_kwargs = {k: v for k, v in tuned.items() if k != "policies"}
    replace_kwargs["policies"] = policies
    if "metric_patterns" in tuned:
        replace_kwargs["metric_patterns"] = tuned["metric_patterns"]

    tuned_config = dc_replace(config, **replace_kwargs)

    return tuned_config, AutoTuneResult(
        profile=profile,
        tuned_params=tuned,
        reasons=reasons,
        locked_params=frozenset(user_locked),
        detected_style=style,
    )


def _get_defaults() -> dict[str, object]:
    """Get default values from MemoSiftConfig for detecting user overrides."""
    from memosift.config import MemoSiftConfig

    d = MemoSiftConfig()
    return {f.name: getattr(d, f.name) for f in fields(d)}
