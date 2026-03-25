# LLM Inspector — post-compression quality feedback via three parallel LLM jobs.
#
# Runs AFTER compression completes, asynchronously. Does not slow down the
# compression pipeline. Produces project-specific protection rules that the
# deterministic engines read on the next session.
#
# Three independent jobs (run in parallel):
#   1. Entity Guardian: identifies entity names lost during compression
#   2. Fact Auditor: evaluates whether compressed context is actionable
#   3. Config Advisor: recommends parameter adjustments from compression history
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memosift.core.types import AnchorLedger, MemoSiftMessage
    from memosift.providers.base import MemoSiftLLMProvider
    from memosift.report import CompressionReport

logger = logging.getLogger("memosift.inspector")


# ── Project Memory ───────────────────────────────────────────────────────


@dataclass
class ProjectMemory:
    """Persistent, project-specific protection rules learned from LLM feedback.

    Read by the deterministic engines at session start. Written by the LLM
    inspector after each compression cycle. Accumulates over time.
    """

    protected_entities: list[str] = field(default_factory=list)
    """Entity names (wells, operators, people, projects) that must survive."""

    protected_path_prefixes: list[str] = field(default_factory=list)
    """File path prefixes that should always be preserved."""

    domain_patterns: list[str] = field(default_factory=list)
    """Domain-specific metric unit patterns (auto-detected)."""

    learned_config: dict[str, object] = field(default_factory=dict)
    """Parameter recommendations from the Config Advisor."""

    audit_scores: list[dict[str, object]] = field(default_factory=list)
    """Historical fact audit scores (last 20 sessions)."""

    sessions_analyzed: int = 0
    """Number of sessions analyzed so far."""

    def save(self, path: str) -> None:
        """Persist to JSON file."""
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> ProjectMemory:
        """Load from JSON file. Returns empty memory if file doesn't exist."""
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(
                protected_entities=data.get("protected_entities", []),
                protected_path_prefixes=data.get("protected_path_prefixes", []),
                domain_patterns=data.get("domain_patterns", []),
                learned_config=data.get("learned_config", {}),
                audit_scores=data.get("audit_scores", [])[-20:],
                sessions_analyzed=data.get("sessions_analyzed", 0),
            )
        except (json.JSONDecodeError, KeyError):
            return cls()

    def get_protection_strings(self) -> frozenset[str]:
        """Return all learned protection strings for the anchor extractor."""
        strings: set[str] = set()
        for entity in self.protected_entities:
            strings.add(entity)
        for prefix in self.protected_path_prefixes:
            strings.add(prefix)
        return frozenset(strings)


# ── LLM Prompts ──────────────────────────────────────────────────────────

_ENTITY_GUARDIAN_PROMPT = """\
You are analyzing a compression result. Compare the ORIGINAL conversation \
with the COMPRESSED version to find important entity names that were LOST.

ORIGINAL (first 50 messages, truncated per message):
{original_sample}

COMPRESSED (all messages):
{compressed_sample}

ANCHOR LEDGER (preserved facts):
{anchor_ledger}

Find entity names (people, companies, well names, project names, locations, \
tool names) that appear in the ORIGINAL but are MISSING from both the \
COMPRESSED version AND the anchor ledger.

Return ONLY a JSON array of lost entity strings. No explanation needed.
Example: ["WHITLEY-DUBOSE UNIT 1H", "Frio County", "EOG Resources"]

If nothing important was lost, return: []"""

_FACT_AUDITOR_PROMPT = """\
You are auditing a compressed AI conversation for quality. The compressed \
version will be used as context for the AI agent's next response.

COMPRESSED CONTEXT:
{compressed_sample}

ANCHOR LEDGER:
{anchor_ledger}

ORIGINAL TASK: {task}

Score each dimension 1-5:
1. **completeness**: Can the agent answer follow-up questions about the work done?
2. **numerical_integrity**: Are key numbers, rates, percentages preserved?
3. **entity_coverage**: Are important names (files, people, projects) present?
4. **tool_continuity**: Can the agent tell what tools were called and what they produced?
5. **actionability**: Could the agent continue the task without re-reading sources?

Return JSON only:
{{"completeness": N, "numerical_integrity": N, "entity_coverage": N, \
"tool_continuity": N, "actionability": N, \
"missing_critical": ["list of facts that should have survived"]}}"""

_CONFIG_ADVISOR_PROMPT = """\
You are analyzing compression performance history to recommend parameter changes.

RECENT AUDIT SCORES (last sessions):
{audit_history}

CURRENT CONFIG:
{current_config}

PATTERNS OBSERVED:
- Lost entities: {lost_entities}
- Compression ratio: {compression_ratio}
- Fact retention: {retention_pct}%

Based on the patterns, recommend parameter adjustments. Only suggest changes \
if there is clear evidence of a recurring problem.

Available parameters:
- entropy_threshold (1.5-2.5): lower = delete more boilerplate lines
- token_prune_keep_ratio (0.3-0.7): lower = prune more aggressively
- dedup_similarity_threshold (0.75-0.95): lower = catch more fuzzy duplicates
- json_array_threshold (2-15): higher = keep more JSON array items
- error_trace_policy ("PRESERVE" or "STACK"): PRESERVE = keep full traces

Return JSON only:
{{"recommendations": {{"param_name": value}}, \
"reasoning": "1-2 sentence explanation"}}

If no changes needed, return:
{{"recommendations": {{}}, "reasoning": "Current config is optimal"}}"""


def _extract_json(text: str) -> dict | list | None:
    """Extract the first JSON object or array from text.

    Handles LLM responses that include explanatory text after the JSON.
    Tries full parse first, then scans for the first { or [ and finds
    its matching closing brace/bracket.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Find first { or [ and try to parse from there.
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find matching closing brace/bracket by counting nesting depth.
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break

    return None


# ── Inspector Jobs ───────────────────────────────────────────────────────


async def _run_entity_guardian(
    original: list[MemoSiftMessage],
    compressed: list[MemoSiftMessage],
    ledger: AnchorLedger,
    llm: MemoSiftLLMProvider,
) -> list[str]:
    """Job 1: Identify entity names lost during compression."""
    original_sample = "\n".join(
        f"[{m.role}]: {(m.content or '')[:300]}"
        for m in original[:50]
    )
    compressed_sample = "\n".join(
        f"[{m.role}]: {(m.content or '')[:300]}"
        for m in compressed
    )
    ledger_text = ledger.render()[:2000]

    prompt = _ENTITY_GUARDIAN_PROMPT.format(
        original_sample=original_sample,
        compressed_sample=compressed_sample,
        anchor_ledger=ledger_text,
    )

    try:
        response = await llm.generate(prompt, max_tokens=512, temperature=0.0)
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(x for x in lines if not x.strip().startswith("```"))
        entities = _extract_json(text)
        if isinstance(entities, list):
            return [e for e in entities if isinstance(e, str) and len(e) >= 2]
    except Exception as e:
        logger.warning("Entity Guardian failed: %s", e)
    return []


async def _run_fact_auditor(
    compressed: list[MemoSiftMessage],
    ledger: AnchorLedger,
    task: str,
    llm: MemoSiftLLMProvider,
) -> dict | None:
    """Job 2: Evaluate compressed context quality."""
    compressed_sample = "\n".join(
        f"[{m.role}]: {(m.content or '')[:400]}"
        for m in compressed
    )
    ledger_text = ledger.render()[:2000]

    prompt = _FACT_AUDITOR_PROMPT.format(
        compressed_sample=compressed_sample,
        anchor_ledger=ledger_text,
        task=task or "Not specified",
    )

    try:
        response = await llm.generate(prompt, max_tokens=512, temperature=0.0)
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(x for x in lines if not x.strip().startswith("```"))
        return _extract_json(text)
    except Exception as e:
        logger.warning("Fact Auditor failed: %s", e)
    return None


async def _run_config_advisor(
    memory: ProjectMemory,
    report: CompressionReport,
    config_dict: dict,
    llm: MemoSiftLLMProvider,
) -> dict | None:
    """Job 3: Recommend parameter adjustments from compression history."""
    # Only advise after 3+ sessions of data.
    if memory.sessions_analyzed < 3:
        return {"recommendations": {}, "reasoning": "Need 3+ sessions before advising"}

    lost = memory.protected_entities[-20:]  # Recent lost entities.
    audit_history = json.dumps(memory.audit_scores[-5:], indent=2)

    prompt = _CONFIG_ADVISOR_PROMPT.format(
        audit_history=audit_history,
        current_config=json.dumps(config_dict, indent=2, default=str),
        lost_entities=json.dumps(lost[:10]),
        compression_ratio=f"{report.compression_ratio:.2f}x",
        retention_pct="n/a",
    )

    try:
        response = await llm.generate(prompt, max_tokens=512, temperature=0.0)
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(x for x in lines if not x.strip().startswith("```"))
        return _extract_json(text)
    except Exception as e:
        logger.warning("Config Advisor failed: %s", e)
    return None


# ── Main Inspector ───────────────────────────────────────────────────────


@dataclass
class InspectionResult:
    """Result of the post-compression LLM inspection."""

    lost_entities: list[str] = field(default_factory=list)
    audit_scores: dict[str, object] | None = None
    config_recommendations: dict[str, object] | None = None
    memory_updated: bool = False


async def inspect_compression(
    original: list[MemoSiftMessage],
    compressed: list[MemoSiftMessage],
    ledger: AnchorLedger,
    report: CompressionReport,
    llm: MemoSiftLLMProvider,
    *,
    memory: ProjectMemory | None = None,
    memory_path: str | None = None,
    task: str | None = None,
    config_dict: dict | None = None,
) -> InspectionResult:
    """Run all three inspector jobs in parallel after compression.

    This is the main entry point. Call it AFTER compress() returns, not during.
    It runs all three LLM jobs concurrently, updates project memory, and
    returns the inspection result.

    Args:
        original: The original (uncompressed) messages.
        compressed: The compressed messages from compress().
        ledger: The anchor ledger from compress().
        report: The compression report from compress().
        llm: LLM provider (Haiku recommended — cheap and fast).
        memory: Optional project memory to update. Created fresh if None.
        memory_path: Optional file path to persist memory after update.
        task: Optional task description for the fact auditor.
        config_dict: Optional current config as dict for the config advisor.

    Returns:
        InspectionResult with all findings.
    """
    if memory is None:
        memory = ProjectMemory()

    # Run all three jobs in parallel.
    entity_task = _run_entity_guardian(original, compressed, ledger, llm)
    audit_task = _run_fact_auditor(compressed, ledger, task, llm)
    config_task = _run_config_advisor(
        memory, report, config_dict or {}, llm
    )

    lost_entities, audit_scores, config_advice = await asyncio.gather(
        entity_task, audit_task, config_task
    )

    result = InspectionResult(
        lost_entities=lost_entities,
        audit_scores=audit_scores,
        config_recommendations=config_advice,
    )

    # Update project memory with findings.
    if lost_entities:
        for entity in lost_entities:
            if entity not in memory.protected_entities:
                memory.protected_entities.append(entity)
        # Cap at 500 entities to prevent unbounded growth.
        memory.protected_entities = memory.protected_entities[-500:]

    if audit_scores and isinstance(audit_scores, dict):
        memory.audit_scores.append(audit_scores)
        memory.audit_scores = memory.audit_scores[-20:]

        # Extract missing_critical items as protection strings.
        missing = audit_scores.get("missing_critical", [])
        if isinstance(missing, list):
            for item in missing:
                if (
                    isinstance(item, str)
                    and len(item) >= 3
                    and item not in memory.protected_entities
                ):
                    memory.protected_entities.append(item)

    if config_advice and isinstance(config_advice, dict):
        recs = config_advice.get("recommendations", {})
        if isinstance(recs, dict) and recs:
            memory.learned_config.update(recs)

    memory.sessions_analyzed += 1
    has_updates = bool(
        lost_entities or audit_scores
        or (config_advice and config_advice.get("recommendations"))
    )
    result.memory_updated = has_updates

    # Persist if path provided.
    if memory_path:
        try:
            memory.save(memory_path)
        except Exception as e:
            logger.warning("Failed to save project memory: %s", e)

    return result
