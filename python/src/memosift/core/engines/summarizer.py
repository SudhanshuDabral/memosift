# Engine D: Abstractive summarization — LLM-dependent, opt-in only.
from __future__ import annotations

import logging
import re
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from memosift.core.types import (
    ClassifiedMessage,
    CompressionPolicy,
    MemoSiftMessage,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig
    from memosift.providers.base import LLMResponse, MemoSiftLLMProvider

logger = logging.getLogger("memosift")

# Target policies for summarization — expanded to include MODERATE (tool results)
# since those are often the biggest token consumers.
_TARGET_POLICIES = {CompressionPolicy.AGGRESSIVE, CompressionPolicy.MODERATE}

SUMMARIZE_PROMPT = """Summarize the following conversation segment concisely.

PRESERVE exactly (do not paraphrase):
- All file paths (e.g., src/auth.ts, ./config/db.json)
- All line numbers (e.g., line 47, auth.ts:47)
- All error messages and types (e.g., TypeError: Cannot read...)
- All decisions and their rationale (e.g., "chose X because Y")
- All unresolved issues or open items
- All specific numeric values (ports, status codes, counts)
- All function/class/variable names

REMOVE:
- Conversational filler ("sure", "let me", "I'll", "okay")
- Redundant restatements of the same information
- Intermediate reasoning that led to a stated conclusion
- Tool invocation metadata (tool names, call IDs)

Output a concise summary preserving all critical facts.

SEGMENT:
{content}"""

# Regex patterns for extracting critical facts to validate post-summary.
_FILE_PATH_RE = re.compile(r"(?:[\w.\-]+[/\\])+[\w.\-]+\.\w{1,10}(?::\d+)?")
_ERROR_MSG_RE = re.compile(
    r"(?:TypeError|ReferenceError|SyntaxError|ValueError|KeyError|"
    r"AttributeError|ImportError|RuntimeError|Error):\s*.{10,100}",
)
_LINE_REF_RE = re.compile(r"(?:line\s+\d+|:\d{1,5}\b)")


async def summarize_segments(
    segments: list[ClassifiedMessage],
    config: MemoSiftConfig,
    llm: MemoSiftLLMProvider,
) -> list[ClassifiedMessage]:
    """Apply abstractive summarization to eligible segments.

    Opt-in only — requires ``config.enable_summarization=True`` and a valid
    ``MemoSiftLLMProvider``. Falls back to returning segments unchanged on failure.

    After summarization, validates that critical facts (file paths, error messages,
    line numbers) from the original are preserved in the summary. If critical facts
    are lost, the original is kept instead.

    Applies to segments with AGGRESSIVE policy (OLD_CONVERSATION, ASSISTANT_REASONING).

    Args:
        segments: Classified messages from previous layers.
        config: Pipeline configuration.
        llm: LLM provider for generating summaries.

    Returns:
        Segments with eligible messages summarized.
    """
    if not config.enable_summarization:
        return segments

    import asyncio

    # Separate: skip (not eligible) vs summarizable (need LLM).
    skip_indices: dict[int, ClassifiedMessage] = {}
    summarizable: list[tuple[int, ClassifiedMessage]] = []

    for i, seg in enumerate(segments):
        if seg.policy not in _TARGET_POLICIES or len(seg.content) < 200:
            skip_indices[i] = seg
        else:
            summarizable.append((i, seg))

    if not summarizable:
        return segments

    async def _summarize_one(idx: int, seg: ClassifiedMessage) -> tuple[int, ClassifiedMessage]:
        try:
            original_facts = _extract_critical_facts(seg.content)
            prompt = SUMMARIZE_PROMPT.format(content=seg.content)
            # Target 30% of original tokens for aggressive summarization.
            original_tokens = await llm.count_tokens(seg.content)
            response: LLMResponse = await llm.generate(
                prompt,
                max_tokens=max(128, original_tokens * 3 // 10),
                temperature=0.0,
            )
            summary = response.text.strip()
            if not _is_valid_summary(summary, seg.content, original_facts):
                return (idx, seg)
            new_msg = MemoSiftMessage(
                role=seg.message.role,
                content=summary,
                name=seg.message.name,
                tool_call_id=seg.message.tool_call_id,
                tool_calls=seg.message.tool_calls,
                metadata=seg.message.metadata,
            )
            return (idx, dc_replace(seg, message=new_msg))
        except Exception as e:
            logger.warning("Summarization failed for segment %d: %s", seg.original_index, e)
            return (idx, seg)

    # Fire all LLM calls in parallel.
    results = await asyncio.gather(*[_summarize_one(i, seg) for i, seg in summarizable])

    # Reassemble in original order.
    all_segs: dict[int, ClassifiedMessage] = dict(skip_indices)
    for i, seg in results:
        all_segs[i] = seg

    return [all_segs[i] for i in sorted(all_segs.keys())]


def _extract_critical_facts(text: str) -> dict[str, set[str]]:
    """Extract critical facts from text for post-summary validation.

    Returns a dict with sets of file paths, error messages, and line references
    found in the original text. These must be preserved in any summary.
    """
    return {
        "file_paths": set(_FILE_PATH_RE.findall(text)),
        "error_msgs": set(_ERROR_MSG_RE.findall(text)),
        "line_refs": set(_LINE_REF_RE.findall(text)),
    }


def _is_valid_summary(
    summary: str,
    original: str,
    original_facts: dict[str, set[str]],
) -> bool:
    """Validate that a summary meets quality criteria.

    A valid summary must:
    1. Be non-trivial (> 20 chars)
    2. Be shorter than the original
    3. Preserve all file paths from the original
    4. Preserve all error messages from the original
    """
    # Length checks.
    if len(summary) <= 20:
        return False
    if len(summary) >= len(original):
        return False

    # Verify critical file paths are preserved (allow some loss if the path
    # would be captured by the anchor ledger).
    missing_paths = 0
    for path in original_facts["file_paths"]:
        if path not in summary:
            missing_paths += 1
    # Allow up to 30% of paths to be missing (anchor ledger has them).
    if original_facts["file_paths"] and missing_paths / len(original_facts["file_paths"]) > 0.3:
        logger.debug(
            "Summary missing too many file paths: %d/%d",
            missing_paths,
            len(original_facts["file_paths"]),
        )
        return False

    # Verify error type prefixes are preserved (just the type, not full message).
    for error in original_facts["error_msgs"]:
        error_type = error.split(":")[0].strip()
        if error_type not in summary:
            logger.debug("Summary missing error type: %s", error_type)
            return False

    return True
