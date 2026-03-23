# Validate cross-language test vectors against the Python implementation.
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python" / "src"))

from memosift.config import MemoSiftConfig
from memosift.core.classifier import classify_messages
from memosift.core.deduplicator import deduplicate
from memosift.core.types import MemoSiftMessage

VECTORS_DIR = Path(__file__).resolve().parent / "test-vectors"


def load_vector(name: str) -> dict:
    """Load a test vector JSON file."""
    with open(VECTORS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def to_messages(raw: list[dict]) -> list[MemoSiftMessage]:
    """Convert raw dicts to MemoSiftMessage objects."""
    return [MemoSiftMessage.from_dict(d) for d in raw]


def validate_classify() -> bool:
    """Validate classify-001 vector."""
    vector = load_vector("classify-001.json")
    messages = to_messages(vector["input"])
    config = MemoSiftConfig(**vector.get("config", {}))

    segments = classify_messages(messages, config)

    expected = vector["expected_classifications"]
    passed = True
    for exp in expected:
        idx = exp["index"]
        expected_type = exp["type"]
        actual_type = segments[idx].content_type.value
        if actual_type != expected_type:
            print(f"  FAIL classify[{idx}]: expected {expected_type}, got {actual_type}")
            passed = False

    return passed


def validate_dedup() -> bool:
    """Validate dedup-001 vector."""
    vector = load_vector("dedup-001.json")
    messages = to_messages(vector["input"])
    config = MemoSiftConfig()

    segments = classify_messages(messages, config)
    deduped, _deps = deduplicate(segments, config)

    expected = vector["expected"]
    passed = True

    # Check that index 3 was deduplicated.
    dedup_idx = expected["deduplicated_indices"][0]
    deduped_content = deduped[dedup_idx].message.content
    check_str = expected["message_3_content_contains"]
    if check_str not in deduped_content:
        print(f"  FAIL dedup[{dedup_idx}]: expected content containing '{check_str}', got '{deduped_content[:80]}'")
        passed = False

    return passed


def validate_compress() -> bool:
    """Validate compress-001 vector — check invariants only (output is non-deterministic across configs)."""
    import asyncio
    from memosift.core.pipeline import compress

    vector = load_vector("compress-001.json")
    messages = to_messages(vector["input"])
    config = MemoSiftConfig(**vector.get("config", {}))

    compressed, report = asyncio.run(compress(messages, config=config))

    invariants = vector["expected_invariants"]
    passed = True

    # System prompt preserved.
    if invariants.get("system_prompt_preserved"):
        system_msgs = [m for m in compressed if m.role == "system"]
        if not system_msgs:
            print("  FAIL compress: system prompt not preserved")
            passed = False

    # Last user message preserved.
    if invariants.get("last_user_message_preserved"):
        user_msgs = [m for m in compressed if m.role == "user"]
        original_last = [m for m in messages if m.role == "user"][-1]
        if not user_msgs or user_msgs[-1].content != original_last.content:
            print("  FAIL compress: last user message not preserved")
            passed = False

    # Token budget respected.
    if invariants.get("compressed_tokens_within_budget") and config.token_budget:
        if report.compressed_tokens > config.token_budget:
            print(f"  FAIL compress: {report.compressed_tokens} tokens exceeds budget {config.token_budget}")
            passed = False

    # Tool call integrity.
    if invariants.get("tool_call_integrity"):
        tc_ids = set()
        tr_ids = set()
        for m in compressed:
            if m.tool_calls:
                for tc in m.tool_calls:
                    tc_ids.add(tc.id)
            if m.tool_call_id:
                tr_ids.add(m.tool_call_id)
        orphaned = tc_ids - tr_ids
        if orphaned:
            print(f"  FAIL compress: orphaned tool_calls without results: {orphaned}")
            passed = False

    return passed


def main() -> int:
    """Run all vector validations."""
    results: dict[str, bool] = {}

    print("Validating cross-language test vectors...\n")

    for name, fn in [
        ("classify-001", validate_classify),
        ("dedup-001", validate_dedup),
        ("compress-001", validate_compress),
    ]:
        try:
            ok = fn()
            results[name] = ok
            status = "PASS" if ok else "FAIL"
        except Exception as e:
            results[name] = False
            status = f"ERROR: {e}"
        print(f"  {status}  {name}")

    print()
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"{passed}/{total} vectors passed.")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
