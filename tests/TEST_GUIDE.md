# MemoSift Test Guide

Overview of the test suite, data sources, and how to run tests.

## Running Tests

```bash
# Python — all 395 tests
python -m pytest tests/python/ -x -q

# Python — skip slow/LLM tests
python -m pytest tests/python/ -x -q -m "not slow and not llm"

# Python — single file
python -m pytest tests/python/test_pipeline.py -x -q

# Python — single test
python -m pytest tests/python/test_pipeline.py::test_function_name -x -q

# Python — with coverage (fails under 80%)
cd python && python -m pytest ../../tests/python/ --cov=memosift --cov-report=term-missing

# TypeScript — all 39 tests
cd typescript && npm test

# Cross-language vector validation
python spec/validate_vectors.py
```

## Test Data Sources

All test data is **synthetic**. No API keys, real user data, session history, or credentials exist anywhere in the test suite.

### Cross-Language Test Vectors (`spec/test-vectors/`)

Shared between Python and TypeScript. Both runtimes must produce identical output.

| Vector | Scenario | Data |
|--------|----------|------|
| `classify-001.json` | 12-message coding session: user asks to fix a bug in `auth.ts` | Synthetic tool calls (read_file, run_tests, edit_file) with a made-up TypeError |
| `dedup-001.json` | Same file read twice — second should be deduplicated | A trivial `add`/`subtract` math utility read twice |
| `compress-001.json` | Full pipeline with 500-token budget | A fake `AuthService` class with bcrypt/JWT methods |

### Shared Fixtures (`tests/python/conftest.py`)

| Fixture | Description |
|---------|-------------|
| `sample_messages` | 6-message conversation with a `read_file` tool call |
| `default_config` | `MemoSiftConfig()` with no overrides |
| `classify_vector` | Loads `classify-001.json` |
| `dedup_vector` | Loads `dedup-001.json` |
| `compress_vector` | Loads `compress-001.json` |

### Inline Data

Most tests build `MemoSiftMessage` or `ClassifiedMessage` objects directly using a helper pattern:

```python
def _make_segment(content, content_type=ContentType.OLD_CONVERSATION, **kwargs):
    return ClassifiedMessage(
        message=MemoSiftMessage(role="assistant", content=content),
        content_type=content_type,
        policy=CompressionPolicy.COMPRESS,
        original_index=0,
        **kwargs,
    )
```

### Mock LLM Providers

Tests that exercise LLM-dependent paths use mock providers — no real API calls:

| Mock | Returns | Used in |
|------|---------|---------|
| `MockLLMProvider` | `"Summary of content."` | `test_engine_summarizer.py` |
| `FailingLLMProvider` | Throws immediately | `test_engine_summarizer.py` (fault tolerance) |
| `MockScorerLLM` | Hardcoded relevance scores | `test_scorer.py` |

## Python Test Files (395 tests)

| File | Tests | Layer/Component | What it validates |
|------|-------|-----------------|-------------------|
| `test_smoke.py` | 32 | All | Imports work, types construct correctly, vectors load |
| `test_classifier.py` | 41 | L1 | Content type classification (SYSTEM_PROMPT, ERROR_TRACE, CODE_BLOCK, etc.) |
| `test_deduplicator.py` | 22 | L2 | SHA-256 exact dedup, TF-IDF fuzzy dedup, dependency maps, cross-window state |
| `test_pipeline.py` | 16 | Full pipeline | 6-layer orchestration, zone partitioning, fault tolerance |
| `test_three_zone.py` | 18 | Pipeline | Zone 1/2/3 partitioning, `_memosift_compressed` passthrough |
| `test_budget.py` | 8 | L6 | Token budget enforcement, dependency-aware dropping |
| `test_scorer.py` | 16 | L4 | Keyword relevance scoring, LLM-based scoring (mocked) |
| `test_positioner.py` | 9 | L5 | Block reordering, tool call atomic grouping |
| `test_engine_verbatim.py` | 27 | Engine 3A | Noise removal, entropy filtering, re-read collapse, boilerplate deletion |
| `test_engine_pruner.py` | 14 | Engine 3B | IDF-based token pruning, keep ratio behavior |
| `test_engine_structural.py` | 13 | Engine 3C | Code to signatures, JSON array truncation, AST fallback to regex |
| `test_engine_summarizer.py` | 22 | Engine 3D | LLM summarization with mock, failure fallback to passthrough |
| `test_adapters.py` | 24 | Adapters | All 5 adapters: OpenAI, Anthropic, Agent SDK, ADK, LangChain |
| `test_adapter_roundtrip.py` | 14 | Adapters | Thinking blocks survive, tool nesting preserved, cache_control kept |
| `test_anchor_ledger.py` | 19 | Anchor Ledger | Fact extraction across 8 categories, append-only behavior |
| `test_performance_tiers.py` | 23 | Pipeline | Tier auto-detection (full/standard/fast/ultra_fast), layer skipping |
| `test_sprint1_improvements.py` | 36 | Anchors | Anchor extraction, reasoning chain detection |
| `test_sprint2_improvements.py` | 25 | Engines 3F, 3G | Importance scoring (6 signals), discourse compression |
| `test_sprint3_improvements.py` | 13 | Engine 3C | Structural engine AST fallback, regex extraction |

## TypeScript Test File (39 tests)

| File | Tests | What it validates |
|------|-------|-------------------|
| `memosift.test.ts` | 39 | Types, config, presets, classifier, pipeline, providers, report, and cross-language vector parity with Python |

## Test Markers

| Marker | Meaning | CI behavior |
|--------|---------|-------------|
| `@pytest.mark.slow` | Takes > 5 seconds | Skipped in fast CI job, included in full test job |
| `@pytest.mark.llm` | Requires real LLM provider | Skipped in fast CI job, included in full test job |
| *(no marker)* | Standard test | Runs in all CI jobs |

Async tests need no decorator — `asyncio_mode = auto` in `pytest.ini` handles them automatically.

## Adding New Tests

1. Place tests in `tests/python/test_<component>.py` or `tests/typescript/`.
2. Use inline `MemoSiftMessage` / `ClassifiedMessage` objects — don't add external data files unless creating a new cross-language test vector.
3. If your test needs an LLM, mock it. If it genuinely requires a real provider, mark it `@pytest.mark.llm`.
4. If your change affects compression output, update the test vectors in `spec/test-vectors/` and ensure both Python and TypeScript produce the same results.
