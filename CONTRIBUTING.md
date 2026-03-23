# Contributing to MemoSift

Thank you for your interest in contributing to MemoSift! This document covers the process for contributing to the project.

## License

MemoSift is licensed under the [MIT License](LICENSE). By contributing, you agree that your contributions will be licensed under the same terms.

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 22+ (for TypeScript)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/memosift/memosift.git
cd memosift

# Install Python package in development mode
cd python && pip install -e ".[dev]"

# Run tests (from repo root)
cd .. && python -m pytest tests/python/ -x -q
```

## Development Workflow

### Before You Start

- Check existing [issues](https://github.com/memosift/memosift/issues) and [pull requests](https://github.com/memosift/memosift/pulls) to avoid duplicate work.
- For non-trivial changes, open an issue first to discuss the approach.

### Making Changes

1. Fork the repository and create a branch from `main`.
2. Write tests for your changes. We use pytest with pytest-asyncio.
3. Run the full test suite and ensure all 395+ tests pass.
4. Run the linter and formatter:
   ```bash
   cd python && ruff format src/ && ruff check src/
   ```
5. Commit with a clear message following [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add new compression engine for markdown content
   fix: handle empty tool_call_id in deduplicator
   refactor: simplify Three-Zone partitioning logic
   ```

### Running Tests

```bash
# All tests
python -m pytest tests/python/ -x -q

# Single file
python -m pytest tests/python/test_pipeline.py -x -q

# Single test
python -m pytest tests/python/test_pipeline.py::test_function_name -x -q

# Skip slow/LLM tests
python -m pytest tests/python/ -x -q -m "not slow and not llm"

# With coverage
cd python && python -m pytest ../../tests/python/ --cov=memosift --cov-report=term-missing
```

### Running TypeScript Tests

```bash
# All tests (from typescript/ directory)
cd typescript && npm test

# Type check without emitting
cd typescript && npx tsc --noEmit

# Lint & format
cd typescript && npx biome check src/
cd typescript && npx biome format src/
```

### Cross-Language Validation

If your change affects compression output, verify test vectors still match:

```bash
python spec/validate_vectors.py
```

## CI Pipeline

Every push to `main` and every PR triggers **4 parallel jobs** in GitHub Actions:

### Job 1: `python-test` (Python 3.12 + 3.13 matrix)

Runs on both Python versions with `fail-fast: false` (both run even if one fails):

1. **Ruff lint** — enforces rules E, F, I, N, UP, B, SIM, TCH
2. **Ruff format check** — fails if any source file isn't formatted
3. **All tests** — runs `pytest -m "not slow and not llm"` (currently all 395 tests, since no tests carry those markers yet)
4. **Coverage gate** — same tests with `--cov=memosift`, **fails if coverage drops below 80%**

### Job 2: `python-full-test` (Python 3.13)

Runs **all** tests including any future `slow` or `llm`-marked tests. This is the safety net that catches everything Job 1 skips.

### Job 3: `typescript-test` (Node.js 22)

1. **Type check** — `tsc --noEmit` ensures all types are valid
2. **Biome lint** — enforces no `any`, consistent style
3. **Biome format** — checks formatting
4. **39 tests** — runs vitest covering types, config, classifier, pipeline, providers, and report

### Job 4: `cross-language-vectors` (Python 3.13)

Runs `spec/validate_vectors.py` against the 3 test vectors in `spec/test-vectors/`:

| Vector | What it validates |
|--------|------------------|
| `classify-001.json` | L1 classifies a 12-message coding session into correct content types (SYSTEM_PROMPT, RECENT_TURN, ERROR_TRACE, etc.) |
| `dedup-001.json` | L2 detects identical file reads and replaces the duplicate with a back-reference |
| `compress-001.json` | Full pipeline preserves system prompt, last user message, respects 500-token budget, and maintains tool call integrity |

If your change affects classification, dedup, or pipeline output, this job will catch regressions.

## Test Architecture

### Test data sources

Tests do **not** use external APIs, databases, or network calls. Everything is self-contained:

- **Inline data** — Most tests build `MemoSiftMessage` or `ClassifiedMessage` objects directly using a `_make_segment()` helper. This is the pattern to follow when adding new tests.
- **Shared fixtures** (`tests/python/conftest.py`) — `sample_messages` (6-message conversation with tool calls), `default_config`, and the 3 test vector loaders.
- **Test vectors** (`spec/test-vectors/*.json`) — Cross-language contract files. If you change compression output, update these and ensure both Python and (future) TypeScript produce the same results.
- **Mock providers** — `MockLLMProvider` and `FailingLLMProvider` in `test_engine_summarizer.py` for testing Engine D without real LLM calls.

### Test file map

| File | Tests | Layer/Component |
|------|-------|-----------------|
| `test_classifier.py` | 41 | L1: content type classification |
| `test_deduplicator.py` | 22 | L2: exact/fuzzy dedup, dependency maps |
| `test_pipeline.py` | 16 | Full 6-layer pipeline, Three-Zone Model |
| `test_three_zone.py` | 18 | Zone partitioning (system/compressed/raw) |
| `test_budget.py` | 8 | L6: token budget enforcement |
| `test_scorer.py` | 16 | L4: relevance scoring |
| `test_positioner.py` | 9 | L5: position optimization |
| `test_engine_verbatim.py` | 27 | Engine 3A: noise removal, entropy filtering |
| `test_engine_pruner.py` | 14 | Engine 3B: IDF-based token pruning |
| `test_engine_structural.py` | 13 | Engine 3C: code signatures, JSON truncation |
| `test_engine_summarizer.py` | 22 | Engine 3D: LLM summarization (mocked) |
| `test_adapters.py` | 24 | 5 framework adapters (OpenAI, Anthropic, Agent SDK, ADK, LangChain) |
| `test_adapter_roundtrip.py` | 14 | Lossless round-trip: thinking blocks, tool nesting, cache control |
| `test_anchor_ledger.py` | 19 | Anchor fact extraction and categories |
| `test_smoke.py` | 32 | Import checks, type construction, vector loading |
| `test_performance_tiers.py` | 23 | Tier auto-detection and layer skipping |
| `test_sprint1_improvements.py` | 36 | Anchor extraction, reasoning chains |
| `test_sprint2_improvements.py` | 25 | Importance scoring, discourse compression |
| `test_sprint3_improvements.py` | 13 | Structural engine AST fallbacks |

### Test markers

- `@pytest.mark.slow` — For tests that take > 5 seconds. Skipped in CI Job 1, included in Job 2.
- `@pytest.mark.llm` — For tests that need a real LLM provider. Skipped in CI Job 1, included in Job 2.
- No markers needed for standard tests — `asyncio_mode = auto` in pytest.ini handles async tests automatically.

### Writing new tests

1. Place tests in `tests/python/test_<component>.py`.
2. Use `_make_segment()` for building `ClassifiedMessage` objects:
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
3. Async tests just need `async def test_...` — no decorator required.
4. If your test needs an LLM, mock it. If it genuinely requires a real provider, mark it `@pytest.mark.llm`.

## Core Invariants

These must never be violated. PRs that break any of these will not be merged.

1. **Zero external model dependencies** in `core/`. No ML models, no torch, no transformers.
2. **Lossless by default.** Default config uses only verbatim deletion and dedup.
3. **Framework-agnostic core.** `core/` never imports from `adapters/` or framework libraries.
4. **Tool call integrity.** If an assistant message with `tool_calls` survives, all matching `tool_result` messages must also survive.
5. **Deterministic layers < 100ms** for 100K tokens.
6. **Layer fault tolerance.** If any layer throws, skip it and pass input unchanged.

## Code Style

### Python

- Modern type hints: `list[X]`, `X | None` (not `Optional[X]`).
- `from __future__ import annotations` at the top of every file.
- All public APIs are `async def`.
- Dataclasses for data, Protocol for interfaces.
- Ruff for formatting and linting (line length 100, rules: E, F, I, N, UP, B, SIM, TCH).
- Every public function has a docstring. Every module has a one-line comment at top.

### TypeScript

- TypeScript 5.7+, strict mode. No `any`.
- Async/await everywhere.
- Biome for formatting and linting.
- Interfaces for contracts, types for data shapes.

## What Makes a Good PR

- **Focused.** One logical change per PR. Don't mix a bug fix with a refactor.
- **Tested.** New features need tests. Bug fixes need a regression test.
- **Documented.** Update docstrings if you change public API signatures.
- **Small.** Prefer multiple small PRs over one large PR.

## Reporting Bugs

Use the [bug report template](https://github.com/memosift/memosift/issues/new?template=bug_report.yml). Include:
- MemoSift version
- Python/Node version
- Minimal reproduction case
- Expected vs actual behavior

## Requesting Features

Use the [feature request template](https://github.com/memosift/memosift/issues/new?template=feature_request.yml). Describe the problem you're solving, not just the solution you want.

## Questions?

Open a [discussion](https://github.com/memosift/memosift/discussions) or ask in our Discord community.
