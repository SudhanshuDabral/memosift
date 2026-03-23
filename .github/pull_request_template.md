## Summary

<!-- What does this PR do? Keep it to 1-3 sentences. -->

## Changes

<!-- Bullet list of what changed and why. -->

-

## Test plan

<!-- How did you verify this works? -->

- [ ] All existing tests pass (`python -m pytest tests/python/ -x -q`)
- [ ] New tests added for changed behavior
- [ ] Linter passes (`cd python && ruff check src/`)
- [ ] Cross-language vectors still valid (`python spec/validate_vectors.py`) *(if compression output changed)*

## Invariant checklist

- [ ] No external dependencies added to `core/`
- [ ] Tool call integrity preserved (tool_calls and tool_results stay paired)
- [ ] Layer fault tolerance maintained (layers skip on error, never crash)
- [ ] Default config remains lossless (no summarization without opt-in)
