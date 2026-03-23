# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-23

### Added

- **Layer 0: Context-Aware Adaptive Compression** — the pipeline now dynamically adjusts compression based on the model's context window utilization. Instead of fixed thresholds, MemoSift reads the model's context window size, estimates current usage, computes a pressure level (NONE/LOW/MEDIUM/HIGH/CRITICAL), and automatically tunes recent-turn protection, token budgets, pruning ratios, engine selection, and observation masking. This means the system never over-compresses when there's room, and aggressively compresses only when the window is genuinely under pressure.
  - `ContextWindowState` — immutable snapshot of model context window capacity and utilization
  - `Pressure` enum — 5 context pressure levels with distinct compression behaviors
  - `AdaptiveOverrides` — computed thresholds, engine gates, and observation masking flags
  - Model context window registry for 18 models across OpenAI, Anthropic, and Google families
  - Percentage-based recent-turn protection (replaces fixed `recent_turns` count)
  - Auto-budget derived from remaining context capacity
  - Pressure-scaled pruning ratios and entropy thresholds
  - Engine gating per pressure level (LOW = dedup+verbatim only, CRITICAL = all engines including Engine D)
  - Auto-enable Engine D (summarization) at CRITICAL pressure when LLM provider is available
  - Adapter-level auto-resolution: Anthropic and Claude Agent SDK adapters detect the model and compute pressure automatically
- `context_window` parameter on `compress()` and all 5 framework adapters
- 58 new Python tests (including Hypothesis property tests) for the adaptive system
- 39 new TypeScript tests mirroring Python
- Adaptive benchmark suite simulating 6 pressure levels across all SDK scenarios

### Changed

- Pipeline engine invocation now gated by `AdaptiveOverrides.engineGates` when Layer 0 is active
- Observation masking activates at HIGH/CRITICAL pressure regardless of tool result count
- Performance tier forced to `"full"` at HIGH/CRITICAL pressure to ensure all engines are available

### Performance

- **4.40x compression at CRITICAL pressure with LLM** (Engine D), up from 1.31x baseline on SDK conversations
- **100% fidelity** maintained across all pressure levels
- **100% tool call integrity** at all pressure levels
- **80% fact retention** at CRITICAL pressure (vs 90% at baseline — a deliberate tradeoff for 3.4x more compression)
- NONE pressure short-circuits the pipeline entirely — zero overhead when the context window has room

## [0.1.0] - 2026-03-22

### Added

- 6-layer compression pipeline with Three-Zone Memory Model
- 7 compression engines: Verbatim (3A), Pruner (3B), Structural (3C), Summarizer (3D), Relevance Pruner (3E), Discourse Compressor (3F), Importance Scorer (3G)
- Framework adapters: OpenAI SDK, Anthropic SDK, Claude Agent SDK, Google ADK, LangChain
- Anchor Ledger for critical fact preservation across compression cycles
- Cross-window dedup state for long-running agents
- Performance tiers with auto-detection (full, standard, fast, ultra_fast)
- Configuration presets: coding, research, support, data, general
- Heuristic token counter (zero external dependencies)
- Cross-language test vectors and JSON schemas
- 395 Python tests with 80%+ coverage
- Full TypeScript implementation mirroring Python (all 6 layers, 7 engines, 5 adapters)
- 39 TypeScript tests (vitest)
- Claude Code hook and transcript parser integrations (TypeScript only)
- GitHub Actions CI for both Python and TypeScript

[0.2.0]: https://github.com/memosift/memosift/releases/tag/v0.2.0
[0.1.0]: https://github.com/memosift/memosift/releases/tag/v0.1.0
