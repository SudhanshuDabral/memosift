# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/memosift/memosift/releases/tag/v0.1.0
