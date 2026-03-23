# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-03-23

### Added

- **`@memosift/mcp-server` — MCP server for agent-discoverable compression tools.** A standalone npm package that exposes MemoSift's capabilities as 8 MCP tools via stdio transport. Any MCP-compatible client (Claude Desktop, Claude Code, Cursor, Windsurf) can connect and use context compression without writing code.
  ```json
  { "mcpServers": { "memosift": { "command": "npx", "args": ["@memosift/mcp-server"] } } }
  ```
  - `memosift_check_pressure` — check if compression is needed before doing it
  - `memosift_compress` — compress messages in any framework format (OpenAI, Anthropic, LangChain, Google ADK) with `compressed_entries` for expand lookup
  - `memosift_configure` — create/update compression sessions with presets and overrides
  - `memosift_get_facts` — retrieve anchor facts (file paths, errors, decisions) extracted during compression
  - `memosift_expand` — re-expand a previously compressed message to see original content
  - `memosift_report` — detailed compression metrics (summary/layers/decisions/full)
  - `memosift_list_sessions` — list all active sessions with metadata
  - `memosift_destroy` — destroy sessions to free memory
  - Session manager with configurable TTL (`MEMOSIFT_SESSION_TTL_MS`), touch-on-access, and periodic cleanup
  - Deterministic-only by default — zero LLM calls. Engine D enabled via `MEMOSIFT_LLM_PROVIDER` env var
  - SDK parity tests: MCP compress output matches direct adapter output for all OpenAI and Anthropic benchmark fixtures
- `MemoSiftSession` public getters: `model`, `preset`, `framework` properties and `set_framework()` / `setFramework()` method (Python/TypeScript)
- MCP server CI job in GitHub Actions (`ci.yml`)
- MCP server publish job in GitHub Actions (`publish.yml`) with `npm-mcp` environment gate
- 30 new MCP server tests (24 tool tests + 6 SDK parity tests) in `tests/mcp-server/`

## [0.4.0] - 2026-03-23

### Added

- **`MemoSiftSession` — the recommended entry point for MemoSift.** A stateful session class that owns the `AnchorLedger`, `CrossWindowState`, and `CompressionCache` internally, collapsing the raw API from 7 objects + 51 knobs into a single constructor + a single `compress()` call. Available in both Python and TypeScript.
  ```python
  from memosift import MemoSiftSession
  session = MemoSiftSession("coding", model="claude-sonnet-4-6")
  compressed, report = await session.compress(messages, usage_tokens=150_000)
  ```
  - `compress()` — accepts framework-native messages (auto-detected), returns compressed messages in the same format
  - `check_pressure()` — check context window pressure without compressing
  - `reconfigure()` — change config (preset/overrides) while preserving accumulated session state
  - `expand()` — re-expand a previously compressed message from cache
  - `save_state()` / `load_state()` — persist ledger + dedup hashes to JSON (cache is session-lifecycle only, not serialized)
  - `ledger`, `facts`, `last_report`, `system` properties
  - State file includes `"version": 1` for forward-compatible schema evolution
- **Framework auto-detection** (`detect_framework()`) — inspects message shape via duck typing to determine the source SDK (OpenAI, Anthropic, Claude Agent SDK, Google ADK, LangChain, or MemoSiftMessage). No framework imports required. Detection is cached after the first `compress()` call.
- `MemoSiftSession` and `detect_framework` exported from both `memosift` (Python) and `memosift` (npm)
- 44 new Python tests (18 detect + 26 session)
- 37 new TypeScript tests (14 detect + 23 session)

## [0.3.0] - 2026-03-23

### Added

- **Adaptive override transparency** — `CompressionReport.adaptive_overrides` now exposes exactly which config fields Layer 0 changed and why. Maps field name to `(original_value, effective_value)` for every overridden field (`recent_turns`, `token_budget`, `token_prune_keep_ratio`, `entropy_threshold`, `performance_tier`, `enable_summarization`). `None`/`null` when Layer 0 is inactive — fully backward compatible.
  - `AdaptiveOverrides.overrides` field (Python: immutable `MappingProxyType`, TypeScript: `Record<string, OverrideEntry>`)
  - `OverrideEntry` type alias in TypeScript: `[original: unknown, effective: unknown]`
- **TypeScript Layer 0 parity** — the full adaptive compression pipeline is now wired into the TypeScript runtime. Previously, `computeAdaptiveThresholds()`, `resolveContextWindow()`, engine gates, and all multiplier tables were implemented in `context-window.ts` but never called from `pipeline.ts`. Now the TypeScript pipeline matches Python exactly: resolve → recalibrate usage → compute thresholds → replace config → short-circuit at NONE → gate engines.
  - `contextWindow` parameter added to TypeScript `CompressOptions` interface
  - Engine gating on all 7 compression engines via L0 `engineGates`
  - `contextWindow` parameter added to all 5 TypeScript adapter convenience functions (`compressOpenAIMessages`, `compressAnthropicMessages`, `compressAgentSdkMessages`, `compressAdkEvents`, `compressLangChainMessages`)
  - Anthropic and Claude Agent SDK TypeScript adapters now auto-resolve `ContextWindowState` from their `model` parameter (matching Python behavior)
- 15 new TypeScript tests for L0 adaptive pipeline integration (`l0-adaptive.test.ts`)

### Fixed

- Layer 0 no longer silently overrides user config values — all overrides are tracked and surfaced in the compression report
- TypeScript pipeline now runs Layer 0 adaptive compression (was dead code in v0.2.0)
- TypeScript adapters now forward `contextWindow` to the pipeline (was missing in v0.2.0)

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
