# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-03-25

### Added

- **Context Intelligence Engine** — MemoSift evolves from a compression utility into an adaptive context intelligence engine. 15 improvements across 5 phases, +83% compression improvement on real production data.

- **Agentic Pattern Detector (Layer 1.5)** — new pipeline layer that detects 5 waste patterns specific to AI agent conversations: duplicate tool calls (collapsed to back-references), failed+retried tool calls (marked for compression), large code arguments (truncated to signatures), thought process blocks (reclassified as ASSISTANT_REASONING), and KPI restatement (marked for pruning). Runs after classification, before deduplication.

- **Contextual Metric Intelligence** — domain-agnostic heuristic that detects significant numerical metrics without hardcoded unit lists. 6 contextual signals: ratio-unit patterns (catches Mcf/d, mg/dL, req/s), high precision numbers, non-common context words (~200 word negative filter), comparison context, table cell detection, JSON key context. Configurable via `metricPatterns` for domain-specific overrides.

- **LLM Inspector** — post-compression quality feedback via 3 parallel LLM jobs (Entity Guardian, Fact Auditor, Config Advisor). Runs AFTER compression, asynchronously, not in the hot path. Produces `ProjectMemory` — persistent, project-specific protection rules that the deterministic engines read on the next session.

- **Content Detection Auto-Tuner** — analyzes incoming message content (code density, error density, JSON structure, numeric data, tool call patterns) and auto-selects optimal compression parameters. Replaces static preset guessing with data-driven configuration. Available via `MemoSiftConfig(autoTune=True)` or `preset("auto")`.

- **Working Memory Categories** — 5 new `AnchorCategory` values: PARAMETERS, CONSTRAINTS, ASSUMPTIONS, DATA_SCHEMA, RELATIONSHIPS. Each with regex extraction patterns for structured domain knowledge capture.

- **Entity Extraction Improvements** — ALL-CAPS multi-word entities (well names, operator names), single ALL-CAPS words with common-abbreviation filter, large comma-separated numbers. Catches "WHITLEY-DUBOSE UNIT 1H", "EOG", "95,467" as anchor facts.

- **New presets**: `energy` (oil & gas with 19 metric patterns), `financial` (7 financial metric patterns), `auto` (content-detection auto-tuning)

- **Cross-session dedup persistence** — `CrossWindowState.saveToFile()` / `loadFromFile()` for multi-session dedup hash reuse

- **Compression Feedback module** — ACON-style failure reporting via `CompressionFeedback.reportMissing()`. Accumulates protection patterns from reported losses.

- **Tiered Memory Architecture** — Hot (Anchor Ledger), Warm (Working Memory Summary via `ledger.workingMemorySummary()`), Cold (CompressionCache for on-demand re-expansion)

- 53 new Python tests (25 metric intelligence + 17 agentic patterns + 11 production benchmarks)
- Production trace fixtures from 3 real debug sessions (144 messages, 139 critical facts)
- Real-LLM benchmark scripts (`engine_d_benchmark.py`, `feedback_loop_benchmark.py`, `feedback_loop_traces.py`)
- Full TypeScript parity for all new features (4 new modules + 12 updated files)

### Changed

- **`coding` preset optimized** — leverages anchor ledger + resolution tracker as safety nets for more aggressive compression. `recentTurns`: 3->2, `entropyThreshold`: 2.5->2.1, `tokenPruneKeepRatio`: 0.7->0.55, `dedupSimilarityThreshold`: 0.90->0.85, `ERROR_TRACE` policy: PRESERVE->STACK, `enableResolutionCompression`: true. Result: +83% more compression with <2pp retention cost.
- **Resolution Tracker graduated** from audit-only to compression-affecting (gated by `enableResolutionCompression` config flag). Resolved deliberation arcs and superseded messages get AGGRESSIVE compression policy.
- **Engine D prompt redesigned** — structured FACTS + SUMMARY output format with brevity bias prevention (rejects summaries losing >30% of numerical values)
- **Observation masking threshold** made adaptive based on segment count (<100: 8, 100-500: 12, 500+: 15)
- **Observation masking** now preserves ALL file paths found in content (not just first match)
- **Importance scoring** split from 6 to 7 signals: generic numerical density (0.05 weight) vs domain metric density (0.10 weight), plus anchor fact coverage (0.10 weight)
- **Discourse compressor** uses keep_ratio=0.6 for causal clauses containing numerical evidence (was 0.2)
- **Relevance pruner** query expansion now includes high-value IDENTIFIERS (Metric:, Amount:, ID:)
- **JSON array truncation** now appends `_stats` dict with min/max/mean/count for numeric fields
- **Engine C** re-enabled protected strings from ProjectMemory (targeted, not broad anchor ledger)
- `ClassifiedMessage.content` returns `""` instead of `None` for null content (fixes NoneType layer errors)
- Version bumped to 0.7.0 (Python + TypeScript)

### Performance

- **5.32x** avg compression on 11 real Claude Code sessions (was 2.91x — **+83% improvement**)
- **2.2M tokens saved** across 11 sessions ($33.15 at Opus pricing — was $29.26)
- **96.0% quality probes** on 9 synthetic datasets (was 94.5% — **+1.5pp**)
- **100% tool call integrity** maintained across all configurations
- TypeScript benchmarks: **5.5x** avg on synthetics, **95.4%** quality probes, **93% parity** with Python

## [0.6.0] - 2026-03-24

### Added

- **Vercel AI SDK adapter** — 6th framework adapter for the largest TypeScript AI ecosystem. Lossless round-trip for `CoreMessage` format: `TextPart`, `ToolCallPart`, `ToolResultPart`, `ImagePart`, `FilePart`. Hyphenated part types (`tool-call`, `tool-result`) correctly distinguished from Anthropic's underscored variants.
  - `compressVercelMessages()` convenience function (Python + TypeScript)
  - `VercelAILLMProvider` wraps Vercel AI SDK's `generateText()` (TypeScript)
  - Auto-detection via `detectFramework()` — `"vercel_ai"` added to `VALID_FRAMEWORKS`
  - `ai >= 3.0.0` optional peer dependency
  - Deep import: `memosift/adapters/vercel-ai`
- **Incremental compression** — `CompressionState` caches IDF vocabulary, classification results, and token counts across `compress()` calls. The Three-Zone Model already skips Zone 2 (previously compressed); state additionally caches per-layer artifacts so Zone 3 processing is faster on subsequent calls.
  - `state` parameter on `compress()` (Python + TypeScript)
  - `MemoSiftSession(incremental=True)` creates and maintains state automatically
  - `CompressionState` exported from both packages
- **`MemoSiftStream`** — real-time compression stream for processing messages as they arrive. Buffers messages until context pressure warrants compression, then compresses only new messages.
  - `push(message)` → returns `StreamEvent` with action/pressure/tokens_saved
  - `flush()` → forces compression regardless of pressure
  - `messages`, `facts`, `session` properties
- **Audit-only resolution tracker** — detects question→deliberation→decision arcs and supersession patterns (corrections, status updates) in conversations. Logged to `CompressionReport.resolution_signals` as informational signals — does NOT modify compression behavior. Provides data for future semantic compression decisions.
- **Real session fidelity probes** — 466 auto-extracted quality probes from 11 real Claude Code sessions (5.5M tokens). Validates that file paths, error messages, tool names, and decisions survive compression on production data. 100% pass rate on coding preset.
  - `benchmarks/session_probe_extractor.py` — auto-extract probes from JSONL sessions
  - `benchmarks/session_fidelity.py` — run probes against compressed real sessions
- 16 new budget enforcement tests including Hypothesis property tests
- 34 new Vercel adapter tests (Python), 20 new incremental tests, 10 new stream tests
- 30+ new TypeScript tests (Vercel adapter, incremental, stream)

### Fixed

- **Mutable metadata sharing in coalescer** — merged messages now use a shallow copy of metadata instead of sharing a reference with the original, preventing cross-contamination between messages
- **Missing error patterns in classifier** — added `throw new Error` (JavaScript/TypeScript), `panic!()` (Rust), `Unhandled rejection` (Node.js), and Node.js stack frame patterns. Non-Python errors now correctly classified as `ERROR_TRACE` instead of `TOOL_RESULT_TEXT`
- **Budget truncation at character boundary** — emergency truncation now snaps to nearest newline boundary instead of splitting mid-word, mid-URL, or mid-identifier
- **Inconsistent content hashing in verbatim engine** — re-read detection now normalizes whitespace before hashing (consistent with deduplicator), preventing false negatives on content with trailing whitespace differences

### Changed

- Version bumped to 0.6.0 (Python + TypeScript)
- `VALID_FRAMEWORKS` now includes `"vercel_ai"` (7 frameworks total)
- `MemoSiftSession` accepts `incremental: bool` parameter
- `CompressionReport` includes `resolution_signals` field (audit-only)

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
