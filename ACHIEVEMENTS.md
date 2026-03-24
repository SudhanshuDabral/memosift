# MemoSift — Verified Achievements

> All numbers from automated benchmarks on real production data. No cherry-picking, no synthetic-only metrics.

## Quality — Proven on Real Data

| Metric | Result | How Verified |
|---|---|---|
| **Fidelity probes (synthetic)** | **335/348 (96.3%)** | 9 domain datasets, 4 presets, 348 probes |
| **Fidelity probes (real sessions)** | **466/466 (100%)** | 11 real Claude Code sessions, auto-extracted probes |
| **Tool call integrity** | **100%** | Every benchmark run — 0 orphaned calls across 5.5M tokens |
| **System prompt preservation** | **100%** | Zone 1 pass-through, never compressed |
| **No hallucination** | **100%** | 27 negative probes — no fabricated content ever found |
| **Deterministic output** | **100%** | Same input + seed = identical output, every time |

## Compression — 11 Real Claude Code Sessions

**Corpus**: 11 production coding agent sessions, 158.9 MB, 10,922 messages, 5.5 million tokens, 4,799 tool calls.

| Preset | Avg Compression | Fact Retention | Tokens Saved | Opus Savings | Integrity |
|---|---|---|---|---|---|
| **coding** (conservative) | **2.91x** | **90.4%** | 1,950,477 | **$29.26** | 11/11 |
| **general** (balanced) | **5.10x** | **89.8%** | 2,168,956 | **$32.53** | 11/11 |

### Per-Session Detail (Coding Preset)

| Session | Size | Tokens | Compression | Retention | Integrity |
|---|---|---|---|---|---|
| Session 1 | 4.6 MB | 339K | 2.2x | 80.9% | OK |
| Session 2 | 5.9 MB | 442K | 2.6x | 98.2% | OK |
| Session 3 | 7.3 MB | 264K | 1.8x | 100.0% | OK |
| Session 4 | 8.8 MB | 464K | 3.9x | 91.7% | OK |
| Session 5 | 9.1 MB | 258K | 2.6x | 76.4% | OK |
| Session 6 | 9.9 MB | 490K | 4.2x | 91.9% | OK |
| Session 7 | 13.0 MB | 354K | 1.8x | 89.0% | OK |
| Session 8 | 18.4 MB | 753K | 3.3x | 87.9% | OK |
| Session 9 | 19.6 MB | 495K | 3.3x | 89.4% | OK |
| Session 10 | 21.5 MB | 608K | 2.6x | 97.2% | OK |
| **Session 11** | **40.9 MB** | **1.04M** | **3.7x** | **91.6%** | **OK** |

## Cost Savings at Scale

All numbers use the observed compression ratios from real production sessions. MemoSift itself costs $0.00 (zero LLM calls in deterministic mode).

| Volume | Opus ($15/MTok) | Sonnet ($3/MTok) | GPT-4o ($2.50/MTok) |
|---|---|---|---|
| 1M tokens/month | $9.82 saved | $1.96 saved | $1.64 saved |
| 10M tokens/month | $98 saved | $20 saved | $16 saved |
| 100M tokens/month | $982 saved | $196 saved | $164 saved |
| **1B tokens/month** | **$9,816/mo saved** | **$1,963/mo saved** | **$1,636/mo saved** |

## Test Coverage

| Suite | Tests | Status |
|---|---|---|
| Python unit tests | 547 | All passing |
| TypeScript unit tests | 160 | All passing |
| Quality probes (synthetic) | 348 | 96.3% pass |
| **Quality probes (real sessions)** | **466** | **100% pass** |
| Budget enforcement (incl. Hypothesis) | 16 | All passing |
| Tool call integrity | Verified every run | 100% |

## Framework Support

6 framework adapters with lossless round-trip:

| Framework | Status | Preserves |
|---|---|---|
| OpenAI SDK | Stable | refusal, annotations |
| Anthropic SDK | Stable | thinking blocks, cache_control, system prompt |
| Claude Agent SDK | Stable | compaction boundaries, tool nesting |
| Google ADK | Stable | function_calls, function_responses |
| LangChain | Stable | additional_kwargs, message types |
| **Vercel AI SDK** | **New in v0.6** | ImagePart, FilePart, isError, ToolCallPart |

## Architecture

- **7-layer pipeline** with Three-Zone Memory Model (system untouched, compressed untouched, new messages compressed)
- **7 compression engines** — each specialized: dedup, verbatim, IDF pruning, code signatures, JSON truncation, relevance filtering, discourse compression
- **Anchor Ledger** — 8-category append-only fact store that survives compression
- **Adaptive L0** — reads model context window, adjusts compression per pressure level
- **Incremental mode** — `CompressionState` caches across calls, `MemoSiftStream` for real-time
- **Audit-only resolution tracking** — detects question→decision arcs without affecting compression

## What Makes This Different

| Dimension | MemoSift | Typical LLM Compaction |
|---|---|---|
| **LLM cost** | $0.00 (deterministic) | Spends tokens to save tokens |
| **Latency** | <200ms | 1-3s (LLM inference) |
| **Tool integrity** | 100% verified | Often drops tool results |
| **Deterministic** | Same input = same output | LLM output varies |
| **Multi-cycle safe** | Zone 2 prevents re-compression | Degrades on repeated compaction |
| **Fact preservation** | Anchor ledger survives drops | Facts lost with source messages |
| **Framework adapters** | 6 (lossless round-trip) | Typically 1 (locked to provider) |
| **Observable** | CompressionReport with every decision | Black box |

---

*All benchmarks run deterministically on real production data. Source: `benchmarks/session_benchmark.py`, `benchmarks/session_fidelity.py`, `benchmarks/quality_probe.py`.*
