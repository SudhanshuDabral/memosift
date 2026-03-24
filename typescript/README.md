# MemoSift

**Your AI agent dies at 150K tokens. MemoSift keeps it alive.**

Context-aware compression engine that sits between your agent and the LLM. Reads the model's context window, detects pressure, and compresses only when needed. Zero LLM calls by default, sub-200ms latency, 100% fidelity, 100% tool call integrity. Drop-in adapters for OpenAI, Anthropic, Claude Agent SDK, Google ADK, LangChain, and Vercel AI SDK. Stateful sessions, real-time streaming, and an MCP server for zero-code integration.

## Why MemoSift

| What | MemoSift | OpenAI Compaction | Anthropic Compaction | LLMLingua-2 |
|------|:--------:|:-----------------:|:--------------------:|:-----------:|
| **Context-aware** | Reads model window, adapts automatically | No | No | No |
| **Compression** | 5.1x (production avg) | ~86-93% removed | ~58.6% savings | 2-14x |
| **Quality** | 4.17/5.0 functional, 100% fidelity | 3.35/5.0 | 3.44/5.0 | 88-100% |
| **Tool integrity** | 100% across 5.5M tokens, 4,799 tool calls | 0% tool result survival | Unknown | Not tracked |
| **LLM required** | No (optional Engine D) | Yes | Yes | Yes (355M params) |
| **Cost** | $0 (deterministic) | LLM cost per call | LLM cost per call | GPU cost |
| **Open source** | Yes | No | No | Yes |
| **Framework adapters** | 6 (lossless round-trip) | None | None | None |

## Install

```bash
npm install memosift
```

Requires Node.js >= 22. Zero runtime dependencies in core.

## Quick Start

```typescript
import { compress, createMessage, createConfig } from "memosift";

const messages = [
  createMessage("system", "You are a helpful assistant."),
  createMessage("user", "What's in app.py?"),
  createMessage("assistant", "Let me check.", { toolCalls: [...] }),
  createMessage("tool", "def main(): ...", { toolCallId: "tc1", name: "read_file" }),
  createMessage("assistant", "Here's what app.py contains..."),
  createMessage("user", "Refactor the main function."),
];

const { messages: compressed, report } = await compress(messages);

console.log(`${report.compressionRatio.toFixed(1)}x compression`);
console.log(`${report.originalTokens} -> ${report.compressedTokens} tokens`);
```

## Stateful Sessions (Recommended)

`MemoSiftSession` is the recommended entry point — it owns the anchor ledger, dedup state, and compression cache internally:

```typescript
import { MemoSiftSession } from "memosift";

const session = new MemoSiftSession({ preset: "coding", model: "claude-sonnet-4-6" });

// Compress framework-native messages (auto-detected format)
const { messages: compressed, report } = await session.compress(messages, { usageTokens: 150_000 });

// Facts survive across compression cycles
for (const fact of session.facts) {
  console.log(`[${fact.category}] ${fact.content}`);
}

// Check pressure without compressing
const pressure = session.checkPressure(150_000);

// Re-expand a collapsed message
const original = session.expand(5);

// Save/restore session state
session.saveState("session.json");
const restored = MemoSiftSession.loadState("session.json");
```

## Real-Time Streaming

`MemoSiftStream` processes messages as they arrive, buffering until context pressure warrants compression:

```typescript
import { MemoSiftStream } from "memosift";

const stream = new MemoSiftStream({ preset: "coding", model: "claude-sonnet-4-6" });

for (const msg of incomingMessages) {
  const event = await stream.push(msg);
  if (event.compressed) {
    console.log(`Saved ${event.tokensSaved} tokens at ${event.pressure} pressure`);
  }
}

// Force compression at end of conversation
await stream.flush();
const compressed = stream.messages;
const facts = stream.facts;
```

## Incremental Compression

`CompressionState` caches IDF vocabulary, classification results, and token counts across `compress()` calls:

```typescript
import { MemoSiftSession } from "memosift";

const session = new MemoSiftSession({ preset: "coding", model: "claude-sonnet-4-6", incremental: true });

// Each call reuses cached state — faster on subsequent compressions
const r1 = await session.compress(window1, { usageTokens: 100_000 });
const r2 = await session.compress(window2, { usageTokens: 150_000 });
```

## Adaptive Compression (Layer 0)

MemoSift reads the model's context window size, estimates current utilization, and dynamically adjusts every compression parameter. No more fixed thresholds that don't know whether you're running GPT-4o (128K) or Claude Opus 4.6 (1M).

```typescript
import { compress, contextWindowFromModel } from "memosift";

// Tell MemoSift the model and current usage
const cw = contextWindowFromModel("claude-haiku-4-5", 150_000);
// Haiku has 200K window, 150K used = 75% consumed = HIGH pressure = aggressive compression

const { messages: compressed, report } = await compress(messages, { contextWindow: cw });
```

| Pressure | Context Remaining | What MemoSift Does |
|----------|:-----------------:|:-------------------|
| **NONE** | >60% | Skips compression entirely -- zero overhead |
| **LOW** | 40-60% | Light: dedup + verbatim noise removal only |
| **MEDIUM** | 25-40% | Standard: adds pruning, structural, discourse |
| **HIGH** | 10-25% | Aggressive: all engines, observation masking |
| **CRITICAL** | <10% | Maximum: auto-enables LLM summarization if available |

**Model registry**: 18 models built-in (GPT-4o/4.1, Claude 4.x, Gemini 2.5). Unknown models use 200K default. When an agent switches models mid-session, MemoSift recalibrates automatically.

## Framework Adapters

6 adapters with lossless round-trip fidelity. All preserve thinking blocks, cache control, annotations, and tool call nesting.

### OpenAI SDK

```typescript
import { compressOpenAIMessages } from "memosift";

const { messages: compressed, report } = await compressOpenAIMessages(messages);
// Feed compressed directly to client.chat.completions.create()
```

### Anthropic SDK

```typescript
import { compressAnthropicMessages } from "memosift";

const { result, report } = await compressAnthropicMessages(messages, {
  system: "You are a coding assistant.",
});
// result.system + result.messages -> client.messages.create()
```

### Claude Agent SDK

```typescript
import { compressAgentSdkMessages } from "memosift";

const { messages: compressed, report } = await compressAgentSdkMessages(messages);
// Handles compaction boundaries, thinking blocks, tool result nesting
```

### Google ADK

```typescript
import { compressAdkEvents } from "memosift";

const { messages: compressed, report } = await compressAdkEvents(events);
// Preserves function_calls and function_responses
```

### LangChain

```typescript
import { compressLangChainMessages } from "memosift";

const { messages: compressed, report } = await compressLangChainMessages(messages);
// Works with HumanMessage, AIMessage, ToolMessage, SystemMessage
```

### Vercel AI SDK

```typescript
import { compressVercelMessages } from "memosift";

const { messages: compressed, report } = await compressVercelMessages(messages);
// Handles TextPart, ToolCallPart, ToolResultPart, ImagePart, FilePart
// Works with CoreMessage format from Vercel AI SDK v3+
```

## Configuration

### Presets

```typescript
import { compress, createPreset } from "memosift";

const config = createPreset("coding");   // Conservative: never lose file paths, errors
// Also: "research", "support", "data", "general"

const { messages: compressed, report } = await compress(messages, { config });
```

### Custom Configuration

```typescript
import { createConfig } from "memosift";

const config = createConfig({
  recentTurns: 3,                    // Protect last 3 conversational turns
  tokenBudget: 100_000,             // Max output tokens (null = no limit)
  dedupSimilarityThreshold: 0.85,   // Cosine similarity for fuzzy dedup
  entropyThreshold: 2.0,            // Min Shannon entropy to keep a line
  tokenPruneKeepRatio: 0.6,         // Fraction of tokens retained in IDF pruning
  enableAnchorLedger: true,         // Extract critical facts during compression
  modelName: "claude-sonnet-4-6",   // Auto-sets budget + pricing
});
```

## How It Works

```
Messages In
  -> L0: Adaptive    -- Read model context window, compute pressure, tune thresholds
  -> L1: Classify    -- Assign content types (system, code, error, tool result, etc.)
  -> L2: Dedup       -- SHA-256 exact + MinHash/TF-IDF fuzzy deduplication
  -> L2.5: Coalesce  -- Merge consecutive short assistant messages
  -> L3: Compress    -- 7 type-specific engines:
       3A: Verbatim  -- Noise removal, re-read collapse, boilerplate deletion
       3B: Pruner    -- IDF-based token pruning (remove low-information words)
       3C: Structural -- Code -> signatures, JSON -> truncated arrays
       3D: Summarizer -- LLM abstractive summarization (opt-in only)
       3E: Relevance  -- Drop segments with low query-keyword overlap
       3F: Discourse  -- Remove elaboration, hedging, filler
       3G: Importance -- 6-signal scoring, shield assignment
  -> L4: Score       -- Task-aware relevance scoring
  -> L5: Position    -- Attention-curve reordering (disabled by default)
  -> L6: Budget      -- Enforce token budget, respect dependencies
Messages Out + CompressionReport
```

### Key Properties

- **Context-aware** -- reads the model's context window, compresses only when needed
- **Zero runtime dependencies** -- no ML models, no torch, no transformers
- **Lossless by default** -- only dedup and verbatim deletion unless you opt in to summarization
- **Tool call integrity** -- if a `tool_call` survives, all matching `tool_result` messages survive
- **Deterministic** -- same input + same config = same output
- **Zero overhead** -- skips compression entirely when the context window has room
- **Model-switching safe** -- recalibrates automatically when the agent switches models
- **Fault-tolerant** -- any layer that throws is skipped, pipeline never crashes

## Benchmarks

### Verified on Real Production Data

| Metric | Result |
|---|---|
| **Compression** | 2.91x (coding) / 5.10x (general) on 11 real Claude Code sessions |
| **Fact retention** | 90.4% (coding) / 466 out of 466 fidelity probes pass |
| **Tool call integrity** | 100% across 5.5 million tokens, 4,799 tool calls |
| **Quality probes** | 96.3% on 9 synthetic domains, **100% on real sessions** |
| **Cost** | $0.00 -- zero LLM calls in deterministic mode |
| **Latency** | <200ms per compression call |
| **Tests** | 547 Python + 160 TypeScript, all passing |

### Adaptive Compression (SDK Conversations)

| Pressure | Compression | Fidelity | Fact Retention | Tool Integrity |
|:---------|:-----------:|:--------:|:--------------:|:--------------:|
| NONE | 1.0x | 100% | 87.9% | ALL PASS |
| LOW | 1.85x | 100% | 91.3% | ALL PASS |
| MEDIUM | 2.58x | 100% | 83.5% | ALL PASS |
| HIGH | 3.17x | 100% | 79.9% | ALL PASS |
| **CRITICAL + LLM** | **4.40x** | **100%** | **79.9%** | **ALL PASS** |

### Production Sessions (11 Real Claude Sessions, 5.5M Tokens)

| Metric | Value |
|:-------|:------|
| Average compression | **5.09x** |
| Fact retention | **90.5%** |
| Tool call integrity | **100% (11/11 sessions)** |
| Total tokens saved | **2.16M** |
| Cost saved (Opus pricing) | **$32.45** |

### Synthetic Benchmarks (9 Domains)

| Config | Avg Ratio | Quality | Cost |
|:-------|:---------:|:-------:|:----:|
| Default (deterministic) | **6.4x** | 96.3% | $0.00 |
| Aggressive | **20.7x** | 89.7% | $0.00 |
| Max observed | **57.7x** | -- | $0.00 |

### Cost Savings at Scale

| Volume | Opus ($15/MTok) | Sonnet ($3/MTok) | GPT-4o ($2.50/MTok) |
|---|---|---|---|
| 1M tokens/month | $9.82 saved | $1.96 saved | $1.64 saved |
| 10M tokens/month | $98 saved | $20 saved | $16 saved |
| 100M tokens/month | $982 saved | $196 saved | $164 saved |
| **1B tokens/month** | **$9,816/mo saved** | **$1,963/mo saved** | **$1,636/mo saved** |

## Three-Zone Memory Model

MemoSift partitions every message list into three zones to prevent re-compression:

| Zone | What | Behavior |
|------|------|----------|
| **Zone 1** | System prompts | Pass through untouched |
| **Zone 2** | Previously compressed | Pass through untouched |
| **Zone 3** | New raw messages | Compressed by the pipeline |

This means MemoSift is safe to call repeatedly -- it never re-compresses its own output.

## Cross-Window Deduplication

For long-running agents, share dedup state across compression cycles:

```typescript
import { compress, createCrossWindowState } from "memosift";

const state = createCrossWindowState();

const result1 = await compress(window1, { crossWindow: state });
const result2 = await compress(window2, { crossWindow: state });
// Window 2 dedup catches content repeated from window 1
```

## MCP Server

Use MemoSift from any MCP-compatible client without writing code:

```bash
npx @memosift/mcp-server
```

```json
{
  "mcpServers": {
    "memosift": {
      "command": "npx",
      "args": ["@memosift/mcp-server"]
    }
  }
}
```

8 tools: `memosift_check_pressure`, `memosift_compress`, `memosift_configure`, `memosift_get_facts`, `memosift_expand`, `memosift_report`, `memosift_list_sessions`, `memosift_destroy`.

## Claude Code Integration

The TypeScript package includes a PreCompact hook for Claude Code:

```json
// ~/.claude/settings.json
{
  "hooks": {
    "PreCompact": [{
      "type": "command",
      "command": "npx memosift-hook"
    }]
  }
}
```

## Development

```bash
cd typescript && npm install && npm test  # 160 tests
cd typescript && npm run lint
```

Python package also available: `pip install memosift` (547 tests, 80%+ coverage).

## License

MIT
