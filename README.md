# MemoSift

**Your AI agent dies at 128K tokens. MemoSift keeps it alive.**

Deterministic context compression engine that sits between your agent and the LLM. Zero LLM calls, sub-2s latency, 97% fact retention. Drop-in adapters for OpenAI, Anthropic, Claude Agent SDK, Google ADK, and LangChain.

## Install

```bash
# Python — core (zero dependencies)
pip install memosift

# With framework adapters
pip install memosift[openai]
pip install memosift[anthropic]
pip install memosift[langchain]
pip install memosift[google-adk]
pip install memosift[all]
```

```bash
# TypeScript / Node.js (>= 22)
npm install memosift
```

## Quick Start

### Python

```python
from memosift import compress, MemoSiftConfig, MemoSiftMessage

messages = [
    MemoSiftMessage(role="system", content="You are a helpful assistant."),
    MemoSiftMessage(role="user", content="What's in app.py?"),
    MemoSiftMessage(role="assistant", content="Let me check.", tool_calls=[...]),
    MemoSiftMessage(role="tool", content="def main(): ...", tool_call_id="tc1", name="read_file"),
    MemoSiftMessage(role="assistant", content="Here's what app.py contains..."),
    MemoSiftMessage(role="user", content="Refactor the main function."),
]

compressed, report = await compress(messages)

print(f"{report.compression_ratio:.1f}x compression")
print(f"{report.original_tokens:,} → {report.compressed_tokens:,} tokens")
print(f"Estimated savings: ${report.estimated_cost_saved:.4f}")
```

### TypeScript

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
console.log(`${report.originalTokens} → ${report.compressedTokens} tokens`);
```

## Framework Adapters

MemoSift converts to and from framework-native message formats with lossless round-trip fidelity. All adapters preserve thinking blocks, cache control, annotations, and tool call nesting.

### OpenAI SDK

```python
# Python
from memosift.adapters.openai_sdk import compress_openai_messages

messages = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Read my config"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "config.yaml"}'}}]},
    {"role": "tool", "tool_call_id": "tc1", "content": "port: 8080\nhost: localhost"},
    {"role": "assistant", "content": "Your config has port 8080 and host localhost."},
    {"role": "user", "content": "Change the port to 3000."},
]

compressed_msgs, report = await compress_openai_messages(messages)
# Feed compressed_msgs directly to client.chat.completions.create()
```

```typescript
// TypeScript
import { compressOpenAIMessages } from "memosift";

const { messages: compressed, report } = await compressOpenAIMessages(messages);
```

### Anthropic SDK

```python
# Python
from memosift.adapters.anthropic_sdk import compress_anthropic_messages

result, report = await compress_anthropic_messages(
    messages,
    system="You are a coding assistant.",  # Anthropic separates system prompts
)
# result.system → system prompt (preserved untouched)
# result.messages → compressed messages for client.messages.create()
```

```typescript
// TypeScript
import { compressAnthropicMessages } from "memosift";

const { result, report } = await compressAnthropicMessages(messages, {
  system: "You are a coding assistant.",
});
```

### Claude Agent SDK

```python
# Python
from memosift.adapters.claude_agent_sdk import compress_agent_sdk_messages

compressed_msgs, report = await compress_agent_sdk_messages(messages)
# Handles compaction boundaries, thinking blocks, and tool result nesting
# Messages with compact_boundary are marked as Zone 2 (never re-compressed)
```

```typescript
// TypeScript
import { compressAgentSdkMessages } from "memosift";

const { messages: compressed, report } = await compressAgentSdkMessages(messages);
```

### Google ADK

```python
# Python
from memosift.adapters.google_adk import compress_adk_events

compressed_events, report = await compress_adk_events(events)
# Preserves function_calls and function_responses
```

```typescript
// TypeScript
import { compressAdkEvents } from "memosift";

const { messages: compressed, report } = await compressAdkEvents(events);
```

### LangChain

```python
# Python
from memosift.adapters.langchain import compress_langchain_messages

compressed_msgs, report = await compress_langchain_messages(messages)
# Works with HumanMessage, AIMessage, ToolMessage, SystemMessage dicts
```

```typescript
// TypeScript
import { compressLangChainMessages } from "memosift";

const { messages: compressed, report } = await compressLangChainMessages(messages);
```

## Configuration

### Presets

MemoSift ships with domain-specific presets that tune all 7 compression engines:

```python
# Python
from memosift import compress, MemoSiftConfig

# Coding — conservative, never loses file paths, line numbers, error messages
config = MemoSiftConfig.preset("coding")

# Research — moderate, aggressive JSON truncation, preserves citations and URLs
config = MemoSiftConfig.preset("research")

# Support — aggressive, keeps recent conversation, summarizes old context
config = MemoSiftConfig.preset("support")

# Data — balanced, preserves numeric values, column names, query results
config = MemoSiftConfig.preset("data")

# General — default balanced compression
config = MemoSiftConfig.preset("general")

compressed, report = await compress(messages, config=config)
```

```typescript
// TypeScript
import { compress, createPreset } from "memosift";

const config = createPreset("coding");
const { messages: compressed, report } = await compress(messages, { config });
```

### Custom Configuration

```python
# Python
config = MemoSiftConfig(
    # Pipeline control
    recent_turns=3,                    # Protect last 3 conversational turns
    token_budget=100_000,              # Max output tokens (None = no limit)
    enable_summarization=False,        # True requires an LLM provider
    reorder_segments=False,            # L5 position optimization (careful with APIs)

    # Engine tuning
    dedup_similarity_threshold=0.85,   # Cosine similarity for fuzzy dedup (0.0–1.0)
    entropy_threshold=2.0,             # Min Shannon entropy to keep a line (bits/char)
    token_prune_keep_ratio=0.6,        # Fraction of tokens retained in IDF pruning
    json_array_threshold=5,            # Max JSON array items before truncation
    code_keep_signatures=True,         # Keep function/class signatures in code blocks
    relevance_drop_threshold=0.05,     # Min keyword overlap to survive relevance scoring

    # Short message coalescence
    coalesce_short_messages=True,      # Merge consecutive short assistant messages
    coalesce_char_threshold=100,       # Max chars for a message to be "short"

    # Anchor Ledger
    enable_anchor_ledger=True,         # Extract critical facts during compression
    anchor_ledger_max_tokens=5000,     # Max token budget for the anchor ledger

    # Model-aware defaults (auto-sets budget + pricing)
    model_name="claude-sonnet-4-6",    # Supported: gpt-4o, gpt-4.1, claude-*, gemini-*

    # Performance
    performance_tier=None,             # None=auto, or: "full", "standard", "fast", "ultra_fast"
    pre_bucket_bypass=True,            # Skip compression for system/recent/compressed messages
)
```

```typescript
// TypeScript
import { createConfig } from "memosift";

const config = createConfig({
  recentTurns: 3,
  tokenBudget: 100_000,
  dedupSimilarityThreshold: 0.85,
  entropyThreshold: 2.0,
  tokenPruneKeepRatio: 0.6,
  enableAnchorLedger: true,
  modelName: "claude-sonnet-4-6",
});
```

### Performance Tiers

MemoSift auto-detects the optimal performance tier based on message count:

| Tier | Messages | Layers | Use case |
|------|----------|--------|----------|
| `full` | ≤50 | All layers including L3G + L3E | Maximum quality |
| `standard` | 51–150 | Skips L3G (importance scoring) | Normal sessions |
| `fast` | 151–300 | Skips L3G + L3E (relevance pruning) | Long sessions |
| `ultra_fast` | >300 | Minimal layers only | Very long agents |

Override with `performance_tier="full"` to force all layers regardless of message count.

## Anchor Ledger

MemoSift extracts critical facts — file paths, error messages, decisions, identifiers — into an append-only ledger that survives compression even when source messages are dropped.

```python
# Python
from memosift import compress, AnchorLedger

ledger = AnchorLedger()
compressed, report = await compress(messages, ledger=ledger)

# Inspect extracted facts
for fact in ledger.facts:
    print(f"[{fact.category.value}] {fact.content}")
    # [FILES] Modified src/auth.py
    # [ERRORS] TypeError: Cannot read property 'id' of undefined
    # [DECISIONS] Using bcrypt for password hashing
    # [IDENTIFIERS] User ID: usr_abc123
```

### Anchor Categories

| Category | What it captures |
|----------|-----------------|
| `INTENT` | User goals and task descriptions |
| `FILES` | File paths mentioned or modified |
| `DECISIONS` | Design/implementation decisions made |
| `ERRORS` | Error messages and stack traces |
| `ACTIVE_CONTEXT` | Current working context (branch, directory) |
| `IDENTIFIERS` | IDs, URLs, keys, version numbers |
| `OUTCOMES` | Completed actions and their results |
| `OPEN_ITEMS` | Unresolved questions, TODOs, pending tasks |

### Persistent Ledger Across Compression Cycles

The ledger is append-only — pass the same instance across multiple compression calls to accumulate facts:

```python
ledger = AnchorLedger()

# First compression cycle
compressed_1, _ = await compress(window_1, ledger=ledger)

# Second cycle — ledger retains facts from window_1
compressed_2, _ = await compress(window_2, ledger=ledger)

# All facts from both windows
print(f"{len(ledger.facts)} total facts preserved")
```

## Cross-Window Deduplication

For long-running agents, share dedup state across compression cycles to catch repeats across windows:

```python
# Python
from memosift import compress, CrossWindowState

state = CrossWindowState()

# Window 1
compressed_1, _ = await compress(window_1, cross_window=state)

# Window 2 — dedup catches content repeated from window 1
compressed_2, _ = await compress(window_2, cross_window=state)
```

```typescript
// TypeScript
import { compress, createCrossWindowState } from "memosift";

const state = createCrossWindowState();

const result1 = await compress(window1, { crossWindow: state });
const result2 = await compress(window2, { crossWindow: state });
```

## Compression Cache (Re-Expansion)

When MemoSift collapses messages, it can store the originals in a cache for selective re-expansion — useful when an agent needs to re-read a file it already saw:

```python
from memosift.core.pipeline import CompressionCache, compress

cache = CompressionCache()
compressed, report = await compress(messages, cache=cache)

# Later, if the agent asks to re-read a collapsed message:
original = cache.get(original_index=5)
if original:
    print(f"Original content: {original}")
```

## Task-Aware Compression

Provide task context so the relevance scorer (L4) knows what matters:

```python
compressed, report = await compress(
    messages,
    task="Fix the authentication bug in the login endpoint",
    config=MemoSiftConfig.preset("coding"),
)
# Messages about auth/login score higher, unrelated content gets pruned
```

## Compression Report

Every compression call returns a detailed report:

```python
compressed, report = await compress(messages, config=config)

# High-level metrics
print(f"Compression ratio: {report.compression_ratio:.1f}x")
print(f"Original tokens: {report.original_tokens:,}")
print(f"Compressed tokens: {report.compressed_tokens:,}")
print(f"Tokens saved: {report.tokens_saved:,}")
print(f"Estimated cost savings: ${report.estimated_cost_saved:.4f}")
print(f"Total time: {report.total_latency_ms:.0f}ms")

# Per-layer breakdown
for layer_report in report.layers:
    print(f"  {layer_report.name}: {layer_report.input_tokens} → {layer_report.output_tokens} tokens ({layer_report.latency_ms:.0f}ms)")

# Individual decisions
for decision in report.decisions:
    print(f"  [{decision.layer}] {decision.action}: {decision.detail}")
```

## Three-Zone Memory Model

MemoSift partitions every message list into three zones to prevent re-compression:

| Zone | What | Behavior |
|------|------|----------|
| **Zone 1** | System prompts | Pass through untouched |
| **Zone 2** | Previously compressed (`_memosift_compressed=True`) | Pass through untouched |
| **Zone 3** | New raw messages | Compressed by the pipeline |

This means MemoSift is safe to call repeatedly — it never re-compresses its own output.

## How It Works

MemoSift compresses through a 6-layer pipeline with 7 compression engines:

```
Messages In
  → L1: Classify    — Assign content types (system, code, error, tool result, etc.)
  → L2: Dedup       — SHA-256 exact + MinHash/TF-IDF fuzzy deduplication
  → L2.5: Coalesce  — Merge consecutive short assistant messages
  → L3: Compress    — 7 type-specific engines:
       3A: Verbatim  — Noise removal, re-read collapse, boilerplate deletion
       3B: Pruner    — IDF-based token pruning (remove low-information words)
       3C: Structural — Code → signatures, JSON → truncated arrays
       3D: Summarizer — LLM abstractive summarization (opt-in only)
       3E: Relevance  — Drop segments with low query-keyword overlap
       3F: Discourse  — Remove elaboration, hedging, filler
       3G: Importance — 6-signal scoring, shield assignment (PRESERVE/MODERATE/COMPRESSIBLE)
  → L4: Score       — Task-aware relevance scoring
  → L5: Position    — Attention-curve reordering (disabled by default)
  → L6: Budget      — Enforce token budget, respect dependencies
Messages Out + CompressionReport
```

### Key Properties

- **Zero runtime dependencies** in core — no ML models, no torch, no transformers
- **Lossless by default** — only dedup and verbatim deletion unless you opt in to summarization
- **Framework-agnostic** — core operates on `MemoSiftMessage[]`, adapters handle conversion
- **Tool call integrity** — if a `tool_call` survives, all matching `tool_result` messages survive too
- **Deterministic** — same input + same config = same output (seeded with `deterministic_seed=42`)
- **Fault-tolerant** — any layer that throws is skipped, pipeline never crashes
- **Sub-100ms** for deterministic layers on 100K tokens

## LLM Provider (Optional)

By default MemoSift is fully deterministic — no LLM calls. To enable Engine D (summarization) or LLM-based relevance scoring, implement the `MemoSiftLLMProvider` protocol:

```python
# Python
from memosift import MemoSiftConfig, compress
from memosift.providers.base import MemoSiftLLMProvider, LLMResponse

class MyLLMProvider(MemoSiftLLMProvider):
    async def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        response = await my_llm_client.complete(prompt, system=system)
        return LLMResponse(text=response.text)

    def count_tokens(self, text: str) -> int:
        return len(text.split())  # or use tiktoken for accuracy

config = MemoSiftConfig(enable_summarization=True)
compressed, report = await compress(messages, llm=MyLLMProvider(), config=config)
```

```typescript
// TypeScript
import type { MemoSiftLLMProvider } from "memosift";

const provider: MemoSiftLLMProvider = {
  complete: async (prompt, system) => {
    const response = await myClient.complete(prompt, { system });
    return { text: response.text };
  },
  countTokens: (text) => text.split(/\s+/).length,
};

const { messages: compressed } = await compress(messages, { llm: provider, config });
```

Without an LLM provider, MemoSift uses a built-in `HeuristicTokenCounter` (word-split approximation) and skips Engine D and LLM-based scoring entirely.

## Claude Code Integration

The TypeScript package includes a PreCompact hook for Claude Code that injects anchor facts before Claude's own compaction:

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

This ensures file paths, error messages, and decisions survive Claude Code's context compaction.

## Development

```bash
# Python
cd python && pip install -e ".[dev]"
cd .. && python -m pytest tests/python/ -x -q   # 395 tests

# TypeScript
cd typescript && npm install && npm test          # 39 tests

# Lint & format
cd python && ruff format src/ && ruff check src/
cd typescript && npm run lint

# Cross-language test vector validation
python spec/validate_vectors.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow, test architecture, and CI pipeline details.

## License

MIT
