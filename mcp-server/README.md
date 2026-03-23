# @memosift/mcp-server

MCP (Model Context Protocol) server for [MemoSift](https://memosift.dev) — context compression tools for AI agents.

MemoSift compresses LLM conversation context through a 7-layer pipeline with 7 compression engines, preserving tool call integrity and critical facts while reducing token usage by 2-4x.

## Quick Start

```bash
npx @memosift/mcp-server
```

### Claude Desktop

Add to `claude_desktop_config.json`:

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

### Claude Code

Add to `~/.claude/settings.json`:

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

### Cursor / Windsurf / VS Code

Same pattern — add to your editor's MCP server configuration.

## Tools

The server exposes 8 tools that agents can discover and invoke:

### `memosift_check_pressure`

Check if your context window needs compression. Returns pressure level (NONE/LOW/MEDIUM/HIGH/CRITICAL) and a recommendation. Call this before compressing to avoid unnecessary work.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model name (default: `claude-sonnet-4-6`) |
| `current_usage_tokens` | integer | Tokens currently consumed |
| `message_count` | integer | Alternative: estimate from message count |

### `memosift_compress`

Compress conversation messages to reduce context window usage. Accepts messages in any supported format (OpenAI, Anthropic, LangChain, Google ADK). Returns compressed messages in the same format plus metrics.

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | object[] | **Required.** Messages in any framework format |
| `preset` | enum | `coding` \| `research` \| `support` \| `data` \| `general` |
| `model` | string | Model name for adaptive compression |
| `framework` | enum | `openai` \| `anthropic` \| `agent_sdk` \| `adk` \| `langchain` (auto-detected) |
| `task` | string | Task description for relevance scoring |
| `session_id` | string | Session ID for persistent state |
| `current_usage_tokens` | integer | Token usage for adaptive pressure |
| `system` | string | System prompt (Anthropic format) |

### `memosift_configure`

Create or update a compression session with specific settings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | **Required.** Session ID |
| `preset` | enum | Domain preset |
| `model` | string | Model name |
| `token_budget` | integer | Hard token limit |
| `recent_turns` | integer | Turns to protect (1-20) |
| `enable_summarization` | boolean | Enable LLM summarization |

### `memosift_get_facts`

Retrieve critical facts (file paths, errors, decisions) extracted during compression. Facts survive even when source messages are dropped.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | **Required.** Session ID |
| `category` | enum | Filter: `INTENT` \| `FILES` \| `DECISIONS` \| `ERRORS` \| `ACTIVE_CONTEXT` \| `IDENTIFIERS` \| `OUTCOMES` \| `OPEN_ITEMS` |

### `memosift_expand`

Re-expand a previously compressed message to see its original content.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | **Required.** Session ID |
| `original_index` | integer | **Required.** Message index from compression report |

### `memosift_report`

Get detailed compression metrics from the most recent compression.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | **Required.** Session ID |
| `detail_level` | enum | `summary` \| `layers` \| `decisions` \| `full` |

### `memosift_list_sessions`

List all active compression sessions with metadata.

### `memosift_destroy`

Destroy a session and free its memory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | **Required.** Session ID to destroy |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMOSIFT_SESSION_TTL_MS` | `3600000` (1h) | Session TTL in milliseconds |
| `MEMOSIFT_DEFAULT_PRESET` | `general` | Default compression preset |
| `MEMOSIFT_MODEL` | — | Default model for new sessions |
| `MEMOSIFT_LLM_PROVIDER` | — | `openai` or `anthropic` for Engine D |
| `OPENAI_API_KEY` | — | Required if LLM provider is `openai` |
| `ANTHROPIC_API_KEY` | — | Required if LLM provider is `anthropic` |

### Presets

| Preset | Optimized For |
|--------|---------------|
| `coding` | Code editing — preserves errors, file paths, signatures |
| `research` | Research — preserves citations, URLs, findings |
| `support` | Customer support — keeps error traces, resolutions |
| `data` | Data analysis — preserves schemas, query results |
| `general` | General-purpose balanced compression |

## How It Works

The MCP server wraps MemoSift's `MemoSiftSession` — a stateful compression session that manages the anchor ledger (fact extraction), cross-window dedup state, and compression cache across multiple tool calls.

Each `session_id` maps to an independent session. Sessions expire after 1 hour of inactivity (configurable). The `_default` session is used when no `session_id` is specified.

By default, the server runs in **deterministic-only mode** — no LLM calls, no external dependencies. All 6 deterministic compression engines are active. Engine D (LLM summarization) is only enabled when an LLM provider is configured.

## Community

- [Discord](https://discord.gg/xtBs7hhgbA) — Join the MemoSift community
  - `#general` — Discussion and announcements
  - `#help` — Get help with integration and configuration
  - `#feature-request` — Suggest new features and improvements
  - `#show-what-you-built` — Share your MemoSift-powered projects
- [GitHub Issues](https://github.com/memosift/memosift/issues) — Bug reports and feature requests
- [memosift.dev](https://memosift.dev) — Documentation

## License

MIT
