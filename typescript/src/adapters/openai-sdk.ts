// OpenAI SDK adapter — convert between OpenAI message format and MemoSiftMessage.

import type { MemoSiftConfig } from "../config.js";
import { createConfig } from "../config.js";
import { compress } from "../core/pipeline.js";
import type { AnchorLedger, MemoSiftMessage, ToolCall, ToolCallFunction } from "../core/types.js";
import { createMessage } from "../core/types.js";
import type { LLMResponse, MemoSiftLLMProvider } from "../providers/base.js";
import type { CompressionReport } from "../report.js";

// ── OpenAI message shape (loose, for interop) ─────────────────────────────

/** A single OpenAI-format message dict. We use Record<string, unknown> in the
 *  public API, but internally cast to this for ergonomic property access. */
interface OpenAIMessage {
  role: string;
  content?: string | null;
  name?: string | null;
  tool_call_id?: string | null;
  tool_calls?: OpenAIToolCall[] | null;
  refusal?: string | null;
  annotations?: unknown[] | null;
  [key: string]: unknown;
}

interface OpenAIToolCall {
  id: string;
  type?: string;
  function: {
    name: string;
    arguments: string;
  };
}

// ── LLM Provider ───────────────────────────────────────────────────────────

/** Wraps an OpenAI-compatible async client into MemoSiftLLMProvider. */
export class OpenAILLMProvider implements MemoSiftLLMProvider {
  private readonly client: unknown;
  private readonly model: string;

  constructor(client: unknown, model = "gpt-4o") {
    this.client = client;
    this.model = model;
  }

  async generate(
    prompt: string,
    options?: { maxTokens?: number; temperature?: number },
  ): Promise<LLMResponse> {
    const maxTokens = options?.maxTokens ?? 2048;
    const temperature = options?.temperature ?? 0.0;

    // The client is expected to be an OpenAI-compatible instance with
    // chat.completions.create(). We cast through unknown to avoid bundling
    // the openai SDK as a dependency.
    const client = this.client as {
      chat: {
        completions: {
          create(params: Record<string, unknown>): Promise<{
            choices: Array<{ message: { content: string | null } }>;
            usage: { prompt_tokens: number; completion_tokens: number };
          }>;
        };
      };
    };

    const resp = await client.chat.completions.create({
      model: this.model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
      temperature,
    });

    return {
      text: resp.choices[0]?.message.content ?? "",
      inputTokens: resp.usage.prompt_tokens,
      outputTokens: resp.usage.completion_tokens,
    };
  }

  async countTokens(text: string): Promise<number> {
    // Heuristic token count — no external tokenizer dependency.
    return Math.ceil(text.length / 3.5);
  }
}

// ── Adapt In ───────────────────────────────────────────────────────────────

/** Convert OpenAI-format message dicts to MemoSiftMessage list.
 *
 *  Handles:
 *  - `content: null` on assistant messages with tool_calls -> `content: ""`
 *  - `tool_calls` list with `id`, `function.name`, `function.arguments`
 *  - `tool_call_id` on tool result messages
 */
export function adaptIn(messages: ReadonlyArray<Record<string, unknown>>): MemoSiftMessage[] {
  const result: MemoSiftMessage[] = [];

  for (const raw of messages) {
    const msg = raw as OpenAIMessage;

    let toolCalls: ToolCall[] | null = null;
    const rawCalls = msg.tool_calls;
    if (rawCalls && Array.isArray(rawCalls) && rawCalls.length > 0) {
      toolCalls = rawCalls.map(
        (tc: OpenAIToolCall): ToolCall => ({
          id: tc.id,
          type: tc.type ?? "function",
          function: {
            name: tc.function.name,
            arguments: tc.function.arguments,
          } satisfies ToolCallFunction,
        }),
      );
    }

    const metadata: Record<string, unknown> = {
      _openai_original_keys: Object.keys(raw),
    };
    if (msg.refusal) {
      metadata._openai_refusal = msg.refusal;
    }
    if (msg.annotations) {
      metadata._openai_annotations = msg.annotations;
    }

    result.push(
      createMessage(msg.role as MemoSiftMessage["role"], msg.content ?? "", {
        name: msg.name ?? null,
        toolCallId: msg.tool_call_id ?? null,
        toolCalls,
        metadata,
      }),
    );
  }

  return result;
}

// ── Adapt Out ──────────────────────────────────────────────────────────────

/** Convert MemoSiftMessage list back to OpenAI-format message dicts.
 *
 *  Preserves tool_calls structure and handles content=null for tool-call
 *  assistant messages per OpenAI API requirements.
 */
export function adaptOut(messages: ReadonlyArray<MemoSiftMessage>): Record<string, unknown>[] {
  const result: Record<string, unknown>[] = [];

  for (const msg of messages) {
    const d: Record<string, unknown> = { role: msg.role };

    if (msg.toolCalls && msg.toolCalls.length > 0) {
      // OpenAI allows content=null for assistant messages with tool_calls.
      d.content = msg.content ? msg.content : null;
      d.tool_calls = msg.toolCalls.map((tc) => ({
        id: tc.id,
        type: tc.type,
        function: {
          name: tc.function.name,
          arguments: tc.function.arguments,
        },
      }));
    } else {
      d.content = msg.content;
    }

    if (msg.name != null) {
      d.name = msg.name;
    }
    if (msg.toolCallId != null) {
      d.tool_call_id = msg.toolCallId;
    }

    // Restore preserved OpenAI-specific fields.
    if (msg.metadata._openai_refusal) {
      d.refusal = msg.metadata._openai_refusal;
    }
    if (msg.metadata._openai_annotations) {
      d.annotations = msg.metadata._openai_annotations;
    }

    result.push(d);
  }

  return result;
}

// ── Convenience Function ───────────────────────────────────────────────────

/** Compress OpenAI-format messages end-to-end.
 *
 *  Converts to MemoSiftMessage, runs the pipeline, converts back.
 *
 *  @param messages - OpenAI-format message dicts.
 *  @param options.llm - Optional MemoSiftLLMProvider (or use `OpenAILLMProvider`).
 *  @param options.config - Pipeline configuration overrides.
 *  @param options.task - Optional task description for relevance scoring.
 *  @param options.ledger - Optional AnchorLedger for fact preservation.
 *  @returns Tuple of compressed OpenAI messages and compression report.
 */
export async function compressOpenAIMessages(
  messages: ReadonlyArray<Record<string, unknown>>,
  options?: {
    llm?: MemoSiftLLMProvider | null;
    config?: Partial<MemoSiftConfig> | null;
    task?: string | null;
    ledger?: AnchorLedger | null;
  },
): Promise<{ messages: Record<string, unknown>[]; report: CompressionReport }> {
  const memosiftMsgs = adaptIn(messages);

  const { messages: compressed, report } = await compress(memosiftMsgs, {
    llm: options?.llm ?? null,
    config: options?.config ?? null,
    task: options?.task ?? null,
    ledger: options?.ledger ?? null,
  });

  return { messages: adaptOut(compressed), report };
}
