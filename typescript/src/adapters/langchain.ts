// LangChain/LangGraph adapter — handles typed message classes and additional_kwargs.

import type { MemoSiftConfig } from "../config.js";
import { compress } from "../core/pipeline.js";
import type { AnchorLedger, MemoSiftMessage, ToolCall } from "../core/types.js";
import { createMessage } from "../core/types.js";
import type { LLMResponse, MemoSiftLLMProvider } from "../providers/base.js";
import type { CompressionReport } from "../report.js";

// ── LangChain type-name → role mapping ────────────────────────────────────────

const _LANGCHAIN_TYPE_TO_ROLE: Readonly<Record<string, MemoSiftMessage["role"]>> = {
  HumanMessage: "user",
  AIMessage: "assistant",
  SystemMessage: "system",
  ToolMessage: "tool",
  ChatMessage: "user",
  FunctionMessage: "tool",
};

function langchainTypeToRole(typeName: string): MemoSiftMessage["role"] {
  return _LANGCHAIN_TYPE_TO_ROLE[typeName] ?? "user";
}

// ── LLM Provider ──────────────────────────────────────────────────────────────

/**
 * Wraps any LangChain BaseChatModel into a MemoSiftLLMProvider.
 *
 * Uses `.bind()` for model kwargs and `.ainvoke()` (via `invoke` in JS)
 * for generation.
 */
export class LangChainLLMProvider implements MemoSiftLLMProvider {
  private readonly _llm: unknown;

  constructor(llm: unknown) {
    this._llm = llm;
  }

  async generate(
    prompt: string,
    options?: { maxTokens?: number; temperature?: number },
  ): Promise<LLMResponse> {
    const maxTokens = options?.maxTokens ?? 2048;
    const temperature = options?.temperature ?? 0.0;

    // LangChain JS models expose .bind() to set kwargs and .invoke() for async generation.
    const llm = this._llm as Record<string, unknown>;
    const bindFn = llm.bind as (kwargs: Record<string, unknown>) => unknown;
    if (typeof bindFn !== "function") {
      throw new Error("LangChain LLM must expose a .bind() method");
    }
    const bound = bindFn.call(llm, { maxTokens, temperature }) as Record<string, unknown>;

    const invokeFn = bound.invoke as (input: unknown[]) => Promise<unknown>;
    if (typeof invokeFn !== "function") {
      throw new Error("LangChain bound model must expose an .invoke() method");
    }

    const response = (await invokeFn.call(bound, [{ role: "user", content: prompt }])) as Record<
      string,
      unknown
    >;

    // Extract usage metadata.
    const usage = (response.usage_metadata ?? {}) as Record<string, unknown>;
    const inputTokens = typeof usage.input_tokens === "number" ? usage.input_tokens : 0;
    const outputTokens = typeof usage.output_tokens === "number" ? usage.output_tokens : 0;

    const text = typeof response.content === "string" ? response.content : String(response);

    return { text, inputTokens, outputTokens };
  }

  async countTokens(text: string): Promise<number> {
    try {
      const llm = this._llm as Record<string, unknown>;
      const countFn = llm.getNumTokens as (t: string) => number | Promise<number>;
      if (typeof countFn === "function") {
        return await Promise.resolve(countFn.call(llm, text));
      }
    } catch {
      // Fall through to heuristic.
    }
    return Math.ceil(text.length / 3.5);
  }
}

// ── Adapt In ──────────────────────────────────────────────────────────────────

/**
 * Convert LangChain message objects to MemoSiftMessage list.
 *
 * Handles HumanMessage, AIMessage, SystemMessage, ToolMessage and
 * their `additional_kwargs` / `response_metadata`.
 */
export function adaptIn(messages: unknown[]): MemoSiftMessage[] {
  const result: MemoSiftMessage[] = [];

  for (const msg of messages) {
    let role: MemoSiftMessage["role"];
    let content: string;
    let additionalKwargs: Record<string, unknown>;
    let toolCallId: string | null;
    let name: string | null;
    let msgTypeName: string;

    if (msg !== null && typeof msg === "object" && !Array.isArray(msg)) {
      const m = msg as Record<string, unknown>;

      if (typeof m.role === "string" && !m.constructor?.toString().includes("class")) {
        // Dict-style message.
        role = (m.role as MemoSiftMessage["role"]) ?? "user";
        content = typeof m.content === "string" ? m.content : "";
        additionalKwargs = (m.additional_kwargs as Record<string, unknown>) ?? {};
        toolCallId = typeof m.tool_call_id === "string" ? m.tool_call_id : null;
        name = typeof m.name === "string" ? m.name : null;
        msgTypeName = "dict";
      } else {
        // LangChain typed message class.
        const constructor = (msg as { constructor: { name: string } }).constructor;
        msgTypeName = constructor?.name ?? "unknown";
        role = langchainTypeToRole(msgTypeName);
        content = typeof m.content === "string" ? m.content : String(msg);
        additionalKwargs = (m.additional_kwargs as Record<string, unknown>) ?? {};
        toolCallId = typeof m.tool_call_id === "string" ? m.tool_call_id : null;
        name = typeof m.name === "string" ? m.name : null;
      }
    } else {
      // Fallback for unexpected shapes.
      role = "user";
      content = String(msg);
      additionalKwargs = {};
      toolCallId = null;
      name = null;
      msgTypeName = "unknown";
    }

    // Extract tool_calls from additional_kwargs (LangChain pattern).
    let toolCalls: ToolCall[] | null = null;
    const rawCalls = additionalKwargs.tool_calls;
    if (Array.isArray(rawCalls) && rawCalls.length > 0) {
      toolCalls = rawCalls.map((tc: Record<string, unknown>) => {
        const fn = (tc.function ?? {}) as Record<string, unknown>;
        return {
          id: typeof tc.id === "string" ? tc.id : "",
          type: "function",
          function: {
            name:
              typeof fn.name === "string" ? fn.name : typeof tc.name === "string" ? tc.name : "",
            arguments:
              typeof fn.arguments === "string"
                ? fn.arguments
                : typeof tc.args === "string"
                  ? tc.args
                  : "{}",
          },
        };
      });
    }

    // Preserve LangChain metadata for round-tripping.
    const metadata: Record<string, unknown> = {
      _langchain_type: msgTypeName,
    };
    if (Object.keys(additionalKwargs).length > 0) {
      const nonTcKwargs: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(additionalKwargs)) {
        if (k !== "tool_calls") {
          nonTcKwargs[k] = v;
        }
      }
      if (Object.keys(nonTcKwargs).length > 0) {
        metadata._langchain_additional_kwargs = nonTcKwargs;
      }
    }

    const m = msg as Record<string, unknown> | null;
    const responseMetadata =
      m !== null && typeof m === "object"
        ? (m.response_metadata as Record<string, unknown> | undefined)
        : undefined;
    if (responseMetadata && Object.keys(responseMetadata).length > 0) {
      metadata._langchain_response_metadata = responseMetadata;
    }

    result.push(
      createMessage(role, typeof content === "string" ? content : String(content), {
        name,
        toolCallId,
        toolCalls,
        metadata,
      }),
    );
  }

  return result;
}

// ── Adapt Out ─────────────────────────────────────────────────────────────────

/**
 * Convert MemoSiftMessage list back to LangChain-compatible dicts.
 *
 * Returns dicts that can be used with LangChain's message constructors.
 */
export function adaptOut(messages: MemoSiftMessage[]): Record<string, unknown>[] {
  const result: Record<string, unknown>[] = [];

  for (const msg of messages) {
    const d: Record<string, unknown> = {
      role: msg.role,
      content: msg.content,
    };

    if (msg.name != null) {
      d.name = msg.name;
    }
    if (msg.toolCallId != null) {
      d.tool_call_id = msg.toolCallId;
    }

    const additionalKwargs: Record<string, unknown> = {};
    if (msg.toolCalls && msg.toolCalls.length > 0) {
      additionalKwargs.tool_calls = msg.toolCalls.map((tc) => ({
        id: tc.id,
        type: tc.type,
        function: {
          name: tc.function.name,
          arguments: tc.function.arguments,
        },
      }));
    }

    // Restore preserved kwargs.
    const savedKwargs =
      (msg.metadata._langchain_additional_kwargs as Record<string, unknown>) ?? {};
    for (const [k, v] of Object.entries(savedKwargs)) {
      additionalKwargs[k] = v;
    }
    if (Object.keys(additionalKwargs).length > 0) {
      d.additional_kwargs = additionalKwargs;
    }

    const responseMetadata = msg.metadata._langchain_response_metadata as
      | Record<string, unknown>
      | undefined;
    if (responseMetadata) {
      d.response_metadata = responseMetadata;
    }

    result.push(d);
  }

  return result;
}

// ── Convenience function ──────────────────────────────────────────────────────

/**
 * Compress LangChain messages end-to-end.
 *
 * @param messages - LangChain message objects or dicts.
 * @param options - Pipeline options (llm, config, task, ledger).
 * @returns Tuple of compressed message dicts and compression report.
 */
export async function compressLangChainMessages(
  messages: unknown[],
  options?: {
    llm?: MemoSiftLLMProvider | null;
    config?: Partial<MemoSiftConfig> | null;
    task?: string | null;
    ledger?: AnchorLedger | null;
  },
): Promise<[Record<string, unknown>[], CompressionReport]> {
  const memosiftMsgs = adaptIn(messages);
  const { messages: compressed, report } = await compress(memosiftMsgs, {
    llm: options?.llm,
    config: options?.config,
    task: options?.task,
    ledger: options?.ledger,
  });
  return [adaptOut(compressed), report];
}
