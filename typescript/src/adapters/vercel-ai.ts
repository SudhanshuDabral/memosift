// Vercel AI SDK adapter — lossless round-trip for CoreMessage format.

import type { MemoSiftConfig } from "../config.js";
import type { ContextWindowState } from "../core/context-window.js";
import { compress } from "../core/pipeline.js";
import type { AnchorLedger, MemoSiftMessage, ToolCall, ToolCallFunction } from "../core/types.js";
import { createMessage } from "../core/types.js";
import type { LLMResponse, MemoSiftLLMProvider } from "../providers/base.js";
import type { CompressionReport } from "../report.js";

// ── Vercel AI SDK message shapes (loose, for interop) ───────────────────────

/** Text content part. */
interface TextPart {
  type: "text";
  text: string;
}

/** Image content part — skipped during compression, preserved via metadata. */
interface ImagePart {
  type: "image";
  image: string | URL | Uint8Array;
  mimeType?: string;
}

/** File content part — skipped during compression, preserved via metadata. */
interface FilePart {
  type: "file";
  data: string | URL | Uint8Array;
  mimeType: string;
}

/** Tool call part within an assistant message. */
interface ToolCallPart {
  type: "tool-call";
  toolCallId: string;
  toolName: string;
  args: unknown;
}

/** Tool result part within a tool message. */
interface ToolResultPart {
  type: "tool-result";
  toolCallId: string;
  toolName: string;
  result: unknown;
  isError?: boolean;
}

/** Union of all Vercel AI SDK content parts. */
type ContentPart = TextPart | ImagePart | FilePart | ToolCallPart | ToolResultPart;

/** Vercel AI SDK CoreMessage (loose shape for duck-typing). */
interface VercelMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | ContentPart[];
  [key: string]: unknown;
}

// ── LLM Provider ────────────────────────────────────────────────────────────

/**
 * Wraps a Vercel AI SDK LanguageModel into MemoSiftLLMProvider.
 *
 * Uses the Vercel AI SDK's `generateText()` function. The caller must pass
 * the `generateText` function and a `model` instance from their Vercel AI SDK
 * setup (to avoid bundling the `ai` package as a dependency).
 */
export class VercelAILLMProvider implements MemoSiftLLMProvider {
  private readonly generateTextFn: (params: Record<string, unknown>) => Promise<{
    text: string;
    usage: { promptTokens: number; completionTokens: number };
  }>;
  private readonly model: unknown;

  constructor(
    generateTextFn: (params: Record<string, unknown>) => Promise<{
      text: string;
      usage: { promptTokens: number; completionTokens: number };
    }>,
    model: unknown,
  ) {
    this.generateTextFn = generateTextFn;
    this.model = model;
  }

  async generate(
    prompt: string,
    options?: { maxTokens?: number; temperature?: number },
  ): Promise<LLMResponse> {
    const maxTokens = options?.maxTokens ?? 2048;
    const temperature = options?.temperature ?? 0.0;

    const result = await this.generateTextFn({
      model: this.model,
      prompt,
      maxTokens,
      temperature,
    });

    return {
      text: result.text,
      inputTokens: result.usage.promptTokens,
      outputTokens: result.usage.completionTokens,
    };
  }

  async countTokens(text: string): Promise<number> {
    // Vercel AI SDK doesn't expose a tokenizer directly — use heuristic.
    return Math.ceil(text.length / 3.5);
  }
}

// ── Adapt In ────────────────────────────────────────────────────────────────

/**
 * Convert Vercel AI SDK CoreMessage array to MemoSiftMessage list.
 *
 * Handles:
 * - `content: string` → direct content
 * - `TextPart` → concatenated text content
 * - `ToolCallPart` → ToolCall with toolCallId → id, toolName → function.name
 * - `ToolResultPart` → tool role message with tool_call_id
 * - `ImagePart`/`FilePart` → skipped, stored in metadata for round-trip
 */
export function adaptIn(messages: ReadonlyArray<Record<string, unknown>>): MemoSiftMessage[] {
  const result: MemoSiftMessage[] = [];

  for (const raw of messages) {
    const msg = raw as VercelMessage;
    const role = msg.role as MemoSiftMessage["role"];

    // String content — simple case.
    if (typeof msg.content === "string") {
      result.push(
        createMessage(role, msg.content, {
          metadata: { _vercel_content_type: "string" },
        }),
      );
      continue;
    }

    // Array content — process parts.
    if (!Array.isArray(msg.content)) {
      result.push(
        createMessage(role, String(msg.content ?? ""), {
          metadata: { _vercel_content_type: "unknown" },
        }),
      );
      continue;
    }

    const parts = msg.content as ContentPart[];

    if (role === "tool") {
      // Tool messages contain ToolResultPart[].
      for (const part of parts) {
        if (part.type === "tool-result") {
          const trp = part as ToolResultPart;
          const contentStr =
            typeof trp.result === "string" ? trp.result : JSON.stringify(trp.result);
          const metadata: Record<string, unknown> = {
            _vercel_content_type: "tool-result",
            _vercel_tool_name: trp.toolName,
          };
          if (trp.isError) {
            metadata._vercel_is_error = true;
          }
          result.push(
            createMessage("tool", contentStr, {
              toolCallId: trp.toolCallId,
              name: trp.toolName,
              metadata,
            }),
          );
        }
      }
      continue;
    }

    // User or assistant message with mixed parts.
    const textParts: string[] = [];
    const toolCalls: ToolCall[] = [];
    const preservedParts: ContentPart[] = [];

    for (const part of parts) {
      if (part.type === "text") {
        textParts.push((part as TextPart).text);
      } else if (part.type === "tool-call") {
        const tcp = part as ToolCallPart;
        toolCalls.push({
          id: tcp.toolCallId,
          type: "function",
          function: {
            name: tcp.toolName,
            arguments: typeof tcp.args === "string" ? tcp.args : JSON.stringify(tcp.args),
          } satisfies ToolCallFunction,
        });
      } else {
        // Image, file, or unknown parts — preserve for round-trip.
        preservedParts.push(part);
      }
    }

    const content = textParts.join("\n");
    const metadata: Record<string, unknown> = {
      _vercel_content_type: "parts",
    };
    if (preservedParts.length > 0) {
      metadata._vercel_preserved_parts = preservedParts;
    }

    result.push(
      createMessage(role, content, {
        toolCalls: toolCalls.length > 0 ? toolCalls : null,
        metadata,
      }),
    );
  }

  return result;
}

// ── Adapt Out ───────────────────────────────────────────────────────────────

/**
 * Convert MemoSiftMessage list back to Vercel AI SDK CoreMessage format.
 *
 * Restores:
 * - String content → `content: string`
 * - ToolCall → `ToolCallPart` with toolCallId/toolName
 * - Tool messages → `ToolResultPart[]`
 * - Preserved ImagePart/FilePart from metadata
 */
export function adaptOut(messages: ReadonlyArray<MemoSiftMessage>): Record<string, unknown>[] {
  const result: Record<string, unknown>[] = [];

  for (const msg of messages) {
    const contentType = msg.metadata._vercel_content_type as string | undefined;

    if (msg.role === "tool") {
      // Reconstruct ToolResultPart.
      const toolResult: Record<string, unknown> = {
        type: "tool-result",
        toolCallId: msg.toolCallId,
        toolName: msg.name ?? msg.metadata._vercel_tool_name ?? "unknown",
        result: _tryParseJson(msg.content),
      };
      if (msg.metadata._vercel_is_error) {
        toolResult.isError = true;
      }
      result.push({ role: "tool", content: [toolResult] });
      continue;
    }

    // String content — simple reconstruction.
    if (contentType === "string" || (!msg.toolCalls && !msg.metadata._vercel_preserved_parts)) {
      result.push({ role: msg.role, content: msg.content });
      continue;
    }

    // Reconstruct parts array.
    const parts: Record<string, unknown>[] = [];

    // Restore preserved parts (images, files) at the beginning.
    const preserved = msg.metadata._vercel_preserved_parts as ContentPart[] | undefined;
    if (preserved) {
      for (const p of preserved) {
        parts.push(p as unknown as Record<string, unknown>);
      }
    }

    // Add text content.
    if (msg.content) {
      parts.push({ type: "text", text: msg.content });
    }

    // Add tool calls.
    if (msg.toolCalls) {
      for (const tc of msg.toolCalls) {
        parts.push({
          type: "tool-call",
          toolCallId: tc.id,
          toolName: tc.function.name,
          args: _tryParseJson(tc.function.arguments),
        });
      }
    }

    result.push({ role: msg.role, content: parts });
  }

  return result;
}

// ── Convenience Function ────────────────────────────────────────────────────

/**
 * Compress Vercel AI SDK CoreMessage array end-to-end.
 *
 * Converts to MemoSiftMessage, runs the pipeline, converts back.
 */
export async function compressVercelMessages(
  messages: ReadonlyArray<Record<string, unknown>>,
  options?: {
    llm?: MemoSiftLLMProvider | null;
    config?: Partial<MemoSiftConfig> | null;
    task?: string | null;
    ledger?: AnchorLedger | null;
    contextWindow?: ContextWindowState | null;
  },
): Promise<{ messages: Record<string, unknown>[]; report: CompressionReport }> {
  const internal = adaptIn(messages);

  const { messages: compressed, report } = await compress(internal, {
    llm: options?.llm ?? null,
    config: options?.config ?? null,
    task: options?.task ?? null,
    ledger: options?.ledger ?? null,
    contextWindow: options?.contextWindow ?? null,
  });

  return { messages: adaptOut(compressed), report };
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Try to parse a string as JSON, return the original string if it fails. */
function _tryParseJson(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}
