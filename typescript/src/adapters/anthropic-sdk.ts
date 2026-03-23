// Anthropic/Claude SDK adapter — handles content blocks, nested tool results, separate system.

import type { MemoSiftConfig } from "../config.js";
import { createConfig } from "../config.js";
import type { CompressOptions } from "../core/pipeline.js";
import { compress } from "../core/pipeline.js";
import {
  type AnchorLedger,
  type MemoSiftMessage,
  type ToolCall,
  createMessage,
} from "../core/types.js";
import type { LLMResponse, MemoSiftLLMProvider } from "../providers/base.js";
import type { CompressionReport } from "../report.js";

/**
 * Result of compressing Anthropic-format messages.
 *
 * Anthropic's API takes `system` as a separate parameter, so the
 * compressed result separates it from the message list.
 */
export interface AnthropicCompressedResult {
  system: string;
  messages: Record<string, unknown>[];
}

/** Wraps the Anthropic AsyncAnthropic client into MemoSiftLLMProvider. */
export class AnthropicLLMProvider implements MemoSiftLLMProvider {
  private readonly _client: unknown;
  private readonly _model: string;

  constructor(client: unknown, model = "claude-sonnet-4-6") {
    this._client = client;
    this._model = model;
  }

  async generate(
    prompt: string,
    options?: { maxTokens?: number; temperature?: number },
  ): Promise<LLMResponse> {
    const maxTokens = options?.maxTokens ?? 2048;
    const temperature = options?.temperature ?? 0.0;
    // biome-ignore lint/suspicious/noExplicitAny: untyped SDK client
    const client = this._client as any;
    const resp = await client.messages.create({
      model: this._model,
      max_tokens: maxTokens,
      temperature,
      messages: [{ role: "user", content: prompt }],
    });
    return {
      text: resp.content[0].text as string,
      inputTokens: resp.usage.input_tokens as number,
      outputTokens: resp.usage.output_tokens as number,
    };
  }

  async countTokens(text: string): Promise<number> {
    try {
      // biome-ignore lint/suspicious/noExplicitAny: untyped SDK client
      const client = this._client as any;
      const resp = await client.messages.count_tokens({
        model: this._model,
        messages: [{ role: "user", content: text }],
      });
      return resp.input_tokens as number;
    } catch {
      // Fallback to heuristic if Anthropic count_tokens API fails.
      return Math.ceil(text.length / 3.5);
    }
  }
}

// ── Internal types for content block processing ─────────────────────────────

interface ToolResultTuple {
  toolUseId: string;
  content: string;
  name: string;
  isError: boolean;
  cacheControl: unknown;
}

// ── adapt_in ────────────────────────────────────────────────────────────────

/**
 * Convert Anthropic-format messages to MemoSiftMessage list.
 *
 * Anthropic differences:
 * - `content` is an ARRAY of blocks: `[{"type": "text", "text": "..."}]`
 * - Tool use blocks appear in assistant messages as `{"type": "tool_use", ...}`
 * - Tool results are NESTED in user messages as `{"type": "tool_result", ...}`
 * - System prompt is a separate parameter, not in the messages array
 */
export function adaptIn(
  messages: Record<string, unknown>[],
  system?: string | null,
): MemoSiftMessage[] {
  const result: MemoSiftMessage[] = [];

  // Add system prompt as a system message.
  if (system) {
    result.push(createMessage("system", system));
  }

  for (const msg of messages) {
    const role = msg.role as MemoSiftMessage["role"];
    const contentBlocks = msg.content ?? [];

    if (typeof contentBlocks === "string") {
      result.push(createMessage(role, contentBlocks));
      continue;
    }

    // Preserve original blocks for lossless round-trip reconstruction.
    const originalBlocks = Array.isArray(contentBlocks) ? [...contentBlocks] : null;

    // Process content blocks.
    const textParts: string[] = [];
    const toolCalls: ToolCall[] = [];
    const toolResults: ToolResultTuple[] = [];

    if (Array.isArray(contentBlocks)) {
      for (const block of contentBlocks) {
        if (typeof block === "string") {
          textParts.push(block);
        } else if (isRecord(block)) {
          const btype = block.type as string | undefined;

          if (btype === "text") {
            textParts.push((block.text as string) ?? "");
          } else if (btype === "tool_use") {
            toolCalls.push({
              id: block.id as string,
              type: "function",
              function: {
                name: block.name as string,
                arguments: JSON.stringify(block.input ?? {}),
              },
            });
          } else if (btype === "thinking") {
          } else if (btype === "tool_result") {
            let trContent = block.content ?? "";
            if (Array.isArray(trContent)) {
              trContent = (trContent as Record<string, unknown>[])
                .filter((b): b is Record<string, unknown> => isRecord(b))
                .map((b) => (b.text as string) ?? "")
                .join(" ");
            }
            toolResults.push({
              toolUseId: (block.tool_use_id as string) ?? "",
              content: trContent as string,
              name: (block.name as string) ?? "",
              isError: (block.is_error as boolean) ?? false,
              cacheControl: block.cache_control ?? null,
            });
          }
        }
      }
    }

    // Emit the main message (assistant with text + tool_use, or user with text).
    const combinedText = textParts.length > 0 ? textParts.join("\n") : "";
    const mainMsg = createMessage(role, combinedText, {
      toolCalls: toolCalls.length > 0 ? toolCalls : null,
      metadata: {
        _anthropic_blocks: true,
        _original_blocks: originalBlocks,
        _original_block_format: "anthropic",
      },
    });
    result.push(mainMsg);

    // Emit tool results as separate tool messages (MemoSift's internal format).
    for (const tr of toolResults) {
      const trMeta: Record<string, unknown> = {};
      if (tr.isError) {
        trMeta._anthropic_is_error = true;
      }
      if (tr.cacheControl) {
        trMeta._anthropic_cache_control = tr.cacheControl;
      }
      result.push(
        createMessage("tool", tr.content, {
          toolCallId: tr.toolUseId,
          name: tr.name,
          metadata: trMeta,
        }),
      );
    }
  }

  return result;
}

// ── adapt_out ───────────────────────────────────────────────────────────────

/**
 * Convert MemoSiftMessage list back to Anthropic format.
 *
 * Re-nests tool results inside user messages and separates the system prompt.
 */
export function adaptOut(messages: MemoSiftMessage[]): AnthropicCompressedResult {
  let system = "";
  const anthropicMsgs: Record<string, unknown>[] = [];

  // Collect tool results for re-nesting.
  const toolResultMap = new Map<string, MemoSiftMessage>();
  for (const msg of messages) {
    if (msg.role === "tool" && msg.toolCallId) {
      toolResultMap.set(msg.toolCallId, msg);
    }
  }

  const consumedToolIds = new Set<string>();

  for (const msg of messages) {
    if (msg.role === "system") {
      system = msg.content;
      continue;
    }

    if (msg.role === "tool") {
      // Tool results will be nested into user messages.
      continue;
    }

    let blocks: Record<string, unknown>[] = [];

    const originalBlocks = msg.metadata._original_blocks as unknown[] | null | undefined;

    if (originalBlocks != null) {
      // Lossless round-trip: walk the original blocks and patch in
      // compressed content / surviving tool_calls.
      const tcMap = new Map<string, ToolCall>();
      if (msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          tcMap.set(tc.id, tc);
        }
      }

      let textReplaced = false;
      for (const origBlock of originalBlocks) {
        const btype = isRecord(origBlock) ? (origBlock.type as string | undefined) : null;

        if (btype === "text") {
          if (!textReplaced) {
            // First text block gets the (possibly compressed) content.
            blocks.push({ type: "text", text: msg.content });
            textReplaced = true;
          }
          // Subsequent text blocks are dropped — their content was
          // merged into msg.content during adaptIn.
        } else if (btype === "thinking") {
          // Pass thinking blocks through unchanged.
          blocks.push({ ...(origBlock as Record<string, unknown>) });
        } else if (btype === "tool_use") {
          const tcId = ((origBlock as Record<string, unknown>).id as string) ?? "";
          if (tcMap.has(tcId)) {
            const tc = tcMap.get(tcId)!;
            let inputData: unknown;
            try {
              inputData = JSON.parse(tc.function.arguments);
            } catch {
              inputData = { raw: tc.function.arguments };
            }
            blocks.push({
              type: "tool_use",
              id: tc.id,
              name: tc.function.name,
              input: inputData,
            });
          }
        } else {
          // Unknown / future block types — pass through unchanged.
          if (isRecord(origBlock)) {
            blocks.push({ ...(origBlock as Record<string, unknown>) });
          } else {
            blocks.push(origBlock as Record<string, unknown>);
          }
        }
      }

      // If no text block existed in originals but we have content, add it.
      if (!textReplaced && msg.content) {
        blocks = [{ type: "text", text: msg.content }, ...blocks];
      }
    } else {
      // Backward-compatible fallback: reconstruct from scratch.
      if (msg.content) {
        blocks.push({ type: "text", text: msg.content });
      }

      if (msg.role === "assistant" && msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          let inputData: unknown;
          try {
            inputData = JSON.parse(tc.function.arguments);
          } catch {
            inputData = { raw: tc.function.arguments };
          }
          blocks.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input: inputData,
          });
        }
      }
    }

    anthropicMsgs.push({ role: msg.role, content: blocks });

    // After an assistant message with tool_calls, nest tool results into
    // the next user message (or create one).
    if (msg.role === "assistant" && msg.toolCalls) {
      const trBlocks: Record<string, unknown>[] = [];
      for (const tc of msg.toolCalls) {
        if (toolResultMap.has(tc.id) && !consumedToolIds.has(tc.id)) {
          const trMsg = toolResultMap.get(tc.id)!;
          trBlocks.push(buildToolResultBlock(tc.id, trMsg));
          consumedToolIds.add(tc.id);
        }
      }
      if (trBlocks.length > 0) {
        anthropicMsgs.push({ role: "user", content: trBlocks });
      }
    }
  }

  return { system, messages: anthropicMsgs };
}

// ── Helper: build a tool_result block ───────────────────────────────────────

function buildToolResultBlock(tcId: string, trMsg: MemoSiftMessage): Record<string, unknown> {
  const block: Record<string, unknown> = {
    type: "tool_result",
    tool_use_id: tcId,
    content: trMsg.content,
  };
  if (trMsg.metadata._anthropic_is_error) {
    block.is_error = true;
  }
  const cacheControl = trMsg.metadata._anthropic_cache_control;
  if (cacheControl) {
    block.cache_control = cacheControl;
  }
  return block;
}

// ── Convenience function ────────────────────────────────────────────────────

/**
 * Compress Anthropic-format messages end-to-end.
 *
 * @param messages - Anthropic-format message dicts (content is array of blocks).
 * @param options - Configuration options.
 * @returns Tuple of [AnthropicCompressedResult, CompressionReport].
 */
export async function compressAnthropicMessages(
  messages: Record<string, unknown>[],
  options?: {
    system?: string | null;
    llm?: MemoSiftLLMProvider | null;
    config?: Partial<MemoSiftConfig> | null;
    task?: string | null;
    ledger?: AnchorLedger | null;
    client?: unknown;
    model?: string;
  },
): Promise<[AnthropicCompressedResult, CompressionReport]> {
  const system = options?.system ?? null;
  const task = options?.task ?? null;
  const ledger = options?.ledger ?? null;
  const model = options?.model ?? "claude-sonnet-4-6";

  let provider: MemoSiftLLMProvider | null = options?.llm ?? null;
  if (provider === null && options?.client != null) {
    provider = new AnthropicLLMProvider(options.client, model);
  }

  const memosiftMsgs = adaptIn(messages, system);

  const compressOpts: CompressOptions = {
    llm: provider,
    config: options?.config ?? undefined,
    task,
    ledger,
  };

  const { messages: compressed, report } = await compress(memosiftMsgs, compressOpts);

  return [adaptOut(compressed), report];
}

// ── Type guard ──────────────────────────────────────────────────────────────

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
