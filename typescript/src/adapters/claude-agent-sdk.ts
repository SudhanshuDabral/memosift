// Claude Agent SDK adapter — handles stateful sessions, compaction boundaries, hooks.

import type { MemoSiftConfig } from "../config.js";
import type { CompressResult } from "../core/pipeline.js";
import { compress } from "../core/pipeline.js";
import type { AnchorLedger, MemoSiftMessage, ToolCall } from "../core/types.js";
import { createMessage } from "../core/types.js";
import type { LLMResponse, MemoSiftLLMProvider } from "../providers/base.js";

/**
 * Wraps a Claude Agent SDK client into MemoSiftLLMProvider.
 *
 * Uses the underlying Anthropic client for generation and token counting.
 */
export class ClaudeAgentLLMProvider implements MemoSiftLLMProvider {
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

    // The Agent SDK wraps an Anthropic client internally.
    const anthropicClient = _getNestedProp(this._client, "_client") ?? this._client;
    const messages = anthropicClient as Record<string, unknown>;
    const messagesApi = _getNestedProp(messages, "messages") as Record<string, unknown>;

    const resp = (await (messagesApi.create as (args: unknown) => Promise<unknown>)({
      model: this._model,
      max_tokens: maxTokens,
      temperature,
      messages: [{ role: "user", content: prompt }],
    })) as {
      content: Array<{ text: string }>;
      usage: { input_tokens: number; output_tokens: number };
    };

    return {
      text: resp.content[0]!.text,
      inputTokens: resp.usage.input_tokens,
      outputTokens: resp.usage.output_tokens,
    };
  }

  async countTokens(text: string): Promise<number> {
    return Math.ceil(text.length / 3.5);
  }
}

// ── adaptIn ────────────────────────────────────────────────────────────────

/**
 * Convert Claude Agent SDK message objects to MemoSiftMessage list.
 *
 * Handles:
 * - AssistantMessage with TextBlock and ToolUseBlock content
 * - UserMessage with ToolResultBlock content
 * - SystemMessage with subtype (init, compact_boundary)
 * - Plain dicts (from get_session_messages())
 * - Compaction boundaries are tagged as already compressed
 */
export function adaptIn(messages: unknown[]): MemoSiftMessage[] {
  const result: MemoSiftMessage[] = [];

  for (const msg of messages) {
    // Handle dict-style messages (from session history).
    if (_isPlainObject(msg)) {
      result.push(..._adaptDictMessage(msg as Record<string, unknown>));
      continue;
    }

    // Handle typed Agent SDK message objects.
    const msgType = _getConstructorName(msg);
    const role = _getNestedProp(msg, "role") as string | null;
    const contentBlocks = _getNestedProp(msg, "content") ?? [];

    if (msgType === "SystemMessage") {
      const subtype = (_getNestedProp(msg, "subtype") as string) ?? "";
      let content = _getNestedProp(msg, "content");
      if (Array.isArray(content)) {
        content = content
          .map((b: unknown) =>
            typeof b === "string" ? b : ((_getNestedProp(b, "text") as string) ?? String(b)),
          )
          .join(" ");
      }
      const dm = createMessage(
        "system",
        typeof content === "string" ? content : String(content ?? ""),
        {
          metadata: { _agent_sdk_subtype: subtype },
        },
      );
      // Mark compaction boundaries as already compressed.
      if (subtype === "compact_boundary") {
        dm._memosiftCompressed = true;
      }
      result.push(dm);
      continue;
    }

    if (msgType === "ResultMessage") {
      // Terminal message with metrics — skip (not conversation content).
      continue;
    }

    // Process content blocks (AssistantMessage, UserMessage).
    // Preserve original blocks for lossless round-trip reconstruction.
    const originalBlocks: unknown[] | null = Array.isArray(contentBlocks)
      ? [...(contentBlocks as unknown[])]
      : null;

    const textParts: string[] = [];
    const toolCalls: ToolCall[] = [];
    const toolResults: Array<[string, string]> = [];

    if (typeof contentBlocks === "string") {
      textParts.push(contentBlocks);
    } else if (Array.isArray(contentBlocks)) {
      for (const block of contentBlocks as unknown[]) {
        const blockType = _getBlockType(block);

        if (blockType === "thinking") {
        } else if (blockType === "text") {
          textParts.push(_getBlockText(block));
        } else if (blockType === "tool_use") {
          const tcId = _getBlockProp(block, "id");
          const tcName = _getBlockProp(block, "name");
          const tcInput =
            _getNestedProp(block, "input") ??
            (_isPlainObject(block) ? (block as Record<string, unknown>).input : {});
          toolCalls.push({
            id: tcId,
            type: "function",
            function: {
              name: tcName,
              arguments: typeof tcInput === "object" ? JSON.stringify(tcInput) : String(tcInput),
            },
          });
        } else if (blockType === "tool_result") {
          const trId = _getBlockProp(block, "tool_use_id");
          let trContent =
            _getNestedProp(block, "content") ??
            (_isPlainObject(block) ? (block as Record<string, unknown>).content : "");
          if (Array.isArray(trContent)) {
            trContent = (trContent as unknown[])
              .map((b: unknown) => {
                if (_isPlainObject(b)) return (b as Record<string, unknown>).text ?? String(b);
                return (_getNestedProp(b, "text") as string) ?? String(b);
              })
              .join(" ");
          }
          toolResults.push([trId, String(trContent)]);
        } else if (typeof block === "string") {
          textParts.push(block);
        }
      }
    }

    const combinedText = textParts.length > 0 ? textParts.join("\n") : "";
    const effectiveRole = (role ??
      (toolCalls.length > 0 ? "assistant" : "user")) as MemoSiftMessage["role"];

    result.push(
      createMessage(effectiveRole, combinedText, {
        toolCalls: toolCalls.length > 0 ? toolCalls : null,
        metadata: {
          _agent_sdk_type: msgType,
          _original_blocks: originalBlocks,
          _original_block_format: "claude_agent_sdk",
        },
      }),
    );

    // Emit tool results as separate tool messages.
    for (const [toolUseId, trContent] of toolResults) {
      result.push(
        createMessage("tool", trContent, {
          toolCallId: toolUseId,
        }),
      );
    }
  }

  return result;
}

// ── adaptOut ───────────────────────────────────────────────────────────────

/**
 * Convert MemoSiftMessage list back to Anthropic-compatible format.
 *
 * Returns dicts suitable for the Anthropic Messages API, which the
 * Agent SDK uses internally. Tool results are re-nested into user messages.
 */
export function adaptOut(messages: MemoSiftMessage[]): Record<string, unknown>[] {
  const result: Record<string, unknown>[] = [];

  // Build tool result lookup.
  const toolResultMap = new Map<string, MemoSiftMessage>();
  for (const msg of messages) {
    if (msg.role === "tool" && msg.toolCallId) {
      toolResultMap.set(msg.toolCallId, msg);
    }
  }

  const consumedToolIds = new Set<string>();

  for (const msg of messages) {
    if (msg.role === "tool") {
      continue; // Re-nested below.
    }

    const blocks: Record<string, unknown>[] = [];
    const originalBlocks = msg.metadata._original_blocks as unknown[] | undefined;

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
        const btype = _getBlockType(origBlock);

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
          if (_isPlainObject(origBlock)) {
            blocks.push({ ...(origBlock as Record<string, unknown>) });
          } else {
            blocks.push({
              type: "thinking",
              thinking: (_getNestedProp(origBlock, "thinking") as string) ?? "",
            });
          }
        } else if (btype === "tool_use") {
          const tcId = _getBlockProp(origBlock, "id");
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
          if (_isPlainObject(origBlock)) {
            blocks.push({ ...(origBlock as Record<string, unknown>) });
          } else if (typeof origBlock === "string") {
            blocks.push(origBlock as unknown as Record<string, unknown>);
          } else {
            blocks.push(btype ? { type: btype } : (origBlock as Record<string, unknown>));
          }
        }
      }

      // If no text block existed in originals but we have content, add it.
      if (!textReplaced && msg.content) {
        blocks.unshift({ type: "text", text: msg.content });
      }
    } else {
      // Fallback: reconstruct from scratch.
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

    result.push({ role: msg.role, content: blocks });

    // Re-nest tool results after assistant tool_use messages.
    if (msg.role === "assistant" && msg.toolCalls) {
      const trBlocks: Record<string, unknown>[] = [];
      for (const tc of msg.toolCalls) {
        if (toolResultMap.has(tc.id) && !consumedToolIds.has(tc.id)) {
          const trMsg = toolResultMap.get(tc.id)!;
          trBlocks.push({
            type: "tool_result",
            tool_use_id: tc.id,
            content: trMsg.content,
          });
          consumedToolIds.add(tc.id);
        }
      }
      if (trBlocks.length > 0) {
        result.push({ role: "user", content: trBlocks });
      }
    }
  }

  return result;
}

// ── Convenience function ───────────────────────────────────────────────────

export interface CompressAgentSdkOptions {
  llm?: MemoSiftLLMProvider | null;
  config?: Partial<MemoSiftConfig> | null;
  task?: string | null;
  ledger?: AnchorLedger | null;
  client?: unknown;
  model?: string;
  contextWindow?: import("../core/context-window.js").ContextWindowState | null;
}

/**
 * Compress Claude Agent SDK session messages end-to-end.
 *
 * Respects compaction boundaries (compact_boundary markers are
 * tagged _memosiftCompressed=true and skip re-compression).
 *
 * @param messages - Agent SDK message objects or session history dicts.
 * @param options - Configuration and provider options.
 * @returns Tuple of compressed Anthropic-format messages and compression report.
 */
export async function compressAgentSdkMessages(
  messages: unknown[],
  options?: CompressAgentSdkOptions,
): Promise<[Record<string, unknown>[], CompressResult["report"]]> {
  let provider = options?.llm ?? null;
  if (provider == null && options?.client != null) {
    provider = new ClaudeAgentLLMProvider(options.client, options.model ?? "claude-sonnet-4-6");
  }

  // Auto-resolve context window from model when not explicitly provided.
  let contextWindow = options?.contextWindow ?? null;
  if (contextWindow === null) {
    const { resolveContextWindow, estimateTokensHeuristic } = await import(
      "../core/context-window.js"
    );
    const model = options?.model ?? "claude-sonnet-4-6";
    const contentStrings = messages.map((m) => {
      if (typeof m === "object" && m !== null && "content" in m) {
        const c = (m as Record<string, unknown>).content;
        if (typeof c === "string") return c;
      }
      return "";
    });
    contextWindow = resolveContextWindow(null, model, estimateTokensHeuristic(contentStrings));
  }

  const memosiftMsgs = adaptIn(messages);
  const { messages: compressed, report } = await compress(memosiftMsgs, {
    llm: provider,
    config: options?.config,
    task: options?.task,
    ledger: options?.ledger,
    contextWindow,
  });
  return [adaptOut(compressed), report];
}

// ── Private helpers ────────────────────────────────────────────────────────

/**
 * Convert a plain dict message (from session history) to MemoSiftMessage(s).
 *
 * Returns a list because a single Anthropic-format user message with
 * tool_result blocks expands into multiple MemoSiftMessages (one per result).
 */
function _adaptDictMessage(msg: Record<string, unknown>): MemoSiftMessage[] {
  const role = (msg.role as string) ?? "user";
  const content = msg.content;

  if (!Array.isArray(content)) {
    return [
      createMessage(
        role as MemoSiftMessage["role"],
        typeof content === "string" ? content : String(content ?? ""),
      ),
    ];
  }

  // Preserve original blocks for lossless round-trip reconstruction.
  const originalBlocks = [...(content as unknown[])];

  const textParts: string[] = [];
  const toolCalls: ToolCall[] = [];
  const toolResults: Array<[string, string]> = [];

  for (const block of content as unknown[]) {
    if (!_isPlainObject(block)) continue;
    const b = block as Record<string, unknown>;

    if (b.type === "thinking") {
    } else if (b.type === "text") {
      textParts.push((b.text as string) ?? "");
    } else if (b.type === "tool_use") {
      toolCalls.push({
        id: (b.id as string) ?? "",
        type: "function",
        function: {
          name: (b.name as string) ?? "",
          arguments: JSON.stringify(b.input ?? {}),
        },
      });
    } else if (b.type === "tool_result") {
      let trContent = b.content;
      if (Array.isArray(trContent)) {
        trContent = (trContent as unknown[])
          .filter((item): item is Record<string, unknown> => _isPlainObject(item))
          .map((item) => (item.text as string) ?? String(item))
          .join(" ");
      }
      toolResults.push([(b.tool_use_id as string) ?? "", String(trContent ?? "")]);
    }
  }

  const result: MemoSiftMessage[] = [];
  const combinedText = textParts.length > 0 ? textParts.join("\n") : "";

  // Main message (assistant with tool_calls, or user with text).
  if (combinedText || toolCalls.length > 0) {
    result.push(
      createMessage(role as MemoSiftMessage["role"], combinedText, {
        toolCalls: toolCalls.length > 0 ? toolCalls : null,
        metadata: {
          _original_blocks: originalBlocks,
          _original_block_format: "claude_agent_sdk",
        },
      }),
    );
  }

  // Emit tool results as separate tool messages.
  for (const [toolUseId, trContent] of toolResults) {
    result.push(
      createMessage("tool", trContent, {
        toolCallId: toolUseId,
      }),
    );
  }

  // If we produced nothing, emit an empty message to preserve the role.
  if (result.length === 0) {
    result.push(createMessage(role as MemoSiftMessage["role"], ""));
  }

  return result;
}

/** Check if a value is a plain object (not an array, not null). */
function _isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Get constructor name from an object instance. */
function _getConstructorName(obj: unknown): string {
  if (obj == null) return "";
  if (typeof obj !== "object") return "";
  return (obj as { constructor?: { name?: string } }).constructor?.name ?? "";
}

/** Safely access a nested property on an unknown object. */
function _getNestedProp(obj: unknown, key: string): unknown {
  if (obj == null || typeof obj !== "object") return undefined;
  return (obj as Record<string, unknown>)[key];
}

/** Get the block type from either a typed object or a dict. */
function _getBlockType(block: unknown): string | null {
  if (_isPlainObject(block)) return (block as Record<string, unknown>).type as string | null;
  if (typeof block === "object" && block != null) {
    return (_getNestedProp(block, "type") as string) ?? null;
  }
  return null;
}

/** Get the text content from a block (typed object or dict). */
function _getBlockText(block: unknown): string {
  if (_isPlainObject(block)) return ((block as Record<string, unknown>).text as string) ?? "";
  return (_getNestedProp(block, "text") as string) ?? "";
}

/** Get a string property from a block (typed object or dict). */
function _getBlockProp(block: unknown, key: string): string {
  if (_isPlainObject(block)) return ((block as Record<string, unknown>)[key] as string) ?? "";
  return (_getNestedProp(block, key) as string) ?? "";
}
