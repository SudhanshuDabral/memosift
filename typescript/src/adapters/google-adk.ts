// Google ADK adapter — handles Events with function_calls/function_responses.

import { createHash } from "node:crypto";

import { type MemoSiftConfig, createConfig } from "../config.js";
import { type CompressResult, compress } from "../core/pipeline.js";
import {
  type AnchorLedger,
  type MemoSiftMessage,
  type ToolCall,
  type ToolCallFunction,
  createMessage,
} from "../core/types.js";
import type { MemoSiftLLMProvider } from "../providers/base.js";

/** Map ADK roles to MemoSift roles. */
function adkRoleToMemoSift(role: string): MemoSiftMessage["role"] {
  const mapping: Record<string, MemoSiftMessage["role"]> = {
    model: "assistant",
    function: "tool",
  };
  return mapping[role] ?? (role as MemoSiftMessage["role"]);
}

/** Map MemoSift roles to ADK roles. */
function memosiftRoleToAdk(role: string): string {
  const mapping: Record<string, string> = {
    assistant: "model",
    tool: "function",
  };
  return mapping[role] ?? role;
}

/**
 * Convert Google ADK events to MemoSiftMessage list.
 *
 * ADK uses Events with `function_calls` and `function_responses`
 * instead of tool_calls/tool_results.
 */
export function adaptIn(events: ReadonlyArray<Record<string, unknown>>): MemoSiftMessage[] {
  const result: MemoSiftMessage[] = [];

  for (const event of events) {
    const role = (event.role as string | undefined) ?? "user";

    // Handle function calls (ADK's equivalent of tool_calls).
    const functionCalls = (event.function_calls ?? []) as Record<string, unknown>[];
    if (Array.isArray(functionCalls) && functionCalls.length > 0) {
      const toolCalls: ToolCall[] = functionCalls.map((fc: Record<string, unknown>): ToolCall => {
        const fcId =
          (fc.id as string | undefined) ??
          `adk_${createHash("sha256")
            .update(JSON.stringify(fc, Object.keys(fc).sort()))
            .digest("hex")
            .slice(0, 12)}`;
        const fn: ToolCallFunction = {
          name: (fc.name as string | undefined) ?? "",
          arguments: JSON.stringify(fc.args ?? {}),
        };
        return { id: fcId, type: "function", function: fn };
      });

      const text = (event.text as string | undefined) ?? "";
      result.push(
        createMessage("assistant", text, {
          toolCalls,
          metadata: {
            _adk_event: true,
            _original_event: { ...event },
            _original_block_format: "google_adk",
          },
        }),
      );
      continue;
    }

    // Handle function responses (ADK's equivalent of tool results).
    const functionResponses = (event.function_responses ?? []) as Record<string, unknown>[];
    if (Array.isArray(functionResponses) && functionResponses.length > 0) {
      for (const fr of functionResponses) {
        let content = fr.response as unknown;
        if (content !== null && typeof content === "object") {
          content = JSON.stringify(content);
        }
        result.push(
          createMessage("tool", String(content ?? ""), {
            toolCallId: (fr.id as string | undefined) ?? "",
            name: (fr.name as string | undefined) ?? "",
            metadata: {
              _adk_event: true,
              _original_event: { ...event },
              _original_block_format: "google_adk",
            },
          }),
        );
      }
      continue;
    }

    // Regular text event.
    const parts = (event.parts ?? []) as Record<string, unknown>[];
    let text = (event.text as string | undefined) ?? "";
    if (!text && Array.isArray(parts) && parts.length > 0) {
      text = parts
        .filter(
          (p): p is Record<string, unknown> => p !== null && typeof p === "object" && "text" in p,
        )
        .map((p) => (p.text as string | undefined) ?? "")
        .join(" ");
    }

    result.push(
      createMessage(adkRoleToMemoSift(role), text, {
        metadata: {
          _adk_event: true,
          _original_event: { ...event },
          _original_block_format: "google_adk",
        },
      }),
    );
  }

  return result;
}

/**
 * Convert MemoSiftMessage list back to ADK event format.
 *
 * Uses the stored `_original_event` for lossless round-trip when available,
 * falling back to reconstruction from message fields.
 */
export function adaptOut(messages: ReadonlyArray<MemoSiftMessage>): Record<string, unknown>[] {
  const result: Record<string, unknown>[] = [];

  for (const msg of messages) {
    const originalEvent = msg.metadata._original_event as Record<string, unknown> | undefined;

    if (originalEvent != null) {
      // Lossless round-trip: reconstruct from original event,
      // replacing only the text field with (possibly compressed) content.
      const event: Record<string, unknown> = { ...originalEvent };
      if ("text" in event || (!event.function_calls && !event.function_responses)) {
        event.text = msg.content;
      }
      result.push(event);
      continue;
    }

    // Fallback: reconstruct from scratch.
    const event: Record<string, unknown> = {
      role: memosiftRoleToAdk(msg.role),
    };

    if (msg.toolCalls && msg.toolCalls.length > 0) {
      event.function_calls = msg.toolCalls.map((tc) => ({
        id: tc.id,
        name: tc.function.name,
        args: tc.function.arguments ? JSON.parse(tc.function.arguments) : {},
      }));
      if (msg.content) {
        event.text = msg.content;
      }
    } else if (msg.role === "tool" && msg.toolCallId) {
      event.function_responses = [
        {
          id: msg.toolCallId,
          name: msg.name ?? "",
          response: msg.content,
        },
      ];
      event.role = "function";
    } else {
      event.text = msg.content;
    }

    result.push(event);
  }

  return result;
}

/** Compress Google ADK events end-to-end. */
export async function compressAdkEvents(
  events: ReadonlyArray<Record<string, unknown>>,
  options?: {
    llm?: MemoSiftLLMProvider | null;
    config?: Partial<MemoSiftConfig> | null;
    task?: string | null;
    ledger?: AnchorLedger | null;
  },
): Promise<{ events: Record<string, unknown>[]; report: CompressResult["report"] }> {
  const memosiftMsgs = adaptIn(events);
  const { messages: compressed, report } = await compress(memosiftMsgs, {
    llm: options?.llm,
    config: options?.config,
    task: options?.task,
    ledger: options?.ledger,
  });
  return { events: adaptOut(compressed), report };
}
