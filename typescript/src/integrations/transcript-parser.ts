// Parse Claude Code session transcripts (JSONL) into MemoSiftMessages.

import { existsSync, readFileSync } from "node:fs";
import { type MemoSiftMessage, type ToolCall, createMessage } from "../core/types.js";

export function parseTranscript(path: string): MemoSiftMessage[] {
  if (!existsSync(path)) return [];
  const content = readFileSync(path, "utf-8");
  return parseTranscriptFromString(content);
}

export function parseTranscriptFromString(content: string): MemoSiftMessage[] {
  const messages: MemoSiftMessage[] = [];
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const entry = JSON.parse(trimmed) as Record<string, unknown>;
      const msg = parseEntry(entry);
      if (msg) messages.push(msg);
    } catch {}
  }
  return messages;
}

function parseEntry(entry: Record<string, unknown>): MemoSiftMessage | null {
  const role = entry.role as string | undefined;
  if (!role || !["system", "user", "assistant", "tool"].includes(role)) return null;

  let content = entry.content;

  // Handle Anthropic-style content blocks (array of dicts).
  if (Array.isArray(content)) {
    const textParts: string[] = [];
    for (const block of content) {
      if (typeof block === "string") {
        textParts.push(block);
      } else if (typeof block === "object" && block !== null) {
        const b = block as Record<string, unknown>;
        if (b.type === "text") textParts.push((b.text as string) ?? "");
        else if (b.type === "tool_result") textParts.push((b.content as string) ?? "");
      }
    }
    content = textParts.join("\n");
  }

  if (typeof content !== "string") content = String(content ?? "");

  // Parse tool calls.
  let toolCalls: ToolCall[] | null = null;
  const rawCalls = entry.tool_calls;
  if (Array.isArray(rawCalls) && rawCalls.length > 0) {
    toolCalls = [];
    for (const tc of rawCalls) {
      const t = tc as Record<string, unknown>;
      if (t.id) {
        const func = (t.function ?? {}) as Record<string, unknown>;
        toolCalls.push({
          id: t.id as string,
          type: (t.type as string) ?? "function",
          function: {
            name: (func.name as string) ?? "",
            arguments: (func.arguments as string) ?? "{}",
          },
        });
      }
    }
  }

  return createMessage(role as MemoSiftMessage["role"], content as string, {
    name: entry.name as string | undefined,
    toolCallId: entry.tool_call_id as string | undefined,
    toolCalls: toolCalls && toolCalls.length > 0 ? toolCalls : undefined,
  });
}
