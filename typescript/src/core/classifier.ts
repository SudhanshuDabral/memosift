// Layer 1: Content segmentation and classification.

import type { MemoSiftConfig } from "../config.js";
import {
  type ClassifiedMessage,
  ContentType,
  DEFAULT_POLICIES,
  type MemoSiftMessage,
  createClassified,
} from "./types.js";

const ERROR_PATTERNS = [
  /Traceback \(most recent/,
  /^\s+at .+\(.+:\d+\)/m,
  /Error: .+\n\s+at/m,
  /panic:/,
  /Exception in thread/,
  /^\s+File ".+", line \d+/m,
  /raise \w+Error/,
];

const MIN_ERROR_LINES = 3;
const FENCED_CODE_RE = /```[\s\S]*?```/;
const CODE_TOOL_NAMES = new Set([
  "read_file",
  "cat",
  "view_file",
  "get_file_contents",
  "ReadFileTool",
  "read",
  "Read",
]);

export function classifyMessages(
  messages: MemoSiftMessage[],
  config: MemoSiftConfig,
): ClassifiedMessage[] {
  const recentBoundary = findNthUserMessageFromEnd(messages, config.recentTurns);
  const lastUser = lastUserIndex(messages);
  const result: ClassifiedMessage[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]!;
    const inRecentWindow = i > recentBoundary;
    let ctype: ContentType;

    if (msg._memosiftCompressed) {
      ctype = ContentType.PREVIOUSLY_COMPRESSED;
    } else if (msg.role === "system") {
      ctype = ContentType.SYSTEM_PROMPT;
    } else if (msg.role === "user" && i === lastUser) {
      ctype = ContentType.USER_QUERY;
    } else if (msg.role === "tool") {
      const toolType = classifyToolResult(msg);
      ctype =
        inRecentWindow && toolType === ContentType.TOOL_RESULT_TEXT
          ? ContentType.RECENT_TURN
          : toolType;
    } else if (inRecentWindow) {
      ctype = ContentType.RECENT_TURN;
    } else if (msg.role === "assistant") {
      ctype = ContentType.ASSISTANT_REASONING;
    } else {
      ctype = ContentType.OLD_CONVERSATION;
    }

    const policy = config.policies[ctype] ?? DEFAULT_POLICIES[ctype];
    const isProtected = [
      ContentType.SYSTEM_PROMPT,
      ContentType.USER_QUERY,
      ContentType.RECENT_TURN,
      ContentType.PREVIOUSLY_COMPRESSED,
    ].includes(ctype);

    result.push(
      createClassified(msg, ctype, policy, {
        originalIndex: i,
        protected: isProtected,
      }),
    );
  }

  return result;
}

export function findNthUserMessageFromEnd(messages: MemoSiftMessage[], n: number): number {
  if (n <= 0) return messages.length;
  let userCount = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]!.role === "user") {
      userCount++;
      if (userCount === n) return i;
    }
  }
  return 0;
}

export function lastUserIndex(messages: MemoSiftMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]!.role === "user") return i;
  }
  return -1;
}

function classifyToolResult(msg: MemoSiftMessage): ContentType {
  if (containsErrorTrace(msg.content)) return ContentType.ERROR_TRACE;
  if (isValidJson(msg.content)) return ContentType.TOOL_RESULT_JSON;
  if (containsCode(msg.content, msg.name ?? undefined)) return ContentType.CODE_BLOCK;
  return ContentType.TOOL_RESULT_TEXT;
}

function isValidJson(text: string): boolean {
  const stripped = text.trim();
  if (!stripped || (stripped[0] !== "{" && stripped[0] !== "[")) return false;
  try {
    const parsed: unknown = JSON.parse(stripped);
    return typeof parsed === "object" && parsed !== null;
  } catch {
    return false;
  }
}

function containsErrorTrace(text: string): boolean {
  let count = 0;
  for (const pattern of ERROR_PATTERNS) {
    const matches = text.match(new RegExp(pattern.source, `${pattern.flags}g`));
    count += matches?.length ?? 0;
    if (count >= MIN_ERROR_LINES) return true;
  }
  return false;
}

function containsCode(text: string, toolName?: string): boolean {
  if (FENCED_CODE_RE.test(text)) return true;
  if (toolName && CODE_TOOL_NAMES.has(toolName)) return true;
  return false;
}
