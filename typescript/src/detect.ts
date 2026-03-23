// Framework auto-detection — inspect message shape to determine the source SDK.

/**
 * Detect which framework produced the given messages.
 *
 * Uses duck-typing heuristics — no framework imports required.
 *
 * Detection order (first match wins):
 * 1. MemoSiftMessage instances → "memosift"
 * 2. Dict with "function_calls" or "function_responses" → "adk"
 * 3. Object with .additional_kwargs property → "langchain"
 * 4. Object with Agent SDK type names → "agent_sdk"
 * 5. Dict with "content" as array of typed blocks → "anthropic"
 * 6. Default → "openai"
 */
export function detectFramework(
  messages: readonly unknown[],
): "openai" | "anthropic" | "agent_sdk" | "adk" | "langchain" | "memosift" {
  if (messages.length === 0) {
    throw new Error("Cannot detect framework from empty message list");
  }

  const samples = messages.slice(0, 5).filter((m) => m != null);
  if (samples.length === 0) return "openai";

  for (const msg of samples) {
    if (typeof msg !== "object" || msg === null) continue;
    const rec = msg as Record<string, unknown>;

    // 1. MemoSiftMessage — has _memosiftCompressed field.
    if ("_memosiftCompressed" in rec && "role" in rec && "content" in rec) {
      return "memosift";
    }

    // 2. Google ADK — uses function_calls/function_responses.
    if ("function_calls" in rec || "function_responses" in rec) {
      return "adk";
    }
    const parts = rec.parts;
    if (Array.isArray(parts)) {
      for (const part of parts) {
        if (
          typeof part === "object" &&
          part !== null &&
          ("function_call" in part || "function_response" in part)
        ) {
          return "adk";
        }
      }
    }

    // 3. LangChain — has additional_kwargs property.
    // Check before Agent SDK because LangChain's SystemMessage shares a name
    // with Agent SDK's SystemMessage, but has additional_kwargs.
    if ("additional_kwargs" in rec) {
      return "langchain";
    }

    // 4. Claude Agent SDK — typed objects with known class names.
    const typeName = (msg as { constructor?: { name?: string } }).constructor?.name;
    if (
      typeName !== undefined &&
      ["SystemMessage", "AssistantMessage", "UserMessage", "ResultMessage"].includes(typeName) &&
      typeName !== "Object"
    ) {
      return "agent_sdk";
    }

    // 5. Anthropic — content is array of blocks with Anthropic-specific type values.
    const content = rec.content;
    if (Array.isArray(content) && content.length > 0) {
      const firstBlock = content[0];
      if (typeof firstBlock === "object" && firstBlock !== null && "type" in firstBlock) {
        const blockType = (firstBlock as Record<string, unknown>).type;
        if (
          blockType === "text" ||
          blockType === "tool_use" ||
          blockType === "tool_result" ||
          blockType === "thinking" ||
          blockType === "image"
        ) {
          return "anthropic";
        }
      }
    }
  }

  // 6. Default — OpenAI.
  return "openai";
}

/** Valid framework identifiers. */
export const VALID_FRAMEWORKS = new Set([
  "openai",
  "anthropic",
  "agent_sdk",
  "adk",
  "langchain",
  "memosift",
] as const);

export type Framework = "openai" | "anthropic" | "agent_sdk" | "adk" | "langchain" | "memosift";
