// Tests for framework auto-detection.

import { describe, expect, it } from "vitest";
import { createMessage } from "../../typescript/src/core/types.js";
import { VALID_FRAMEWORKS, detectFramework } from "../../typescript/src/detect.js";

describe("detectFramework", () => {
  it("throws on empty array", () => {
    expect(() => detectFramework([])).toThrow("empty");
  });

  it("returns openai for all-null messages", () => {
    expect(detectFramework([null, null])).toBe("openai");
  });

  // ── MemoSiftMessage ──────────────────────────────────────────────

  it("detects MemoSiftMessage instances", () => {
    const msgs = [createMessage("user", "hello"), createMessage("assistant", "hi")];
    expect(detectFramework(msgs)).toBe("memosift");
  });

  // ── OpenAI ───────────────────────────────────────────────────────

  it("detects OpenAI simple messages", () => {
    const msgs = [
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hello" },
    ];
    expect(detectFramework(msgs)).toBe("openai");
  });

  it("detects OpenAI with tool_calls", () => {
    const msgs = [
      {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "tc1", type: "function", function: { name: "read", arguments: "{}" } }],
      },
      { role: "tool", content: "result", tool_call_id: "tc1" },
    ];
    expect(detectFramework(msgs)).toBe("openai");
  });

  // ── Anthropic ────────────────────────────────────────────────────

  it("detects Anthropic content blocks", () => {
    const msgs = [
      { role: "user", content: [{ type: "text", text: "Hello" }] },
      { role: "assistant", content: [{ type: "text", text: "Hi" }] },
    ];
    expect(detectFramework(msgs)).toBe("anthropic");
  });

  it("detects Anthropic tool_use blocks", () => {
    const msgs = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let me check." },
          { type: "tool_use", id: "tu1", name: "read_file", input: {} },
        ],
      },
    ];
    expect(detectFramework(msgs)).toBe("anthropic");
  });

  it("Anthropic string content falls to openai", () => {
    const msgs = [{ role: "user", content: "Hello" }];
    expect(detectFramework(msgs)).toBe("openai");
  });

  // ── Google ADK ───────────────────────────────────────────────────

  it("detects ADK function_calls", () => {
    const msgs = [{ role: "model", function_calls: [{ name: "search", args: { q: "test" } }] }];
    expect(detectFramework(msgs)).toBe("adk");
  });

  it("detects ADK function_responses", () => {
    const msgs = [{ role: "function", function_responses: [{ name: "search", response: {} }] }];
    expect(detectFramework(msgs)).toBe("adk");
  });

  it("detects ADK parts with function_call", () => {
    const msgs = [{ role: "model", parts: [{ function_call: { name: "search", args: {} } }] }];
    expect(detectFramework(msgs)).toBe("adk");
  });

  // ── LangChain ────────────────────────────────────────────────────

  it("detects LangChain dict with additional_kwargs", () => {
    const msgs = [{ role: "user", content: "Hi", additional_kwargs: {} }];
    expect(detectFramework(msgs)).toBe("langchain");
  });

  // ── Priority / Edge Cases ────────────────────────────────────────

  it("ADK takes priority over anthropic content blocks", () => {
    const msgs = [{ role: "model", content: [{ type: "text", text: "hi" }], function_calls: [] }];
    expect(detectFramework(msgs)).toBe("adk");
  });

  it("VALID_FRAMEWORKS has all 7 frameworks", () => {
    expect(VALID_FRAMEWORKS.size).toBe(7);
    for (const fw of [
      "openai",
      "anthropic",
      "agent_sdk",
      "adk",
      "langchain",
      "memosift",
      "vercel_ai",
    ]) {
      expect(VALID_FRAMEWORKS.has(fw as never)).toBe(true);
    }
  });
});
