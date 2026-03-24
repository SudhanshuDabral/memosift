// Tests for the Vercel AI SDK adapter (TypeScript).

import { describe, expect, it } from "vitest";
import {
  adaptIn,
  adaptOut,
  compressVercelMessages,
} from "../../typescript/src/adapters/vercel-ai.js";
import { createMessage } from "../../typescript/src/core/types.js";

describe("Vercel AI SDK Adapter", () => {
  describe("adaptIn", () => {
    it("converts string content", () => {
      const messages = [{ role: "user", content: "Hello, world!" }];
      const result = adaptIn(messages);
      expect(result).toHaveLength(1);
      expect(result[0]!.role).toBe("user");
      expect(result[0]!.content).toBe("Hello, world!");
      expect(result[0]!.metadata._vercel_content_type).toBe("string");
    });

    it("concatenates TextPart array", () => {
      const messages = [
        {
          role: "assistant",
          content: [
            { type: "text", text: "Hello" },
            { type: "text", text: "World" },
          ],
        },
      ];
      const result = adaptIn(messages);
      expect(result).toHaveLength(1);
      expect(result[0]!.content).toBe("Hello\nWorld");
    });

    it("converts ToolCallPart to ToolCall", () => {
      const messages = [
        {
          role: "assistant",
          content: [
            { type: "text", text: "Checking." },
            {
              type: "tool-call",
              toolCallId: "tc1",
              toolName: "read_file",
              args: { path: "test.py" },
            },
          ],
        },
      ];
      const result = adaptIn(messages);
      expect(result[0]!.toolCalls).toHaveLength(1);
      expect(result[0]!.toolCalls![0]!.id).toBe("tc1");
      expect(result[0]!.toolCalls![0]!.function.name).toBe("read_file");
    });

    it("converts ToolResultPart to tool message", () => {
      const messages = [
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "tc1",
              toolName: "read_file",
              result: "file content here",
            },
          ],
        },
      ];
      const result = adaptIn(messages);
      expect(result).toHaveLength(1);
      expect(result[0]!.role).toBe("tool");
      expect(result[0]!.content).toBe("file content here");
      expect(result[0]!.toolCallId).toBe("tc1");
    });

    it("preserves ImagePart in metadata", () => {
      const messages = [
        {
          role: "user",
          content: [
            { type: "text", text: "What is this?" },
            { type: "image", image: "base64data", mimeType: "image/png" },
          ],
        },
      ];
      const result = adaptIn(messages);
      expect(result[0]!.content).toBe("What is this?");
      const preserved = result[0]!.metadata._vercel_preserved_parts as unknown[];
      expect(preserved).toHaveLength(1);
    });

    it("preserves isError flag", () => {
      const messages = [
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "tc3",
              toolName: "run_test",
              result: "AssertionError",
              isError: true,
            },
          ],
        },
      ];
      const result = adaptIn(messages);
      expect(result[0]!.metadata._vercel_is_error).toBe(true);
    });
  });

  describe("adaptOut", () => {
    it("outputs string content for simple messages", () => {
      const msgs = [
        createMessage("user", "Hello", {
          metadata: { _vercel_content_type: "string" },
        }),
      ];
      const result = adaptOut(msgs);
      expect(result[0]).toEqual({ role: "user", content: "Hello" });
    });

    it("outputs ToolCallPart for tool calls", () => {
      const msgs = [
        createMessage("assistant", "Checking.", {
          toolCalls: [
            {
              id: "tc1",
              type: "function",
              function: { name: "read_file", arguments: '{"path":"x.py"}' },
            },
          ],
          metadata: { _vercel_content_type: "parts" },
        }),
      ];
      const result = adaptOut(msgs);
      const parts = result[0]!.content as Record<string, unknown>[];
      const toolParts = parts.filter((p) => p.type === "tool-call");
      expect(toolParts).toHaveLength(1);
      expect(toolParts[0]!.toolCallId).toBe("tc1");
      expect(toolParts[0]!.toolName).toBe("read_file");
    });

    it("outputs ToolResultPart for tool messages", () => {
      const msgs = [
        createMessage("tool", '{"result": true}', {
          toolCallId: "tc1",
          name: "check",
          metadata: { _vercel_content_type: "tool-result", _vercel_tool_name: "check" },
        }),
      ];
      const result = adaptOut(msgs);
      expect(result[0]!.role).toBe("tool");
      const parts = result[0]!.content as Record<string, unknown>[];
      expect(parts[0]!.type).toBe("tool-result");
      expect(parts[0]!.result).toEqual({ result: true });
    });
  });

  describe("round-trip", () => {
    it("simple messages survive adapt_in → adapt_out", () => {
      const original = [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there!" },
      ];
      const internal = adaptIn(original);
      const restored = adaptOut(internal);
      for (let i = 0; i < original.length; i++) {
        expect(restored[i]!.role).toBe(original[i]!.role);
        expect(restored[i]!.content).toBe(original[i]!.content);
      }
    });

    it("tool call messages survive round-trip", () => {
      const original = [
        {
          role: "assistant",
          content: [
            { type: "text", text: "Let me check." },
            {
              type: "tool-call",
              toolCallId: "tc1",
              toolName: "read_file",
              args: { path: "test.py" },
            },
          ],
        },
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "tc1",
              toolName: "read_file",
              result: "def hello(): pass",
            },
          ],
        },
      ];
      const internal = adaptIn(original);
      const restored = adaptOut(internal);

      // Assistant message.
      const parts = restored[0]!.content as Record<string, unknown>[];
      const textParts = parts.filter((p) => p.type === "text");
      const toolParts = parts.filter((p) => p.type === "tool-call");
      expect(textParts[0]!.text).toBe("Let me check.");
      expect(toolParts[0]!.toolCallId).toBe("tc1");

      // Tool result.
      const resultParts = restored[1]!.content as Record<string, unknown>[];
      expect(resultParts[0]!.toolCallId).toBe("tc1");
    });
  });

  describe("compressVercelMessages", () => {
    it("compresses end-to-end", async () => {
      const messages = [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there!" },
        { role: "user", content: "What is Python?" },
      ];
      const { messages: compressed, report } = await compressVercelMessages(messages);
      expect(compressed.length).toBeGreaterThanOrEqual(1);
      expect(report.originalTokens).toBeGreaterThan(0);
    });
  });
});
