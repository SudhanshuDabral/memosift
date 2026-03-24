// Tests for MemoSiftStream (TypeScript).

import { describe, expect, it } from "vitest";
import { Pressure } from "../../typescript/src/core/context-window.js";
import { createMessage } from "../../typescript/src/core/types.js";
import { MemoSiftStream } from "../../typescript/src/stream.js";

describe("MemoSiftStream", () => {
  it("creates an incremental session", () => {
    const stream = new MemoSiftStream("coding", { model: "claude-sonnet-4-6" });
    expect(stream.session.incremental).toBe(true);
    expect(stream.messageCount).toBe(0);
  });

  it("buffers messages when no model configured (NONE pressure)", async () => {
    const stream = new MemoSiftStream("general");
    const event = await stream.push(createMessage("user", "Hello"));
    expect(event.action).toBe("buffered");
    expect(event.compressed).toBe(false);
    expect(stream.messageCount).toBe(1);
  });

  it("accumulates multiple messages", async () => {
    const stream = new MemoSiftStream("general");
    await stream.push(createMessage("user", "Hello"));
    await stream.push(createMessage("assistant", "Hi!"));
    await stream.push(createMessage("user", "What is Python?"));
    expect(stream.messageCount).toBe(3);
  });

  it("flush forces compression", async () => {
    const stream = new MemoSiftStream("general");
    await stream.push(createMessage("system", "You are helpful."));
    await stream.push(createMessage("user", "Hello"));
    await stream.push(createMessage("assistant", "Hi!"));
    await stream.push(createMessage("user", "Thanks"));

    const event = await stream.flush();
    expect(event.action).toBe("compressed");
    expect(event.compressed).toBe(true);
    expect(stream.messageCount).toBeGreaterThanOrEqual(1);
  });

  it("messages property returns a copy", async () => {
    const stream = new MemoSiftStream("general");
    await stream.push(createMessage("user", "Hello"));
    const msgs = stream.messages;
    expect(msgs).toHaveLength(1);
    // Should be a copy.
    (msgs as unknown[]).push(createMessage("user", "extra"));
    expect(stream.messageCount).toBe(1);
  });

  it("facts are empty initially", () => {
    const stream = new MemoSiftStream("general");
    expect(stream.facts).toEqual([]);
  });

  it("flush on empty stream returns buffered", async () => {
    const stream = new MemoSiftStream("general");
    const event = await stream.flush();
    expect(event.action).toBe("buffered");
    expect(event.compressed).toBe(false);
  });

  it("works with plain object messages", async () => {
    const stream = new MemoSiftStream("general");
    await stream.push({ role: "system", content: "You are helpful." });
    await stream.push({ role: "user", content: "Hello" });
    expect(stream.messageCount).toBe(2);

    const event = await stream.flush();
    expect(event.compressed).toBe(true);
  });
});
