// Tests for incremental compression (CompressionState + session incremental mode).

import { describe, expect, it } from "vitest";
import { createCompressionState } from "../../typescript/src/core/state.js";
import {
  bumpSequence,
  cacheClassification,
  cacheTokenCount,
  getCachedClassification,
  getCachedTokenCount,
  hasContent,
  recordContentHash,
  setOutputHash,
} from "../../typescript/src/core/state.js";
import { ContentType, createMessage } from "../../typescript/src/core/types.js";
import { MemoSiftSession } from "../../typescript/src/session.js";

describe("CompressionState", () => {
  it("caches and retrieves classifications", () => {
    const state = createCompressionState();
    cacheClassification(state, "def hello(): pass", ContentType.CODE_BLOCK);
    expect(getCachedClassification(state, "def hello(): pass")).toBe(ContentType.CODE_BLOCK);
    expect(getCachedClassification(state, "different content")).toBeNull();
  });

  it("caches and retrieves token counts", () => {
    const state = createCompressionState();
    cacheTokenCount(state, "hello world", 3);
    expect(getCachedTokenCount(state, "hello world")).toBe(3);
    expect(getCachedTokenCount(state, "other")).toBeNull();
  });

  it("tracks content hashes", () => {
    const state = createCompressionState();
    expect(hasContent(state, "test content")).toBe(false);
    recordContentHash(state, "test content", 0);
    expect(hasContent(state, "test content")).toBe(true);
  });

  it("bumps sequence numbers", () => {
    const state = createCompressionState();
    expect(state.sequence).toBe(0);
    expect(bumpSequence(state)).toBe(1);
    expect(bumpSequence(state)).toBe(2);
    expect(state.sequence).toBe(2);
  });

  it("computes output hash", () => {
    const state = createCompressionState();
    setOutputHash(state, ["hello", "world"]);
    const hash1 = state.outputHash;
    expect(hash1).toHaveLength(32);

    setOutputHash(state, ["hello", "world", "!"]);
    expect(state.outputHash).not.toBe(hash1);
  });
});

describe("MemoSiftSession incremental mode", () => {
  it("creates state when incremental=true", () => {
    const session = new MemoSiftSession("general", { incremental: true });
    expect(session.incremental).toBe(true);
    expect(session.state).not.toBeNull();
  });

  it("has no state when incremental=false", () => {
    const session = new MemoSiftSession("general");
    expect(session.incremental).toBe(false);
    expect(session.state).toBeNull();
  });

  it("populates state after compress", async () => {
    const session = new MemoSiftSession("general", { incremental: true });
    const messages = [
      createMessage("system", "You are a helpful assistant."),
      createMessage("user", "Hello"),
      createMessage("assistant", "Hi! How can I help?"),
      createMessage("user", "Tell me about Python"),
    ];

    await session.compress(messages);
    expect(session.state!.sequence).toBe(1);
    expect(session.state!.outputHash).toHaveLength(32);
  });

  it("increments sequence on subsequent calls", async () => {
    const session = new MemoSiftSession("general", { incremental: true });
    const messages = [
      createMessage("system", "You are a helpful assistant."),
      createMessage("user", "Hello"),
      createMessage("assistant", "Hi!"),
      createMessage("user", "What is Python?"),
    ];

    const { messages: compressed1 } = await session.compress(messages);
    expect(session.state!.sequence).toBe(1);

    const newMessages = [
      ...compressed1,
      createMessage("assistant", "Python is a language."),
      createMessage("user", "Tell me more."),
    ];
    await session.compress(newMessages);
    expect(session.state!.sequence).toBe(2);
  });

  it("produces same output as batch mode", async () => {
    const messages = [
      createMessage("system", "You are a helpful assistant."),
      createMessage("user", "Hello"),
      createMessage("assistant", "Hi there!"),
      createMessage("user", "What is Python?"),
    ];

    const batchSession = new MemoSiftSession("general");
    const { messages: batchResult } = await batchSession.compress(messages);

    const incSession = new MemoSiftSession("general", { incremental: true });
    const { messages: incResult } = await incSession.compress(messages);

    expect(batchResult.length).toBe(incResult.length);
    for (let i = 0; i < batchResult.length; i++) {
      expect(batchResult[i]!.role).toBe(incResult[i]!.role);
      expect(batchResult[i]!.content).toBe(incResult[i]!.content);
    }
  });
});
