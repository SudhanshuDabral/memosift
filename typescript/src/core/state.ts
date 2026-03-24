// CompressionState — cached pipeline artifacts for incremental compression.

import { createHash } from "node:crypto";
import type { ContentType } from "./types.js";

/**
 * Cached pipeline artifacts for incremental compression.
 *
 * Stores intermediate results across `compress()` calls so subsequent
 * invocations can skip redundant work on already-compressed messages.
 */
export interface CompressionState {
  /** IDF scores from Engine B (pruner). Reused across calls. */
  idfVocabulary: Map<string, number>;
  /** SHA-256 hashes of message content → original message index. */
  contentHashes: Map<string, number>;
  /** Mapping of SHA-256(content) → ContentType. */
  classificationCache: Map<string, ContentType>;
  /** Mapping of SHA-256(content) → estimated token count. */
  tokenCache: Map<string, number>;
  /** Incremented on each compress() call. */
  sequence: number;
  /** SHA-256 hash of the last compressed output. */
  outputHash: string;
}

/** Create a new empty CompressionState. */
export function createCompressionState(): CompressionState {
  return {
    idfVocabulary: new Map(),
    contentHashes: new Map(),
    classificationCache: new Map(),
    tokenCache: new Map(),
    sequence: 0,
    outputHash: "",
  };
}

/** Compute a short SHA-256 hash of content for cache keying. */
function contentHash(content: string): string {
  return createHash("sha256").update(content).digest("hex").slice(0, 16);
}

/** Cache the classification result for a message's content. */
export function cacheClassification(
  state: CompressionState,
  content: string,
  contentType: ContentType,
): void {
  state.classificationCache.set(contentHash(content), contentType);
}

/** Retrieve a cached classification, or null if not cached. */
export function getCachedClassification(
  state: CompressionState,
  content: string,
): ContentType | null {
  return state.classificationCache.get(contentHash(content)) ?? null;
}

/**
 * Two-tier cache key: length for fast discrimination, hash on collision.
 *
 * Most messages have unique lengths, so `L{n}:{prefix}` is enough.
 * For longer content, the SHA-256 prefix ensures correctness.
 */
function tokenCacheKey(content: string): string {
  const n = content.length;
  if (n < 256) return `L${n}:${content.slice(0, 32)}`;
  const h = createHash("sha256").update(content).digest("hex").slice(0, 12);
  return `L${n}:${h}`;
}

/** Cache the token count for a message's content. */
export function cacheTokenCount(state: CompressionState, content: string, count: number): void {
  state.tokenCache.set(tokenCacheKey(content), count);
}

/** Retrieve a cached token count, or null if not cached. */
export function getCachedTokenCount(state: CompressionState, content: string): number | null {
  return state.tokenCache.get(tokenCacheKey(content)) ?? null;
}

/** Record a content hash for dedup cross-referencing. */
export function recordContentHash(state: CompressionState, content: string, index: number): void {
  const key = contentHash(content);
  if (!state.contentHashes.has(key)) {
    state.contentHashes.set(key, index);
  }
}

/** Check if content has been seen before (exact match). */
export function hasContent(state: CompressionState, content: string): boolean {
  return state.contentHashes.has(contentHash(content));
}

/** Increment and return the new sequence number. */
export function bumpSequence(state: CompressionState): number {
  state.sequence += 1;
  return state.sequence;
}

/** Compute and store the output hash from compressed message contents. */
export function setOutputHash(state: CompressionState, messagesContent: string[]): void {
  const combined = messagesContent.join("\n---\n");
  state.outputHash = createHash("sha256").update(combined).digest("hex").slice(0, 32);
}
