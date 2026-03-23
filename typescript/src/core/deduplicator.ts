// Layer 2: Semantic deduplication — exact hash + MinHash/LSH fuzzy + TF-IDF fallback.

import { createHash } from "node:crypto";
import type { MemoSiftConfig } from "../config.js";
import {
  type ClassifiedMessage,
  CompressionPolicy,
  ContentType,
  type CrossWindowState,
  type DependencyMap,
  type MemoSiftMessage,
  createClassified,
  createDependencyMap,
  createMessage,
  depMapAdd,
} from "./types.js";

// Policies that skip deduplication entirely.
const SKIP_DEDUP_POLICIES = new Set([CompressionPolicy.PRESERVE, CompressionPolicy.LIGHT]);

// MinHash parameters tuned for 0.85 Jaccard threshold.
const NUM_HASHES = 128;
const NUM_BANDS = 16;
const ROWS_PER_BAND = NUM_HASHES / NUM_BANDS; // 8

// Use TF-IDF for groups smaller than this; MinHash overhead isn't worth it.
const MINHASH_MIN_GROUP_SIZE = 5;

// Shingle size for MinHash (character n-grams).
const SHINGLE_SIZE = 5;

// 32-bit safe prime for MinHash hash functions.
const PRIME = 2147483647; // 2^31 - 1

// Large primes for hash coefficient generation.
const LARGE_PRIME_A = 1610612741;
const LARGE_PRIME_B = 805306457;

// Pre-computed hash function coefficients (deterministic seed).
const HASH_A: number[] = [];
const HASH_B: number[] = [];
for (let i = 0; i < NUM_HASHES; i++) {
  HASH_A.push((((i + 1) * LARGE_PRIME_A + 1) % PRIME) >>> 0);
  HASH_B.push((((i + 1) * LARGE_PRIME_B + 7) % PRIME) >>> 0);
}

export function deduplicate(
  segments: ClassifiedMessage[],
  config: MemoSiftConfig,
  crossWindow?: CrossWindowState | null,
  exactOnly?: boolean,
): { segments: ClassifiedMessage[]; deps: DependencyMap } {
  const deps: DependencyMap = createDependencyMap();
  let result = exactDedup(segments, deps, crossWindow);
  if (!exactOnly) {
    result = fuzzyDedup(result, config.dedupSimilarityThreshold, deps);
    result = chunkDedup(result, deps);
  }
  return { segments: result, deps };
}

function normalizeForHash(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function contentHash(text: string): string {
  return createHash("sha256").update(normalizeForHash(text)).digest("hex");
}

function exactDedup(
  segments: ClassifiedMessage[],
  deps: DependencyMap,
  crossWindow?: CrossWindowState | null,
): ClassifiedMessage[] {
  const seen = new Map<string, number>(); // hash → index in input segments list
  const result: ClassifiedMessage[] = [];

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i]!;
    if (SKIP_DEDUP_POLICIES.has(seg.policy)) {
      result.push(seg);
      continue;
    }
    const h = contentHash(seg.message.content);
    if (seen.has(h)) {
      const originalSegIdx = seen.get(h)!;
      const original = segments[originalSegIdx]!;
      const toolName = seg.message.name ?? "content";
      const refText = `[${toolName} was read earlier in this session. Content unchanged. See message #${original.originalIndex}.]`;
      const newMsg = createMessage(seg.message.role, refText, {
        name: seg.message.name,
        toolCallId: seg.message.toolCallId,
        toolCalls: seg.message.toolCalls,
        metadata: seg.message.metadata,
      });
      depMapAdd(deps, seg.originalIndex, original.originalIndex);
      result.push(
        createClassified(newMsg, seg.contentType, seg.policy, {
          originalIndex: seg.originalIndex,
          relevanceScore: seg.relevanceScore,
          protected: seg.protected,
        }),
      );
    } else if (crossWindow?.contentHashes.has(h)) {
      // Seen in a previous window.
      const toolName = seg.message.name ?? "content";
      const refText = `[${toolName} was read earlier in this session. Content unchanged (seen in previous context window).]`;
      const newMsg = createMessage(seg.message.role, refText, {
        name: seg.message.name,
        toolCallId: seg.message.toolCallId,
        toolCalls: seg.message.toolCalls,
        metadata: seg.message.metadata,
      });
      result.push(
        createClassified(newMsg, seg.contentType, seg.policy, {
          originalIndex: seg.originalIndex,
          relevanceScore: seg.relevanceScore,
          protected: seg.protected,
        }),
      );
    } else {
      seen.set(h, i);
      result.push(seg);
    }
  }

  // Add all hashes to cross-window state.
  if (crossWindow) {
    for (const h of seen.keys()) {
      crossWindow.contentHashes.add(h);
    }
  }

  return result;
}

function fuzzyDedup(
  segments: ClassifiedMessage[],
  threshold: number,
  deps: DependencyMap,
): ClassifiedMessage[] {
  const groups = new Map<string, number[]>();
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i]!;
    if (SKIP_DEDUP_POLICIES.has(seg.policy)) continue;
    const key = seg.contentType;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(i);
  }

  for (const indices of groups.values()) {
    if (indices.length < 2) continue;
    const docs = indices.map((i) => segments[i]!.message.content);
    if (!docs.some((d) => d)) continue;

    if (indices.length >= MINHASH_MIN_GROUP_SIZE) {
      fuzzyDedupMinhash(segments, indices, docs, threshold, deps);
    } else {
      fuzzyDedupTfidf(segments, indices, docs, threshold, deps);
    }
  }

  return segments;
}

/** Mark older message (aPos) as duplicate of newer (bPos). Mutates segments array. */
function markAsDedup(
  segments: ClassifiedMessage[],
  indices: number[],
  aPos: number,
  bPos: number,
  deps: DependencyMap,
): void {
  const olderIdx = indices[aPos]!;
  const newerIdx = indices[bPos]!;
  const older = segments[olderIdx]!;
  const newer = segments[newerIdx]!;
  const toolName = older.message.name ?? "content";
  const refText = `[${toolName} was read earlier in this session. Content unchanged. See message #${newer.originalIndex}.]`;
  const newMsg = createMessage(older.message.role, refText, {
    name: older.message.name,
    toolCallId: older.message.toolCallId,
    toolCalls: older.message.toolCalls,
    metadata: older.message.metadata,
  });
  segments[olderIdx] = createClassified(newMsg, older.contentType, older.policy, {
    originalIndex: older.originalIndex,
    relevanceScore: older.relevanceScore,
    protected: older.protected,
  });
  depMapAdd(deps, older.originalIndex, newer.originalIndex);
}

/** MinHash/LSH fuzzy dedup — O(n) after preprocessing. */
function fuzzyDedupMinhash(
  segments: ClassifiedMessage[],
  indices: number[],
  docs: string[],
  threshold: number,
  deps: DependencyMap,
): void {
  // Compute MinHash signatures.
  const signatures = docs.map((doc) => minhashSignature(doc));

  // LSH banding: hash each band to find candidate pairs.
  const candidates = new Set<string>(); // "aPos,bPos" encoded as string
  for (let band = 0; band < NUM_BANDS; band++) {
    const buckets = new Map<string, number[]>();
    const start = band * ROWS_PER_BAND;
    const end = start + ROWS_PER_BAND;

    for (let pos = 0; pos < signatures.length; pos++) {
      const sig = signatures[pos]!;
      // Create a deterministic band hash from the signature slice.
      const bandKey = sig.slice(start, end).join(",");
      if (!buckets.has(bandKey)) buckets.set(bandKey, []);
      buckets.get(bandKey)!.push(pos);
    }

    for (const members of buckets.values()) {
      if (members.length < 2) continue;
      for (let a = 0; a < members.length; a++) {
        for (let b = a + 1; b < members.length; b++) {
          const pa = members[a]!;
          const pb = members[b]!;
          // Always store smaller index first for dedup consistency.
          const key = pa < pb ? `${pa},${pb}` : `${pb},${pa}`;
          candidates.add(key);
        }
      }
    }
  }

  // Verify candidates with full Jaccard similarity.
  const deduped = new Set<number>();
  // Sort candidates for deterministic processing order.
  const sortedCandidates = [...candidates].sort();

  for (const key of sortedCandidates) {
    const [aStr, bStr] = key.split(",");
    const aPos = Number(aStr);
    const bPos = Number(bStr);

    if (deduped.has(aPos) || deduped.has(bPos)) continue;

    const sim = jaccardFromMinhash(signatures[aPos]!, signatures[bPos]!);
    if (sim >= threshold) {
      markAsDedup(segments, indices, aPos, bPos, deps);
      deduped.add(aPos);
    }
  }
}

/** TF-IDF cosine similarity fuzzy dedup — fallback for small groups. */
function fuzzyDedupTfidf(
  segments: ClassifiedMessage[],
  indices: number[],
  docs: string[],
  threshold: number,
  deps: DependencyMap,
): void {
  const vectors = tfidfVectors(docs);
  const deduped = new Set<number>();

  for (let aPos = 0; aPos < indices.length; aPos++) {
    if (deduped.has(aPos)) continue;
    for (let bPos = aPos + 1; bPos < indices.length; bPos++) {
      if (deduped.has(bPos)) continue;
      const sim = cosineSimilarity(vectors[aPos]!, vectors[bPos]!);
      if (sim >= threshold) {
        markAsDedup(segments, indices, aPos, bPos, deps);
        deduped.add(aPos);
        break;
      }
    }
  }
}

// ── MinHash implementation (zero dependencies) ──────────────────────────────

/** Simple 32-bit string hash (FNV-1a variant). */
function hashString32(s: string): number {
  let hash = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < s.length; i++) {
    hash ^= s.charCodeAt(i);
    // FNV-1a multiply: multiply by FNV prime 16777619, keep 32-bit.
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0; // ensure unsigned 32-bit
}

/** Generate character k-gram shingle hashes from text. */
export function shingles(text: string, k: number = SHINGLE_SIZE): Set<number> {
  const t = text.toLowerCase().trim();
  if (t.length < k) {
    return t ? new Set([hashString32(t)]) : new Set();
  }
  const result = new Set<number>();
  for (let i = 0; i <= t.length - k; i++) {
    result.add(hashString32(t.slice(i, i + k)));
  }
  return result;
}

/** Compute MinHash signature (array of NUM_HASHES minimum hash values). */
export function minhashSignature(text: string): number[] {
  const shingleSet = shingles(text);
  if (shingleSet.size === 0) {
    return new Array(NUM_HASHES).fill(PRIME);
  }

  const sig = new Array<number>(NUM_HASHES).fill(PRIME);
  for (const shingle of shingleSet) {
    for (let i = 0; i < NUM_HASHES; i++) {
      // (a * shingle + b) mod PRIME, using 32-bit safe Math.imul + modulo.
      // Since PRIME = 2^31-1, intermediate results fit in safe integer range
      // when we compute step by step.
      const ax = Math.imul(HASH_A[i]!, shingle >>> 0);
      const h = (((ax >>> 0) + HASH_B[i]!) % PRIME) >>> 0;
      if (h < sig[i]!) {
        sig[i] = h;
      }
    }
  }
  return sig;
}

/** Estimate Jaccard similarity from two MinHash signatures. */
export function jaccardFromMinhash(sigA: number[], sigB: number[]): number {
  if (sigA.length === 0 || sigB.length === 0) return 0;
  let matches = 0;
  for (let i = 0; i < sigA.length; i++) {
    if (sigA[i] === sigB[i]) matches++;
  }
  return matches / sigA.length;
}

// ── TF-IDF implementation ──────────────────────────────────────────────────

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/\b\w+\b/g) ?? [];
}

function tfidfVectors(documents: string[]): Map<string, number>[] {
  const nDocs = documents.length;
  const tokenized = documents.map(tokenize);
  const docFreq = new Map<string, number>();

  for (const tokens of tokenized) {
    const unique = new Set(tokens);
    for (const token of unique) {
      docFreq.set(token, (docFreq.get(token) ?? 0) + 1);
    }
  }

  const idf = new Map<string, number>();
  for (const [token, freq] of docFreq) {
    idf.set(token, Math.log((nDocs + 1) / (freq + 1)) + 1);
  }

  return tokenized.map((tokens) => {
    if (tokens.length === 0) return new Map();
    const tf = new Map<string, number>();
    for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
    const vec = new Map<string, number>();
    for (const [token, count] of tf) {
      vec.set(token, (count / tokens.length) * (idf.get(token) ?? 1));
    }
    return vec;
  });
}

function cosineSimilarity(a: Map<string, number>, b: Map<string, number>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let dot = 0;
  for (const [k, v] of a) {
    const bv = b.get(k);
    if (bv !== undefined) dot += v * bv;
  }
  let normA = 0;
  for (const v of a.values()) normA += v * v;
  let normB = 0;
  for (const v of b.values()) normB += v * v;
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}

// ── Chunk-level deduplication ──────────────────────────────────────────────

const CHUNK_DEDUP_MIN_TOKENS = 1000;
const CHUNK_TARGET_TOKENS = 200;

function chunkDedup(segments: ClassifiedMessage[], deps: DependencyMap): ClassifiedMessage[] {
  const seenChunks = new Map<string, [number, string]>();
  const result: ClassifiedMessage[] = [];

  for (const seg of segments) {
    if (SKIP_DEDUP_POLICIES.has(seg.policy)) {
      result.push(seg);
      continue;
    }

    const estTokens = (seg.message.content || "").split(/\s+/).length;
    if (estTokens < CHUNK_DEDUP_MIN_TOKENS) {
      result.push(seg);
      continue;
    }

    const chunks =
      seg.contentType === ContentType.CODE_BLOCK
        ? splitCodeChunks(seg.message.content)
        : splitParagraphChunks(seg.message.content);

    if (chunks.length <= 1) {
      result.push(seg);
      continue;
    }

    const newChunks: string[] = [];
    let anyDeduped = false;
    for (const chunk of chunks) {
      const chunkHash = createHash("sha256")
        .update(chunk.replace(/\s+/g, " ").trim())
        .digest("hex");

      if (seenChunks.has(chunkHash)) {
        const [origIdx, label] = seenChunks.get(chunkHash)!;
        newChunks.push(`[Duplicate chunk — see message ${origIdx}: ${label}]`);
        anyDeduped = true;
      } else {
        seenChunks.set(chunkHash, [seg.originalIndex, chunk.slice(0, 60).trim()]);
        newChunks.push(chunk);
      }
    }

    if (anyDeduped) {
      const newMsg = createMessage(seg.message.role, newChunks.join("\n\n"), {
        name: seg.message.name,
        toolCallId: seg.message.toolCallId,
        toolCalls: seg.message.toolCalls,
        metadata: seg.message.metadata,
      });
      result.push(
        createClassified(newMsg, seg.contentType, seg.policy, {
          originalIndex: seg.originalIndex,
          relevanceScore: seg.relevanceScore,
          protected: seg.protected,
          importanceScore: seg.importanceScore,
          shield: seg.shield,
        }),
      );
    } else {
      result.push(seg);
    }
  }

  return result;
}

function splitParagraphChunks(text: string): string[] {
  const paragraphs = text.split(/\n\s*\n/);
  const chunks: string[] = [];
  let current: string[] = [];
  let currentTokens = 0;

  for (const para of paragraphs) {
    const paraTokens = para.split(/\s+/).length;
    if (currentTokens + paraTokens > CHUNK_TARGET_TOKENS && current.length > 0) {
      chunks.push(current.join("\n\n"));
      current = [para];
      currentTokens = paraTokens;
    } else {
      current.push(para);
      currentTokens += paraTokens;
    }
  }
  if (current.length > 0) chunks.push(current.join("\n\n"));
  return chunks;
}

function splitCodeChunks(text: string): string[] {
  const parts = text
    .split(/^(?=\s*(?:def |class |function |async |export ))/m)
    .filter((p) => p.trim());
  if (parts.length <= 1) return splitParagraphChunks(text);

  const chunks: string[] = [];
  let current: string[] = [];
  let currentTokens = 0;
  for (const part of parts) {
    const partTokens = part.split(/\s+/).length;
    if (currentTokens + partTokens > CHUNK_TARGET_TOKENS && current.length > 0) {
      chunks.push(current.join("\n"));
      current = [part];
      currentTokens = partTokens;
    } else {
      current.push(part);
      currentTokens += partTokens;
    }
  }
  if (current.length > 0) chunks.push(current.join("\n"));
  return chunks;
}
