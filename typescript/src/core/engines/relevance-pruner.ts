// Layer 3E: Query-relevance pruning — TF-IDF + causal dependency check.

import type { MemoSiftConfig } from "../../config.js";
import {
  AnchorCategory,
  type AnchorLedger,
  type ClassifiedMessage,
  CompressionPolicy,
  type DependencyMap,
  Shield,
  createClassified,
  createMessage,
  depMapHasDependents,
  depMapHasLogicalDependents,
} from "../types.js";

/** Policies that skip relevance pruning. */
const SKIP_POLICIES: ReadonlySet<CompressionPolicy> = new Set([
  CompressionPolicy.PRESERVE,
  CompressionPolicy.LIGHT,
]);

/** Stop words excluded from TF-IDF vectorization. */
const STOP_WORDS: ReadonlySet<string> = new Set([
  "a",
  "an",
  "the",
  "and",
  "or",
  "but",
  "in",
  "on",
  "at",
  "to",
  "for",
  "of",
  "with",
  "by",
  "from",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "have",
  "has",
  "had",
  "do",
  "does",
  "did",
  "will",
  "would",
  "could",
  "should",
  "this",
  "that",
  "it",
  "its",
  "i",
  "me",
  "my",
  "we",
  "you",
  "he",
  "she",
  "they",
  "not",
  "no",
  "so",
  "if",
  "then",
  "just",
  "also",
]);

/** Low relevance threshold — messages below this are candidates for compression. */
const LOW_RELEVANCE_THRESHOLD = 0.15;

/** High anchor coverage threshold — if facts are in ledger, safe to collapse. */
const ANCHOR_COVERAGE_THRESHOLD = 0.8;

/** Sparse TF-IDF vector represented as a map from term to weight. */
type TfIdfVector = Map<string, number>;

/**
 * Prune low-relevance messages using TF-IDF cosine similarity against recent queries.
 *
 * Scores each non-protected segment against the last 2 user queries (expanded
 * with anchor ledger terms). Low-relevance segments with COMPRESSIBLE shield
 * are either collapsed to ledger references or heavily pruned.
 *
 * Safety mechanisms (3 guards):
 * 1. DependencyMap dedup check — don't remove messages that are referenced.
 * 2. DependencyMap logical check — don't break reasoning chains.
 * 3. Shield level — respect PRESERVE shields from importance scoring.
 */
export function queryRelevancePrune(
  segments: ClassifiedMessage[],
  config: MemoSiftConfig,
  deps: DependencyMap,
  ledger?: AnchorLedger | null,
): ClassifiedMessage[] {
  // Extract last 2 user queries.
  const userQueries = extractRecentUserQueries(segments, 2);
  if (userQueries.length === 0) {
    return segments; // No queries to score against.
  }

  // Build expanded query from user queries + anchor ledger context.
  const expandedQuery = buildExpandedQuery(userQueries, ledger ?? null);

  // Compute TF-IDF vectors for all segments + query.
  const allTexts = [...segments.map((seg) => seg.message.content || ""), expandedQuery];
  const vectors = tfidfVectors(allTexts);
  const queryVector = vectors[vectors.length - 1]!;
  const segmentVectors = vectors.slice(0, -1);

  const result: ClassifiedMessage[] = [];
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i]!;

    // Skip protected segments.
    if (SKIP_POLICIES.has(seg.policy) || seg.protected) {
      result.push(seg);
      continue;
    }

    // Skip PRESERVE-shielded segments.
    if (seg.shield === Shield.PRESERVE) {
      result.push(seg);
      continue;
    }

    // Compute relevance score.
    let score = cosineSimilarity(segmentVectors[i]!, queryVector);

    // Distractor detection: high similarity but low overlap with active context.
    if (score > 0.3 && ledger != null) {
      const activeFacts = ledger.factsByCategory(AnchorCategory.ACTIVE_CONTEXT);
      if (activeFacts.length > 0) {
        const activeText = activeFacts.map((f) => f.content).join(" ");
        const activeOverlap = factOverlap(seg.message.content || "", activeText);
        if (activeOverlap < 0.2) {
          score *= 0.5; // Penalize distractor.
        }
      }
    }

    // Check dependencies before compressing.
    if (depMapHasDependents(deps, seg.originalIndex)) {
      result.push(seg);
      continue;
    }
    if (depMapHasLogicalDependents(deps, seg.originalIndex)) {
      result.push(seg);
      continue;
    }

    // Apply compression based on score and shield.
    if (score < LOW_RELEVANCE_THRESHOLD && seg.shield === Shield.COMPRESSIBLE) {
      if (
        ledger != null &&
        anchorCoverage(seg.message.content || "", ledger) >= ANCHOR_COVERAGE_THRESHOLD
      ) {
        // Facts are in the ledger — safe to collapse.
        const newMsg = createMessage(seg.message.role, "[Facts preserved in anchor ledger]", {
          name: seg.message.name,
          toolCallId: seg.message.toolCallId,
          toolCalls: seg.message.toolCalls,
          metadata: seg.message.metadata,
        });
        result.push(
          createClassified(newMsg, seg.contentType, seg.policy, {
            originalIndex: seg.originalIndex,
            relevanceScore: score,
            estimatedTokens: seg.estimatedTokens,
            protected: seg.protected,
            importanceScore: seg.importanceScore,
            shield: seg.shield,
          }),
        );
      } else {
        // Apply heavy token pruning (keep 30% of tokens).
        const pruned = heavyPrune(seg.message.content || "", 0.3);
        const newMsg = createMessage(seg.message.role, pruned, {
          name: seg.message.name,
          toolCallId: seg.message.toolCallId,
          toolCalls: seg.message.toolCalls,
          metadata: seg.message.metadata,
        });
        result.push(
          createClassified(newMsg, seg.contentType, seg.policy, {
            originalIndex: seg.originalIndex,
            relevanceScore: score,
            estimatedTokens: seg.estimatedTokens,
            protected: seg.protected,
            importanceScore: seg.importanceScore,
            shield: seg.shield,
          }),
        );
      }
    } else {
      result.push(
        createClassified(seg.message, seg.contentType, seg.policy, {
          originalIndex: seg.originalIndex,
          relevanceScore: Math.max(seg.relevanceScore, score),
          estimatedTokens: seg.estimatedTokens,
          protected: seg.protected,
          importanceScore: seg.importanceScore,
          shield: seg.shield,
        }),
      );
    }
  }

  return result;
}

/** Extract the last N user messages from segments. */
function extractRecentUserQueries(segments: ClassifiedMessage[], n: number): string[] {
  const queries: string[] = [];
  for (let i = segments.length - 1; i >= 0; i--) {
    const seg = segments[i]!;
    if (seg.message.role === "user" && seg.message.content) {
      queries.push(seg.message.content);
      if (queries.length >= n) break;
    }
  }
  queries.reverse();
  return queries;
}

/** Build expanded query from user queries + anchor ledger ACTIVE_CONTEXT and ERRORS. */
function buildExpandedQuery(queries: string[], ledger: AnchorLedger | null): string {
  const parts = [...queries];
  if (ledger != null) {
    for (const fact of ledger.factsByCategory(AnchorCategory.ACTIVE_CONTEXT)) {
      parts.push(fact.content);
    }
    for (const fact of ledger.factsByCategory(AnchorCategory.ERRORS)) {
      parts.push(fact.content);
    }
  }
  return parts.join(" ");
}

/** Compute word overlap between text and reference (0.0-1.0). */
function factOverlap(text: string, reference: string): number {
  const textWords = new Set(
    (text.toLowerCase().match(/\b\w+\b/g) ?? []).filter((w) => !STOP_WORDS.has(w)),
  );
  const refWords = new Set(
    (reference.toLowerCase().match(/\b\w+\b/g) ?? []).filter((w) => !STOP_WORDS.has(w)),
  );
  if (refWords.size === 0) return 0.0;
  let overlap = 0;
  for (const word of textWords) {
    if (refWords.has(word)) overlap++;
  }
  return overlap / refWords.size;
}

/** Compute what fraction of the segment's entities appear in the anchor ledger. */
function anchorCoverage(text: string, ledger: AnchorLedger): number {
  if (!text) return 0.0;
  const protectedStrings = ledger.getProtectedStrings();
  if (protectedStrings.size === 0) return 0.0;

  // Extract entities from text.
  const entities = new Set<string>();

  // File paths.
  const filePathPattern = /(?:[A-Za-z]:)?[\w.\-]+(?:[/\\][\w.\-]+)+/g;
  for (const match of text.matchAll(filePathPattern)) {
    entities.add(match[0]);
  }

  // Code identifiers (camelCase).
  const camelCasePattern = /\b[a-z]+(?:[A-Z][a-z]+)+\b/g;
  for (const match of text.matchAll(camelCasePattern)) {
    entities.add(match[0]);
  }

  // Code identifiers (snake_case).
  const snakeCasePattern = /\b[a-z]+(?:_[a-z]+)+\b/g;
  for (const match of text.matchAll(snakeCasePattern)) {
    entities.add(match[0]);
  }

  if (entities.size === 0) return 1.0; // No entities to check — consider fully covered.

  let covered = 0;
  for (const entity of entities) {
    for (const p of protectedStrings) {
      if (p.includes(entity) || entity.includes(p)) {
        covered++;
        break;
      }
    }
  }
  return covered / entities.size;
}

/**
 * Aggressively prune tokens, keeping only the first `keepRatio` fraction of words per line.
 * Protected patterns (file paths, numbers, identifiers) are always kept.
 */
function heavyPrune(text: string, keepRatio: number): string {
  return text
    .split("\n")
    .map((line) => {
      const words = line.split(/\s+/).filter((w) => w.length > 0);
      if (words.length === 0) return line;
      const keepCount = Math.max(1, Math.floor(words.length * keepRatio));
      return words.slice(0, keepCount).join(" ");
    })
    .join("\n");
}

// -- TF-IDF implementation -------------------------------------------------------

/** Split text into lowercase word tokens, excluding stop words. */
function tokenize(text: string): string[] {
  const tokens = text.toLowerCase().match(/\b\w+\b/g) ?? [];
  return tokens.filter((t) => !STOP_WORDS.has(t));
}

/** Compute sparse TF-IDF vectors for a list of documents. */
function tfidfVectors(documents: string[]): TfIdfVector[] {
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

  const vectors: TfIdfVector[] = [];
  for (const tokens of tokenized) {
    if (tokens.length === 0) {
      vectors.push(new Map());
      continue;
    }
    const tf = new Map<string, number>();
    for (const token of tokens) {
      tf.set(token, (tf.get(token) ?? 0) + 1);
    }
    const total = tokens.length;
    const vec = new Map<string, number>();
    for (const [token, count] of tf) {
      vec.set(token, (count / total) * (idf.get(token) ?? 1.0));
    }
    vectors.push(vec);
  }

  return vectors;
}

/** Cosine similarity between two sparse TF-IDF vectors. */
function cosineSimilarity(a: TfIdfVector, b: TfIdfVector): number {
  if (a.size === 0 || b.size === 0) return 0.0;

  let dot = 0;
  for (const [key, aVal] of a) {
    const bVal = b.get(key);
    if (bVal !== undefined) {
      dot += aVal * bVal;
    }
  }

  let normA = 0;
  for (const v of a.values()) normA += v * v;

  let normB = 0;
  for (const v of b.values()) normB += v * v;

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) return 0.0;
  return dot / (normA * normB);
}
