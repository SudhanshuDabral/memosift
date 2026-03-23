// Engine C: Structural compression — code signature extraction + JSON truncation.

import type { MemoSiftConfig } from "../../config.js";
import {
  type AnchorLedger,
  type ClassifiedMessage,
  CompressionPolicy,
  ContentType,
  createClassified,
  createMessage,
} from "../types.js";

const TARGET_POLICIES = new Set([CompressionPolicy.STRUCTURAL, CompressionPolicy.SIGNATURE]);

const JS_CLASS_RE = /^(\s*(?:export\s+)?class\s+\w+(?:\s+extends\s+\w+)?)/gm;
const JS_FUNC_RE =
  /^(\s*(?:export\s+)?(?:async\s+)?(?:function\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:function|\(.*?\)\s*=>)))/gm;
const JS_METHOD_RE =
  /^(\s*(?:async\s+)?(?:get\s+|set\s+)?(?!if\b|else\b|while\b|for\b|switch\b|catch\b)\w+\s*\(.*?\)(?:\s*:\s*\S+)?)\s*\{/gm;
const PY_CLASS_RE = /^(\s*class\s+\w+(?:\(.*?\))?)\s*:/gm;
const PY_FUNC_RE = /^(\s*(?:async\s+)?def\s+\w+\s*\(.*?\)(?:\s*->\s*.+)?)\s*:/gm;

export function structuralCompress(
  segments: ClassifiedMessage[],
  config: MemoSiftConfig,
  ledger?: AnchorLedger | null,
): ClassifiedMessage[] {
  return segments.map((seg) => {
    if (!TARGET_POLICIES.has(seg.policy)) return seg;

    let newContent: string;
    if (seg.contentType === ContentType.TOOL_RESULT_JSON) {
      newContent = compressJson(seg.message.content, config.jsonArrayThreshold);
    } else if (seg.contentType === ContentType.CODE_BLOCK) {
      newContent = compressCode(seg.message.content, config.codeKeepSignatures);
    } else {
      return seg;
    }

    if (newContent === seg.message.content) return seg;
    const newMsg = createMessage(seg.message.role, newContent, {
      name: seg.message.name,
      toolCallId: seg.message.toolCallId,
      toolCalls: seg.message.toolCalls,
      metadata: seg.message.metadata,
    });
    return createClassified(newMsg, seg.contentType, seg.policy, {
      originalIndex: seg.originalIndex,
      relevanceScore: seg.relevanceScore,
      protected: seg.protected,
    });
  });
}

function compressJson(text: string, arrayThreshold: number): string {
  try {
    const data: unknown = JSON.parse(text.trim());
    return JSON.stringify(truncateJsonValue(data, arrayThreshold), null, 2);
  } catch {
    return text;
  }
}

function truncateJsonValue(value: unknown, threshold: number): unknown {
  if (Array.isArray(value)) {
    if (value.length > threshold) {
      // Check for schema-uniform arrays (all items are dicts with same keys).
      const schema = detectArraySchema(value);

      // Keep first 2 exemplars.
      const exemplars = value.slice(0, 2).map((v) => truncateJsonValue(v, threshold));

      const remaining = value.length - 2;
      if (remaining > 0) {
        if (schema) {
          const keysStr = schema.join(", ");
          return [
            ...exemplars,
            `... ${remaining} more items with same schema ({${keysStr}}) (total: ${value.length})`,
          ];
        }
        return [...exemplars, `... and ${remaining} more items (total: ${value.length})`];
      }
      return exemplars;
    }
    return value.map((v) => truncateJsonValue(v, threshold));
  }
  if (typeof value === "object" && value !== null) {
    const result: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      result[k] = truncateJsonValue(v, threshold);
    }
    return result;
  }
  return value;
}

/**
 * Detect if all items in an array are objects with identical keys.
 * Returns the sorted key list if uniform, null otherwise.
 */
function detectArraySchema(items: unknown[]): string[] | null {
  if (items.length < 3) return null;
  if (!items.every((item) => typeof item === "object" && item !== null && !Array.isArray(item))) {
    return null;
  }

  const firstKeys = Object.keys(items[0] as Record<string, unknown>).sort();
  // Check at least first 5 items (or all if fewer).
  const sample = items.slice(0, Math.min(5, items.length));
  for (const item of sample) {
    const keys = Object.keys(item as Record<string, unknown>).sort();
    if (keys.length !== firstKeys.length || keys.some((k, i) => k !== firstKeys[i])) {
      return null;
    }
  }
  return firstKeys;
}

function compressCode(text: string, keepSignatures: boolean): string {
  if (!keepSignatures) return text;
  return compressCodeRegex(text);
}

function compressCodeRegex(text: string): string {
  const signatures: string[] = [];

  for (const pattern of [JS_CLASS_RE, JS_FUNC_RE, JS_METHOD_RE, PY_CLASS_RE, PY_FUNC_RE]) {
    pattern.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(text)) !== null) {
      signatures.push(match[1]!.trimEnd());
    }
  }

  if (signatures.length === 0) return text;

  const resultParts: string[] = [];
  for (const line of text.split("\n")) {
    const stripped = line.trim();
    if (
      stripped.startsWith("import ") ||
      stripped.startsWith("from ") ||
      stripped.startsWith("require(")
    ) {
      resultParts.push(line);
    } else if (
      stripped.startsWith("export ") &&
      !["class ", "function ", "async ", "default "].some((kw) => stripped.includes(kw))
    ) {
      if (!stripped.includes("=")) resultParts.push(line);
    }
  }

  // Detect JS/TS code to emit braces-style signatures.
  const isJs = [JS_CLASS_RE, JS_FUNC_RE, JS_METHOD_RE].some((p) => {
    p.lastIndex = 0;
    return p.test(text);
  });

  const seen = new Set<string>();
  for (const sig of signatures) {
    if (!seen.has(sig)) {
      seen.add(sig);
      resultParts.push(isJs ? `${sig} { ... }` : `${sig} ...`);
    }
  }

  return resultParts.length > 0 ? resultParts.join("\n") : text;
}
