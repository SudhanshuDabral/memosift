// Engine D: Abstractive summarization — LLM-dependent, opt-in only.

import type { MemoSiftConfig } from "../../config.js";
import type { MemoSiftLLMProvider } from "../../providers/base.js";
import { type ClassifiedMessage, CompressionPolicy, createMessage } from "../types.js";

const TARGET_POLICIES = new Set([CompressionPolicy.AGGRESSIVE]);

const SUMMARIZE_PROMPT = `Summarize the following conversation segment concisely.

PRESERVE exactly (do not paraphrase):
- All file paths (e.g., src/auth.ts, ./config/db.json)
- All line numbers (e.g., line 47, auth.ts:47)
- All error messages and types (e.g., TypeError: Cannot read...)
- All decisions and their rationale (e.g., "chose X because Y")
- All unresolved issues or open items
- All specific numeric values (ports, status codes, counts)
- All function/class/variable names

REMOVE:
- Conversational filler ("sure", "let me", "I'll", "okay")
- Redundant restatements of the same information
- Intermediate reasoning that led to a stated conclusion
- Tool invocation metadata (tool names, call IDs)

Output a concise summary preserving all critical facts.

SEGMENT:
{content}`;

const FILE_PATH_RE = /(?:[\w.\-]+[/\\])+[\w.\-]+\.\w{1,10}(?::\d+)?/g;
const ERROR_MSG_RE =
  /(?:TypeError|ReferenceError|SyntaxError|ValueError|KeyError|AttributeError|ImportError|RuntimeError|Error):\s*.{10,100}/g;

interface CriticalFacts {
  filePaths: Set<string>;
  errorMsgs: Set<string>;
}

function extractCriticalFacts(text: string): CriticalFacts {
  return {
    filePaths: new Set(text.match(FILE_PATH_RE) ?? []),
    errorMsgs: new Set(text.match(ERROR_MSG_RE) ?? []),
  };
}

function isValidSummary(summary: string, original: string, facts: CriticalFacts): boolean {
  if (summary.length <= 20) return false;
  if (summary.length >= original.length) return false;

  for (const path of facts.filePaths) {
    if (!summary.includes(path)) return false;
  }

  for (const error of facts.errorMsgs) {
    const errorType = error.split(":")[0]?.trim() ?? "";
    if (errorType && !summary.includes(errorType)) return false;
  }

  return true;
}

export async function summarizeSegments(
  segments: ClassifiedMessage[],
  config: MemoSiftConfig,
  llm: MemoSiftLLMProvider,
): Promise<ClassifiedMessage[]> {
  if (!config.enableSummarization) return segments;

  const skipMap = new Map<number, ClassifiedMessage>();
  const summarizable: [number, ClassifiedMessage][] = [];

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i]!;
    if (!TARGET_POLICIES.has(seg.policy) || seg.message.content.length < 200) {
      skipMap.set(i, seg);
    } else {
      summarizable.push([i, seg]);
    }
  }

  if (summarizable.length === 0) return segments;

  const results = await Promise.all(
    summarizable.map(async ([idx, seg]): Promise<[number, ClassifiedMessage]> => {
      try {
        const facts = extractCriticalFacts(seg.message.content);
        const prompt = SUMMARIZE_PROMPT.replace("{content}", seg.message.content);
        const maxTokens = Math.max(256, Math.ceil(seg.message.content.length / 3.5 / 2));
        const response = await llm.generate(prompt, { maxTokens, temperature: 0 });
        const summary = response.text.trim();

        if (!isValidSummary(summary, seg.message.content, facts)) {
          return [idx, seg];
        }

        const newMsg = createMessage(seg.message.role, summary, {
          name: seg.message.name,
          toolCallId: seg.message.toolCallId,
          toolCalls: seg.message.toolCalls,
          metadata: seg.message.metadata,
        });
        return [idx, { ...seg, message: newMsg }];
      } catch {
        return [idx, seg];
      }
    }),
  );

  const allSegs = new Map(skipMap);
  for (const [i, seg] of results) {
    allSegs.set(i, seg);
  }

  return [...allSegs.keys()].sort((a, b) => a - b).map((k) => allSegs.get(k)!);
}
