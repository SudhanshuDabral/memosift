// Anchor fact extraction — capture critical facts from messages before compression.

import {
  AnchorCategory,
  type AnchorFact,
  type AnchorLedger,
  type ClassifiedMessage,
  ContentType,
  type DependencyMap,
  type MemoSiftMessage,
  createAnchorFact,
  depMapAddLogical,
} from "./types.js";

const FILE_PATH_PATTERN =
  /(?:^|\s|["'])((?:[a-zA-Z]:)?(?:[./\\])?(?:[\w.\-]+[/\\])*[\w.\-]+\.\w{1,10})(?::\d+)?/g;
const ERROR_PATTERN =
  /(?:TypeError|ReferenceError|SyntaxError|ValueError|KeyError|AttributeError|ImportError|RuntimeError|Exception|Error|FAIL):\s*.{10,200}/gm;
const LINE_REF_PATTERN = /[\w./\\]+:\d+/g;
const URL_PATTERN = /https?:\/\/\S+/g;

const CODE_ENTITY_PATTERNS = [
  /\bclass\s+(\w+)/g,
  /\bdef\s+(\w+)/g,
  /\bfunction\s+(\w+)/g,
  /\basync\s+(\w+)\s*\(/g,
  /(?:get|set)\s+(\w+)\s*\(/g,
];

const EDIT_TOOL_NAMES = new Set([
  "edit_file",
  "write_file",
  "create_file",
  "patch_file",
  "Edit",
  "Write",
  "edit",
  "write",
]);

const DECISION_MARKERS = [
  /\bI'll use\b/i,
  /\bLet's go with\b/i,
  /\bchoosing\b.{1,60}\bbecause\b/i,
  /\bchose\b/i,
  /\bdecided to\b/i,
  /\bI'll go with\b/i,
  /\bwe(?:'ll| will) use\b/i,
];

const HEDGING_PATTERN = /\b(?:maybe|perhaps|could consider|might want to|possibly|not sure)\b|\?/i;

const UUID_PATTERN = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi;
const ORDER_ID_PATTERN = /\b[A-Z]{2,4}-\d{4,}\b/g;

// Date pattern — ISO dates and common written date formats.
const DATE_PATTERN =
  /\b(\d{4}-\d{2}-\d{2})\b|\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b/gi;

// Tracking number pattern — alphanumeric sequences typical of shipping carriers.
const TRACKING_PATTERN = /\b([A-Z0-9]{15,30})\b/g;

// Legal statute/section patterns.
const STATUTE_PATTERN =
  /\b(\d+\s+U\.S\.C\.\s*(?:§|Section)\s*\d+(?:\([a-z]\))?)\b|\b(Section\s+\d+(?:\.\d+)*(?:\([a-z]\))?)\b|\b(Article\s+\d+)\b/gi;

// Monetary amounts.
const MONEY_PATTERN = /\$[\d,]+(?:\.\d{2})?/g;

// Percentage values.
const PERCENT_PATTERN = /\b\d+(?:\.\d+)?%/g;

// Domain-specific term pattern — medical/scientific terms (long, specific words).
const DOMAIN_TERM_PATTERN =
  /\b([a-z]{8,}(?:ide|ine|ase|ose|ate|ism|itis|emia|opathy|amine|mycin|cillin|prazole|sartan|statin))\b/gi;

const REASONING_CHAIN_PATTERN =
  /\b(?:therefore|so we can|which means|building on that|given that|as established|as a result|consequently|based on this|following from|this confirms|thus|hence)\b/i;

const EXTRACTABLE_TYPES = new Set([
  ContentType.TOOL_RESULT_TEXT,
  ContentType.TOOL_RESULT_JSON,
  ContentType.ERROR_TRACE,
  ContentType.CODE_BLOCK,
  ContentType.OLD_CONVERSATION,
  ContentType.ASSISTANT_REASONING,
]);

export function extractAnchorsFromMessage(
  msg: MemoSiftMessage,
  turn: number,
  toolName?: string | null,
): AnchorFact[] {
  if (!msg.content) return [];
  const facts: AnchorFact[] = [];
  const isEdit = toolName != null && EDIT_TOOL_NAMES.has(toolName);
  const action = isEdit ? "modified" : "read";

  FILE_PATH_PATTERN.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = FILE_PATH_PATTERN.exec(msg.content)) !== null) {
    facts.push(
      createAnchorFact(AnchorCategory.FILES, `${match[1]} — ${action} at turn ${turn}`, turn),
    );
  }

  ERROR_PATTERN.lastIndex = 0;
  while ((match = ERROR_PATTERN.exec(msg.content)) !== null) {
    facts.push(createAnchorFact(AnchorCategory.ERRORS, match[0].trim(), turn));
  }

  LINE_REF_PATTERN.lastIndex = 0;
  while ((match = LINE_REF_PATTERN.exec(msg.content)) !== null) {
    facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `Reference: ${match[0]}`, turn, 0.8));
  }

  URL_PATTERN.lastIndex = 0;
  while ((match = URL_PATTERN.exec(msg.content)) !== null) {
    const url = match[0].replace(/[.,;)"']+$/, "");
    facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `URL: ${url}`, turn, 0.7));
  }

  for (const pattern of CODE_ENTITY_PATTERNS) {
    pattern.lastIndex = 0;
    while ((match = pattern.exec(msg.content)) !== null) {
      const name = match[1]!;
      if (name.length > 2) {
        facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `Code entity: ${name}`, turn, 0.7));
      }
    }
  }

  // Extract dates.
  DATE_PATTERN.lastIndex = 0;
  while ((match = DATE_PATTERN.exec(msg.content)) !== null) {
    const dateStr = match[1] ?? match[2];
    if (dateStr) {
      facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `Date: ${dateStr}`, turn, 0.8));
    }
  }

  // Extract tracking numbers (15-30 char alphanumeric with mixed digits/letters).
  TRACKING_PATTERN.lastIndex = 0;
  while ((match = TRACKING_PATTERN.exec(msg.content)) !== null) {
    const tracking = match[1]!;
    // Filter out false positives: must contain both digits and letters.
    if (/\d/.test(tracking) && /[A-Za-z]/.test(tracking)) {
      facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `Tracking: ${tracking}`, turn, 0.8));
    }
  }

  // Extract legal statutes/sections.
  STATUTE_PATTERN.lastIndex = 0;
  while ((match = STATUTE_PATTERN.exec(msg.content)) !== null) {
    const statute = match[1] ?? match[2] ?? match[3];
    if (statute) {
      facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `Statute: ${statute}`, turn, 0.85));
    }
  }

  // Extract monetary amounts.
  MONEY_PATTERN.lastIndex = 0;
  while ((match = MONEY_PATTERN.exec(msg.content)) !== null) {
    facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `Amount: ${match[0]}`, turn, 0.8));
  }

  // Extract percentages.
  PERCENT_PATTERN.lastIndex = 0;
  while ((match = PERCENT_PATTERN.exec(msg.content)) !== null) {
    facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `Metric: ${match[0]}`, turn, 0.7));
  }

  // Extract domain-specific terms (medical, scientific).
  DOMAIN_TERM_PATTERN.lastIndex = 0;
  while ((match = DOMAIN_TERM_PATTERN.exec(msg.content)) !== null) {
    const term = match[1]!;
    if (term.length >= 8) {
      facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `Term: ${term}`, turn, 0.7));
    }
  }

  return facts;
}

/** Extract decision facts from assistant text. Filters out hedging. */
function extractDecisionsFromText(text: string, turn: number): AnchorFact[] {
  const facts: AnchorFact[] = [];
  const sentences = text.split(/(?<=[.!])\s+/);
  for (const sentence of sentences) {
    if (HEDGING_PATTERN.test(sentence)) continue;
    for (const marker of DECISION_MARKERS) {
      if (marker.test(sentence)) {
        const content = sentence.trim().slice(0, 200);
        facts.push(createAnchorFact(AnchorCategory.DECISIONS, content, turn, 0.85));
        break;
      }
    }
  }
  return facts;
}

/** Recursively extract facts from a parsed JSON value. */
function extractFactsFromJsonValue(
  value: unknown,
  key: string | null,
  turn: number,
  toolName: string | null,
): AnchorFact[] {
  const facts: AnchorFact[] = [];

  if (typeof value === "string") {
    FILE_PATH_PATTERN.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = FILE_PATH_PATTERN.exec(value)) !== null) {
      const path = m[1]!;
      const action = toolName && EDIT_TOOL_NAMES.has(toolName) ? "modified" : "referenced";
      const context = key ? ` (key=${key})` : "";
      facts.push(
        createAnchorFact(
          AnchorCategory.FILES,
          `${path} — ${action} at turn ${turn}${context}`,
          turn,
        ),
      );
    }
    URL_PATTERN.lastIndex = 0;
    while ((m = URL_PATTERN.exec(value)) !== null) {
      const url = m[0].replace(/[.,;)"']+$/, "");
      facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `URL: ${url}`, turn, 0.7));
    }
    UUID_PATTERN.lastIndex = 0;
    while ((m = UUID_PATTERN.exec(value)) !== null) {
      facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `UUID: ${m[0]}`, turn, 0.8));
    }
    ORDER_ID_PATTERN.lastIndex = 0;
    while ((m = ORDER_ID_PATTERN.exec(value)) !== null) {
      facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `ID: ${m[0]}`, turn, 0.8));
    }
  } else if (Array.isArray(value)) {
    for (const item of value) {
      facts.push(...extractFactsFromJsonValue(item, key, turn, toolName));
    }
  } else if (value !== null && typeof value === "object") {
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      facts.push(...extractFactsFromJsonValue(v, k, turn, toolName));
    }
  }

  return facts;
}

export function extractAnchorsFromSegments(
  segments: ClassifiedMessage[],
  ledger: AnchorLedger,
): void {
  const userIndices = segments
    .filter((s) => s.message.role === "user")
    .map((s) => s.originalIndex)
    .sort((a, b) => a - b);

  function turnForIndex(idx: number): number {
    let count = 0;
    for (const ui of userIndices) {
      if (ui <= idx) count++;
      else break;
    }
    return Math.max(count, 1);
  }

  // ── Extract INTENT from first user message ──
  const firstUser = segments.find(
    (s) => s.message.role === "user" && !s.message._memosiftCompressed,
  );
  if (firstUser) {
    const intentText = firstUser.message.content.trim().slice(0, 300);
    if (intentText) {
      ledger.add(createAnchorFact(AnchorCategory.INTENT, intentText, 1, 0.9));
    }
  }

  // ── Extract ACTIVE_CONTEXT from last user + last assistant ──
  let lastUser: ClassifiedMessage | null = null;
  let lastAssistant: ClassifiedMessage | null = null;
  for (let i = segments.length - 1; i >= 0; i--) {
    const seg = segments[i]!;
    if (seg.message._memosiftCompressed) continue;
    if (seg.message.role === "user" && !lastUser) lastUser = seg;
    else if (seg.message.role === "assistant" && !lastAssistant) lastAssistant = seg;
    if (lastUser && lastAssistant) break;
  }

  if (lastUser) {
    const turn = turnForIndex(lastUser.originalIndex);
    const text = lastUser.message.content.trim().slice(0, 300);
    if (text)
      ledger.add(
        createAnchorFact(AnchorCategory.ACTIVE_CONTEXT, `Current task: ${text}`, turn, 0.9),
      );
  }
  if (lastAssistant) {
    const turn = turnForIndex(lastAssistant.originalIndex);
    const text = lastAssistant.message.content.trim().slice(0, 300);
    if (text)
      ledger.add(
        createAnchorFact(AnchorCategory.ACTIVE_CONTEXT, `Last response: ${text}`, turn, 0.8),
      );
  }

  // ── Extract facts from all segments ──
  for (const seg of segments) {
    if (seg.message._memosiftCompressed) continue;
    const turn = turnForIndex(seg.originalIndex);

    // Extract DECISIONS from assistant messages.
    if (seg.message.role === "assistant" && seg.message.content) {
      const decisionFacts = extractDecisionsFromText(seg.message.content, turn);
      for (const fact of decisionFacts) ledger.add(fact);
    }

    // Extract from tool_call arguments — parse as JSON.
    if (seg.message.toolCalls) {
      for (const tc of seg.message.toolCalls) {
        if (tc.function.name) {
          ledger.add(
            createAnchorFact(
              AnchorCategory.IDENTIFIERS,
              `Tool used: ${tc.function.name}`,
              turn,
              0.9,
            ),
          );
        }

        const argsStr = tc.function.arguments;
        try {
          const parsed = JSON.parse(argsStr) as unknown;
          const jsonFacts = extractFactsFromJsonValue(parsed, null, turn, tc.function.name);
          for (const fact of jsonFacts) ledger.add(fact);
        } catch {
          // Fallback: regex extraction from raw string.
          FILE_PATH_PATTERN.lastIndex = 0;
          let argMatch: RegExpExecArray | null;
          while ((argMatch = FILE_PATH_PATTERN.exec(argsStr)) !== null) {
            const path = argMatch[1]!;
            const action = EDIT_TOOL_NAMES.has(tc.function.name) ? "modified" : "referenced";
            ledger.add(
              createAnchorFact(AnchorCategory.FILES, `${path} — ${action} at turn ${turn}`, turn),
            );
          }
        }
      }
    }

    if (!EXTRACTABLE_TYPES.has(seg.contentType)) continue;
    const facts = extractAnchorsFromMessage(seg.message, turn, seg.message.name);
    for (const fact of facts) ledger.add(fact);
  }
}

/** Detect logical reasoning dependencies and add edges to DependencyMap. */
export function extractReasoningChains(segments: ClassifiedMessage[], deps: DependencyMap): void {
  const assistantIndices: number[] = [];
  for (const seg of segments) {
    if (seg.message.role === "assistant" && !seg.message._memosiftCompressed) {
      assistantIndices.push(seg.originalIndex);
    }
  }

  for (const seg of segments) {
    if (seg.message._memosiftCompressed || !seg.message.content) continue;
    if (!REASONING_CHAIN_PATTERN.test(seg.message.content)) continue;

    let prior: number | null = null;
    for (let i = assistantIndices.length - 1; i >= 0; i--) {
      if (assistantIndices[i]! < seg.originalIndex) {
        prior = assistantIndices[i]!;
        break;
      }
    }

    if (prior !== null && prior !== seg.originalIndex) {
      depMapAddLogical(deps, seg.originalIndex, prior);
    }
  }
}
