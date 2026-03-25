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

// ── Contextual Metric Intelligence patterns ──────────────────────────

const NUMBER_IN_CONTEXT = /(?<!\w)(\d[\d,]*(?:\.\d+)?)\s+([A-Za-z/][A-Za-z0-9/]{0,15}(?:\s+[A-Za-z]{1,8})?)/g;
const COMPARISON_CONTEXT =
  /(?:\bvs\.?\b|\bversus\b|\bcompared to\b|\bfrom\b.{1,20}\bto\b|\bincreased by\b|\bdecreased by\b|\bdropped\b|\brose\b|\bfell\b|\bhigher\b|\blower\b)/i;

const COMMON_WORDS = new Set([
  "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
  "with", "by", "from", "as", "into", "through", "during", "before", "after",
  "above", "below", "between", "under", "over", "up", "down", "out", "off",
  "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
  "did", "will", "would", "could", "should", "may", "might", "shall", "can",
  "this", "that", "it", "its", "i", "me", "my", "we", "you", "he", "she",
  "they", "them", "their", "our", "your", "his", "her", "who", "which", "what",
  "get", "got", "go", "went", "come", "came", "make", "made", "take", "took",
  "see", "saw", "know", "knew", "think", "thought", "give", "gave", "find",
  "found", "tell", "told", "ask", "asked", "use", "used", "try", "tried",
  "need", "needed", "want", "wanted", "run", "ran", "set", "put", "keep",
  "new", "old", "good", "bad", "great", "small", "large", "big", "long",
  "short", "high", "low", "right", "left", "last", "first", "next", "early",
  "late", "full", "empty", "same", "different", "other", "more", "less",
  "most", "least", "many", "few", "much", "each", "every", "all", "both",
  "such", "own", "only", "just", "also", "very", "too", "quite", "really",
  "things", "items", "people", "days", "times", "ways", "years", "months",
  "weeks", "hours", "minutes", "seconds", "points", "steps", "parts", "types",
  "cases", "lines", "words", "pages", "rows", "columns", "fields", "files",
  "records", "entries", "users", "results", "values", "options", "changes",
  "issues", "errors", "tests", "tasks", "turns", "calls", "attempts",
  "reasons", "examples", "instances", "versions", "levels", "stages",
  "not", "no", "so", "if", "then", "than", "when", "where", "how", "why",
  "here", "there", "now", "still", "already", "yet", "even", "well",
  "about", "like", "some", "any", "per", "total", "main",
  "available", "possible", "specific", "similar", "additional",
]);

// ALL-CAPS entity patterns.
const ALLCAPS_ENTITY_PATTERN = /\b([A-Z][A-Z\s-]{3,}(?:\d+[A-Z]?\b)?)\b/g;
const SINGLE_CAPS_ENTITY_RE = /\b([A-Z]{3,15})\b/g;
const COMMON_ABBREVIATIONS = new Set([
  "THE", "AND", "FOR", "NOT", "BUT", "ARE", "WAS", "HAS", "HAD", "CAN",
  "ALL", "ANY", "FEW", "NEW", "OLD", "USE", "GET", "SET", "RUN", "ADD",
  "API", "URL", "SQL", "CSS", "HTML", "JSON", "HTTP", "HTTPS", "REST",
  "PDF", "CSV", "XML", "SDK", "CLI", "GUI", "IDE", "GIT", "NPM", "PIP",
  "AWS", "GCP", "CPU", "GPU", "RAM", "SSD", "HDD", "USB", "LAN", "WAN",
  "TRUE", "FALSE", "NULL", "NONE", "PASS", "FAIL", "TODO", "NOTE",
  "SYSTEM", "USER", "TOOL", "ERROR", "TRACE", "DEBUG", "INFO", "WARN",
]);

// Large comma-separated numbers.
const LARGE_NUMBER_RE = /\b(\d{1,3}(?:,\d{3})+)\b/g;

// Working memory patterns.
const PARAMETER_PATTERN =
  /\b(?:threshold|limit|cap|minimum|maximum|min|max|target|budget|ceiling|floor|cutoff|baseline|benchmark|setpoint)\s*(?:of|=|:|\bis\b)?\s*(\d[\d,.]*(?:\.\d+)?\s*\S{0,15})/gi;
const CONSTRAINT_PATTERN =
  /((?:must not|do not|don't|should not|shouldn't|cannot|can't|never|only if|exclude|excluding|except|avoid|restrict)\s+.{10,120}?[.!;\n])/gi;
const ASSUMPTION_PATTERN =
  /((?:assum(?:e|ing|ed)|by default|unless (?:specified|otherwise|stated)|we(?:'re| are) treating|for (?:the purposes|simplicity)|taken as|treated as)\s+.{10,120}?[.!;\n])/gi;

// Entity co-occurrence patterns.
const ENTITY_COOCCURRENCE_PATTERNS = [
  /\b([A-Z][A-Za-z\s]{2,30}?)\s+(?:has|shows?|produces?|delivers?|maintains?|outperforms?|exceeds?|averages?)\s+(?:a\s+)?(.{5,60}?)(?:\.|,|;|\n)/gm,
  /\b([A-Z][A-Za-z\s]{2,30}?)\s+(?:is|was|are|were)\s+(?:the\s+)?(.{5,40}?)(?:\.|,|;|\n)/gm,
];

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

  // Extract contextual metrics (number + non-common context word).
  const existingMetrics = new Set(facts.filter((f) => f.content.startsWith("Metric:")).map((f) => f.content));
  NUMBER_IN_CONTEXT.lastIndex = 0;
  while ((match = NUMBER_IN_CONTEXT.exec(msg.content)) !== null) {
    const number = match[1]!;
    const context = match[2]!.trim();
    const contextWord = context.split(/\s+/)[0]?.toLowerCase() ?? "";
    let score = 0;
    if (/[A-Za-z]+\/[A-Za-z]+/.test(context)) score += 0.9;
    if (number.includes(",") || (number.includes(".") && number.split(".")[1]!.length >= 2)) score += 0.4;
    if (contextWord && !COMMON_WORDS.has(contextWord)) score += 0.5;
    if (msg.content.includes("|")) score += 0.3;
    if (score >= 0.5) {
      const matched = `${number} ${context.split(/\s+/)[0]}`.trim();
      const factContent = `Metric: ${matched}`;
      if (!existingMetrics.has(factContent)) {
        facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, factContent, turn, Math.min(score, 1.0)));
        existingMetrics.add(factContent);
      }
    }
  }

  // Extract ALL-CAPS entity names (well names, operators).
  const seenCaps = new Set<string>();
  ALLCAPS_ENTITY_PATTERN.lastIndex = 0;
  while ((match = ALLCAPS_ENTITY_PATTERN.exec(msg.content)) !== null) {
    const entity = match[1]!.trim();
    const words = entity.split(/\s+/);
    const hasMultiple = words.length >= 2;
    const hasDigits = entity.length >= 5 && /\d/.test(entity);
    if ((hasMultiple || hasDigits) && !seenCaps.has(entity) && entity.length <= 50) {
      seenCaps.add(entity);
      facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `Entity: ${entity}`, turn, 0.7));
    }
  }

  // Extract single ALL-CAPS words (operator codes like EOG, FESCO).
  SINGLE_CAPS_ENTITY_RE.lastIndex = 0;
  while ((match = SINGLE_CAPS_ENTITY_RE.exec(msg.content)) !== null) {
    const word = match[1]!;
    if (!COMMON_ABBREVIATIONS.has(word) && !seenCaps.has(word)) {
      seenCaps.add(word);
      facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `Entity: ${word}`, turn, 0.6));
    }
  }

  // Extract large comma-separated numbers (production values).
  LARGE_NUMBER_RE.lastIndex = 0;
  while ((match = LARGE_NUMBER_RE.exec(msg.content)) !== null) {
    const value = match[1]!;
    const factContent = `Metric: ${value}`;
    if (!existingMetrics.has(factContent)) {
      facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, factContent, turn, 0.75));
      existingMetrics.add(factContent);
    }
  }

  // Extract domain-specific terms (medical, scientific).
  DOMAIN_TERM_PATTERN.lastIndex = 0;
  while ((match = DOMAIN_TERM_PATTERN.exec(msg.content)) !== null) {
    const term = match[1]!;
    if (term.length >= 8) {
      facts.push(createAnchorFact(AnchorCategory.IDENTIFIERS, `Term: ${term}`, turn, 0.7));
    }
  }

  // Extract parameters (thresholds, limits, targets).
  PARAMETER_PATTERN.lastIndex = 0;
  while ((match = PARAMETER_PATTERN.exec(msg.content)) !== null) {
    const value = match[1]?.trim().replace(/[.,;]+$/, "") ?? "";
    if (value.length >= 2) {
      facts.push(createAnchorFact(AnchorCategory.PARAMETERS, `Parameter: ${match[0].trim().slice(0, 150)}`, turn, 0.8));
    }
  }

  // Extract constraints (explicit rules).
  CONSTRAINT_PATTERN.lastIndex = 0;
  while ((match = CONSTRAINT_PATTERN.exec(msg.content)) !== null) {
    const constraint = match[1]!.trim().slice(0, 200);
    if (constraint.length >= 15) {
      facts.push(createAnchorFact(AnchorCategory.CONSTRAINTS, constraint, turn, 0.8));
    }
  }

  // Extract assumptions.
  ASSUMPTION_PATTERN.lastIndex = 0;
  while ((match = ASSUMPTION_PATTERN.exec(msg.content)) !== null) {
    const assumption = match[1]!.trim().slice(0, 200);
    if (assumption.length >= 15) {
      facts.push(createAnchorFact(AnchorCategory.ASSUMPTIONS, assumption, turn, 0.75));
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
