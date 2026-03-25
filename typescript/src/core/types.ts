// Core type definitions for the MemoSift compression pipeline.

import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";

export enum ContentType {
  SYSTEM_PROMPT = "SYSTEM_PROMPT",
  USER_QUERY = "USER_QUERY",
  RECENT_TURN = "RECENT_TURN",
  TOOL_RESULT_JSON = "TOOL_RESULT_JSON",
  TOOL_RESULT_TEXT = "TOOL_RESULT_TEXT",
  CODE_BLOCK = "CODE_BLOCK",
  ERROR_TRACE = "ERROR_TRACE",
  ASSISTANT_REASONING = "ASSISTANT_REASONING",
  OLD_CONVERSATION = "OLD_CONVERSATION",
  PREVIOUSLY_COMPRESSED = "PREVIOUSLY_COMPRESSED",
}

export enum CompressionPolicy {
  PRESERVE = "PRESERVE",
  LIGHT = "LIGHT",
  MODERATE = "MODERATE",
  STRUCTURAL = "STRUCTURAL",
  STACK = "STACK",
  AGGRESSIVE = "AGGRESSIVE",
  SIGNATURE = "SIGNATURE",
}

export enum Shield {
  PRESERVE = "PRESERVE",
  MODERATE = "MODERATE",
  COMPRESSIBLE = "COMPRESSIBLE",
}

export const DEFAULT_POLICIES: Record<ContentType, CompressionPolicy> = {
  [ContentType.SYSTEM_PROMPT]: CompressionPolicy.PRESERVE,
  [ContentType.USER_QUERY]: CompressionPolicy.PRESERVE,
  [ContentType.RECENT_TURN]: CompressionPolicy.LIGHT,
  [ContentType.TOOL_RESULT_JSON]: CompressionPolicy.STRUCTURAL,
  [ContentType.TOOL_RESULT_TEXT]: CompressionPolicy.MODERATE,
  [ContentType.CODE_BLOCK]: CompressionPolicy.SIGNATURE,
  [ContentType.ERROR_TRACE]: CompressionPolicy.STACK,
  [ContentType.ASSISTANT_REASONING]: CompressionPolicy.AGGRESSIVE,
  [ContentType.OLD_CONVERSATION]: CompressionPolicy.AGGRESSIVE,
  [ContentType.PREVIOUSLY_COMPRESSED]: CompressionPolicy.PRESERVE,
};

export interface ToolCallFunction {
  name: string;
  arguments: string;
}

export interface ToolCall {
  id: string;
  type: string;
  function: ToolCallFunction;
}

export interface MemoSiftMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  name?: string | null;
  toolCallId?: string | null;
  toolCalls?: ToolCall[] | null;
  metadata: Record<string, unknown>;
  _memosiftContentType?: string | null;
  _memosiftCompressed: boolean;
  _memosiftOriginalTokens?: number | null;
}

export function createMessage(
  role: MemoSiftMessage["role"],
  content: string,
  options?: Partial<Omit<MemoSiftMessage, "role" | "content">>,
): MemoSiftMessage {
  return {
    role,
    content,
    name: options?.name ?? null,
    toolCallId: options?.toolCallId ?? null,
    toolCalls: options?.toolCalls ?? null,
    metadata: options?.metadata ?? {},
    _memosiftContentType: options?._memosiftContentType ?? null,
    _memosiftCompressed: options?._memosiftCompressed ?? false,
    _memosiftOriginalTokens: options?._memosiftOriginalTokens ?? null,
  };
}

export interface ClassifiedMessage {
  message: MemoSiftMessage;
  contentType: ContentType;
  policy: CompressionPolicy;
  originalIndex: number;
  relevanceScore: number;
  estimatedTokens: number;
  protected: boolean;
  importanceScore: number;
  shield: Shield;
}

export function createClassified(
  message: MemoSiftMessage,
  contentType: ContentType,
  policy: CompressionPolicy,
  options?: Partial<
    Pick<
      ClassifiedMessage,
      | "originalIndex"
      | "relevanceScore"
      | "estimatedTokens"
      | "protected"
      | "importanceScore"
      | "shield"
    >
  >,
): ClassifiedMessage {
  return {
    message,
    contentType,
    policy,
    originalIndex: options?.originalIndex ?? 0,
    relevanceScore: options?.relevanceScore ?? 0.5,
    estimatedTokens: options?.estimatedTokens ?? 0,
    protected: options?.protected ?? false,
    importanceScore: options?.importanceScore ?? 0.5,
    shield: options?.shield ?? Shield.MODERATE,
  };
}

export interface DependencyMap {
  references: Map<number, number>;
  logicalDeps: Map<number, number>;
}

export function createDependencyMap(): DependencyMap {
  return { references: new Map(), logicalDeps: new Map() };
}

export function depMapAdd(dm: DependencyMap, dedupedIndex: number, originalIndex: number): void {
  dm.references.set(dedupedIndex, originalIndex);
}

export function depMapAddLogical(
  dm: DependencyMap,
  dependentIndex: number,
  dependencyIndex: number,
): void {
  dm.logicalDeps.set(dependentIndex, dependencyIndex);
}

export function depMapCanDrop(dm: DependencyMap, index: number): boolean {
  for (const v of dm.references.values()) {
    if (v === index) return false;
  }
  for (const v of dm.logicalDeps.values()) {
    if (v === index) return false;
  }
  return true;
}

export function depMapHasDependents(dm: DependencyMap, index: number): boolean {
  for (const v of dm.references.values()) {
    if (v === index) return true;
  }
  return false;
}

export function depMapHasLogicalDependents(dm: DependencyMap, index: number): boolean {
  for (const v of dm.logicalDeps.values()) {
    if (v === index) return true;
  }
  return false;
}

export function depMapDependentsOf(dm: DependencyMap, index: number): number[] {
  const result: number[] = [];
  for (const [k, v] of dm.references.entries()) {
    if (v === index) result.push(k);
  }
  return result;
}

// ── Anchor Ledger types ────────────────────────────────────────────────────

export enum AnchorCategory {
  INTENT = "INTENT",
  FILES = "FILES",
  DECISIONS = "DECISIONS",
  ERRORS = "ERRORS",
  ACTIVE_CONTEXT = "ACTIVE_CONTEXT",
  IDENTIFIERS = "IDENTIFIERS",
  OUTCOMES = "OUTCOMES",
  OPEN_ITEMS = "OPEN_ITEMS",
  PARAMETERS = "PARAMETERS",
  CONSTRAINTS = "CONSTRAINTS",
  ASSUMPTIONS = "ASSUMPTIONS",
  DATA_SCHEMA = "DATA_SCHEMA",
  RELATIONSHIPS = "RELATIONSHIPS",
}

const LEDGER_SECTION_HEADERS: Record<string, string> = {
  [AnchorCategory.INTENT]: "## SESSION INTENT",
  [AnchorCategory.FILES]: "## FILES TOUCHED",
  [AnchorCategory.DECISIONS]: "## KEY DECISIONS",
  [AnchorCategory.ERRORS]: "## ERRORS ENCOUNTERED",
  [AnchorCategory.ACTIVE_CONTEXT]: "## ACTIVE CONTEXT",
  [AnchorCategory.IDENTIFIERS]: "## IDENTIFIERS",
  [AnchorCategory.OUTCOMES]: "## OUTCOMES",
  [AnchorCategory.OPEN_ITEMS]: "## OPEN ITEMS",
  [AnchorCategory.PARAMETERS]: "## PARAMETERS",
  [AnchorCategory.CONSTRAINTS]: "## CONSTRAINTS",
  [AnchorCategory.ASSUMPTIONS]: "## ASSUMPTIONS",
  [AnchorCategory.DATA_SCHEMA]: "## DATA SCHEMA",
  [AnchorCategory.RELATIONSHIPS]: "## RELATIONSHIPS",
};

const LEDGER_PRIMARY_SECTIONS: AnchorCategory[] = [
  AnchorCategory.INTENT,
  AnchorCategory.FILES,
  AnchorCategory.DECISIONS,
  AnchorCategory.ERRORS,
  AnchorCategory.ACTIVE_CONTEXT,
];

export interface AnchorFact {
  readonly category: AnchorCategory;
  readonly content: string;
  readonly turn: number;
  readonly confidence: number;
}

export function createAnchorFact(
  category: AnchorCategory,
  content: string,
  turn: number,
  confidence = 1.0,
): AnchorFact {
  return { category, content, turn, confidence };
}

export interface CrossWindowState {
  contentHashes: Set<string>;
}

export function createCrossWindowState(): CrossWindowState {
  return { contentHashes: new Set() };
}

export class AnchorLedger {
  facts: AnchorFact[] = [];
  private seenHashes: Set<string> = new Set();
  private criticalCache: ReadonlySet<string> | null = null;

  add(fact: AnchorFact): boolean {
    const hash = createHash("sha256").update(fact.content).digest("hex").slice(0, 32);
    if (this.seenHashes.has(hash)) return false;
    this.seenHashes.add(hash);
    this.facts.push(fact);
    this.criticalCache = null;
    return true;
  }

  update(category: AnchorCategory, oldContent: string, newContent: string): void {
    const idx = this.facts.findIndex((f) => f.category === category && f.content === oldContent);
    if (idx === -1) return;
    const old = this.facts[idx]!;
    this.facts[idx] = createAnchorFact(old.category, newContent, old.turn, old.confidence);
    const oldHash = createHash("sha256").update(oldContent).digest("hex").slice(0, 32);
    const newHash = createHash("sha256").update(newContent).digest("hex").slice(0, 32);
    this.seenHashes.delete(oldHash);
    this.seenHashes.add(newHash);
    this.criticalCache = null;
  }

  /** Return the set of core identifiers from all facts. */
  getProtectedStrings(): ReadonlySet<string> {
    const strings = new Set<string>();
    for (const fact of this.facts) {
      let core = (fact.content.split(" \u2014 ")[0] ?? fact.content).split(" — ")[0]!.trim();
      if (fact.category === AnchorCategory.IDENTIFIERS) {
        for (const prefix of ["Tool used: ", "Code entity: ", "Reference: "]) {
          if (core.startsWith(prefix)) {
            core = core.slice(prefix.length);
            break;
          }
        }
      }
      if (core.length >= 3) strings.add(core);
      if (fact.category === AnchorCategory.FILES && (core.includes("/") || core.includes("\\"))) {
        const filename = core.replaceAll("\\", "/").replace(/\/$/, "").split("/").pop() ?? "";
        if (filename.length >= 3) strings.add(filename);
      }
    }
    return strings;
  }

  /** Prefixes for high-value identifier facts included in critical strings. */
  private static readonly HIGH_VALUE_PREFIXES: readonly string[] = [
    "Tracking:",
    "Date:",
    "Statute:",
    "Amount:",
    "Metric:",
    "ID:",
    "UUID:",
    "URL:",
  ];

  /** Return protected strings from FILES, ERRORS, and high-value IDENTIFIERS.
   * Strict filtering to avoid broad matches that kill compression:
   * - FILES: Only paths containing a directory separator AND a file extension.
   * - ERRORS: Only error messages >= 10 chars.
   * - IDENTIFIERS: Only high-value facts (Tracking, Date, Statute, Amount,
   *   Metric, ID, UUID, URL) — not Code entities or tool names which are too broad. */
  getCriticalStrings(): ReadonlySet<string> {
    if (this.criticalCache) return this.criticalCache;
    const strings = new Set<string>();
    for (const fact of this.facts) {
      const core = (fact.content.split(" \u2014 ")[0] ?? fact.content).split(" — ")[0]!.trim();
      if (fact.category === AnchorCategory.FILES) {
        if (!core.includes("/") && !core.includes("\\")) continue;
        const parts = core.replaceAll("\\", "/").replace(/\/$/, "").split("/");
        const filename = parts[parts.length - 1] ?? "";
        if (!filename.includes(".") || filename.length < 3) continue;
        const namePart = filename.split(".").slice(0, -1).join(".");
        if (/^[\d.\-_]+$/.test(namePart)) continue;
        if (core.length >= 8) strings.add(core);
        if (filename.length >= 5) strings.add(filename);
      } else if (fact.category === AnchorCategory.ERRORS) {
        if (core.length >= 10) strings.add(core);
      } else if (fact.category === AnchorCategory.IDENTIFIERS) {
        // Only include high-value identifiers, not broad ones.
        for (const prefix of AnchorLedger.HIGH_VALUE_PREFIXES) {
          if (core.startsWith(prefix)) {
            const value = core.slice(prefix.length).trim();
            if (value.length >= 3) {
              strings.add(value);
            }
            break;
          }
        }
      }
    }
    this.criticalCache = strings;
    return strings;
  }

  /** Return true if any critical fact (FILES or ERRORS) appears in text. */
  containsAnchorFact(text: string): boolean {
    if (this.facts.length === 0) return false;
    const critical = this.getCriticalStrings();
    if (critical.size === 0) return false;
    const lower = text.toLowerCase();
    for (const s of critical) {
      if (lower.includes(s.toLowerCase())) return true;
    }
    return false;
  }

  render(): string {
    const lines: string[] = ["[SESSION MEMORY — preserved across compressions]", ""];
    const allCategories = [
      ...LEDGER_PRIMARY_SECTIONS,
      ...Object.values(AnchorCategory).filter((c) => !LEDGER_PRIMARY_SECTIONS.includes(c)),
    ];
    for (const category of allCategories) {
      const categoryFacts = this.facts.filter((f) => f.category === category);
      if (categoryFacts.length > 0) {
        const header = LEDGER_SECTION_HEADERS[category] ?? `## ${category}`;
        lines.push(header);
        for (const fact of categoryFacts) {
          lines.push(`- ${fact.content}`);
        }
        lines.push("");
      }
    }
    return lines.join("\n");
  }

  tokenEstimate(): number {
    return Math.floor(this.render().length / 4);
  }

  workingMemorySummary(): string {
    const warmCategories = [
      AnchorCategory.OUTCOMES,
      AnchorCategory.DECISIONS,
      AnchorCategory.PARAMETERS,
      AnchorCategory.CONSTRAINTS,
      AnchorCategory.RELATIONSHIPS,
    ];
    const lines: string[] = [];
    for (const category of warmCategories) {
      const categoryFacts = this.facts.filter((f) => f.category === category);
      if (categoryFacts.length > 0) {
        const header = LEDGER_SECTION_HEADERS[category] ?? `## ${category}`;
        lines.push(header);
        for (const fact of categoryFacts.slice(0, 10)) {
          lines.push(`- ${fact.content}`);
        }
      }
    }
    if (lines.length === 0) return "";
    return "[WORKING MEMORY]\n" + lines.join("\n");
  }

  factsByCategory(category: AnchorCategory): AnchorFact[] {
    return this.facts.filter((f) => f.category === category);
  }

  save(path: string): void {
    const data = {
      facts: this.facts.map((f) => ({
        category: f.category,
        content: f.content,
        turn: f.turn,
        confidence: f.confidence,
      })),
    };
    writeFileSync(path, JSON.stringify(data, null, 2), "utf-8");
  }

  static load(path: string): AnchorLedger {
    if (!existsSync(path)) return new AnchorLedger();
    const data = JSON.parse(readFileSync(path, "utf-8")) as {
      facts: { category: string; content: string; turn: number; confidence?: number }[];
    };
    const ledger = new AnchorLedger();
    for (const item of data.facts) {
      ledger.add(
        createAnchorFact(
          item.category as AnchorCategory,
          item.content,
          item.turn,
          item.confidence ?? 1.0,
        ),
      );
    }
    return ledger;
  }
}

// ── CrossWindowState persistence ─────────────────────────────────────────────

export function saveCrossWindowState(state: CrossWindowState, path: string): void {
  const data = { contentHashes: [...state.contentHashes].sort() };
  writeFileSync(path, JSON.stringify(data), "utf-8");
}

export function loadCrossWindowState(path: string): CrossWindowState {
  if (!existsSync(path)) return createCrossWindowState();
  const data = JSON.parse(readFileSync(path, "utf-8"));
  return { contentHashes: new Set(data.contentHashes ?? []) };
}

// ── ClassifiedMessage content helper ─────────────────────────────────────────

export function getContent(seg: ClassifiedMessage): string {
  return seg.message.content || "";
}
