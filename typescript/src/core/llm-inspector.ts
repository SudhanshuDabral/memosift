// LLM Inspector — post-compression quality feedback via three parallel LLM jobs.
//
// Runs AFTER compression completes, asynchronously. Does not slow down the
// compression pipeline. Produces project-specific protection rules that the
// deterministic engines read on the next session.
//
// Three independent jobs (run in parallel):
//   1. Entity Guardian: identifies entity names lost during compression
//   2. Fact Auditor: evaluates whether compressed context is actionable
//   3. Config Advisor: recommends parameter adjustments from compression history

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import type { MemoSiftLLMProvider } from "../providers/base.js";
import type { CompressionReport } from "../report.js";
import type { AnchorLedger, MemoSiftMessage } from "./types.js";

// ── Project Memory ───────────────────────────────────────────────────────

/**
 * Persistent, project-specific protection rules learned from LLM feedback.
 *
 * Read by the deterministic engines at session start. Written by the LLM
 * inspector after each compression cycle. Accumulates over time.
 */
export class ProjectMemory {
  /** Entity names (wells, operators, people, projects) that must survive. */
  protectedEntities: string[];

  /** File path prefixes that should always be preserved. */
  protectedPathPrefixes: string[];

  /** Domain-specific metric unit patterns (auto-detected). */
  domainPatterns: string[];

  /** Parameter recommendations from the Config Advisor. */
  learnedConfig: Record<string, unknown>;

  /** Historical fact audit scores (last 20 sessions). */
  auditScores: Record<string, unknown>[];

  /** Number of sessions analyzed so far. */
  sessionsAnalyzed: number;

  constructor(options?: {
    protectedEntities?: string[];
    protectedPathPrefixes?: string[];
    domainPatterns?: string[];
    learnedConfig?: Record<string, unknown>;
    auditScores?: Record<string, unknown>[];
    sessionsAnalyzed?: number;
  }) {
    this.protectedEntities = options?.protectedEntities ?? [];
    this.protectedPathPrefixes = options?.protectedPathPrefixes ?? [];
    this.domainPatterns = options?.domainPatterns ?? [];
    this.learnedConfig = options?.learnedConfig ?? {};
    this.auditScores = options?.auditScores ?? [];
    this.sessionsAnalyzed = options?.sessionsAnalyzed ?? 0;
  }

  /** Persist to JSON file. */
  save(path: string): void {
    const data = {
      protectedEntities: this.protectedEntities,
      protectedPathPrefixes: this.protectedPathPrefixes,
      domainPatterns: this.domainPatterns,
      learnedConfig: this.learnedConfig,
      auditScores: this.auditScores,
      sessionsAnalyzed: this.sessionsAnalyzed,
    };
    writeFileSync(path, JSON.stringify(data, null, 2), "utf-8");
  }

  /** Load from JSON file. Returns empty memory if file doesn't exist. */
  static load(path: string): ProjectMemory {
    if (!existsSync(path)) return new ProjectMemory();
    try {
      const raw = readFileSync(path, "utf-8");
      const data = JSON.parse(raw) as Record<string, unknown>;
      const auditScores = Array.isArray(data["auditScores"])
        ? (data["auditScores"] as Record<string, unknown>[]).slice(-20)
        : [];
      return new ProjectMemory({
        protectedEntities: Array.isArray(data["protectedEntities"])
          ? (data["protectedEntities"] as string[])
          : [],
        protectedPathPrefixes: Array.isArray(data["protectedPathPrefixes"])
          ? (data["protectedPathPrefixes"] as string[])
          : [],
        domainPatterns: Array.isArray(data["domainPatterns"])
          ? (data["domainPatterns"] as string[])
          : [],
        learnedConfig:
          typeof data["learnedConfig"] === "object" && data["learnedConfig"] !== null
            ? (data["learnedConfig"] as Record<string, unknown>)
            : {},
        auditScores,
        sessionsAnalyzed: typeof data["sessionsAnalyzed"] === "number"
          ? (data["sessionsAnalyzed"] as number)
          : 0,
      });
    } catch {
      return new ProjectMemory();
    }
  }

  /** Return all learned protection strings for the anchor extractor. */
  getProtectionStrings(): ReadonlySet<string> {
    const strings = new Set<string>();
    for (const entity of this.protectedEntities) {
      strings.add(entity);
    }
    for (const prefix of this.protectedPathPrefixes) {
      strings.add(prefix);
    }
    return strings;
  }
}

// ── LLM Prompts ──────────────────────────────────────────────────────────

const ENTITY_GUARDIAN_PROMPT = `\
You are analyzing a compression result. Compare the ORIGINAL conversation \
with the COMPRESSED version to find important entity names that were LOST.

ORIGINAL (first 50 messages, truncated per message):
{original_sample}

COMPRESSED (all messages):
{compressed_sample}

ANCHOR LEDGER (preserved facts):
{anchor_ledger}

Find entity names (people, companies, well names, project names, locations, \
tool names) that appear in the ORIGINAL but are MISSING from both the \
COMPRESSED version AND the anchor ledger.

Return ONLY a JSON array of lost entity strings. No explanation needed.
Example: ["WHITLEY-DUBOSE UNIT 1H", "Frio County", "EOG Resources"]

If nothing important was lost, return: []`;

const FACT_AUDITOR_PROMPT = `\
You are auditing a compressed AI conversation for quality. The compressed \
version will be used as context for the AI agent's next response.

COMPRESSED CONTEXT:
{compressed_sample}

ANCHOR LEDGER:
{anchor_ledger}

ORIGINAL TASK: {task}

Score each dimension 1-5:
1. **completeness**: Can the agent answer follow-up questions about the work done?
2. **numerical_integrity**: Are key numbers, rates, percentages preserved?
3. **entity_coverage**: Are important names (files, people, projects) present?
4. **tool_continuity**: Can the agent tell what tools were called and what they produced?
5. **actionability**: Could the agent continue the task without re-reading sources?

Return JSON only:
{{"completeness": N, "numerical_integrity": N, "entity_coverage": N, \
"tool_continuity": N, "actionability": N, \
"missing_critical": ["list of facts that should have survived"]}}`;

const CONFIG_ADVISOR_PROMPT = `\
You are analyzing compression performance history to recommend parameter changes.

RECENT AUDIT SCORES (last sessions):
{audit_history}

CURRENT CONFIG:
{current_config}

PATTERNS OBSERVED:
- Lost entities: {lost_entities}
- Compression ratio: {compression_ratio}
- Fact retention: {retention_pct}%

Based on the patterns, recommend parameter adjustments. Only suggest changes \
if there is clear evidence of a recurring problem.

Available parameters:
- entropy_threshold (1.5-2.5): lower = delete more boilerplate lines
- token_prune_keep_ratio (0.3-0.7): lower = prune more aggressively
- dedup_similarity_threshold (0.75-0.95): lower = catch more fuzzy duplicates
- json_array_threshold (2-15): higher = keep more JSON array items
- error_trace_policy ("PRESERVE" or "STACK"): PRESERVE = keep full traces

Return JSON only:
{{"recommendations": {{"param_name": value}}, \
"reasoning": "1-2 sentence explanation"}}

If no changes needed, return:
{{"recommendations": {{}}, "reasoning": "Current config is optimal"}}`;

// ── JSON extraction ──────────────────────────────────────────────────────

/**
 * Extract the first JSON object or array from text.
 *
 * Handles LLM responses that include explanatory text after the JSON.
 * Tries full parse first, then scans for the first `{` or `[` and finds
 * its matching closing brace/bracket.
 */
export function extractJson(text: string): Record<string, unknown> | unknown[] | null {
  const trimmed = text.trim();
  try {
    return JSON.parse(trimmed) as Record<string, unknown> | unknown[];
  } catch {
    // Fall through to scanning approach.
  }

  const pairs: [string, string][] = [
    ["{", "}"],
    ["[", "]"],
  ];

  for (const [startChar, endChar] of pairs) {
    const start = trimmed.indexOf(startChar);
    if (start === -1) continue;

    let depth = 0;
    let inString = false;
    let escape = false;

    for (let i = start; i < trimmed.length; i++) {
      const ch = trimmed[i]!;
      if (escape) {
        escape = false;
        continue;
      }
      if (ch === "\\") {
        escape = true;
        continue;
      }
      if (ch === '"') {
        inString = !inString;
        continue;
      }
      if (inString) continue;
      if (ch === startChar) depth++;
      else if (ch === endChar) {
        depth--;
        if (depth === 0) {
          try {
            return JSON.parse(trimmed.slice(start, i + 1)) as
              | Record<string, unknown>
              | unknown[];
          } catch {
            break;
          }
        }
      }
    }
  }

  return null;
}

// ── Inspector Jobs ───────────────────────────────────────────────────────

/** Job 1: Identify entity names lost during compression. */
async function runEntityGuardian(
  original: readonly MemoSiftMessage[],
  compressed: readonly MemoSiftMessage[],
  ledger: AnchorLedger,
  llm: MemoSiftLLMProvider,
): Promise<string[]> {
  const originalSample = original
    .slice(0, 50)
    .map((m) => `[${m.role}]: ${(m.content ?? "").slice(0, 300)}`)
    .join("\n");
  const compressedSample = compressed
    .map((m) => `[${m.role}]: ${(m.content ?? "").slice(0, 300)}`)
    .join("\n");
  const ledgerText = ledger.render().slice(0, 2000);

  const prompt = ENTITY_GUARDIAN_PROMPT.replace("{original_sample}", originalSample)
    .replace("{compressed_sample}", compressedSample)
    .replace("{anchor_ledger}", ledgerText);

  try {
    const response = await llm.generate(prompt, { maxTokens: 512, temperature: 0.0 });
    let text = response.text.trim();
    if (text.startsWith("```")) {
      const lines = text.split("\n");
      text = lines.filter((x) => !x.trim().startsWith("```")).join("\n");
    }
    const entities = extractJson(text);
    if (Array.isArray(entities)) {
      return entities.filter(
        (e): e is string => typeof e === "string" && e.length >= 2,
      );
    }
  } catch {
    // Entity Guardian failed — return empty.
  }
  return [];
}

/** Job 2: Evaluate compressed context quality. */
async function runFactAuditor(
  compressed: readonly MemoSiftMessage[],
  ledger: AnchorLedger,
  task: string | null,
  llm: MemoSiftLLMProvider,
): Promise<Record<string, unknown> | null> {
  const compressedSample = compressed
    .map((m) => `[${m.role}]: ${(m.content ?? "").slice(0, 400)}`)
    .join("\n");
  const ledgerText = ledger.render().slice(0, 2000);

  const prompt = FACT_AUDITOR_PROMPT.replace("{compressed_sample}", compressedSample)
    .replace("{anchor_ledger}", ledgerText)
    .replace("{task}", task ?? "Not specified");

  try {
    const response = await llm.generate(prompt, { maxTokens: 512, temperature: 0.0 });
    let text = response.text.trim();
    if (text.startsWith("```")) {
      const lines = text.split("\n");
      text = lines.filter((x) => !x.trim().startsWith("```")).join("\n");
    }
    const result = extractJson(text);
    if (result !== null && !Array.isArray(result)) {
      return result;
    }
  } catch {
    // Fact Auditor failed — return null.
  }
  return null;
}

/** Job 3: Recommend parameter adjustments from compression history. */
async function runConfigAdvisor(
  memory: ProjectMemory,
  report: CompressionReport,
  configDict: Record<string, unknown>,
  llm: MemoSiftLLMProvider,
): Promise<Record<string, unknown> | null> {
  // Only advise after 3+ sessions of data.
  if (memory.sessionsAnalyzed < 3) {
    return { recommendations: {}, reasoning: "Need 3+ sessions before advising" };
  }

  const lost = memory.protectedEntities.slice(-20);
  const auditHistory = JSON.stringify(memory.auditScores.slice(-5), null, 2);

  const prompt = CONFIG_ADVISOR_PROMPT.replace("{audit_history}", auditHistory)
    .replace("{current_config}", JSON.stringify(configDict, null, 2))
    .replace("{lost_entities}", JSON.stringify(lost.slice(0, 10)))
    .replace("{compression_ratio}", `${report.compressionRatio.toFixed(2)}x`)
    .replace("{retention_pct}", "n/a");

  try {
    const response = await llm.generate(prompt, { maxTokens: 512, temperature: 0.0 });
    let text = response.text.trim();
    if (text.startsWith("```")) {
      const lines = text.split("\n");
      text = lines.filter((x) => !x.trim().startsWith("```")).join("\n");
    }
    const result = extractJson(text);
    if (result !== null && !Array.isArray(result)) {
      return result;
    }
  } catch {
    // Config Advisor failed — return null.
  }
  return null;
}

// ── Inspection Result ───────────────────────────────────────────────────

/** Result of the post-compression LLM inspection. */
export interface InspectionResult {
  readonly lostEntities: string[];
  readonly auditScores: Record<string, unknown> | null;
  readonly configRecommendations: Record<string, unknown> | null;
  readonly memoryUpdated: boolean;
}

// ── Main Inspector ───────────────────────────────────────────────────────

/**
 * Run all three inspector jobs in parallel after compression.
 *
 * This is the main entry point. Call it AFTER compress() returns, not during.
 * It runs all three LLM jobs concurrently, updates project memory, and
 * returns the inspection result.
 *
 * @param original - The original (uncompressed) messages.
 * @param compressed - The compressed messages from compress().
 * @param ledger - The anchor ledger from compress().
 * @param report - The compression report from compress().
 * @param llm - LLM provider (Haiku recommended — cheap and fast).
 * @param options - Optional parameters for inspection.
 * @returns InspectionResult with all findings.
 */
export async function inspectCompression(
  original: readonly MemoSiftMessage[],
  compressed: readonly MemoSiftMessage[],
  ledger: AnchorLedger,
  report: CompressionReport,
  llm: MemoSiftLLMProvider,
  options?: {
    memory?: ProjectMemory | null;
    memoryPath?: string | null;
    task?: string | null;
    configDict?: Record<string, unknown> | null;
  },
): Promise<InspectionResult> {
  const memory = options?.memory ?? new ProjectMemory();

  // Run all three jobs in parallel.
  const [lostEntities, auditScores, configAdvice] = await Promise.all([
    runEntityGuardian(original, compressed, ledger, llm),
    runFactAuditor(compressed, ledger, options?.task ?? null, llm),
    runConfigAdvisor(memory, report, options?.configDict ?? {}, llm),
  ]);

  const result: InspectionResult = {
    lostEntities,
    auditScores,
    configRecommendations: configAdvice,
    memoryUpdated: false,
  };

  // Update project memory with findings.
  if (lostEntities.length > 0) {
    for (const entity of lostEntities) {
      if (!memory.protectedEntities.includes(entity)) {
        memory.protectedEntities.push(entity);
      }
    }
    // Cap at 500 entities to prevent unbounded growth.
    memory.protectedEntities = memory.protectedEntities.slice(-500);
  }

  if (auditScores !== null && typeof auditScores === "object") {
    memory.auditScores.push(auditScores);
    memory.auditScores = memory.auditScores.slice(-20);

    // Extract missing_critical items as protection strings.
    const missing = auditScores["missing_critical"];
    if (Array.isArray(missing)) {
      for (const item of missing) {
        if (
          typeof item === "string" &&
          item.length >= 3 &&
          !memory.protectedEntities.includes(item)
        ) {
          memory.protectedEntities.push(item);
        }
      }
    }
  }

  if (configAdvice !== null && typeof configAdvice === "object") {
    const recs = configAdvice["recommendations"];
    if (typeof recs === "object" && recs !== null && Object.keys(recs).length > 0) {
      Object.assign(memory.learnedConfig, recs as Record<string, unknown>);
    }
  }

  memory.sessionsAnalyzed += 1;

  // Persist if path provided.
  if (options?.memoryPath) {
    try {
      memory.save(options.memoryPath);
    } catch {
      // Failed to save project memory — non-fatal.
    }
  }

  return {
    ...result,
    memoryUpdated: true,
  };
}
