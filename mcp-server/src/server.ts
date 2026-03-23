// MCP server setup — registers all 8 MemoSift tools.

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import {
  Pressure,
  contextWindowFromModel,
  pressure as computePressure,
  remainingRatio,
} from "memosift";
import type { CompressionReport, Decision, LayerReport } from "memosift";
import { SessionManager } from "./session-manager.js";

const DEFAULT_SESSION = "_default";

export function createMemoSiftServer(): { server: McpServer; sessionManager: SessionManager } {
  const server = new McpServer({
    name: "memosift",
    version: "0.1.0",
  });

  const sessionManager = new SessionManager();

  // ── Tool 1: memosift_check_pressure ──────────────────────────────────────

  server.tool(
    "memosift_check_pressure",
    "Check context window pressure and get a recommendation on whether compression is needed. Returns pressure level (NONE/LOW/MEDIUM/HIGH/CRITICAL) and a recommendation. Call this before compressing to avoid unnecessary work.",
    {
      model: z.string().default("claude-sonnet-4-6").describe("Model name for context window lookup"),
      current_usage_tokens: z.number().int().min(0).optional().describe("Tokens currently consumed"),
      message_count: z.number().int().min(0).optional().describe("Number of messages (alternative to token count)"),
    },
    async (params) => {
      try {
        const usage = params.current_usage_tokens ?? (params.message_count ? params.message_count * 200 : 0);
        const state = contextWindowFromModel(params.model, usage);
        const p = computePressure(state);
        const remaining = remainingRatio(state);

        const recommendations: Record<string, string> = {
          [Pressure.NONE]: "No compression needed — context window has plenty of room.",
          [Pressure.LOW]: "Light compression recommended — dedup and verbatim cleanup only.",
          [Pressure.MEDIUM]: "Standard compression recommended — pruning and structural engines active.",
          [Pressure.HIGH]: "Aggressive compression needed — all engines active, observation masking on.",
          [Pressure.CRITICAL]: "Maximum compression urgently needed — all engines including summarization.",
        };

        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              pressure: p,
              remaining_pct: Math.round(remaining * 100),
              should_compress: p !== Pressure.NONE,
              recommendation: recommendations[p],
              model: params.model,
            }, null, 2),
          }],
        };
      } catch (err) {
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ error: String(err) }) }],
          isError: true,
        };
      }
    },
  );

  // ── Tool 2: memosift_compress ────────────────────────────────────────────

  server.tool(
    "memosift_compress",
    "Compress conversation messages to reduce context window usage. Accepts messages in any supported format (OpenAI, Anthropic, LangChain, Google ADK). Returns compressed messages in the same format plus a compression report with metrics. This is the main MemoSift action.",
    {
      messages: z.array(z.record(z.string(), z.unknown())).describe("Conversation messages in any supported framework format"),
      preset: z.enum(["coding", "research", "support", "data", "general"]).default("general").describe("Compression preset tuned for specific agent types"),
      model: z.string().optional().describe("Model name for adaptive compression"),
      framework: z.enum(["openai", "anthropic", "agent_sdk", "adk", "langchain"]).optional().describe("Message format — auto-detected if omitted"),
      task: z.string().optional().describe("Current task description for relevance-aware compression"),
      session_id: z.string().optional().describe("Session ID for persistent state (ledger/dedup). Omit for stateless."),
      current_usage_tokens: z.number().int().optional().describe("Token usage for adaptive pressure calculation"),
      system: z.string().optional().describe("System prompt (for Anthropic format)"),
    },
    async (params) => {
      try {
        const sid = params.session_id ?? DEFAULT_SESSION;
        const session = sessionManager.getOrCreate(sid, {
          preset: params.preset,
          model: params.model,
        });

        // Set framework explicitly if provided.
        if (params.framework) {
          session.setFramework(params.framework);
        }

        const { messages: compressed, report } = await session.compress(
          params.messages,
          {
            task: params.task,
            usageTokens: params.current_usage_tokens,
            system: params.system,
          },
        );

        // Build compressed_entries from report decisions.
        const compressedEntries = report.decisions
          .filter((d: Decision) => d.action !== "computed" && d.action !== "skipped" && d.messageIndex >= 0)
          .map((d: Decision) => ({
            original_index: d.messageIndex,
            action: d.action,
            tokens_saved: d.originalTokens - d.resultTokens,
            expandable: session.expand(d.messageIndex) !== undefined,
          }));

        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              success: true,
              messages: compressed,
              report: {
                original_tokens: report.originalTokens,
                compressed_tokens: report.compressedTokens,
                compression_ratio: Math.round(report.compressionRatio * 100) / 100,
                tokens_saved: report.tokensSaved,
                latency_ms: Math.round(report.totalLatencyMs * 100) / 100,
                anchor_facts_extracted: session.facts.length,
                adaptive_overrides: report.adaptiveOverrides,
                layers: report.layers.map((l: LayerReport) => ({
                  name: l.name,
                  tokens_removed: l.tokensRemoved,
                  latency_ms: Math.round(l.latencyMs * 100) / 100,
                })),
              },
              compressed_entries: compressedEntries,
            }, null, 2),
          }],
        };
      } catch (err) {
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              success: false,
              error: String(err),
              messages: params.messages,
            }, null, 2),
          }],
          isError: true,
        };
      }
    },
  );

  // ── Tool 3: memosift_configure ───────────────────────────────────────────

  server.tool(
    "memosift_configure",
    "Create or update a MemoSift compression session with specific settings. Use before compressing to customize behavior, or between compressions to change strategy.",
    {
      session_id: z.string().describe("Session ID to create or update"),
      preset: z.enum(["coding", "research", "support", "data", "general"]).optional().describe("Domain preset"),
      model: z.string().optional().describe("Model name for context window lookup"),
      token_budget: z.number().int().optional().describe("Hard token limit for output"),
      recent_turns: z.number().int().min(1).max(20).optional().describe("Number of recent turns to protect"),
      enable_summarization: z.boolean().optional().describe("Enable LLM-based summarization"),
    },
    async (params) => {
      const existing = sessionManager.get(params.session_id);

      if (existing) {
        // Reconfigure existing session.
        const overrides: Record<string, unknown> = {};
        if (params.token_budget !== undefined) overrides.tokenBudget = params.token_budget;
        if (params.recent_turns !== undefined) overrides.recentTurns = params.recent_turns;
        if (params.enable_summarization !== undefined) overrides.enableSummarization = params.enable_summarization;
        existing.reconfigure(params.preset, overrides);
      } else {
        // Create new session.
        sessionManager.getOrCreate(params.session_id, {
          preset: params.preset,
          model: params.model,
        });
      }

      const session = sessionManager.get(params.session_id)!;

      return {
        content: [{
          type: "text" as const,
          text: JSON.stringify({
            session_id: params.session_id,
            status: existing ? "updated" : "created",
            fact_count: session.facts.length,
          }, null, 2),
        }],
      };
    },
  );

  // ── Tool 4: memosift_get_facts ───────────────────────────────────────────

  server.tool(
    "memosift_get_facts",
    "Retrieve critical facts (file paths, errors, decisions, identifiers) extracted during compression. Facts survive even when source messages are dropped. Useful for understanding what information MemoSift preserved.",
    {
      session_id: z.string().describe("Session ID to retrieve facts from"),
      category: z.enum(["INTENT", "FILES", "DECISIONS", "ERRORS", "ACTIVE_CONTEXT", "IDENTIFIERS", "OUTCOMES", "OPEN_ITEMS"]).optional().describe("Filter by fact category"),
    },
    async (params) => {
      const session = sessionManager.get(params.session_id);
      if (!session) {
        const available = sessionManager.list().map((s) => s.id);
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              error: `Session "${params.session_id}" not found. Available: [${available.join(", ")}]`,
            }),
          }],
          isError: true,
        };
      }

      let facts = [...session.facts];
      if (params.category) {
        facts = facts.filter((f) => f.category === params.category);
      }

      return {
        content: [{
          type: "text" as const,
          text: JSON.stringify({
            total_facts: facts.length,
            facts: facts.map((f) => ({
              category: f.category,
              content: f.content,
              turn: f.turn,
              confidence: f.confidence,
            })),
            rendered: session.ledger.render(),
          }, null, 2),
        }],
      };
    },
  );

  // ── Tool 5: memosift_expand ──────────────────────────────────────────────

  server.tool(
    "memosift_expand",
    "Re-expand a previously compressed message to see its original full content. Use when you need details that were summarized or truncated. Check compressed_entries from the compress response to find expandable indices.",
    {
      session_id: z.string().describe("Session ID"),
      original_index: z.number().int().describe("Message index from the compression report"),
    },
    async (params) => {
      const session = sessionManager.get(params.session_id);
      if (!session) {
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              found: false,
              reason: `Session "${params.session_id}" not found.`,
            }),
          }],
          isError: true,
        };
      }

      const content = session.expand(params.original_index);
      if (content === undefined) {
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              found: false,
              reason: "Cache not available — original content was not cached for this index, or session was restored from saved state.",
            }),
          }],
        };
      }

      return {
        content: [{
          type: "text" as const,
          text: JSON.stringify({ found: true, original_content: content }),
        }],
      };
    },
  );

  // ── Tool 6: memosift_report ──────────────────────────────────────────────

  server.tool(
    "memosift_report",
    "Get detailed compression metrics from the most recent compression in a session. Includes per-layer breakdowns, individual decisions, and cost savings. Useful for debugging compression behavior.",
    {
      session_id: z.string().describe("Session ID"),
      detail_level: z.enum(["summary", "layers", "decisions", "full"]).default("summary").describe("Level of detail in the report"),
    },
    async (params) => {
      const session = sessionManager.get(params.session_id);
      if (!session) {
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({ error: `Session "${params.session_id}" not found.` }),
          }],
          isError: true,
        };
      }

      const report = session.lastReport;
      if (!report) {
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({ info: "No compression has been performed in this session yet." }),
          }],
        };
      }

      const summary = {
        original_tokens: report.originalTokens,
        compressed_tokens: report.compressedTokens,
        compression_ratio: Math.round(report.compressionRatio * 100) / 100,
        tokens_saved: report.tokensSaved,
        estimated_cost_saved: report.estimatedCostSaved,
        latency_ms: Math.round(report.totalLatencyMs * 100) / 100,
        performance_tier: report.performanceTier,
        adaptive_overrides: report.adaptiveOverrides,
      };

      let result: Record<string, unknown> = summary;

      if (params.detail_level === "layers" || params.detail_level === "full") {
        result.layers = report.layers.map((l: LayerReport) => ({
          name: l.name,
          input_tokens: l.inputTokens,
          output_tokens: l.outputTokens,
          tokens_removed: l.tokensRemoved,
          latency_ms: Math.round(l.latencyMs * 100) / 100,
        }));
      }

      if (params.detail_level === "decisions" || params.detail_level === "full") {
        result.decisions = report.decisions.map((d: Decision) => ({
          layer: d.layer,
          action: d.action,
          message_index: d.messageIndex,
          original_tokens: d.originalTokens,
          result_tokens: d.resultTokens,
          reason: d.reason,
        }));
      }

      if (params.detail_level === "full") {
        result.segment_counts = report.segmentCounts;
      }

      return {
        content: [{
          type: "text" as const,
          text: JSON.stringify(result, null, 2),
        }],
      };
    },
  );

  // ── Tool 7: memosift_list_sessions ───────────────────────────────────────

  server.tool(
    "memosift_list_sessions",
    "List all active compression sessions with their last access time and fact count.",
    {},
    async () => {
      const sessions = sessionManager.list();
      return {
        content: [{
          type: "text" as const,
          text: JSON.stringify({ sessions }, null, 2),
        }],
      };
    },
  );

  // ── Tool 8: memosift_destroy ─────────────────────────────────────────────

  server.tool(
    "memosift_destroy",
    "Destroy a compression session and free its memory (ledger, cache, dedup state). Use when done with a conversation thread to prevent memory leaks.",
    {
      session_id: z.string().describe("Session ID to destroy"),
    },
    async (params) => {
      const destroyed = sessionManager.destroy(params.session_id);
      return {
        content: [{
          type: "text" as const,
          text: JSON.stringify({
            destroyed,
            message: destroyed
              ? `Session "${params.session_id}" destroyed.`
              : `Session "${params.session_id}" not found.`,
          }),
        }],
      };
    },
  );

  return { server, sessionManager };
}
