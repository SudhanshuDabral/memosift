#!/usr/bin/env node
/**
 * MemoSift PreCompact hook for Claude Code.
 *
 * Injects an Anchor Ledger (critical facts) before Claude Code's own
 * compaction runs, ensuring file paths, error messages, and decisions
 * survive compression.
 *
 * Install by adding to ~/.claude/settings.json:
 *
 *     {
 *       "hooks": {
 *         "PreCompact": [{
 *           "type": "command",
 *           "command": "npx memosift-hook"
 *         }]
 *       }
 *     }
 */

import { existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { createConfig } from "../config.js";
import { extractAnchorsFromSegments } from "../core/anchor-extractor.js";
import { classifyMessages } from "../core/classifier.js";
import { AnchorLedger } from "../core/types.js";
import { parseTranscript } from "./transcript-parser.js";

const LEDGER_DIR = join(homedir(), ".memosift", "ledgers");

export interface HookInput {
  session_id?: string;
  transcript_path?: string;
  trigger?: string;
}

export interface HookOutput {
  systemMessage?: string;
}

export function runHook(input: HookInput): HookOutput {
  const sessionId = input.session_id ?? "unknown";
  const transcriptPath = input.transcript_path ?? "";

  if (!transcriptPath) return {};

  const messages = parseTranscript(transcriptPath);
  if (messages.length === 0) return {};

  // Load or create ledger for this session.
  mkdirSync(LEDGER_DIR, { recursive: true });
  const ledgerPath = join(LEDGER_DIR, `${sessionId}.json`);
  const ledger = AnchorLedger.load(ledgerPath);

  // Classify messages and extract anchors.
  const config = createConfig({
    recentTurns: 3,
    entropyThreshold: 2.5,
    enableAnchorLedger: true,
  });
  const classified = classifyMessages(messages, config);
  extractAnchorsFromSegments(classified, ledger);

  // Save the updated ledger.
  ledger.save(ledgerPath);

  if (ledger.facts.length === 0) return {};

  return { systemMessage: ledger.render() };
}

/** Entry point — reads JSON from stdin, writes JSON to stdout. */
async function main(): Promise<void> {
  try {
    const chunks: Buffer[] = [];
    for await (const chunk of process.stdin) {
      chunks.push(chunk as Buffer);
    }
    const raw = Buffer.concat(chunks).toString("utf-8").trim();
    if (!raw) {
      process.stdout.write("{}");
      return;
    }
    const input = JSON.parse(raw) as HookInput;
    const result = runHook(input);
    process.stdout.write(JSON.stringify(result));
  } catch {
    process.stdout.write("{}");
  }
}

main();
