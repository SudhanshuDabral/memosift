// MemoSiftStream — real-time compression stream, process messages as they arrive.

import {
  Pressure,
  pressure as computePressure,
  contextWindowFromModel,
  estimateTokensHeuristic,
} from "./core/context-window.js";
import type { AnchorFact } from "./core/types.js";
import type { CompressionReport } from "./report.js";
import { MemoSiftSession } from "./session.js";
import type { MemoSiftSessionOptions } from "./session.js";

/** Result of pushing a message to the stream. */
export interface StreamEvent {
  /** Either "buffered" (no compression triggered) or "compressed". */
  readonly action: "buffered" | "compressed";
  /** Whether compression was triggered on this push. */
  readonly compressed: boolean;
  /** Tokens saved by compression (0 if not compressed). */
  readonly tokensSaved: number;
  /** The current context pressure level. */
  readonly pressure: Pressure;
}

/**
 * Real-time compression stream — process messages as they arrive.
 *
 * Wraps a `MemoSiftSession` with incremental mode enabled.
 * Messages are buffered until context pressure warrants compression,
 * at which point compression runs only on new messages (Zone 3).
 *
 * ```typescript
 * const stream = new MemoSiftStream("coding", { model: "claude-sonnet-4-6" });
 *
 * for (const message of incomingMessages) {
 *   const event = await stream.push(message);
 *   if (event.compressed) console.log(`Saved ${event.tokensSaved} tokens`);
 * }
 *
 * const compressed = stream.messages;
 * const facts = stream.facts;
 * ```
 */
export class MemoSiftStream {
  private readonly _session: MemoSiftSession;
  private _messages: unknown[] = [];

  constructor(preset = "general", options?: Omit<MemoSiftSessionOptions, "incremental">) {
    this._session = new MemoSiftSession(preset, {
      ...options,
      incremental: true,
    });
  }

  /**
   * Push a new message and get a compression decision.
   *
   * The message is appended to the internal buffer. Compression is triggered
   * only when context pressure exceeds NONE.
   */
  async push(message: unknown): Promise<StreamEvent> {
    this._messages.push(message);

    const pressure = this._session.checkPressure(this.estimateTokens());
    if (pressure === Pressure.NONE) {
      return { action: "buffered", compressed: false, tokensSaved: 0, pressure };
    }

    const { messages, report } = await this._session.compress(this._messages);
    this._messages = [...messages];
    return {
      action: "compressed",
      compressed: true,
      tokensSaved: report.tokensSaved,
      pressure,
    };
  }

  /**
   * Force compression regardless of pressure.
   *
   * Useful at the end of a conversation or before a long pause.
   */
  async flush(): Promise<StreamEvent> {
    if (this._messages.length === 0) {
      return { action: "buffered", compressed: false, tokensSaved: 0, pressure: Pressure.NONE };
    }

    const { messages, report } = await this._session.compress(this._messages);
    this._messages = [...messages];
    const pressure = this._session.checkPressure(this.estimateTokens());
    return {
      action: "compressed",
      compressed: true,
      tokensSaved: report.tokensSaved,
      pressure,
    };
  }

  /** Current message state (may include compressed messages). */
  get messages(): readonly unknown[] {
    return [...this._messages];
  }

  /** Accumulated anchor facts. */
  get facts(): readonly AnchorFact[] {
    return this._session.facts;
  }

  /** The underlying session (for advanced configuration). */
  get session(): MemoSiftSession {
    return this._session;
  }

  /** Number of messages in the buffer. */
  get messageCount(): number {
    return this._messages.length;
  }

  private estimateTokens(): number {
    const contents: string[] = [];
    for (const msg of this._messages) {
      if (typeof msg === "object" && msg !== null && "content" in msg) {
        const c = (msg as Record<string, unknown>).content;
        contents.push(typeof c === "string" ? c : JSON.stringify(c ?? ""));
      } else {
        contents.push(String(msg));
      }
    }
    return estimateTokensHeuristic(contents);
  }
}
