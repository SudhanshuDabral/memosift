// Session lifecycle management — maintains MemoSiftSession instances across MCP tool calls.

import { MemoSiftSession } from "memosift";
import type { MemoSiftSessionOptions } from "memosift";

export interface SessionInfo {
  id: string;
  lastAccessedMs: number;
  factCount: number;
  model: string | null;
  preset: string;
}

/**
 * Manages MemoSiftSession instances keyed by session ID.
 *
 * - Touch-on-access: every get/getOrCreate resets the TTL timer.
 * - Configurable TTL via MEMOSIFT_SESSION_TTL_MS env var (default 1 hour).
 * - Default session "_default" is used when session_id is omitted.
 */
export class SessionManager {
  private readonly sessions = new Map<string, MemoSiftSession>();
  private readonly timestamps = new Map<string, number>();
  private readonly ttlMs: number;
  private cleanupTimer: ReturnType<typeof setInterval> | null = null;

  constructor() {
    const envTtl = process.env.MEMOSIFT_SESSION_TTL_MS;
    const parsed = envTtl ? parseInt(envTtl, 10) : NaN;
    this.ttlMs = Number.isNaN(parsed) ? 3_600_000 : parsed; // 1 hour default

    // Periodic cleanup every 5 minutes.
    this.cleanupTimer = setInterval(() => this.cleanup(), 300_000);
    // Don't hold the process open for cleanup.
    if (this.cleanupTimer.unref) this.cleanupTimer.unref();
  }

  /** Get an existing session or create a new one. Resets TTL. */
  getOrCreate(
    sessionId: string,
    options?: { preset?: string; model?: string },
  ): MemoSiftSession {
    const existing = this.sessions.get(sessionId);
    if (existing) {
      this.timestamps.set(sessionId, Date.now());
      return existing;
    }

    const preset = options?.preset ?? process.env.MEMOSIFT_DEFAULT_PRESET ?? "general";
    const model = options?.model ?? process.env.MEMOSIFT_MODEL ?? undefined;

    const sessionOpts: MemoSiftSessionOptions = {};
    if (model) sessionOpts.model = model;

    const session = new MemoSiftSession(preset, sessionOpts);
    this.sessions.set(sessionId, session);
    this.timestamps.set(sessionId, Date.now());
    return session;
  }

  /** Get an existing session. Returns undefined if not found. Resets TTL. */
  get(sessionId: string): MemoSiftSession | undefined {
    const session = this.sessions.get(sessionId);
    if (session) {
      this.timestamps.set(sessionId, Date.now());
    }
    return session;
  }

  /** Destroy a session and free its memory. */
  destroy(sessionId: string): boolean {
    const existed = this.sessions.delete(sessionId);
    this.timestamps.delete(sessionId);
    return existed;
  }

  /** List all active sessions with metadata. */
  list(): SessionInfo[] {
    const result: SessionInfo[] = [];
    for (const [id, session] of this.sessions) {
      result.push({
        id,
        lastAccessedMs: this.timestamps.get(id) ?? 0,
        factCount: session.facts.length,
        model: session.model,
        preset: session.preset,
      });
    }
    return result;
  }

  /** Evict sessions that have exceeded their TTL. */
  cleanup(): void {
    const now = Date.now();
    for (const [id, ts] of this.timestamps) {
      if (now - ts > this.ttlMs) {
        this.sessions.delete(id);
        this.timestamps.delete(id);
      }
    }
  }

  /** Stop the periodic cleanup timer. */
  dispose(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = null;
    }
  }

  /** Number of active sessions. */
  get size(): number {
    return this.sessions.size;
  }
}
