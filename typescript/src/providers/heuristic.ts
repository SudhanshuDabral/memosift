// Heuristic token counter — fallback when no LLM provider is available.

import type { LLMResponse, MemoSiftLLMProvider } from "./base.js";

export class HeuristicTokenCounter implements MemoSiftLLMProvider {
  private charsPerToken: number;

  constructor(charsPerToken = 3.5) {
    this.charsPerToken = charsPerToken;
  }

  async countTokens(text: string): Promise<number> {
    if (!text) return 0;
    return Math.ceil(text.length / this.charsPerToken);
  }

  async generate(): Promise<LLMResponse> {
    throw new Error(
      "HeuristicTokenCounter does not support text generation. " +
        "Provide a real MemoSiftLLMProvider to use LLM-dependent features.",
    );
  }
}
