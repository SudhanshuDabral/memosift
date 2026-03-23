// MemoSiftLLMProvider interface — the dependency-injection contract for LLM access.

export interface LLMResponse {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

export interface MemoSiftLLMProvider {
  generate(
    prompt: string,
    options?: { maxTokens?: number; temperature?: number },
  ): Promise<LLMResponse>;

  countTokens(text: string): Promise<number>;
}
