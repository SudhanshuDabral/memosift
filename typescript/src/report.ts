// Compression report — observability and metrics for every compress() call.

export interface Decision {
  layer: string;
  action: string;
  messageIndex: number;
  originalTokens: number;
  resultTokens: number;
  reason: string;
}

export interface LayerReport {
  name: string;
  inputTokens: number;
  outputTokens: number;
  tokensRemoved: number;
  latencyMs: number;
  llmCallsMade: number;
  llmTokensConsumed: number;
}

export class CompressionReport {
  originalTokens = 0;
  compressedTokens = 0;
  compressionRatio = 1.0;
  tokensSaved = 0;
  estimatedCostSaved = 0;
  totalLatencyMs = 0;
  performanceTier = "full";
  layers: LayerReport[] = [];
  segmentCounts: Record<string, number> = {};
  decisions: Decision[] = [];
  /**
   * Fields overridden by Layer 0 adaptive compression.
   * Maps field name to [original_value, effective_value].
   * null when Layer 0 is not active (no contextWindow provided).
   */
  adaptiveOverrides: Record<string, [unknown, unknown]> | null = null;

  addLayer(
    name: string,
    inputTokens: number,
    outputTokens: number,
    latencyMs: number,
    options?: { llmCallsMade?: number; llmTokensConsumed?: number },
  ): void {
    this.layers.push({
      name,
      inputTokens,
      outputTokens,
      tokensRemoved: inputTokens - outputTokens,
      latencyMs,
      llmCallsMade: options?.llmCallsMade ?? 0,
      llmTokensConsumed: options?.llmTokensConsumed ?? 0,
    });
    this.totalLatencyMs += latencyMs;
  }

  addLayerFailure(name: string, error: string, latencyMs: number): void {
    this.layers.push({
      name,
      inputTokens: 0,
      outputTokens: 0,
      tokensRemoved: 0,
      latencyMs,
      llmCallsMade: 0,
      llmTokensConsumed: 0,
    });
    this.decisions.push({
      layer: name,
      action: "skipped",
      messageIndex: -1,
      originalTokens: 0,
      resultTokens: 0,
      reason: `Layer failed: ${error}`,
    });
    this.totalLatencyMs += latencyMs;
  }

  addDecision(
    layer: string,
    action: string,
    messageIndex: number,
    originalTokens: number,
    resultTokens: number,
    reason: string,
  ): void {
    this.decisions.push({ layer, action, messageIndex, originalTokens, resultTokens, reason });
  }

  finalize(originalTokens: number, compressedTokens: number, costPer1kTokens = 0.003): void {
    this.originalTokens = originalTokens;
    this.compressedTokens = compressedTokens;
    this.tokensSaved = originalTokens - compressedTokens;
    this.compressionRatio =
      compressedTokens === 0
        ? originalTokens > 0
          ? Number.POSITIVE_INFINITY
          : 1.0
        : originalTokens / compressedTokens;
    this.estimatedCostSaved = (this.tokensSaved / 1000) * costPer1kTokens;
  }
}
