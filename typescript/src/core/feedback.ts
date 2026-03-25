// Failure-driven compression feedback — learns from compression failures.
//
// When the agent discovers a fact was lost during compression, the caller
// reports it via CompressionFeedback.reportMissing(). The feedback store
// accumulates "do not compress" patterns that are applied as Shield overrides
// in future compression cycles.

/**
 * Accumulates feedback about compression failures for future cycles.
 *
 * When the agent needs a fact that was compressed away, call
 * `reportMissing(factText)` to teach the system to protect similar
 * content in future cycles.
 *
 * The `protectionPatterns` getter returns a set of strings that should
 * be treated as critical by the importance scorer and budget enforcer.
 */
export class CompressionFeedback {
  private readonly missingFacts: string[] = [];
  private readonly protectionStrings: Set<string> = new Set();

  /**
   * Report a fact that was needed but lost during compression.
   *
   * Extracts key tokens and numbers from the fact text and adds them
   * to the protection set for future cycles.
   */
  reportMissing(factText: string): void {
    this.missingFacts.push(factText);

    // Extract tokens worth protecting.
    // Numbers with context.
    const numberRe = /\b\d[\d,.]*(?:\.\d+)?\s*\S{0,12}/g;
    let match: RegExpExecArray | null;
    while ((match = numberRe.exec(factText)) !== null) {
      this.protectionStrings.add(match[0].trim());
    }

    // Capitalized entity names.
    const entityRe = /\b[A-Z][A-Za-z]{2,20}(?:\s+[A-Z][A-Za-z]{2,20})*\b/g;
    while ((match = entityRe.exec(factText)) !== null) {
      this.protectionStrings.add(match[0]);
    }
  }

  /** Return the set of strings that should be protected in future cycles. */
  get protectionPatterns(): ReadonlySet<string> {
    return new Set(this.protectionStrings);
  }

  /** Number of missing facts reported. */
  get missingCount(): number {
    return this.missingFacts.length;
  }

  /** Reset all feedback. */
  clear(): void {
    this.missingFacts.length = 0;
    this.protectionStrings.clear();
  }
}
