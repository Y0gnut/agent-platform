# ADR 001 — Model Routing Strategy

**Date:** 2026-09  
**Status:** Accepted  
**Deciders:** Project author

---

## Context

The gateway needs to route incoming prompts to one of two local models:

- `llama3.2:3b` (small, fast, ~2 GB VRAM): adequate for single-hop factual lookups and short answers.
- `llama3.1:8b` (large, slower, ~6 GB VRAM): better for multi-step reasoning, comparisons, and longer context.

Routing must be fast (< 1 ms overhead, not a bottleneck) and understandable without needing to read model internals. It must also degrade gracefully: if the large model is slow or unavailable, the small model should silently take over.

---

## Decision

Use a **rule-based heuristic** with three signals:

1. **Complexity keyword presence**: a fixed set of ~15 words (`explain`, `compare`, `analyze`, `summarize`, ...) that appear in multi-step prompts.
2. **Prompt word count >= 30**: longer prompts carry sufficient context to benefit from higher capacity models.
3. **Multiple question marks (>= 2)**: indicates the user is asking several independent sub-questions.

Any single signal routes to the large model. If no signal matches, route to the small model.

Implemented in `router.py` as `classify_prompt()`.

---

## Consequences

**Advantages:**
- Zero latency overhead (pure string operations, no external model invocation).
- Fully interpretable: operators can trace and audit every routing decision immediately.
- Straightforward to tune: add keywords or modify thresholds in a single configuration block.
- No training data or model serving infrastructure required.

**Limitations:**
- Signals are proxies rather than true semantic comprehension. For instance, "Explain the capital of France" triggers the keyword rule despite being simple.
- The keyword list was determined empirically rather than optimized against a labelled benchmark.
- Fixed length thresholds (30 words, 2 question marks) do not adapt dynamically to varied prompt distributions.

**Production Considerations:**
Replace the heuristic with a lightweight binary classifier trained on prompt-routing outcome pairs, served as a fast inference dependency (< 10 ms CPU latency). The gateway interface (`classify_prompt() -> "simple" | "complex"`) remains unchanged.
