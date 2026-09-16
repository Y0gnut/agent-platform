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

1. **Complexity keyword presence** — a fixed set of ~15 words (`explain`, `compare`, `analyze`, `summarize`, …) that almost exclusively appear in multi-step prompts.
2. **Prompt word count ≥ 30** — longer prompts carry enough context to benefit from a stronger model.
3. **Multiple question marks (≥ 2)** — indicates the user is asking several independent sub-questions.

Any single signal routes to the large model. No signal → small model.

Implemented in [`router.py`](file:///c:/Users/kunyu/OneDrive%20-%20UWA/Desktop/agent-platform/agent-platform/router.py) as `classify_prompt()`.

---

## Consequences

**Pros:**
- Zero latency overhead (pure string operations, no model call).
- Fully interpretable — an interviewer or operator can trace any routing decision in seconds.
- Easy to tune: add a keyword or change a threshold in one place.
- No training data or model serving infrastructure required.

**Cons / Honest Limitations:**
- Signals are proxies, not ground truth. "Explain the capital of France" is classified as complex by the keyword rule despite being a simple question.
- The keyword list was chosen by inspection, not measured against a labelled routing dataset. There is no principled recall/precision trade-off.
- Threshold values (30 words, 2 question marks) were set empirically and are not validated across different prompt distributions.
- A trained intent classifier (e.g. fine-tuned DistilBERT on a routing dataset) would generalise significantly better but requires a training pipeline, labelled data, and an additional model to serve.

**What we'd do at scale:** Replace the heuristic with a lightweight binary classifier trained on prompt-routing outcome pairs, served as a FastAPI dependency (< 10 ms inference on CPU). The gateway interface (`classify_prompt() → "simple" | "complex"`) would stay the same; only the implementation changes.
