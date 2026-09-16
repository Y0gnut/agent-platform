# ADR 004 — Evaluation Scoring Method

**Date:** 2026-09  
**Status:** Accepted  
**Deciders:** Project author

---

## Context

The evaluation pipeline (`mlops/run_eval.py`) needs to score each model response against a hand-written expected answer. The scoring method determines what "correct" means and what the pass rate measures.

Options considered:

| Method | Pros | Cons |
|---|---|---|
| **Exact string match** | Simple, zero cost | Fails on any valid paraphrase; too strict for free-form LLM output |
| **ROUGE-L** | Recall-focused, standard in NLP | Sensitive to word order; ignores semantic equivalence |
| **LLM-as-judge** (GPT-4 / local model) | High quality, flexible rubric | Requires API key or large model; not reproducible; judge can hallucinate |
| **Cosine similarity (sentence embeddings)** | Free, local, paraphrase-robust, reproducible | Cannot detect factual errors; threshold is a heuristic |

---

## Decision

Use **cosine similarity between sentence embeddings** (`all-MiniLM-L6-v2`). A question passes if `similarity >= 0.60`.

Both expected and actual answers are embedded and compared in the same vector space used for retrieval - no external model or API required. The 0.60 threshold was calibrated against typical score distributions where accurate answers scored >= 0.65, and inaccurate answers scored <= 0.45. Pass rate metrics are logged to MLflow per run so regressions are visible over time.

---

## Consequences

**Advantages:**
- **Local and cost-free**: No external API cost, no network dependency, functions offline.
- **Paraphrase-robust**: Syntactically different sentences conveying identical semantics score with high similarity (typically >= 0.90).
- **Deterministic**: Same input texts and model weights yield identical scores.
- **Regression detection**: Meaningful drops in aggregate pass rate are readily surfaced across evaluation iterations.

**Limitations:**
- **Topical similarity does not guarantee factual accuracy**: Answers sharing key domain terminology can achieve passing scores while containing factual errors. Embedding similarity cannot detect hallucinations.
- **Sensitivity to brief reference answers**: If reference answers are concise, extraneous generation containing target keywords may pass the similarity threshold.
- **Threshold calibration**: Thresholds reflect the specific calibration set and may require recalibration across different problem domains.

**Production Considerations:**
Integrate LLM-as-judge scoring (using structured evaluation rubrics) alongside embedding similarity, tracking both dimensions in MLflow. Cosine similarity provides fast, deterministic gating in CI pipelines, while model-based judge scoring validates nuanced correctness before production deployments.
