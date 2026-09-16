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

Both the expected and actual answer are embedded and compared in the same vector space used for retrieval — no additional model or API needed. The 0.60 threshold was selected by visual inspection of the score distribution: most clearly-correct answers scored ≥ 0.65, most clearly-wrong answers scored ≤ 0.45. The pass rate is logged to MLflow per run so regressions are visible as a drop in `pass_rate` over time.

---

## Consequences

**Pros:**
- **Free and fully local**: no API cost, no network dependency, works offline.
- **Paraphrase-robust**: "Use async def when calling libraries that require await" and "async def should be used with libraries requiring await" score ≈ 0.92 against each other.
- **Reproducible**: same model + same code → identical scores.
- **Regression-sensitive**: a 10-point drop in pass rate is detectable even when individual answers vary due to LLM stochasticity.

**Cons / Honest Limitations (be explicit about these):**
- **Topical similarity ≠ factual accuracy**: "Use async def for CPU-bound tasks" scores ~0.75 against the correct answer because they share key tokens (`async def`), even though it is factually wrong. The scorer **cannot detect hallucination**.
- **Short expected answers are easy to fool**: if the expected answer is "Use uv", almost any response mentioning "uv" clears the threshold.
- **Threshold is not principled**: moving from 0.60 to 0.65 changes the reported pass rate without changing actual answer quality. The 0.60 value was tuned to our specific gold set and would not generalise to a different domain.
- **Does not measure retrieval quality independently**: a response synthesised from the wrong source document can still pass if the LLM's hallucination happens to resemble the expected answer.

**Intended use:** Regression detection, not quality certification. A stable or rising `avg_similarity` indicates the system is not getting worse; it does not guarantee correctness of individual answers.

**What we'd do at scale:** Add LLM-as-judge scoring (GPT-4o with a rubric) as a second signal, and track both `semantic_similarity` and `judge_score` in MLflow. Use human spot-checks to calibrate judge reliability. Keep cosine similarity for fast, cheap regression gating and add the judge score for release decisions.
