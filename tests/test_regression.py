"""
tests/test_regression.py — Regression Gate Against the Gold Set
===============================================================
Loads mlops/gold_set.jsonl, runs each question through the full mcp_client
pipeline, scores answers with the same semantic similarity scorer used in
mlops/run_eval.py, and asserts the overall pass rate is >= REGRESSION_THRESHOLD.

Design decisions:
  - Imports scoring logic from mlops/run_eval.py (load_gold_set, score_answer,
    load_embedding_model) rather than duplicating it. This ensures the regression
    test and the official eval pipeline use identical scoring.
  - Does NOT rerun MLflow logging (run_eval() is not called). This test is a
    quality gate, not an experiment tracker.
  - Threshold is set at 0.60 (10 percentage points below the best known run of
    66.7%). Adjust REGRESSION_THRESHOLD if you establish a higher baseline.

This is the slowest test module (~5-10 minutes for 24 questions over a local
LLM). It is marked with @pytest.mark.regression and skipped if the gateway is
not running.

Run only this test:
    pytest tests/test_regression.py -v -s -m regression
"""

import json
import pathlib
import sys

import pytest

# sys.path is set by conftest.py
from tests.conftest import GOLD_SET_PATH, gateway_is_up

# ── Regression threshold ───────────────────────────────────────────────────────
# Best known pass rate (Week 3 run): 16/24 = 66.7% at threshold=0.60.
# We set the gate 10 points below to absorb run-to-run LLM stochasticity
# without masking real regressions (e.g., a broken tool would drop this to ~0%).
REGRESSION_THRESHOLD = 0.60

# ── Skip condition ─────────────────────────────────────────────────────────────
pytestmark = [
    pytest.mark.regression,
    pytest.mark.skipif(
        not gateway_is_up(),
        reason="Gateway not running at localhost:8000 — start with: python router.py",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Import eval scoring helpers from mlops/run_eval.py
# ═══════════════════════════════════════════════════════════════════════════════

# run_eval.py adds _REPO to sys.path itself on import, but conftest already
# handles that. We just import the public functions we need.
from run_eval import (  # noqa: E402  (import after sys.path modification)
    load_embedding_model,
    load_gold_set,
    score_answer,
)

# Import the synchronous pipeline entry point
import mcp_client  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
#  Regression test
# ═══════════════════════════════════════════════════════════════════════════════

def test_gold_set_pass_rate(capsys):
    """
    Run all gold-set questions through the pipeline and assert pass rate >= threshold.

    Scoring: cosine similarity between expected and actual answer embeddings,
    using the same model (all-MiniLM-L6-v2) and threshold (0.60) as run_eval.py.
    A question passes if similarity >= 0.60.
    """
    gold_set = load_gold_set(GOLD_SET_PATH)
    embed_model = load_embedding_model()

    SIMILARITY_THRESHOLD = 0.60  # must match run_eval.SIMILARITY_THRESHOLD

    results = []
    for i, record in enumerate(gold_set, start=1):
        question = record.get("question", "")
        expected = record.get("expected_answer", "")

        try:
            pipeline_out = mcp_client.run_query(question)
            actual = pipeline_out.get("response", "").strip()
            tool_used = pipeline_out.get("tool_used", "unknown")
        except Exception as exc:
            actual = f"[ERROR: {exc}]"
            tool_used = "error"

        try:
            similarity = score_answer(embed_model, expected, actual)
        except Exception:
            similarity = 0.0

        passed = similarity >= SIMILARITY_THRESHOLD
        results.append(
            {
                "id": i,
                "question": question[:60],
                "tool_used": tool_used,
                "similarity": round(similarity, 4),
                "passed": passed,
            }
        )

    n = len(results)
    n_passed = sum(1 for r in results if r["passed"])
    pass_rate = n_passed / n if n > 0 else 0.0
    avg_similarity = sum(r["similarity"] for r in results) / n if n > 0 else 0.0

    # ── Print table for -s output ──────────────────────────────────────────────
    divider = "-" * 90
    print(f"\n{'=' * 90}")
    print("  REGRESSION TEST SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Questions : {n}")
    print(f"  Passed    : {n_passed}  ({pass_rate:.1%})")
    print(f"  Failed    : {n - n_passed}")
    print(f"  Avg Sim   : {avg_similarity:.4f}  (threshold={SIMILARITY_THRESHOLD})")
    print(f"  Gate      : pass_rate >= {REGRESSION_THRESHOLD:.0%}")
    print(f"\n{divider}")
    print(f"  {'#':>3}  {'PASS':^4}  {'SIM':>5}  {'TOOL':<20}  QUESTION")
    print(divider)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['id']:>3}  {status:^4}  {r['similarity']:.3f}  {r['tool_used']:<20}  {r['question']}")
    print(f"{divider}\n")

    # ── Regression assertion ───────────────────────────────────────────────────
    assert pass_rate >= REGRESSION_THRESHOLD, (
        f"Regression detected! pass_rate={pass_rate:.1%} is below "
        f"threshold={REGRESSION_THRESHOLD:.0%}.\n"
        f"  {n_passed}/{n} questions passed.\n"
        f"  Avg similarity: {avg_similarity:.4f}\n"
        f"  Check the table above for which questions failed."
    )
