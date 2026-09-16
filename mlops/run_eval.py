"""
mlops/run_eval.py - Automated Evaluation Pipeline with MLflow Tracking
=======================================================================
Executes the evaluation dataset (mlops/gold_set.jsonl) across the agent
orchestration pipeline, scores answers using embedding cosine similarity,
and records run parameters, artifacts, and metrics to MLflow.

Usage:
    # Gateway service must be running:
    #   python router.py
    python mlops/run_eval.py

Scoring Methodology and Trade-offs
----------------------------------
Answers are evaluated by computing cosine similarity between sentence
embeddings generated with all-MiniLM-L6-v2 against the expected answer.

Key considerations:
1. Paraphrase robustness: Captures semantic alignment without requiring
   brittle surface-form exact match or n-gram overlap metrics.
2. Local execution: Operates without reliance on external judge models or APIs.
3. Caveats: High semantic similarity reflects topical overlap and may not
   detect fine-grained factual inversions. The pipeline is designed primarily
   for regression detection across model, prompt, or retrieval updates.
"""

import asyncio
import csv
import hashlib
import json
import logging
import pathlib
import sys
from datetime import datetime, timezone

import mlflow

# ---- Paths -------------------------------------------------------------------
# Allow running as: python mlops/run_eval.py  OR  python run_eval.py
_HERE = pathlib.Path(__file__).parent.resolve()
_REPO = _HERE.parent
_GOLD_SET_PATH = _HERE / "gold_set.jsonl"
_MLRUNS_DIR = _REPO / "mlruns"  # local MLflow tracking directory

# Add the repo root to sys.path so we can import mcp_client.py
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---- Configuration -----------------------------------------------------------
# Centralised constants -- change ONE of these to create an A/B comparison run.

# Cosine similarity threshold above which an answer is classified as "pass".
# Chosen to be strict enough to reject clearly wrong answers while allowing
# legitimate paraphrases.  0.75 is a commonly used cutoff in semantic search
# evaluation (see BEIR benchmark papers), but it is not scientifically derived
# for this specific task -- adjust based on empirical inspection of failures.
SIMILARITY_THRESHOLD = 0.6

# The sentence-transformers model used for embedding. MUST match the model
# used in mcp_servers/retrieval_server.py and scripts/ingest.py so that all
# embeddings live in the same vector space.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# MLflow experiment name. All runs from this eval script land here so you can
# compare them side-by-side in the UI.
MLFLOW_EXPERIMENT_NAME = "agent-platform-eval"

# ---- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] eval | %(message)s",
)
logger = logging.getLogger("eval")


# ---- Scoring -----------------------------------------------------------------

def load_embedding_model():
    """
    Load all-MiniLM-L6-v2 from local cache.

    Using local_files_only=True avoids Hugging Face's remote metadata checks,
    which involve ~20 HTTP HEAD requests and can take 60+ seconds on a slow
    connection.  If the model is not cached yet, fall back to downloading it.
    """
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model '%s' (local cache)...", EMBEDDING_MODEL_NAME)
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception:
        logger.warning("Local cache miss -- downloading %s...", EMBEDDING_MODEL_NAME)
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model ready.")
    return model


def cosine_similarity(vec_a, vec_b) -> float:
    """
    Compute cosine similarity between two numpy vectors.

    Returns a float in [-1, 1]; 1 = identical direction, 0 = orthogonal.
    We use manual computation rather than scipy to keep dependencies minimal.
    """
    import numpy as np
    a, b = vec_a.flatten(), vec_b.flatten()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def score_answer(model, expected: str, actual: str) -> float:
    """
    Embed both strings with the same model and return cosine similarity.

    Both strings are embedded together in a single forward pass for efficiency
    (sentence-transformers batches them internally).
    """
    embeddings = model.encode([expected, actual])
    return cosine_similarity(embeddings[0], embeddings[1])


# ---- Gold set loading --------------------------------------------------------

def load_gold_set(path: pathlib.Path) -> list:
    """Load JSONL -- one JSON object per line, skip blank lines."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error("Bad JSON on line %d: %s", line_num, e)
                raise
    logger.info("Loaded %d questions from %s", len(records), path)
    return records


def gold_set_version(path: pathlib.Path) -> str:
    """
    Return a short content hash of the gold set file as a version marker.

    Using a content hash (not a filename or date) means the version string
    changes exactly when the gold set changes -- no false positives from
    filesystem timestamps, no false negatives from forgotten version bumps.
    """
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return "sha256:" + sha256


# ---- Pipeline integration ----------------------------------------------------

async def run_question_async(query: str) -> dict:
    """
    Run one question through the full mcp_client pipeline.

    Returns the dict produced by mcp_client.run_query_async, which includes:
      query, tool_used, tool_args, tool_result, response, latency_ms
    """
    from mcp_client import run_query_async
    return await run_query_async(query)


# ---- Routing config snapshot -------------------------------------------------

def get_router_config() -> dict:
    """
    Import the router module and read its configuration constants.

    We snapshot these at run time so the MLflow run record reflects the exact
    config that was active during evaluation -- critical for A/B comparisons
    where one parameter changes between runs.
    """
    try:
        import router
        return {
            "large_model": router.LARGE_MODEL,
            "small_model": router.SMALL_MODEL,
            "long_prompt_word_threshold": router.LONG_PROMPT_WORD_THRESHOLD,
            "multi_question_threshold": router.MULTI_QUESTION_THRESHOLD,
            "large_model_timeout_s": router.LARGE_MODEL_TIMEOUT,
        }
    except Exception as e:
        logger.warning("Could not read router config: %s", e)
        return {}


# ---- Main evaluation loop ----------------------------------------------------

def run_eval():
    """
    Main entry point.

    Flow:
      1. Load gold set and embedding model.
      2. Start an MLflow run.
      3. For each question: run pipeline -> score answer -> record result.
      4. Compute aggregate metrics.
      5. Log params, metrics, and per-question artifact to MLflow.
      6. Print console summary.
    """
    gold_set = load_gold_set(_GOLD_SET_PATH)
    embed_model = load_embedding_model()
    router_cfg = get_router_config()
    gold_version = gold_set_version(_GOLD_SET_PATH)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---- MLflow setup --------------------------------------------------------
    # Point MLflow at a local ./mlruns directory so no remote server is needed.
    # After running, launch `mlflow ui` in the repo root to inspect results.
    mlflow.set_tracking_uri("file:///" + str(_MLRUNS_DIR))
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("Starting MLflow run under experiment '%s'", MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run ID: %s", run_id)

        # ---- Log parameters --------------------------------------------------
        # Parameters describe WHAT was being evaluated -- the config, not the
        # result.  These are the knobs you change between A/B runs.
        params = {
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "gold_set_version": gold_version,
            "gold_set_size": len(gold_set),
            "run_timestamp": run_timestamp,
        }
        for k, v in router_cfg.items():
            params["router_" + k] = v
        mlflow.log_params(params)

        # ---- Per-question evaluation ------------------------------------------
        results = []
        for i, record in enumerate(gold_set, start=1):
            question = record.get("question", "")
            expected = record.get("expected_answer", "")
            source_doc = record.get("source_doc", "")

            logger.info("[%d/%d] Querying pipeline: %r", i, len(gold_set), question[:60])

            # Run the question through the full mcp_client pipeline.
            # asyncio.run() starts a fresh event loop for each question.
            # This is slightly less efficient than reusing one loop but avoids
            # "loop already running" errors from the mcp_client internals.
            try:
                pipeline_out = asyncio.run(run_question_async(question))
                actual = pipeline_out.get("response", "").strip()
                tool_used = pipeline_out.get("tool_used", "unknown")
                latency_ms = pipeline_out.get("latency_ms", 0.0)
            except Exception as e:
                logger.error("Pipeline error on question %d: %s", i, e)
                actual = "[ERROR: " + str(e) + "]"
                tool_used = "error"
                latency_ms = 0.0

            # Score: cosine similarity between expected and actual answer.
            try:
                similarity = score_answer(embed_model, expected, actual)
            except Exception as e:
                logger.error("Scoring error on question %d: %s", i, e)
                similarity = 0.0

            passed = similarity >= SIMILARITY_THRESHOLD
            results.append({
                "id": i,
                "question": question,
                "source_doc": source_doc,
                "expected_answer": expected,
                "actual_answer": actual,
                "tool_used": tool_used,
                "similarity": round(similarity, 4),
                "passed": passed,
                "latency_ms": round(latency_ms, 1),
            })
            logger.info(
                "  similarity=%.3f  pass=%s  tool=%s",
                similarity, "YES" if passed else "NO", tool_used,
            )

        # ---- Aggregate metrics -----------------------------------------------
        n = len(results)
        n_passed = sum(1 for r in results if r["passed"])
        pass_rate = n_passed / n if n > 0 else 0.0
        avg_similarity = sum(r["similarity"] for r in results) / n if n > 0 else 0.0

        # Log metrics to MLflow -- these are the numbers you compare across runs.
        mlflow.log_metrics({
            "pass_rate": round(pass_rate, 4),
            "avg_similarity": round(avg_similarity, 4),
            "n_passed": float(n_passed),
            "n_failed": float(n - n_passed),
            "n_total": float(n),
        })

        # ---- Artifact: per-question CSV --------------------------------------
        # Save the detailed results so you can inspect individual failures
        # without re-running the full eval.  Stored as an MLflow artifact so
        # it is versioned together with the run's metrics and parameters.
        artifact_path = _HERE / "results_latest.csv"
        _write_results_csv(results, artifact_path)
        mlflow.log_artifact(str(artifact_path), artifact_path="eval_results")
        logger.info("Per-question results saved to %s", artifact_path)

        # ---- Console summary -------------------------------------------------
        _print_summary(results, pass_rate, avg_similarity, run_id)


def _write_results_csv(results: list, path: pathlib.Path) -> None:
    """Write per-question results to a CSV file for human inspection."""
    fieldnames = [
        "id", "question", "source_doc", "expected_answer", "actual_answer",
        "tool_used", "similarity", "passed", "latency_ms",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _print_summary(results: list, pass_rate: float, avg_similarity: float, run_id: str) -> None:
    """Print a readable summary table to stdout."""
    n = len(results)
    n_passed = sum(1 for r in results if r["passed"])
    divider = "-" * 100

    print("\n" + "=" * 100)
    print("  EVALUATION SUMMARY")
    print("=" * 100)
    print("  MLflow Run ID : " + run_id)
    print("  Total         : " + str(n) + " questions")
    print("  Passed        : " + str(n_passed) + "  (" + "{:.1%}".format(pass_rate) + ")")
    print("  Failed        : " + str(n - n_passed))
    print("  Avg Similarity: " + "{:.4f}".format(avg_similarity) + "  (threshold=" + str(SIMILARITY_THRESHOLD) + ")")
    print("\n  Scoring note: cosine similarity > " + str(SIMILARITY_THRESHOLD) + " = PASS.")
    print("  WARNING: high similarity does NOT guarantee factual correctness.")
    print("  A topically on-target but factually wrong answer can still pass.")
    print("\n" + divider)
    print("  {:>3}  {:^4}  {:>5}  {:<18}  QUESTION".format("#", "PASS", "SIM", "TOOL"))
    print(divider)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        q = r["question"]
        q_trunc = q[:55] + "..." if len(q) > 55 else q
        print("  {:>3}  {:^4}  {:.3f}  {:<18}  {}".format(
            r["id"], status, r["similarity"], r["tool_used"], q_trunc
        ))
    print(divider)
    print("\n  View in MLflow UI: run `mlflow ui` then open http://localhost:5000")
    print("=" * 100 + "\n")


# ---- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    run_eval()
