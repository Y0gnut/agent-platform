"""
mlops/compare_runs.py -- A/B Comparison of Two MLflow Evaluation Runs
======================================================================
Pulls the metrics and parameters from two MLflow runs and prints a
side-by-side comparison table.

Usage
-----
Step 1: Run the baseline eval:
    python mlops/run_eval.py
    # Note the Run ID printed at the end, e.g. "abc123..."

Step 2: Change ONE thing in the system.  Examples:
    - In router.py, change LONG_PROMPT_WORD_THRESHOLD from 30 to 20
      (makes more queries route to the large model)
    - In router.py, change LARGE_MODEL to a different alias
    - In mlops/run_eval.py, change SIMILARITY_THRESHOLD

Step 3: Run eval again:
    python mlops/run_eval.py
    # Note the second Run ID

Step 4: Compare:
    python mlops/compare_runs.py <run_id_A> <run_id_B>

    Or compare the two most recent runs automatically:
    python mlops/compare_runs.py

You can also see both runs side-by-side in the MLflow UI:
    mlflow ui   # open http://localhost:5000
    # Select both runs in the experiment view -> click "Compare"
"""

import pathlib
import sys

import mlflow

# ---- Paths -------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent.resolve()
_REPO = _HERE.parent
_MLRUNS_DIR = _REPO / "mlruns"

MLFLOW_EXPERIMENT_NAME = "agent-platform-eval"

# Metrics to compare (shown in the table in this order).
METRICS_TO_COMPARE = [
    "pass_rate",
    "avg_similarity",
    "n_passed",
    "n_failed",
    "n_total",
]

# Router parameters to compare (the knobs most likely changed between runs).
PARAMS_TO_COMPARE = [
    "similarity_threshold",
    "router_large_model",
    "router_small_model",
    "router_long_prompt_word_threshold",
    "router_large_model_timeout_s",
    "gold_set_version",
    "gold_set_size",
    "run_timestamp",
]


def get_recent_run_ids(n: int = 2) -> list:
    """Return the IDs of the n most recent runs in the experiment, newest first."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            "Experiment '" + MLFLOW_EXPERIMENT_NAME + "' not found. "
            "Run 'python mlops/run_eval.py' at least once first."
        )
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=n,
    )
    if len(runs) < n:
        raise RuntimeError(
            "Need at least " + str(n) + " runs to compare, but found " + str(len(runs)) + ". "
            "Run 'python mlops/run_eval.py' at least " + str(n) + " times."
        )
    return [r.info.run_id for r in runs]


def fetch_run(run_id: str):
    """Fetch a single MLflow run by ID and return the Run object."""
    client = mlflow.tracking.MlflowClient()
    return client.get_run(run_id)


def compare_runs(run_id_a: str, run_id_b: str) -> None:
    """Pull two runs and print a formatted side-by-side comparison."""
    mlflow.set_tracking_uri("file:///" + str(_MLRUNS_DIR))

    run_a = fetch_run(run_id_a)
    run_b = fetch_run(run_id_b)

    label_a = "Run A (" + run_id_a[:8] + "...)"
    label_b = "Run B (" + run_id_b[:8] + "...)"

    print("\n" + "=" * 90)
    print("  A/B COMPARISON")
    print("=" * 90)
    print("  {:<35}  {:<22}  {:<22}  {}".format("METRIC", label_a, label_b, "DELTA (B - A)"))
    print("-" * 90)

    for metric in METRICS_TO_COMPARE:
        val_a = run_a.data.metrics.get(metric)
        val_b = run_b.data.metrics.get(metric)
        if val_a is None and val_b is None:
            continue
        val_a_str = "{:.4f}".format(val_a) if val_a is not None else "N/A"
        val_b_str = "{:.4f}".format(val_b) if val_b is not None else "N/A"
        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            # Mark improvements in pass_rate and avg_similarity with a + sign
            sign = "+" if delta > 0 else ""
            delta_str = sign + "{:.4f}".format(delta)
        else:
            delta_str = "N/A"
        print("  {:<35}  {:<22}  {:<22}  {}".format(metric, val_a_str, val_b_str, delta_str))

    print("\n" + "-" * 90)
    print("  PARAMETERS")
    print("-" * 90)
    print("  {:<35}  {:<22}  {}".format("PARAMETER", label_a, label_b))
    print("-" * 90)

    for param in PARAMS_TO_COMPARE:
        val_a = run_a.data.params.get(param, "N/A")
        val_b = run_b.data.params.get(param, "N/A")
        # Highlight changed parameters with an asterisk
        changed = " *" if val_a != val_b else ""
        print("  {:<35}  {:<22}  {}{}".format(param, str(val_a)[:20], str(val_b)[:20], changed))

    print("-" * 90)
    print("  * = parameter changed between runs")
    print("\n  Tip: open the MLflow UI with `mlflow ui` and select both runs")
    print("       to see interactive charts and the full artifact list.")
    print("=" * 90 + "\n")


# ---- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///" + str(_MLRUNS_DIR))

    if len(sys.argv) == 3:
        # User supplied two run IDs explicitly on the command line
        run_id_a, run_id_b = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        # Auto-detect the two most recent runs
        print("No run IDs supplied -- using the two most recent runs.")
        run_id_a, run_id_b = get_recent_run_ids(n=2)
        print("  Run A (older): " + run_id_a)
        print("  Run B (newer): " + run_id_b)
    else:
        print("Usage:")
        print("  python mlops/compare_runs.py <run_id_A> <run_id_B>")
        print("  python mlops/compare_runs.py   # compares two most recent runs")
        sys.exit(1)

    compare_runs(run_id_a, run_id_b)
