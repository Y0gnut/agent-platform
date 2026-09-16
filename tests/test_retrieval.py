"""
tests/test_retrieval.py — Retrieval Hit-Rate Tests
===================================================
For 5 known questions from the gold set, verifies that search_documents()
returns the CORRECT source document (per gold_set.jsonl) somewhere in its
top-3 results, and reports the retrieval hit rate.

Design rationale
----------------
This test layer sits between pure unit tests (test_tools.py, which checks
result shape) and full integration tests (test_agent_output.py, which checks
the final LLM answer). It answers the question: "Is the retrieval step finding
the right document at all?" independently of whether the LLM synthesises a
good answer from that document.

A low hit rate here (e.g., 3/5) immediately tells you the retrieval layer
is the failure mode — as opposed to a low score in test_regression.py, which
could be either retrieval OR LLM synthesis quality.

Requirements:
  - ChromaDB initialised (python scripts/ingest.py)
  - No gateway needed

Run:
    pytest tests/test_retrieval.py -v -s    # -s shows the hit-rate summary
"""

import json
import pathlib

import pytest

# sys.path is set by conftest.py
from tests.conftest import GOLD_SET_PATH, chroma_db_exists

# ── Skip if ChromaDB not present ───────────────────────────────────────────────
pytestmark = pytest.mark.skipif(
    not chroma_db_exists(),
    reason="ChromaDB not initialised — run: python scripts/ingest.py",
)

# ── Select 5 representative questions from the gold set ───────────────────────
# We pick questions whose source_doc mapping is unambiguous (the answer comes
# squarely from ONE file). This makes retrieval hit/miss binary and clear.
_SELECTED_GOLD_QUESTIONS = [
    {
        "question": "When should you define a path operation function with async def in FastAPI?",
        "source_doc": "async.md",
    },
    {
        "question": "What is the default IP address that the 'fastapi dev' command listens on?",
        "source_doc": "fastapi-cli.md",
    },
    {
        "question": "Which Python decorator is required to create a middleware function?",
        "source_doc": "tutorial__middleware.md",
    },
    {
        "question": "Which ORM does the official Full Stack FastAPI Template use for SQL database interactions?",
        "source_doc": "project-generation.md",
    },
    {
        "question": "What is the recommended tool for managing a FastAPI project and its virtual environment?",
        "source_doc": "virtual-environments.md",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def retrieval_warmup():
    """
    Pre-warm the embedding model and ChromaDB collection ONCE per test module.
    Avoids re-loading the 90 MB SentenceTransformer on every parametrised call.
    """
    from mcp_servers.retrieval_server import _get_collection, _get_model
    _get_model()
    _get_collection()


# ═══════════════════════════════════════════════════════════════════════════════
#  Retrieval hit-rate tests
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level counters so we can print a hit-rate summary at the end.
_hits = []
_misses = []


@pytest.mark.needs_chroma
@pytest.mark.parametrize(
    "entry",
    _SELECTED_GOLD_QUESTIONS,
    ids=[e["source_doc"] for e in _SELECTED_GOLD_QUESTIONS],
)
def test_correct_source_in_top3(entry, retrieval_warmup):
    """
    For each gold question, the expected source_doc must appear somewhere in
    the top-3 search_documents results.

    We strip the .md extension and match case-insensitively so that minor
    differences (e.g. "Async.md" vs "async.md") don't cause spurious failures.
    """
    from mcp_servers.retrieval_server import search_documents

    question = entry["question"]
    expected_source = entry["source_doc"]
    expected_stem = pathlib.Path(expected_source).stem.lower()

    result_text = search_documents(question)

    # Normalise for matching
    result_lower = result_text.lower()
    hit = expected_stem in result_lower

    if hit:
        _hits.append(expected_source)
    else:
        _misses.append(expected_source)

    assert hit, (
        f"Expected source '{expected_source}' (stem: '{expected_stem}') "
        f"not found in top-3 results for question:\n  {question!r}\n\n"
        f"Top-3 result text (first 600 chars):\n  {result_text[:600]!r}"
    )


@pytest.mark.needs_chroma
def test_hit_rate_summary(retrieval_warmup, capsys):
    """
    Prints a retrieval hit-rate summary. Run with -s to see it in the console.
    This test always passes — it's informational only.

    NOTE: Because parametrised tests and this test run in the same session,
    the counters in _hits/_misses are accumulated by the parametrised tests
    above. If this test runs before them, the counters will be empty.
    The -s flag shows the print() output in real time.
    """
    n_total = len(_SELECTED_GOLD_QUESTIONS)
    n_hits = len(_hits)
    n_misses = len(_misses)
    hit_rate = n_hits / n_total if n_total > 0 else 0.0

    summary = (
        f"\n{'=' * 60}\n"
        f"  RETRIEVAL HIT-RATE SUMMARY\n"
        f"{'=' * 60}\n"
        f"  Evaluated: {n_total} gold questions\n"
        f"  Hits  (correct source in top-3): {n_hits}\n"
        f"  Misses:                          {n_misses}\n"
        f"  Hit Rate:                        {hit_rate:.0%}\n"
        f"{'=' * 60}"
    )
    print(summary)

    # We don't assert a minimum here — that would duplicate test_correct_source_in_top3.
    # The hit rate is displayed for diagnostic purposes.
    assert True
