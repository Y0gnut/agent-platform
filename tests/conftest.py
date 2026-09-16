"""
tests/conftest.py — shared pytest configuration and fixtures
============================================================
Ensures that the repo root is on sys.path before any test module is imported,
so ``import mcp_client``, ``import mcp_servers.calculator_server``, and
``from mlops import run_eval`` all work regardless of the working directory
pytest is invoked from.
"""

import pathlib
import sys

import pytest
import requests

# ── sys.path setup ─────────────────────────────────────────────────────────────
# Insert repo root (the directory that contains router.py, mcp_client.py, etc.)
# as the first entry so our local modules shadow any installed packages.
_REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Also add mlops/ so ``from mlops import run_eval`` style imports work.
_MLOPS_DIR = _REPO_ROOT / "mlops"
if str(_MLOPS_DIR) not in sys.path:
    sys.path.insert(0, str(_MLOPS_DIR))


# ── Shared constants ───────────────────────────────────────────────────────────
GATEWAY_URL = "http://localhost:8000/chat/completions"
CHROMA_DB_PATH = _REPO_ROOT / "chroma_db"
GOLD_SET_PATH = _REPO_ROOT / "mlops" / "gold_set.jsonl"


# ── Helpers ────────────────────────────────────────────────────────────────────

def gateway_is_up() -> bool:
    """Return True if the gateway is reachable (used in skipif markers)."""
    try:
        resp = requests.post(
            GATEWAY_URL,
            json={"prompt": "ping"},
            timeout=5,
        )
        return resp.status_code < 500
    except Exception:
        return False


def chroma_db_exists() -> bool:
    """Return True if the ChromaDB directory has been initialised."""
    return CHROMA_DB_PATH.exists() and any(CHROMA_DB_PATH.iterdir())


# ── pytest markers ─────────────────────────────────────────────────────────────
# Register custom marks to avoid PytestUnknownMarkWarning.

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring a running gateway at localhost:8000",
    )
    config.addinivalue_line(
        "markers",
        "regression: mark test as a full gold-set regression gate (slow, needs gateway)",
    )
    config.addinivalue_line(
        "markers",
        "needs_chroma: mark test as requiring an initialised ChromaDB at ./chroma_db",
    )
