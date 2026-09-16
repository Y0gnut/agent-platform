"""
tests/conftest.py - Shared Test Fixtures and Environment Configuration
======================================================================
Configures sys.path and common fixtures across unit, integration, and regression suites.
"""

import pathlib
import sys

import pytest
import requests

_REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MLOPS_DIR = _REPO_ROOT / "mlops"
if str(_MLOPS_DIR) not in sys.path:
    sys.path.insert(0, str(_MLOPS_DIR))

GATEWAY_URL = "http://localhost:8000/chat/completions"
CHROMA_DB_PATH = _REPO_ROOT / "chroma_db"
GOLD_SET_PATH = _REPO_ROOT / "mlops" / "gold_set.jsonl"


# ── Helpers ────────────────────────────────────────────────────────────────────

def gateway_is_up() -> bool:
    """Return True if the gateway is reachable (used in skipif markers)."""
    try:
        resp = requests.get("http://localhost:8000/health", timeout=2)
        if resp.status_code == 200:
            return True
    except Exception:
        pass
    try:
        resp = requests.post(
            GATEWAY_URL,
            json={"prompt": "ping"},
            timeout=15,
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
