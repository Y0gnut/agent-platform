"""
test_router.py - Integration tests for the Local LLM Gateway
=============================================================
These tests send real HTTP requests to the running router (localhost:8000).

They demonstrate three scenarios:
  1. A simple prompt  -> should route to the SMALL model (llama3.2:3b).
  2. A complex prompt -> should route to the LARGE model (llama3.1:8b).
  3. A forced fallback -> simulates large model failure (e.g. wrong port)
     and verifies automatic fallback to the small model.

Run with:
    python test_router.py
"""

import sys

# Ensure stdout handles UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import requests

ROUTER_URL = "http://localhost:8000/chat/completions"


def send_prompt(prompt: str, label: str, simulate_large_failure: bool = False) -> dict:
    """
    Helper: POST a prompt to the router and pretty-print the response.
    Returns the parsed JSON dict.
    """
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"PROMPT: {prompt!r}")
    if simulate_large_failure:
        print("SIMULATING: Large model connection failure (target port 9999)")
    print("-" * 60)

    payload = {"prompt": prompt}
    if simulate_large_failure:
        payload["simulate_large_failure"] = True

    try:
        resp = requests.post(
            ROUTER_URL,
            json=payload,
            timeout=120,  # Allow time for model generation on local hardware
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to router. Is it running on port 8000?")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP error: {e}\n   Body: {resp.text}")
        sys.exit(1)

    print(f"MODEL USED : {data['model_used']}")
    print(f"FELL BACK  : {data['fell_back']}")
    print(f"LATENCY    : {data['latency_ms']:.0f} ms")
    preview = data['response'][:200].replace('\n', ' ')
    print(f"RESPONSE   : {preview}{'...' if len(data['response']) > 200 else ''}")
    return data


def assert_field(data: dict, field: str, expected, label: str):
    """Assertion helper that prints PASS/FAIL status cleanly."""
    actual = data.get(field)
    if actual == expected:
        print(f"[PASS] [{label}] {field} == {expected!r}")
    else:
        print(f"[FAIL] [{label}] {field}: expected {expected!r}, got {actual!r}")
        sys.exit(1)


# -----------------------------------------------------------------------------
# Test 1: Simple prompt -> should route to SMALL model, no fallback
# -----------------------------------------------------------------------------
# "What is Python?" is short, contains no complexity keywords, and asks a
# single question. The classifier flags it as "simple" and sends it directly
# to llama3.2:3b.
result_1 = send_prompt(
    prompt="What is Python?",
    label="Simple prompt -> expect small-model",
)
assert_field(result_1, "model_used", "small-model", "Test 1")
assert_field(result_1, "fell_back",  False,         "Test 1")


# -----------------------------------------------------------------------------
# Test 2: Complex prompt -> should route to LARGE model, no fallback
# -----------------------------------------------------------------------------
# Contains keywords "compare" and "analyze" and exceeds word threshold.
# The classifier routes it to llama3.1:8b.
result_2 = send_prompt(
    prompt=(
        "Compare and analyze the architectural differences between transformer-based "
        "language models and recurrent neural networks. Explain how attention mechanisms "
        "address the vanishing gradient problem, and evaluate which is better suited for long contexts."
    ),
    label="Complex prompt -> expect large-model",
)
assert_field(result_2, "model_used", "large-model", "Test 2")
assert_field(result_2, "fell_back",  False,         "Test 2")


# -----------------------------------------------------------------------------
# Test 3: Large model failure -> should FALL BACK to small model
# -----------------------------------------------------------------------------
# Simulates the scenario where the large model is unreachable (e.g. timeout,
# dead container, or wrong port 9999).
# The router detects the failure, logs the fallback warning, retries against
# the small model, and returns fell_back=True.
result_3 = send_prompt(
    prompt=(
        "Explain the mathematical principles behind gradient descent and compare "
        "stochastic gradient descent with Adam optimizer."
    ),
    label="Simulated large-model failure -> expect automatic fallback to small-model",
    simulate_large_failure=True,
)
assert_field(result_3, "model_used", "small-model", "Test 3")
assert_field(result_3, "fell_back",  True,          "Test 3")


print(f"\n{'='*60}")
print("[SUCCESS] All 3 gateway integration tests passed successfully!")
print("="*60)
