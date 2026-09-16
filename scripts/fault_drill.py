"""
scripts/fault_drill.py — Fault Tolerance Demonstration
=======================================================
Demonstrates the gateway's fallback behaviour by simulating a large-model
failure and confirming the system degrades gracefully to the small model.

How the simulation works
------------------------
router.py's ChatRequest schema includes a ``simulate_large_failure`` boolean
(default False). When True, the router redirects the large-model call to
http://localhost:9999 — an intentionally unreachable port — which triggers an
immediate connection error and fires the fallback path. No container
manipulation is needed; the flag is per-request so normal traffic is
unaffected while the drill is running.

What this script captures and prints
-------------------------------------
  BEFORE (normal operation):
    - Which model handled the request (should be large-model for a complex query)
    - Latency in ms
    - fell_back: False

  DURING (simulated failure):
    - Whether the request STILL succeeded (HTTP 200)
    - Which model actually handled it (should be small-model via fallback)
    - Latency (usually longer — timeout + retry overhead)
    - fell_back: True

  AFTER: No restore step needed — the failure flag is per-request only.

Usage:
    # Gateway must be running:
    python router.py

    python scripts/fault_drill.py

Paste the printed output directly into your README or demo recording.
"""

import sys
import time
import pathlib

import requests

# ── Configuration ──────────────────────────────────────────────────────────────
GATEWAY_URL = "http://localhost:8000/chat/completions"
GATEWAY_TIMEOUT = 90  # seconds — must exceed large-model timeout (45s)

# A query that is unambiguously "complex" by the router's heuristic:
# - Contains complexity keyword ("compare", "analyze")
# - Well over 30 words
# - Multiple question marks
TEST_QUERY = (
    "Compare and analyze the trade-offs between using async def and regular def "
    "in FastAPI path operations. When should you use each, and what happens "
    "if you block the event loop? What are the performance implications?"
)


def call_gateway(query: str, simulate_failure: bool = False) -> dict:
    """
    POST a query to the gateway and return the full response dict.

    Args:
        query: The user prompt.
        simulate_failure: If True, sends simulate_large_failure=True in the
                          request body, triggering the fallback path.

    Returns:
        dict with keys: response, model_used, fell_back, latency_ms
        Plus a synthetic 'http_success' bool and 'drill_latency_ms' (wall-clock
        time for the full round-trip including fallback retry).

    Raises:
        SystemExit if the gateway is unreachable (before the drill starts).
    """
    payload = {
        "prompt": query,
        "simulate_large_failure": simulate_failure,
    }

    wall_start = time.monotonic()
    try:
        resp = requests.post(GATEWAY_URL, json=payload, timeout=GATEWAY_TIMEOUT)
        wall_latency_ms = (time.monotonic() - wall_start) * 1000
        http_success = resp.status_code == 200

        if http_success:
            data = resp.json()
        else:
            data = {
                "response": f"[HTTP {resp.status_code}] {resp.text[:200]}",
                "model_used": "unknown",
                "fell_back": False,
                "latency_ms": 0.0,
            }

    except requests.exceptions.ConnectionError:
        raise SystemExit(
            "\n[ERROR] Cannot connect to gateway at http://localhost:8000.\n"
            "Start it with:  python router.py\n"
        )
    except requests.exceptions.Timeout:
        wall_latency_ms = (time.monotonic() - wall_start) * 1000
        data = {
            "response": "[TIMEOUT] Gateway did not respond within the drill timeout.",
            "model_used": "unknown",
            "fell_back": False,
            "latency_ms": 0.0,
        }
        http_success = False

    return {
        "http_success": http_success,
        "model_used": data.get("model_used", "unknown"),
        "fell_back": data.get("fell_back", False),
        "gateway_latency_ms": data.get("latency_ms", 0.0),  # reported by gateway
        "drill_latency_ms": round(wall_latency_ms, 0),       # wall-clock this script measured
        "response_snippet": data.get("response", "")[:120],
    }


def _status_icon(ok: bool) -> str:
    return "✓" if ok else "✗"


def print_summary(before: dict, during: dict) -> None:
    """Print a clean before/after table suitable for README or screen recording."""
    sep = "=" * 70
    thin = "-" * 70

    print(f"\n{sep}")
    print("  FAULT DRILL — Gateway Fallback Demonstration")
    print(sep)
    print(f"\n  Query (complex routing trigger):\n  {TEST_QUERY[:80]}...")
    print(f"\n{thin}")
    print(f"  {'Metric':<30}  {'BEFORE (normal)':<22}  {'DURING (large model down)'}")
    print(thin)

    rows = [
        ("HTTP success",
         _status_icon(before["http_success"]) + "  " + str(before["http_success"]),
         _status_icon(during["http_success"]) + "  " + str(during["http_success"])),

        ("Model used",
         before["model_used"],
         during["model_used"]),

        ("fell_back",
         str(before["fell_back"]),
         str(during["fell_back"]) + ("  ← FALLBACK TRIGGERED" if during["fell_back"] else "")),

        ("Wall-clock latency (ms)",
         f"{before['drill_latency_ms']:.0f} ms",
         f"{during['drill_latency_ms']:.0f} ms"
         + (f"  (+ ~{during['drill_latency_ms'] - before['drill_latency_ms']:.0f} ms overhead)"
            if during['drill_latency_ms'] > before['drill_latency_ms'] else "")),

        ("Gateway-reported latency (ms)",
         f"{before['gateway_latency_ms']:.0f} ms",
         f"{during['gateway_latency_ms']:.0f} ms"),

        ("Response snippet",
         before["response_snippet"][:35] + "…",
         during["response_snippet"][:35] + "…"),
    ]

    for label, before_val, during_val in rows:
        print(f"  {label:<30}  {before_val:<22}  {during_val}")

    print(thin)

    # ── Result verdict ─────────────────────────────────────────────────────────
    fallback_ok = during["fell_back"] is True
    still_succeeded = during["http_success"] is True
    model_changed = before["model_used"] != during["model_used"]

    print("\n  VERDICT:")
    print(f"    {_status_icon(still_succeeded)} Request still succeeded during simulated failure: {still_succeeded}")
    print(f"    {_status_icon(fallback_ok)}  fell_back=True confirmed in response:          {fallback_ok}")
    print(f"    {_status_icon(model_changed)}  Model changed from {before['model_used']} → {during['model_used']}: {model_changed}")

    if still_succeeded and fallback_ok:
        print("\n  RESULT: PASS — System degraded gracefully. Users saw no error.")
    else:
        print("\n  RESULT: FAIL — Fallback did not trigger as expected.")
        print("  Check that router.py is running and LARGE_MODEL_TIMEOUT is set.")

    print(f"\n{sep}\n")


def main() -> None:
    print("\n[fault_drill] Starting fault tolerance demonstration...")
    print(f"[fault_drill] Gateway: {GATEWAY_URL}")
    print(f"[fault_drill] Simulation method: simulate_large_failure=True (per-request flag)")

    # ── Step 1: Normal request ─────────────────────────────────────────────────
    print("\n[1/2] Sending complex query under NORMAL conditions (large model active)...")
    before = call_gateway(TEST_QUERY, simulate_failure=False)
    print(f"      → model_used={before['model_used']}  fell_back={before['fell_back']}"
          f"  latency={before['drill_latency_ms']:.0f}ms")

    # ── Step 2: Simulated failure ──────────────────────────────────────────────
    print("\n[2/2] Sending same query with large model SIMULATED UNAVAILABLE...")
    print("      (router will attempt large model → connect to port 9999 → fail → fallback)")
    during = call_gateway(TEST_QUERY, simulate_failure=True)
    print(f"      → model_used={during['model_used']}  fell_back={during['fell_back']}"
          f"  latency={during['drill_latency_ms']:.0f}ms  success={during['http_success']}")

    # ── Step 3: Print summary ──────────────────────────────────────────────────
    print_summary(before, during)


if __name__ == "__main__":
    main()
