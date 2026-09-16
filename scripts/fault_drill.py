"""
scripts/fault_drill.py - Gateway Fallback and Fault Tolerance Validation
=========================================================================
Validates the gateway fallback path by simulating large-model failure
and asserting automatic failover to the small model without client disruption.
"""

import pathlib
import sys
import time

import requests

GATEWAY_URL = "http://localhost:8000/chat/completions"
GATEWAY_TIMEOUT = 90

TEST_QUERY = (
    "Compare and analyze the trade-offs between using async def and regular def "
    "in FastAPI path operations. When should you use each, and what happens "
    "if you block the event loop? What are the performance implications?"
)


def call_gateway(query: str, simulate_failure: bool = False) -> dict:
    """
    Dispatch a request to the gateway and collect execution metrics.

    Args:
        query: Prompt text to submit.
        simulate_failure: If True, sets simulate_large_failure=True in payload.

    Returns:
        dict containing response status, model used, fallback state, and latencies.
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
            "Ensure the router service is running: python router.py\n"
        )
    except requests.exceptions.Timeout:
        wall_latency_ms = (time.monotonic() - wall_start) * 1000
        data = {
            "response": "[TIMEOUT] Gateway did not respond within the configured timeout.",
            "model_used": "unknown",
            "fell_back": False,
            "latency_ms": 0.0,
        }
        http_success = False

    return {
        "http_success": http_success,
        "model_used": data.get("model_used", "unknown"),
        "fell_back": data.get("fell_back", False),
        "gateway_latency_ms": data.get("latency_ms", 0.0),
        "drill_latency_ms": round(wall_latency_ms, 0),
        "response_snippet": data.get("response", "")[:120],
    }


def _status_icon(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def print_summary(before: dict, during: dict) -> None:
    """Print tabular evaluation comparison of normal vs. degraded states."""
    sep = "=" * 70
    thin = "-" * 70

    print(f"\n{sep}")
    print("  FAULT DRILL: Gateway Fallback Verification")
    print(sep)
    print(f"\n  Query (complex routing trigger):\n  {TEST_QUERY[:80]}...")
    print(f"\n{thin}")
    print(f"  {'Metric':<30}  {'BEFORE (normal)':<22}  {'DURING (simulated failure)'}")
    print(thin)

    rows = [
        ("HTTP success",
         _status_icon(before["http_success"]) + " " + str(before["http_success"]),
         _status_icon(during["http_success"]) + " " + str(during["http_success"])),

        ("Model used",
         before["model_used"],
         during["model_used"]),

        ("fell_back",
         str(before["fell_back"]),
         str(during["fell_back"]) + ("  <- FALLBACK TRIGGERED" if during["fell_back"] else "")),

        ("Wall-clock latency (ms)",
         f"{before['drill_latency_ms']:.0f} ms",
         f"{during['drill_latency_ms']:.0f} ms"
         + (f"  (+ ~{during['drill_latency_ms'] - before['drill_latency_ms']:.0f} ms overhead)"
            if during['drill_latency_ms'] > before['drill_latency_ms'] else "")),

        ("Gateway-reported latency (ms)",
         f"{before['gateway_latency_ms']:.0f} ms",
         f"{during['gateway_latency_ms']:.0f} ms"),

        ("Response snippet",
         before["response_snippet"][:35] + "...",
         during["response_snippet"][:35] + "..."),
    ]

    for label, before_val, during_val in rows:
        print(f"  {label:<30}  {before_val:<22}  {during_val}")

    print(thin)

    fallback_ok = during["fell_back"] is True
    still_succeeded = during["http_success"] is True
    model_changed = before["model_used"] != during["model_used"]

    print("\n  VERIFICATION:")
    print(f"    {_status_icon(still_succeeded)} Request succeeded during simulated failure: {still_succeeded}")
    print(f"    {_status_icon(fallback_ok)} Fallback flag confirmed in response payload:   {fallback_ok}")
    print(f"    {_status_icon(model_changed)} Model transitioned: {before['model_used']} -> {during['model_used']}")

    if still_succeeded and fallback_ok:
        print("\n  OVERALL RESULT: PASS - System degraded gracefully without client error.")
    else:
        print("\n  OVERALL RESULT: FAIL - Fallback did not execute as expected.")

    print(f"\n{sep}\n")


def main() -> None:
    print("\n[fault_drill] Starting fault tolerance demonstration...")
    print(f"[fault_drill] Gateway: {GATEWAY_URL}")
    print(f"[fault_drill] Simulation method: simulate_large_failure=True (per-request flag)")

    # ── Step 1: Normal request ─────────────────────────────────────────────────
    print("\n[1/2] Sending complex query under NORMAL conditions (large model active)...")
    before = call_gateway(TEST_QUERY, simulate_failure=False)
    print(f"      -> model_used={before['model_used']}  fell_back={before['fell_back']}"
          f"  latency={before['drill_latency_ms']:.0f}ms")

    # Step 2: Simulated failure
    print("\n[2/2] Sending same query with large model SIMULATED UNAVAILABLE...")
    print("      (router will attempt large model -> connect to port 9999 -> fail -> fallback)")
    during = call_gateway(TEST_QUERY, simulate_failure=True)
    print(f"      -> model_used={during['model_used']}  fell_back={during['fell_back']}"
          f"  latency={during['drill_latency_ms']:.0f}ms  success={during['http_success']}")

    # ── Step 3: Print summary ──────────────────────────────────────────────────
    print_summary(before, during)


if __name__ == "__main__":
    main()
