"""
test_mcp_flow.py — Integration tests for the Week 2 MCP Tool Layer
===================================================================
Three test scenarios that validate the full tool-calling flow:
  1. Doc question  → search_documents tool should be selected
  2. Math question → calculate tool should be selected and answer correct
  3. Conversational → no tool, direct answer

Test design choices
-------------------
Layer 1 (routing tests): Call decide_tool() directly.
  - Tests the LLM routing decision in isolation
  - Requires: router.py running on port 8000 (and LiteLLM on port 4000)
  - Fast: one gateway call per test

Layer 2 (tool execution tests): Call the MCP tools directly via asyncio.
  - Tests the tool servers in isolation (no gateway needed)
  - Fast: spawns the server subprocess, makes one call, exits
  - For retrieval: requires chroma_db/ to be built (run scripts/ingest.py)
  - For calculator: no external dependencies

Why split the layers?
  Full end-to-end tests (decide → call tool → final answer) are slow (~30s)
  and fragile (require gateway + LiteLLM + Ollama + Chroma all running).
  Splitting lets CI run the fast tool-execution tests even without an LLM.

Run all tests:
    pytest test_mcp_flow.py -v

Run only routing tests (requires gateway):
    pytest test_mcp_flow.py -v -m routing

Run only tool tests (no LLM needed):
    pytest test_mcp_flow.py -v -m tool_only
"""

import asyncio
import pytest
import requests

# ── Fixtures and helpers ───────────────────────────────────────────────────────

def _gateway_is_up() -> bool:
    """Check if the Week 1 gateway is reachable."""
    try:
        requests.get("http://localhost:8000/docs", timeout=3)
        return True
    except requests.exceptions.ConnectionError:
        return False


def _chroma_is_built() -> bool:
    """Check if the chroma_db directory exists and has content."""
    import pathlib
    chroma_dir = pathlib.Path(__file__).parent / "chroma_db"
    return chroma_dir.exists() and any(chroma_dir.iterdir())


# ── Pytest markers ─────────────────────────────────────────────────────────────
# Register custom marks (avoids PytestUnknownMarkWarning)
# Run: pytest test_mcp_flow.py -m routing   or   -m tool_only


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1 — Doc question → search_documents should be called
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _gateway_is_up(), reason="Gateway not running (start with: python router.py)")
def test_routing_doc_question():
    """
    A question about FastAPI internals should trigger the search_documents tool.

    This tests the LLM routing decision, not the retrieval result itself.
    The LLM sees "path parameter" and "FastAPI" → should choose search_documents.
    """
    from mcp_client import decide_tool

    query = "How do you define a path parameter in FastAPI?"
    decision = decide_tool(query)

    assert decision["tool"] == "search_documents", (
        f"Expected 'search_documents' but got '{decision['tool']}'\n"
        f"  Full decision: {decision}\n"
        "  The routing LLM should recognise this as a FastAPI docs question."
    )
    assert "query" in decision.get("args", {}), (
        "Expected 'args.query' to be set for search_documents"
    )
    print(f"\n[PASS] tool=search_documents | search query: {decision['args']['query']!r}")


@pytest.mark.skipif(not _chroma_is_built(), reason="ChromaDB not built (run: python scripts/ingest.py)")
def test_tool_search_returns_relevant_content():
    """
    The retrieval tool should return content containing path-parameter information.

    This tests the search_documents tool directly (no LLM involved).
    It requires the Chroma DB to be built from FastAPI docs.
    """
    from mcp_client import call_mcp_tool, RETRIEVAL_SERVER

    result = asyncio.run(
        call_mcp_tool(
            RETRIEVAL_SERVER,
            "search_documents",
            {"query": "path parameter definition"},
        )
    )

    assert result, "Tool returned empty string — is chroma_db built?"
    # The FastAPI docs should mention path parameters in the retrieved content
    assert any(
        keyword in result.lower()
        for keyword in ("path", "parameter", "route", "{", "fastapi")
    ), (
        f"Retrieved content doesn't mention path parameters.\nGot: {result[:500]}"
    )
    print(f"\n[PASS] Retrieved {len(result)} chars — mentions path/parameter/route keywords.")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2 — Math question → calculate tool should be called, answer must be correct
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _gateway_is_up(), reason="Gateway not running (start with: python router.py)")
def test_routing_math_question():
    """
    A numeric aggregation question should trigger the calculate tool.

    "average of 4, 8, 15, 16, 23, 42" is clearly mathematical — no docs needed.
    """
    from mcp_client import decide_tool

    query = "What's the average of 4, 8, 15, 16, 23, 42?"
    decision = decide_tool(query)

    assert decision["tool"] == "calculate", (
        f"Expected 'calculate' but got '{decision['tool']}'\n"
        f"  Full decision: {decision}\n"
        "  The routing LLM should recognise this as a math question."
    )
    print(f"\n[PASS] tool=calculate | expression: {decision.get('args', {}).get('expression')!r}")


def test_tool_calculator_average():
    """
    The calculator tool should compute the correct average.

    Average of [4, 8, 15, 16, 23, 42] = 108 / 6 = 18.0
    This test requires NO external services — purely tests the math logic.
    """
    from mcp_client import call_mcp_tool, CALCULATOR_SERVER

    NUMBERS = [4, 8, 15, 16, 23, 42]
    EXPECTED = sum(NUMBERS) / len(NUMBERS)  # = 18.0

    # Test function-style input (most likely output from the routing LLM)
    result_func = asyncio.run(
        call_mcp_tool(
            CALCULATOR_SERVER,
            "calculate",
            {"expression": "average(4, 8, 15, 16, 23, 42)"},
        )
    )
    assert float(result_func) == pytest.approx(EXPECTED), (
        f"Expected {EXPECTED}, got {result_func!r}"
    )

    # Also test arithmetic expression format
    result_arith = asyncio.run(
        call_mcp_tool(
            CALCULATOR_SERVER,
            "calculate",
            {"expression": "(4 + 8 + 15 + 16 + 23 + 42) / 6"},
        )
    )
    assert float(result_arith) == pytest.approx(EXPECTED), (
        f"Expected {EXPECTED}, got {result_arith!r}"
    )

    print(f"\n[PASS] calculate('average(...)') = {result_func}, arith = {result_arith}")


def test_tool_calculator_other_operations():
    """
    Smoke test other calculator operations: sum, max, min, simple arithmetic.
    These are tool-only tests — no gateway or LLM required.
    """
    from mcp_client import call_mcp_tool, CALCULATOR_SERVER

    test_cases = [
        ("sum(1, 2, 3, 4, 5)", 15.0),
        ("max(3, 1, 4, 1, 5, 9, 2, 6)", 9.0),
        ("min(3, 1, 4, 1, 5)", 1.0),
        ("2 + 2", 4.0),
        ("100 / 4", 25.0),
        ("2 ** 10", 1024.0),
    ]

    for expression, expected in test_cases:
        result = asyncio.run(
            call_mcp_tool(
                CALCULATOR_SERVER,
                "calculate",
                {"expression": expression},
            )
        )
        assert float(result) == pytest.approx(expected), (
            f"calculate({expression!r}): expected {expected}, got {result!r}"
        )
        print(f"  {expression} = {result}  [OK]")

    print(f"\n[PASS] All {len(test_cases)} calculator operations correct.")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — Conversational question → NO tool should be called
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _gateway_is_up(), reason="Gateway not running (start with: python router.py)")
def test_routing_conversational_no_tool():
    """
    A greeting or conversational query should NOT trigger any tool.

    "Hello, what can you help with?" requires no retrieval and no calculation.
    The routing LLM should return {"tool": "none"}.
    """
    from mcp_client import decide_tool

    query = "Hello, what can you help with?"
    decision = decide_tool(query)

    assert decision["tool"] == "none", (
        f"Expected 'none' but got '{decision['tool']}'\n"
        f"  Full decision: {decision}\n"
        "  The routing LLM should not invoke a tool for a greeting."
    )
    print(f"\n[PASS] tool=none for conversational query {query!r}")
