"""
tests/test_agent_output.py - Integration Tests for the Full MCP Pipeline
========================================================================
Dispatches representative queries through the complete mcp_client.py pipeline
(gateway -> routing decision -> optional tool call -> final answer) and
asserts structural and semantic properties of the responses.
"""

import pytest
import requests

from tests.conftest import GATEWAY_URL, chroma_db_exists, gateway_is_up
import mcp_client

_NEEDS_GATEWAY = pytest.mark.skipif(
    not gateway_is_up(),
    reason="Gateway not running at localhost:8000. Start with: python router.py",
)

_INTEGRATION_QUERIES = [
    (
        "What is the default IP address that the fastapi dev command listens on?",
        ["127.0.0.1", "localhost"],
        "fastapi-cli IP address lookup - should use search_documents",
    ),
    (
        "What is the average of 4, 8, 15, 16, 23, 42?",
        ["18"],
        "arithmetic average - should use calculate tool",
    ),
    (
        "Hello! What can you help me with today?",
        None,
        "conversational greeting - should answer directly without tool",
    ),
    (
        "Which Python decorator is required to create a middleware function in FastAPI?",
        ["middleware", "@app.middleware", "decorator"],
        "middleware decorator question - should use search_documents",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Integration tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestAgentOutput:
    """End-to-end output quality checks through the full mcp_client pipeline."""

    @_NEEDS_GATEWAY
    @pytest.mark.parametrize(
        "query,expected_keywords,description",
        _INTEGRATION_QUERIES,
        ids=[q[2] for q in _INTEGRATION_QUERIES],
    )
    def test_response_basic_quality(self, query, expected_keywords, description):
        """
        For each fixed query:
        1. Response is non-empty
        2. Response does not start with an error marker
        3. Latency is recorded (> 0 ms)
        4. If expected_keywords is set, at least one appears in the response
        """
        result = mcp_client.run_query(query)

        # ── Basic shape ────────────────────────────────────────────────────────
        assert "response" in result, "Result dict missing 'response' key"
        assert "tool_used" in result, "Result dict missing 'tool_used' key"
        assert "latency_ms" in result, "Result dict missing 'latency_ms' key"

        response_text = result["response"]

        # ── Non-empty ──────────────────────────────────────────────────────────
        assert response_text, f"Response was empty for query: {query!r}"
        assert len(response_text.strip()) > 0

        # ── No error prefix ────────────────────────────────────────────────────
        # The pipeline uses "[ERROR" as a prefix for caught exceptions.
        assert not response_text.strip().startswith("[ERROR"), (
            f"Response contains error marker for query {query!r}: {response_text[:200]!r}"
        )

        # ── Latency recorded ──────────────────────────────────────────────────
        assert result["latency_ms"] > 0, "latency_ms should be > 0"

        # ── Keyword check (where applicable) ──────────────────────────────────
        if expected_keywords:
            response_lower = response_text.lower()
            found = any(kw.lower() in response_lower for kw in expected_keywords)
            assert found, (
                f"Expected at least one of {expected_keywords!r} in response for query "
                f"{query!r}.\nGot: {response_text[:300]!r}"
            )

    @_NEEDS_GATEWAY
    def test_math_query_uses_calculate_tool(self):
        """
        A direct arithmetic query should route to the calculate tool, not answer
        from model knowledge. The tool_used field tells us which path was taken.
        """
        result = mcp_client.run_query("What is 256 divided by 8?")
        assert result["tool_used"] == "calculate", (
            f"Expected tool_used='calculate', got: {result['tool_used']!r}. "
            f"Response: {result['response'][:200]!r}"
        )

    @_NEEDS_GATEWAY
    @pytest.mark.skipif(
        not chroma_db_exists(),
        reason="ChromaDB not initialised — run: python scripts/ingest.py",
    )
    def test_fastapi_query_uses_search_documents_tool(self):
        """
        A FastAPI-specific question should route to search_documents for retrieval.
        """
        result = mcp_client.run_query(
            "When should you use async def in a FastAPI path operation?"
        )
        assert result["tool_used"] == "search_documents", (
            f"Expected tool_used='search_documents', got: {result['tool_used']!r}. "
            f"Response: {result['response'][:200]!r}"
        )

    @_NEEDS_GATEWAY
    def test_result_schema_complete(self):
        """All expected keys are present in the pipeline result dict."""
        result = mcp_client.run_query("What is 2 + 2?")
        required_keys = {"query", "tool_used", "tool_args", "tool_result", "response", "latency_ms"}
        missing = required_keys - set(result.keys())
        assert not missing, f"Result dict missing keys: {missing}"
