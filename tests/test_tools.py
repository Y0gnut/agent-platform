"""
tests/test_tools.py — Unit Tests for MCP Tool Handler Functions
===============================================================
Tests the calculator and retrieval tool logic DIRECTLY — not through the MCP
subprocess protocol or the gateway. This keeps tests fast (<2s for calculator,
~10s for retrieval which must load the embedding model once) and eliminates
infrastructure dependencies for the pure-logic cases.

What is tested:
  Calculator:
    - Basic arithmetic via evaluate_expression()
    - Function-call style via _parse_function_call() + _apply_function()
    - Edge cases: division by zero, empty list, negative numbers, floordiv
    - The calculate() tool function itself (end-to-end within the module)

  Retrieval (requires ChromaDB):
    - search_documents() returns a non-empty string
    - Result contains expected formatting markers (--- Result N ---)
    - Result contains source: metadata
    - Results are specific to the query (smoke-level check)

Run without gateway or Ollama:
    pytest tests/test_tools.py -v
"""

import math
import pytest

# ── sys.path is set by conftest.py ─────────────────────────────────────────────
from tests.conftest import chroma_db_exists

# ── Import calculator internals directly (no MCP subprocess) ──────────────────
from mcp_servers.calculator_server import (
    _apply_function,
    _parse_function_call,
    calculate,
    evaluate_expression,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Calculator — evaluate_expression()
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluateExpression:
    """Direct tests of the AST-safe arithmetic evaluator."""

    def test_simple_addition(self):
        assert evaluate_expression("4 + 8") == pytest.approx(12.0)

    def test_simple_subtraction(self):
        assert evaluate_expression("100 - 37") == pytest.approx(63.0)

    def test_multiplication(self):
        assert evaluate_expression("6 * 7") == pytest.approx(42.0)

    def test_float_division(self):
        assert evaluate_expression("10 / 4") == pytest.approx(2.5)

    def test_floor_division(self):
        assert evaluate_expression("10 // 3") == pytest.approx(3.0)

    def test_modulo(self):
        assert evaluate_expression("17 % 5") == pytest.approx(2.0)

    def test_power(self):
        assert evaluate_expression("2 ** 10") == pytest.approx(1024.0)

    def test_negative_unary(self):
        assert evaluate_expression("-5 + 3") == pytest.approx(-2.0)

    def test_parentheses(self):
        assert evaluate_expression("(4 + 8 + 15 + 16 + 23 + 42) / 6") == pytest.approx(18.0)

    def test_complex_expression(self):
        result = evaluate_expression("3 * (2 + 4) - 1")
        assert result == pytest.approx(17.0)

    def test_division_by_zero_raises(self):
        """Division by zero should raise ValueError (not silently return inf)."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            evaluate_expression("1 / 0")

    def test_invalid_expression_raises(self):
        """A non-numeric expression should raise ValueError."""
        with pytest.raises(ValueError):
            evaluate_expression("hello + world")

    def test_empty_string_raises(self):
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError):
            evaluate_expression("")


# ═══════════════════════════════════════════════════════════════════════════════
#  Calculator — _parse_function_call() + _apply_function()
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseFunctionCall:
    """Tests for the function-style expression parser."""

    def test_sum_parentheses(self):
        result = _parse_function_call("sum(4, 8, 15, 16, 23, 42)")
        assert result is not None
        func_name, numbers = result
        assert func_name == "sum"
        assert numbers == pytest.approx([4, 8, 15, 16, 23, 42])

    def test_average_parentheses(self):
        result = _parse_function_call("average(4, 8, 15, 16, 23, 42)")
        assert result is not None
        assert result[0] in ("average", "avg", "mean")
        assert result[1] == pytest.approx([4, 8, 15, 16, 23, 42])

    def test_avg_alias(self):
        result = _parse_function_call("avg(10, 20)")
        assert result is not None
        assert result[0] == "avg"

    def test_mean_alias(self):
        result = _parse_function_call("mean(1, 2, 3)")
        assert result is not None
        assert result[0] == "mean"

    def test_max(self):
        result = _parse_function_call("max(3, 1, 4, 1, 5, 9)")
        assert result is not None
        assert result[0] == "max"

    def test_min(self):
        result = _parse_function_call("min(3, 1, 4, 1, 5, 9)")
        assert result is not None
        assert result[0] == "min"

    def test_count(self):
        result = _parse_function_call("count(10, 20, 30)")
        assert result is not None
        assert result[0] == "count"

    def test_sum_of_keyword(self):
        result = _parse_function_call("sum of 4, 8, 15")
        assert result is not None

    def test_case_insensitive(self):
        result = _parse_function_call("AVERAGE(10, 20, 30)")
        assert result is not None

    def test_plain_arithmetic_returns_none(self):
        """Pure arithmetic should NOT match the function pattern."""
        assert _parse_function_call("4 + 8") is None

    def test_no_match_text(self):
        assert _parse_function_call("what is the average") is None


class TestApplyFunction:
    """Tests for the function evaluator."""

    def test_sum(self):
        assert _apply_function("sum", [4.0, 8.0, 15.0]) == pytest.approx(27.0)

    def test_average(self):
        assert _apply_function("average", [4.0, 8.0, 15.0, 16.0, 23.0, 42.0]) == pytest.approx(18.0)

    def test_max(self):
        assert _apply_function("max", [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]) == pytest.approx(9.0)

    def test_min(self):
        assert _apply_function("min", [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]) == pytest.approx(1.0)

    def test_count(self):
        assert _apply_function("count", [10.0, 20.0, 30.0]) == pytest.approx(3.0)

    def test_empty_list_raises(self):
        """Edge case: empty list should raise ValueError, not silently return 0."""
        with pytest.raises(ValueError, match="No numbers provided"):
            _apply_function("sum", [])

    def test_average_empty_list_raises(self):
        with pytest.raises(ValueError, match="No numbers provided"):
            _apply_function("average", [])


# ═══════════════════════════════════════════════════════════════════════════════
#  Calculator — calculate() tool (end-to-end within the module)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateTool:
    """
    Tests the calculate() function that the MCP server exposes as a tool.
    These go through the full in-module dispatch (function-style → arithmetic).
    """

    def test_arithmetic_expression(self):
        result = calculate("4 + 8 + 15 + 16 + 23 + 42")
        # Result is returned as a string
        assert result == "108"

    def test_average_function(self):
        result = calculate("average(4, 8, 15, 16, 23, 42)")
        assert float(result) == pytest.approx(18.0)

    def test_division(self):
        result = calculate("(4 + 8 + 15 + 16 + 23 + 42) / 6")
        assert float(result) == pytest.approx(18.0)

    def test_division_by_zero_returns_error_string(self):
        """
        Edge case: division by zero.
        The calculate() tool must NOT raise an exception — it returns an
        error-description string so the LLM can relay it to the user gracefully.
        """
        result = calculate("1 / 0")
        assert isinstance(result, str)
        assert "[ERROR]" in result or "division by zero" in result.lower()

    def test_max_function(self):
        result = calculate("max(10, 3, 7, 1, 9)")
        assert float(result) == pytest.approx(10.0)

    def test_min_function(self):
        result = calculate("min(10, 3, 7, 1, 9)")
        assert float(result) == pytest.approx(1.0)

    def test_count_function(self):
        result = calculate("count(10, 20, 30, 40, 50)")
        assert float(result) == pytest.approx(5.0)

    def test_negative_numbers(self):
        result = calculate("-5 + 10")
        assert float(result) == pytest.approx(5.0)

    def test_power(self):
        result = calculate("2 ** 8")
        assert float(result) == pytest.approx(256.0)

    def test_nonsense_returns_error_string(self):
        """Non-numeric gibberish should return an error string, not raise."""
        result = calculate("hello world")
        assert isinstance(result, str)
        assert "[ERROR]" in result


# ═══════════════════════════════════════════════════════════════════════════════
#  Retrieval — search_documents() shape tests (requires chroma_db/)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.needs_chroma
@pytest.mark.skipif(
    not chroma_db_exists(),
    reason="ChromaDB not initialised — run: python scripts/ingest.py",
)
class TestSearchDocumentsShape:
    """
    Call search_documents() directly (not via MCP subprocess) and verify the
    returned string has the expected structure.

    These tests import the function from retrieval_server.py directly, bypassing
    the MCP protocol entirely — which is intentional for unit tests (faster,
    no subprocess overhead, no JSON-RPC framing).
    """

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def warmup(cls):
        """Pre-warm the model and DB once for all retrieval tests in this class."""
        from mcp_servers.retrieval_server import _get_collection, _get_model
        _get_model()
        _get_collection()

    def test_returns_non_empty_string(self):
        from mcp_servers.retrieval_server import search_documents
        result = search_documents("async def")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_result_headers(self):
        from mcp_servers.retrieval_server import search_documents
        result = search_documents("async def")
        assert "--- Result" in result, (
            f"Expected '--- Result' headers in output, got: {result[:200]!r}"
        )

    def test_contains_source_metadata(self):
        from mcp_servers.retrieval_server import search_documents
        result = search_documents("path parameter")
        assert "source:" in result, (
            f"Expected 'source:' metadata in output, got: {result[:200]!r}"
        )

    def test_contains_similarity_metadata(self):
        from mcp_servers.retrieval_server import search_documents
        result = search_documents("middleware")
        assert "similarity:" in result

    def test_known_query_returns_relevant_content(self):
        """
        For a well-known query, the result should mention FastAPI-related content.
        This is a loose smoke test — we're not asserting exact text, just that
        the response is topically relevant.
        """
        from mcp_servers.retrieval_server import search_documents
        result = search_documents("how to use async def in FastAPI")
        result_lower = result.lower()
        assert any(kw in result_lower for kw in ["async", "await", "fastapi", "def"]), (
            f"Expected async/await keywords in result, got: {result[:300]!r}"
        )

    def test_no_documents_query_still_returns_string(self):
        """
        Even for a query with no good match, the function must return a string
        (not raise). Poor matches return low-similarity results, not errors.
        """
        from mcp_servers.retrieval_server import search_documents
        result = search_documents("xyzzy frobnicator quux")
        assert isinstance(result, str)
        assert len(result) > 0
