"""
mcp_servers/calculator_server.py — Arithmetic Calculator MCP Server
====================================================================
Exposes one MCP tool: calculate(expression) → numeric result as string.

Supports two input formats:
  1. Arithmetic expression:  "4 + 8 + 15 + 16 + 23 + 42"  →  "108"
  2. Function call style:    "average(4, 8, 15, 16, 23, 42)"  →  "18.0"
     Supported functions: sum, average (or avg/mean), max, min, count

Design: intentionally simple. The purpose of this server is to prove the
MCP pattern generalises beyond retrieval — a second tool with a different
domain (math vs. text) demonstrates the protocol's value as a standard
interface. If we added more tools later, the client code wouldn't change.

Security: arithmetic evaluation uses ast.parse() + a restricted node
visitor. We never call eval() on arbitrary user input.

IMPORTANT: stdout is reserved for MCP JSON-RPC — all logging goes to stderr.
"""

import ast
import sys
import logging
import re
import operator
from typing import Union

# ── Logging must go to stderr ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] calculator_server | %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("calculator_server")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="calculator-server")


# ── Safe arithmetic evaluator ─────────────────────────────────────────────────
# Why not eval()? eval("__import__('os').system('rm -rf /')") is a real risk.
# We parse the expression into an AST and then walk only the allowed node types.

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node) -> Union[int, float]:
    """Recursively evaluate an ast node, allowing only numeric operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Non-numeric literal: {node.value!r}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _ALLOWED_OPS[op_type](_safe_eval(node.operand))
    else:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def evaluate_expression(expression: str) -> float:
    """
    Parse and evaluate a simple arithmetic expression string safely.
    Returns a float result.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return float(result)
    except Exception as e:
        raise ValueError(f"Could not evaluate '{expression}': {e}") from e


# ── Function-style parser ─────────────────────────────────────────────────────
# Handles: "average(4, 8, 15, 16, 23, 42)" or "sum of 4, 8, 15"
_FUNCTION_PATTERN = re.compile(
    r"^\s*(sum|average|avg|mean|max|min|count)\s*"
    r"(?:of\s+)?"       # optional "of" keyword: "average of 4, 8"
    r"\(?([0-9., ]+)\)?\s*$",
    re.IGNORECASE,
)


def _parse_function_call(expression: str) -> tuple[str, list[float]] | None:
    """
    Try to parse a function-style expression.
    Returns (function_name, [numbers]) or None if pattern doesn't match.
    """
    m = _FUNCTION_PATTERN.match(expression)
    if not m:
        return None
    func_name = m.group(1).lower()
    numbers_str = m.group(2)
    try:
        numbers = [float(n.strip()) for n in numbers_str.split(",") if n.strip()]
    except ValueError:
        return None
    return func_name, numbers


def _apply_function(func_name: str, numbers: list[float]) -> float:
    """Apply sum/average/max/min/count to a list of numbers."""
    if not numbers:
        raise ValueError("No numbers provided to function.")
    if func_name in ("sum",):
        return sum(numbers)
    elif func_name in ("average", "avg", "mean"):
        return sum(numbers) / len(numbers)
    elif func_name == "max":
        return max(numbers)
    elif func_name == "min":
        return min(numbers)
    elif func_name == "count":
        return float(len(numbers))
    else:
        raise ValueError(f"Unknown function: {func_name!r}")


# ── Tool definition ────────────────────────────────────────────────────────────

@mcp.tool()
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the numeric result.

    Supports:
      - Arithmetic:    "4 + 8 + 15 + 16 + 23 + 42"       → "108.0"
      - Division:      "(4 + 8 + 15 + 16 + 23 + 42) / 6"  → "18.0"
      - Functions:     "average(4, 8, 15, 16, 23, 42)"    → "18.0"
      - Named ops:     "sum of 4, 8, 15"                  → "27.0"
      Supported functions: sum, average (avg/mean), max, min, count

    Args:
        expression: An arithmetic expression or function call string.

    Returns:
        The numeric result as a string (e.g. "18.0").
        Returns an error description string if evaluation fails.
    """
    logger.info("calculate called | expression=%r", expression)

    # Try function-style first (e.g. "average(4, 8, 15)")
    func_result = _parse_function_call(expression)
    if func_result is not None:
        func_name, numbers = func_result
        try:
            result = _apply_function(func_name, numbers)
            # Format: drop trailing .0 for whole numbers to keep output clean
            formatted = str(int(result)) if result == int(result) else str(result)
            logger.info("Function result: %s(%s) = %s", func_name, numbers, formatted)
            return formatted
        except ValueError as e:
            logger.warning("Function evaluation failed: %s", e)
            # Fall through to arithmetic evaluator

    # Try arithmetic expression (e.g. "(4+8+15+16+23+42)/6")
    try:
        result = evaluate_expression(expression)
        formatted = str(int(result)) if result == int(result) else str(round(result, 10))
        logger.info("Arithmetic result: %r = %s", expression, formatted)
        return formatted
    except ValueError as e:
        error_msg = f"[ERROR] Could not evaluate expression: {e}"
        logger.error(error_msg)
        return error_msg


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("calculator_server starting (stdio transport)…")
    mcp.run()
