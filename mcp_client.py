"""
mcp_client.py — MCP Orchestration Client
=========================================
Wires the two MCP tool servers (retrieval + calculator) into the Week 1
LLM gateway to create an agent that can:
  1. Decide whether a user query needs a tool (search_documents / calculate)
  2. Call that tool via the MCP protocol
  3. Use the tool result as context for a final LLM answer
  4. Answer directly if no tool is needed

Architecture
------------
User query
    │
    ▼ decide_tool()  ─── POST /chat/completions ──► Week 1 gateway (port 8000)
    │                         (routing + LLM call)
    │  {"tool": "search_documents", "args": {...}}
    │     or
    │  {"tool": "calculate", "args": {...}}
    │     or
    │  {"tool": "none"}
    │
    ▼ call_mcp_tool()  ─── MCP stdio protocol ──► retrieval_server.py
    │                                          ──► calculator_server.py
    │  tool result (string)
    │
    ▼ generate_final_answer()  ─── POST /chat/completions ──► gateway
    │
    ▼ return {response, tool_used, tool_result, latency_ms}

MCP transport: stdio (subprocess)
  Each tool server is spawned as a child process when a tool call is needed.
  The client communicates via stdin/stdout using JSON-RPC 2.0.
  Why subprocess instead of HTTP? Stdio is the standard MCP transport for
  local tools — no port allocation, no firewall rules, no authentication.
  The protocol handles framing and schema negotiation automatically.

Run interactively:
    python mcp_client.py

Or import decide_tool() and call_mcp_tool() from tests:
    from mcp_client import decide_tool, run_query_async
"""

import asyncio
import json
import logging
import re
import sys
import time
import pathlib

import requests  # sync — used for the gateway calls (simpler than httpx here)

# ── Langfuse observability (optional — no-op if not configured) ─────────────────
from gateway.langfuse_client import flush, get_langfuse, start_trace, timed_span
_lf = get_langfuse()  # None if LANGFUSE_PUBLIC_KEY/SECRET_KEY not set

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] mcp_client | %(message)s",
)
logger = logging.getLogger("mcp_client")

# ── Configuration ─────────────────────────────────────────────────────────────
GATEWAY_URL = "http://localhost:8000/chat/completions"
GATEWAY_API_KEY = "sk-local-dev"  # matches LiteLLM master key

# Resolve server script paths relative to this file's location
_HERE = pathlib.Path(__file__).parent.resolve()
RETRIEVAL_SERVER = str(_HERE / "mcp_servers" / "retrieval_server.py")
CALCULATOR_SERVER = str(_HERE / "mcp_servers" / "calculator_server.py")

# Map tool name → server script path
TOOL_SERVER_MAP = {
    "search_documents": RETRIEVAL_SERVER,
    "calculate": CALCULATOR_SERVER,
}

# Timeout for gateway HTTP calls (seconds)
GATEWAY_TIMEOUT = 90


# ── Step 1: Tool routing decision ─────────────────────────────────────────────

_ROUTING_SYSTEM_PROMPT = """\
You are a tool router. Given a user query, decide which tool — if any — to use.

Available tools:
- search_documents: Search FastAPI documentation. Use for questions about FastAPI features, syntax, or concepts.
- calculate: Evaluate arithmetic expressions. Use for math questions involving numbers and operations (sum, average, min, max, arithmetic).
- none: No tool needed. Use for greetings, general conversation, opinion questions, or things you can answer from common knowledge.

IMPORTANT: Respond ONLY with a single line of valid JSON. No explanation. No markdown.

Format:
{"tool": "search_documents", "args": {"query": "<search phrase>"}}
{"tool": "calculate", "args": {"expression": "<math expression>"}}
{"tool": "none", "args": {}}

User query: {query}
JSON:"""


def decide_tool(query: str, trace_id: str = "") -> dict:
    """
    Ask the LLM gateway to classify the query and return the tool routing decision.

    Returns a dict: {"tool": str, "args": dict}
    where tool is one of: "search_documents", "calculate", "none".

    Uses the Week 1 gateway's /chat/completions endpoint so the routing LLM
    benefits from the gateway's model selection and fallback logic — we don't
    need to manage that here.

    Args:
        query: The user query string.
        trace_id: Optional Langfuse trace ID to propagate to the router for
                  linked tracing. Passed via X-Langfuse-Trace-Id header.
    """
    routing_prompt = _ROUTING_SYSTEM_PROMPT.replace("{query}", query)
    logger.info("Requesting tool decision for query: %r", query)

    try:
        headers = {}
        if trace_id:
            headers["x-langfuse-trace-id"] = trace_id
        resp = requests.post(
            GATEWAY_URL,
            json={"prompt": routing_prompt},
            headers=headers,
            timeout=GATEWAY_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to gateway at http://localhost:8000. "
            "Start it with: python router.py"
        )

    raw_response = resp.json().get("response", "")
    logger.debug("Raw LLM routing response: %r", raw_response)

    # Extract JSON from the response — the LLM might add surrounding text
    decision = _parse_tool_json(raw_response)
    logger.info(
        "Tool decision: tool=%r args=%r", decision.get("tool"), decision.get("args")
    )
    return decision


def _parse_tool_json(text: str) -> dict:
    """
    Extract and parse the JSON tool decision from an LLM response.

    Why a custom extractor? Local LLMs sometimes add a preamble like
    "Sure! Here's my answer:" before the JSON even when instructed not to.
    We scan for the first '{' and then walk forward tracking brace depth
    to find the matching '}', extracting the outermost JSON object.
    This is more robust than a regex that can't handle nested braces.
    """
    # Try direct parse first (happy path — no preamble)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Find the outermost {...} by tracking brace depth
    start = text.find('{')
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # Default: no tool if we can't parse the response
    logger.warning("Could not parse tool JSON from: %r — defaulting to no-tool", text)
    return {"tool": "none", "args": {}}



# ── Step 2: MCP tool invocation ───────────────────────────────────────────────

async def call_mcp_tool(
    server_script: str, tool_name: str, args: dict, timeout: float = 60.0
) -> str:
    """
    Spawn an MCP server as a subprocess and call a single tool.

    The MCP SDK handles:
      - Spawning the child process
      - Establishing the stdio transport (JSON-RPC over stdin/stdout)
      - Protocol handshake (initialize / initialized)
      - Tool call framing and result parsing

    We spawn a fresh process per call (not per session) because:
      - Startup cost is acceptable (~1-2s including model load for retrieval)
      - Simpler lifecycle: no need to track long-lived subprocess handles
      - Clean state: avoids any memory leaks from keeping servers warm
    In production, you'd maintain a persistent connection to each server.

    Args:
        server_script: Path to the MCP server Python script.
        tool_name: Name of the tool to call.
        args: Arguments to pass to the tool.
        timeout: Maximum seconds to wait for the server to respond (default 60s).
                 Raise asyncio.TimeoutError if exceeded — prevents silent hangs when
                 the retrieval server is slow to load embeddings.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,   # use the same Python interpreter as mcp_client.py
        args=[server_script],
    )

    async def _run() -> str:
        logger.info("Spawning MCP server: %s", server_script)
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.info(
                    "MCP session initialized. Calling tool %r with args %r", tool_name, args
                )
                return await session.call_tool(tool_name, args)

    try:
        result = await asyncio.wait_for(_run(), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"MCP tool '{tool_name}' did not respond within {timeout:.0f}s. "
            f"The server script may be slow to start or hanging: {server_script}"
        )

    # result.content is a list of ContentBlock objects.
    # Each block has a .text attribute for text content.
    texts = []
    for block in result.content:
        if hasattr(block, "text"):
            texts.append(block.text)
    tool_output = "\n".join(texts)
    logger.info("Tool %r returned %d chars", tool_name, len(tool_output))
    return tool_output


# ── Step 3: Generate final answer using tool context ─────────────────────────

_FINAL_ANSWER_SYSTEM_PROMPT = (
    "You must answer the user's question directly and concisely in 1 to 2 sentences. "
    "Do NOT use conversational filler like 'According to the documentation' or "
    "'The answer is'. Just provide the raw facts."
)

_FINAL_ANSWER_PROMPT = """\
System: {system}

User question: {query}

Context from {tool_name}:
{tool_result}

Answer:"""


def generate_final_answer(query: str, tool_name: str, tool_result: str, trace_id: str = "") -> str:
    """
    Feed the tool result back to the LLM as context and produce a final answer.

    This is the "augmented generation" step in RAG (Retrieval-Augmented Generation).
    The LLM receives both the user's question and the retrieved/computed context,
    allowing it to synthesize a more accurate and grounded response.

    Args:
        query: The original user query.
        tool_name: Name of the tool that was called.
        tool_result: Output from the tool call.
        trace_id: Optional Langfuse trace ID to propagate for linked tracing.
    """
    prompt = _FINAL_ANSWER_PROMPT.format(
        system=_FINAL_ANSWER_SYSTEM_PROMPT,
        query=query,
        tool_name=tool_name,
        tool_result=tool_result,
    )
    try:
        headers = {}
        if trace_id:
            headers["x-langfuse-trace-id"] = trace_id
        resp = requests.post(
            GATEWAY_URL,
            json={"prompt": prompt},
            headers=headers,
            timeout=GATEWAY_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.RequestException as e:
        logger.error("Gateway error during final answer: %s", e)
        # Return the raw tool result if the LLM call fails — better than nothing
        return f"(Tool result — LLM synthesis unavailable)\n{tool_result}"


_DIRECT_ANSWER_PROMPT = (
    "{system}\n\nUser question: {query}\n\nAnswer:"
)

def generate_direct_answer(query: str, trace_id: str = "") -> str:
    """Call the gateway directly (no tool) for a conversational response."""
    prompt = _DIRECT_ANSWER_PROMPT.format(
        system=_FINAL_ANSWER_SYSTEM_PROMPT,
        query=query,
    )
    try:
        headers = {}
        if trace_id:
            headers["x-langfuse-trace-id"] = trace_id
        resp = requests.post(
            GATEWAY_URL,
            json={"prompt": prompt},
            headers=headers,
            timeout=GATEWAY_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.RequestException as e:
        logger.error("Gateway error on direct query: %s", e)
        raise


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def run_query_async(query: str) -> dict:
    """
    Full orchestration: decide → maybe call tool → generate answer.

    Returns:
        {
            "query": str,
            "tool_used": str,        # "search_documents" | "calculate" | "none"
            "tool_args": dict,
            "tool_result": str,      # "" if no tool called
            "response": str,
            "latency_ms": float,
        }
    """
    start = time.monotonic()

    # ── Langfuse: create root trace for this user request ──────────────────────
    lf_trace = start_trace(_lf, name="mcp_client:run_query", user_query=query)
    trace_id = lf_trace.id  # "" if Langfuse is disabled (NoopTrace)

    # ── Step 1: decide tool ─────────────────────────────────────────────────────
    with timed_span(lf_trace, "decide_tool", input_data=query) as dt_span:
        decision = decide_tool(query, trace_id=trace_id)
        tool_name = decision.get("tool", "none")
        tool_args = decision.get("args", {})
        dt_span.update(
            output=str(decision),
            metadata={"tool_selected": tool_name, "tool_args": str(tool_args)},
        )

    tool_result = ""
    final_response = ""

    # ── Step 2: call tool if needed ─────────────────────────────────────────────
    if tool_name != "none" and tool_name in TOOL_SERVER_MAP:
        logger.info("==> Using tool: %s | args: %s", tool_name, tool_args)
        server_script = TOOL_SERVER_MAP[tool_name]
        with timed_span(lf_trace, f"mcp_tool:{tool_name}", input_data=str(tool_args)) as tool_span:
            tool_result = await call_mcp_tool(server_script, tool_name, tool_args)
            tool_span.update(
                output=tool_result[:500],
                metadata={
                    "tool_name": tool_name,
                    "tool_args": str(tool_args),
                    "result_length_chars": len(tool_result),
                },
            )

        # ── Step 3: generate answer with tool context ────────────────────────────
        logger.info("==> Generating final answer with tool context…")
        with timed_span(lf_trace, "generate_final_answer", input_data=query) as ans_span:
            final_response = generate_final_answer(query, tool_name, tool_result, trace_id=trace_id)
            ans_span.update(
                output=final_response[:500],
                metadata={"response_length_chars": len(final_response)},
            )
    else:
        # No tool needed — answer directly
        logger.info("==> No tool needed. Answering directly.")
        tool_name = "none"
        with timed_span(lf_trace, "generate_direct_answer", input_data=query) as ans_span:
            final_response = generate_direct_answer(query, trace_id=trace_id)
            ans_span.update(
                output=final_response[:500],
                metadata={"response_length_chars": len(final_response)},
            )

    latency_ms = (time.monotonic() - start) * 1000

    # ── Langfuse: finalise root trace ──────────────────────────────────────────────
    lf_trace.update(
        output=final_response[:500],
        metadata={
            "tool_used": tool_name,
            "latency_ms": round(latency_ms, 1),
            "response_length_chars": len(final_response),
            "tool_result_length": len(tool_result),
        },
    )
    flush(_lf)

    result = {
        "query": query,
        "tool_used": tool_name,
        "tool_args": tool_args,
        "tool_result": tool_result,
        "response": final_response,
        "latency_ms": round(latency_ms, 2),
    }

    logger.info(
        "[DONE] tool=%s latency=%.0f ms", tool_name, latency_ms
    )
    return result


def run_query(query: str) -> dict:
    """Synchronous wrapper around run_query_async for non-async callers."""
    return asyncio.run(run_query_async(query))


# ── Interactive REPL ──────────────────────────────────────────────────────────

def main() -> None:
    """Simple interactive loop for manual testing."""
    print("\n" + "=" * 60)
    print("  MCP Agent Client — Week 2 Local LLM Platform")
    print("=" * 60)
    print("  Available tools: search_documents | calculate")
    print("  Type 'quit' or Ctrl-C to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        try:
            result = run_query(query)
        except RuntimeError as e:
            print(f"\n[ERROR] {e}\n")
            continue
        except Exception as e:
            logger.exception("Unexpected error")
            print(f"\n[ERROR] {e}\n")
            continue

        print(f"\n[Tool used: {result['tool_used']}]")
        if result["tool_result"]:
            # Truncate tool result for display — it can be very long
            snippet = result["tool_result"][:300] + ("…" if len(result["tool_result"]) > 300 else "")
            print(f"[Tool result snippet]: {snippet}")
        print(f"\nAssistant: {result['response']}")
        print(f"\n[Latency: {result['latency_ms']:.0f} ms]\n")


if __name__ == "__main__":
    main()
