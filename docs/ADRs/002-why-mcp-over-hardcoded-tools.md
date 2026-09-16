# ADR 002 - Why MCP Over Hardcoded Tool Calls

**Date:** 2026-09  
**Status:** Accepted  
**Deciders:** Architecture Team

---

## Context

The agent needs to invoke external capabilities - document retrieval and arithmetic evaluation - from within the LLM orchestration loop. Several architectural approaches exist:

1. **Hardcoded function calls**: import `search()` and `calculate()` directly into `mcp_client.py` and call them as synchronous Python functions.
2. **HTTP microservices**: expose each tool as an independent REST endpoint; the orchestrator invokes them via HTTP requests.
3. **Model Context Protocol (MCP)**: each tool runs as an isolated process communicating via JSON-RPC 2.0 over stdio; the orchestrator discovers and invokes tools through standard protocol sessions.

---

## Decision

Use **MCP (stdio transport)**. Each tool server (`retrieval_server.py`, `calculator_server.py`) operates as an independent process declaring its schema dynamically. `mcp_client.py` discovers available tools at runtime via the MCP handshake and executes them using standard `session.call_tool()` invocations.

---

## Consequences

**Advantages:**
- **Decoupled by protocol**: replacing the retrieval server with a live search service requires zero modifications to `mcp_client.py`. The orchestrator only relies on tool names and input schemas.
- **Independent execution**: tool servers can run in isolated environments or distinct runtime languages without introducing process-level coupling.
- **No port contention**: the stdio transport executes each tool as a managed subprocess, avoiding socket allocation and local port conflicts.
- **Ecosystem compatibility**: adhering to the Model Context Protocol ensures compatibility with external agent ecosystems and standardized client runners.

**Limitations:**
- **Per-call process initialization**: spawning a process per invocation adds 1-2 seconds of startup overhead, predominantly from embedding model initialization in the retrieval server.
- **Abstraction overhead**: for static local utilities, standard subprocess framing introduces additional architectural layers compared to direct Python imports.
- **Error diagnostics**: JSON-RPC errors reported through transport layers require structured logging to trace runtime execution issues.

**Production Considerations:**
Maintain persistent `ClientSession` connections with warm subprocess pools rather than re-instantiating processes on each invocation, and register servers with a centralized service discovery catalog.
