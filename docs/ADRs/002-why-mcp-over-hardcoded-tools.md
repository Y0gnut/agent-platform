# ADR 002 — Why MCP Over Hardcoded Tool Calls

**Date:** 2026-09  
**Status:** Accepted  
**Deciders:** Project author

---

## Context

The agent needs to call external capabilities — document retrieval and arithmetic evaluation — from within the LLM orchestration loop. There are several ways to do this:

1. **Hardcoded function calls**: import `search()` and `calculate()` directly into `mcp_client.py` and call them as Python functions.
2. **HTTP microservices**: expose each tool as a REST endpoint; the orchestrator calls them via HTTP.
3. **Model Context Protocol (MCP)**: each tool runs as a separate process that speaks JSON-RPC 2.0 over stdio; the orchestrator uses the MCP client SDK to discover and invoke tools.

---

## Decision

Use **MCP (stdio transport)**. Each tool server (`retrieval_server.py`, `calculator_server.py`) is an independent process that declares its own schema. `mcp_client.py` discovers tools at runtime via the MCP handshake and calls them with `session.call_tool()`.

---

## Consequences

**Pros:**
- **Decoupled by protocol, not by convention**: replacing the retrieval server with a web-search server requires zero changes to `mcp_client.py`. The orchestrator does not know or care what the tool does — only its name and input schema.
- **Independent deployability**: tool servers can be in different languages, environments, or even remote systems. The protocol handles framing, versioning, and error propagation.
- **No port management**: the stdio transport spawns each server as a subprocess; no port allocation, firewall rules, or authentication needed for local development.
- **Industry alignment**: MCP is an emerging standard backed by Anthropic and major IDE vendors. Demonstrating familiarity with it is directly portfolio-relevant.

**Cons / Honest Limitations:**
- **Per-call subprocess startup cost**: each tool call spawns a new Python process (~1-2s, dominated by model loading in the retrieval server). In production, a persistent connection pool would eliminate this.
- **Additional complexity vs. direct imports**: for two local tools that will never change, the MCP abstraction is over-engineered. A production engineer might reasonably choose direct imports for simplicity.
- **Debugging is harder**: JSON-RPC errors surfaced through the MCP SDK are less readable than a Python traceback from a direct function call.

**What we'd do at scale:** Keep MCP but maintain a long-lived `ClientSession` per tool server (rather than spawning per call) and add a tool registry service so new tools can be added without restarting the orchestrator.
