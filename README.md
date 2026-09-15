# agent-platform — Week 1 + 2: Local LLM Gateway & MCP Tool Layer

A **local, free, production-inspired AI agent platform** built across two weeks,
demonstrating core AI infrastructure skills: LLM routing, RAG pipelines, and
tool-calling via the Model Context Protocol (MCP).

---

## Architecture Overview

```
User / Test Script
       │  question or command
       ▼
┌────────────────────────────────────┐
│          mcp_client.py             │  Orchestrator
│  1. Ask LLM: which tool to use?   │
│  2. Call tool via MCP (if needed) │
│  3. Augment prompt with result    │
│  4. Generate final answer         │
└───┬──────────────┬─────────────────┘
    │ stdio/MCP    │ POST /chat/completions
    │              ▼
    │  ┌─────────────────────────────────┐
    │  │         router.py               │  FastAPI – port 8000
    │  │  1. Classify prompt complexity  │
    │  │  2. Route to small or large     │
    │  │  3. Fallback on timeout/error   │
    │  └────────────┬────────────────────┘
    │               │  OpenAI-compatible REST
    │               ▼
    │  ┌─────────────────────────────────┐
    │  │        LiteLLM Proxy            │  Docker – port 4000
    │  │  Aliases: small-model,          │
    │  │           large-model           │
    │  └────────────┬────────────────────┘
    │               │  http://host.docker.internal:11434
    │               ▼
    │  ┌─────────────────────────────────┐
    │  │    Ollama (Windows host)        │  port 11434
    │  │  llama3.2:3b  (small-model)    │
    │  │  llama3.1:8b  (large-model)    │
    │  └─────────────────────────────────┘
    │
    │  MCP stdio (subprocess)
    ├──► retrieval_server.py   ──► ChromaDB (./chroma_db/)
    │                               all-MiniLM-L6-v2 embeddings
    │                               FastAPI docs (~20 files)
    │
    └──► calculator_server.py  ──► safe AST arithmetic evaluator
```

---

## Project Structure

```
agent-platform/
├── docker-compose.yml           # LiteLLM proxy container
├── litellm_config.yaml          # Model aliases + host.docker.internal routing
│
├── router.py                    # Week 1 – FastAPI routing gateway
├── test_router.py               # Week 1 – integration tests (3 scenarios)
│
├── mcp_client.py                # Week 2 – MCP orchestration client
├── test_mcp_flow.py             # Week 2 – MCP tool layer tests
│
├── mcp_servers/
│   ├── retrieval_server.py      # MCP server: search_documents tool
│   └── calculator_server.py     # MCP server: calculate tool
│
├── scripts/
│   ├── fetch_docs.sh            # Shallow-clone FastAPI repo, extract .md files
│   └── ingest.py                # Chunk + embed docs → persist to ChromaDB
│
├── data/
│   └── raw_docs/                # FastAPI markdown docs (committed, ~25 files)
│
├── chroma_db/                   # Local vector DB (generated — add to .gitignore)
│
├── requirements.txt             # All Python dependencies
└── README.md
```

---

## Quick Start

### Prerequisites

- [Ollama](https://ollama.ai) installed natively on Windows with models pulled:
  ```bash
  ollama pull llama3.2:3b
  ollama pull llama3.1:8b
  ```
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Python 3.10+, WSL (for the data fetch script)

---

### Week 1 Setup — LLM Gateway

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the LiteLLM proxy (Docker):**
   ```bash
   docker compose up -d
   ```
   > LiteLLM starts on port 4000 and connects to Ollama on the Windows host via
   > `http://host.docker.internal:11434`.

3. **Start the gateway router (Terminal 2):**
   ```bash
   python router.py
   # Gateway listens on http://localhost:8000
   ```

4. **Run Week 1 tests:**
   ```bash
   python test_router.py
   ```

---

### Week 2 Setup — MCP Tool Layer + RAG

5. **Fetch FastAPI docs** (WSL required on Windows; runs a shallow git clone):
   ```bash
   wsl bash scripts/fetch_docs.sh
   ```
   This populates `data/raw_docs/` with ~25 markdown files (~100 KB total).

6. **Ingest docs into ChromaDB** (downloads ~90 MB model on first run):
   ```bash
   python scripts/ingest.py
   ```
   Creates `./chroma_db/` with embedded document chunks.

7. **Run the MCP agent interactively:**
   ```bash
   python mcp_client.py
   # Try: "How do you define a path parameter in FastAPI?"
   # Try: "What is the average of 4, 8, 15, 16, 23, 42?"
   # Try: "Hello, what can you help with?"
   ```

8. **Run Week 2 tests:**
   ```bash
   # Tool-only tests (no gateway needed):
   pytest test_mcp_flow.py -v -k "tool"

   # All tests (requires gateway + Chroma DB):
   pytest test_mcp_flow.py -v
   ```

---

## Data Source

The `data/raw_docs/` directory contains a curated subset of the
[FastAPI official documentation](https://github.com/tiangolo/fastapi),
specifically the top-level conceptual docs and core tutorial files under
`docs/en/docs/`. These are plain markdown files licensed under the
[MIT License](https://github.com/tiangolo/fastapi/blob/master/LICENSE).

To re-fetch or update the docs:
```bash
wsl bash scripts/fetch_docs.sh   # re-clones and overwrites raw_docs/
python scripts/ingest.py         # rebuilds the Chroma collection
```

---

## Why MCP?

Before MCP (Model Context Protocol), adding a new capability to an LLM agent
meant modifying the agent's core code: hardcoding the tool's calling convention,
response format, and error handling directly into the orchestrator. Changing or
replacing a tool required touching every agent that used it.

MCP solves this by **standardising how agents discover and invoke tools**. Each
tool server declares its own schema — tool name, input types, description. Any
MCP-compliant client can discover and call any MCP server without prior
knowledge of its internals. In this project: if we replaced `retrieval_server.py`
with a web-search server tomorrow, `mcp_client.py` would call it identically,
using the same `session.call_tool()` method. The LLM routing prompt would need a
one-line description update, but zero code changes in the orchestrator or the
other tool servers. This is the same principle as REST or gRPC for microservices —
a protocol contract that decouples producers from consumers.

---

## Routing Strategy (Week 1)

The router uses three heuristic signals — keyword detection, prompt length ≥ 30
words, or multiple question marks — to route to `llama3.1:8b` (large), otherwise
defaulting to `llama3.2:3b` (small). See the Week 1 design notes below.

### Fallback Behaviour

If the large model times out (default: 45 seconds) or errors:
1. The router logs a clear warning with the exception type.
2. The same prompt is retried against `llama3.2:3b`.
3. The response includes `"fell_back": true` for observability.

---

## API Reference

### `POST /chat/completions` (router.py, port 8000)

**Request:**
```json
{ "prompt": "Your question here" }
```

**Response:**
```json
{
  "response":   "The model's answer text",
  "model_used": "small-model | large-model",
  "fell_back":  false,
  "latency_ms": 1234.56
}
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LITELLM_BASE_URL` | `http://localhost:4000` | LiteLLM proxy URL |
| `LITELLM_API_KEY` | `sk-local-dev` | Must match `LITELLM_MASTER_KEY` in docker-compose |
| `LARGE_MODEL_TIMEOUT` | `45` seconds | Timeout before fallback triggers |
| `COLLECTION_NAME` | `docs` | ChromaDB collection name (ingest + retrieval) |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Must match across ingest + retrieval |
| `TOP_K` | `3` | Retrieval results returned per query |

---

## Design Decisions (Interview Notes)

### Week 1

| Decision | Rationale |
|---|---|
| **LiteLLM as proxy** | Normalises Ollama's near-OpenAI API to the full OpenAI spec; swapping a cloud model requires only a YAML change |
| **FastAPI for the gateway** | Async by default; automatic OpenAPI docs at `/docs`; Pydantic enforces schema |
| **httpx over openai SDK** | Fine-grained per-call timeout control; no real API key object required |
| **Model aliases in LiteLLM** | Decouples routing logic from model version strings |
| **Fallback to small, not retry large** | A timed-out model is likely still slow; retrying doubles wait time |

### Week 2

| Decision | Rationale |
|---|---|
| **all-MiniLM-L6-v2 embeddings** | 90 MB, free, local, ~5k sent/sec on CPU — fits a demo corpus without any API cost |
| **Heading-based chunking** | Splits on `##` boundaries to preserve semantic coherence; sliding window handles oversized sections |
| **350-word chunks** | ≈450 tokens — fits within MiniLM's 512-token context window with headroom |
| **50-word overlap** | Prevents sentences straddling chunk boundaries from being missed in retrieval |
| **Top-K = 3** | Enough context for multi-part questions; small enough not to overwhelm the LLM's context window |
| **stdio MCP transport** | No port allocation or firewall config; subprocess lifecycle is simple; standard for local tools |
| **Spawn-per-call subprocess** | Simpler lifecycle; clean state; acceptable latency (~1-2s startup) for a demo |
| **Safe AST arithmetic** | `ast.parse()` + restricted node visitor — never `eval()` on user input |