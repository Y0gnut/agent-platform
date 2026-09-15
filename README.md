# agent-platform — Week 1: Local LLM Gateway

A **local, free, production-inspired LLM gateway** that routes requests between
two Ollama models based on query complexity, with automatic fallback. Built as
Week 1 of a portfolio project demonstrating AI infrastructure and architecture
skills.

---

## Architecture Overview

```
User / Test Script
       │  POST /chat/completions {"prompt": "..."}
       ▼
┌─────────────────────────────────┐
│          router.py              │  FastAPI – port 8000
│  1. Classify prompt complexity  │
│  2. Route to small or large     │
│  3. Fallback on timeout/error   │
└────────────┬────────────────────┘
             │  OpenAI-compatible REST
             ▼
┌─────────────────────────────────┐
│        LiteLLM Proxy            │  Docker – port 4000
│  Unified API for both models    │
│  Aliases: small-model,          │
│           large-model           │
└────────────┬────────────────────┘
             │  Ollama API
             ▼
┌─────────────────────────────────┐
│           Ollama                │  Docker – port 11434
│  llama3.2:3b  (small-model)     │
│  llama3.1:8b  (large-model)     │
└─────────────────────────────────┘
```

---

## Project Structure

```
agent-platform/
├── docker-compose.yml     # Ollama + LiteLLM containers
├── litellm_config.yaml    # Model aliases and proxy settings
├── router.py              # FastAPI routing gateway (main logic)
├── test_router.py         # Integration tests (3 scenarios)
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Quick Start

### Prerequisites
- [Ollama](https://ollama.ai) installed and running locally with models pulled:
  ```bash
  ollama pull llama3.2:3b
  ollama pull llama3.1:8b
  ```
- Python 3.10+
- *(Optional)* [Docker Desktop](https://www.docker.com/products/docker-desktop/) if running via containers

---

### Option A: Running Natively (Recommended for Local Dev & No-Docker Environments)

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start LiteLLM Proxy (Terminal 1):**
   ```bash
   litellm --config litellm_config.yaml --port 4000
   ```

3. **Start the Router Gateway (Terminal 2):**
   ```bash
   python router.py
   # or: uvicorn router:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Run the Test Suite (Terminal 3):**
   ```bash
   python test_router.py
   ```

---

### Option B: Running with Docker Compose

1. **Start the containers:**
   ```bash
   docker compose up -d
   ```
2. **Start the router & run tests:**
   ```bash
   python router.py
   python test_router.py
   ```

### 4 – Test it

```bash
# Quick smoke test with curl
curl -s -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Python?"}' | python -m json.tool

# Run the full test suite
python test_router.py
```

---

## Routing Strategy

### How the heuristic works

The router uses three independent signals, any one of which triggers routing to
the **large model**:

| Signal | Condition | Rationale |
|---|---|---|
| **Keyword detection** | Prompt contains words like `explain`, `compare`, `analyze`, `evaluate`, `summarize`, `contrast`, `discuss` | These words imply multi-step reasoning, comparison, or synthesis — tasks where a larger model's broader "knowledge" and deeper context window justify the extra latency |
| **Prompt length** | Word count ≥ 30 | A long prompt usually carries complex context or multiple sub-requirements. Short factual queries ("What is X?") are almost always under 20 words |
| **Multiple questions** | Two or more `?` characters | Multiple questions require the model to track independent sub-tasks simultaneously — a known weakness of smaller models |

**Default behaviour:** if no signal fires, the prompt goes to `llama3.2:3b`.

### Why I chose this approach

1. **Zero latency overhead.** The classifier runs in microseconds. A
   ML-based classifier (e.g., a fine-tuned BERT) would add 50–200 ms of
   inference time on every request — often more than the routing benefit.
2. **Fully interpretable.** Every routing decision can be explained in a
   sentence. That matters for debugging and for justifying decisions to a team.
3. **No training data required.** A trained classifier needs labelled examples
   of "simple" vs "complex" queries. For a portfolio project, collecting and
   curating that dataset would dwarf the implementation work.

### Honest limitations

> This is a **rule-based classifier**. It is deliberately simple to make the
> routing logic readable and auditable, but it has real weaknesses you should
> know about before a production deployment:

- **False positives on keywords.** `"Don't explain it to me"` contains
  `explain` and would be over-routed to the large model, even though it's
  a short negative command.
- **Prompt length is a crude proxy.** A 35-word prompt asking for a shopping
  list is not inherently complex. A 10-word prompt asking "What is the Riemann
  Hypothesis?" may warrant a large model.
- **No semantic understanding.** The heuristic cannot distinguish
  `"Compare apples and oranges"` (trivial) from
  `"Compare transformer and SSM architectures for long-context reasoning"` (non-trivial).
- **Language-dependent.** All keywords are English. A multilingual gateway would
  need per-language keyword sets or a language-agnostic approach.

### What a production system would use instead

1. **A trained routing classifier** — fine-tuned on query–complexity labels
   (e.g., a DistilBERT or TinyBERT model). Adds latency but much higher accuracy.
2. **LLM self-assessment** — ask the small model to rate its own confidence
   (`0–1`) and escalate to the large model if confidence < threshold. Uses
   token log-probabilities. Elegant but adds one model call per request.
3. **Semantic embedding similarity** — embed the prompt and compare to
   a pre-computed centroid of "complex" vs "simple" query embeddings. Fast
   after the embedding step, language-agnostic, but requires an embedding
   model and labelled cluster centroids.
4. **Cost-aware routing** — combine complexity with a cost/latency budget.
   Route to the large model only when complexity is high AND the caller has
   not exceeded their latency SLA. Used in systems like [RouteLLM](https://github.com/lm-sys/RouteLLM).

---

## Fallback Behaviour

If the **large model times out** (default: 15 seconds) or returns any error:

1. The router logs a clear warning:
   ```
   ⚠️  FALLBACK triggered | Large model failed: TimeoutException: ... | Retrying with small model.
   ```
2. The same prompt is immediately retried against `llama3.2:3b`.
3. The response includes `"fell_back": true` so callers and dashboards can
   detect degraded operation.

To simulate fallback manually:
```bash
# Start the router pointing at a dead LiteLLM port
LITELLM_BASE_URL=http://localhost:9999 uvicorn router:app --port 8001

# Send a complex query – you'll see the fallback warning in the router logs
curl -s -X POST http://localhost:8001/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Compare and analyze the differences between RNNs and Transformers."}' \
  | python -m json.tool
```

Expected response:
```json
{
  "response": "...",
  "model_used": "small-model",
  "fell_back": true,
  "latency_ms": 3241.5
}
```

---

## API Reference

### `POST /chat/completions`

**Request body:**
```json
{ "prompt": "Your question here" }
```

**Response:**
```json
{
  "response":    "The model's answer text",
  "model_used":  "small-model | large-model",
  "fell_back":   false,
  "latency_ms":  1234.56
}
```

**Error (503):** Both models unavailable.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LITELLM_BASE_URL` | `http://localhost:4000` | LiteLLM proxy URL |
| `LITELLM_API_KEY` | `sk-local-dev` | Must match `LITELLM_MASTER_KEY` in docker-compose |
| `LARGE_MODEL_TIMEOUT` | `15` seconds | Timeout before fallback triggers |
| `LONG_PROMPT_WORD_THRESHOLD` | `30` words | Complexity signal: prompt length |
| `MULTI_QUESTION_THRESHOLD` | `2` question marks | Complexity signal: multiple questions |

---

## Week 1 Design Decisions (Interview Notes)

| Decision | Rationale |
|---|---|
| **LiteLLM as proxy** | Normalises Ollama's near-OpenAI API to the actual OpenAI spec; allows swapping Ollama for a cloud provider without touching the router |
| **FastAPI for the gateway** | Async by default (matches httpx's async client); automatic OpenAPI docs at `/docs`; Pydantic models enforce the request/response schema |
| **httpx over the openai SDK** | Fine-grained timeout control per call; no need for a real API key object; lighter dependency |
| **Model aliases in LiteLLM** | Decouples routing logic from model version strings; upgrading a model = one YAML change |
| **Fallback to small, not retry large** | A timed-out large model is likely still slow; retrying doubles user wait time for no benefit |
| **Structured JSON response with metadata** | Observability without a full APM stack; callers can log routing decisions and detect degraded mode |