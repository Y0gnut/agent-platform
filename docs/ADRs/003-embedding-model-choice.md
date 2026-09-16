# ADR 003 — Embedding Model Choice

**Date:** 2026-09  
**Status:** Accepted  
**Deciders:** Project author

---

## Context

The retrieval pipeline (RAG) needs to embed document chunks at ingest time and query strings at query time, then find the nearest neighbours in ChromaDB. The embedding model determines retrieval quality, inference speed, and infrastructure cost.

Options considered:

| Model | Size | Quality | Cost | Requires |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | ~90 MB | Good for English text | Free | Local CPU |
| `text-embedding-ada-002` (OpenAI) | Remote | Excellent | ~$0.10/1M tokens | API key, internet |
| `all-mpnet-base-v2` | ~420 MB | Better than MiniLM | Free | More RAM |
| `nomic-embed-text` (Ollama) | ~270 MB | Competitive | Free | Ollama running |

---

## Decision

Use **`all-MiniLM-L6-v2`** from the `sentence-transformers` library.

- ~90 MB download, fits in CPU memory on any laptop.
- ~5,000 sentences/second on CPU (fast enough that ingest is not the bottleneck).
- 512-token context window — sufficient for our 350-word chunks with headroom.
- No API key, no network dependency, fully reproducible.
- Must be the **same model** at ingest time (`scripts/ingest.py`), retrieval time (`retrieval_server.py`), and eval time (`mlops/run_eval.py`). Using different models produces incompatible embedding spaces; enforced by a shared constant `EMBEDDING_MODEL_NAME`.

---

## Consequences

**Pros:**
- Zero ongoing cost, works offline.
- Identical model across ingest, retrieval, and eval removes a class of subtle bugs (cross-model embedding drift).
- Well-documented, widely used — easy to explain in an interview.

**Cons / Honest Limitations:**
- MiniLM-L6-v2 lags behind larger models (e.g. `text-embedding-3-large`, `nomic-embed-text`) on standard retrieval benchmarks by 5-15% NDCG. For a 24-question gold set over FastAPI docs, the difference is not measurable in practice.
- 512-token limit means chunks must be carefully sized. Our 350-word target stays within this, but documents with very long paragraphs may be truncated mid-sentence at chunk boundaries.
- The model is English-only and optimised for semantic similarity, not exact keyword match. Queries using API names or exact code snippets may retrieve less relevant results than BM25 would.

**What we'd do at scale:** Run an offline retrieval benchmark (e.g. BEIR) on a sample of the corpus to choose between MiniLM, `all-mpnet-base-v2`, and a proprietary embedding API. Use a hybrid retrieval approach (dense + BM25) for queries that mix semantic intent with exact identifiers.
