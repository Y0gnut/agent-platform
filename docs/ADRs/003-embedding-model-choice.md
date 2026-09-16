# ADR 003 - Embedding Model Choice

**Date:** 2026-09  
**Status:** Accepted  
**Deciders:** Architecture Team

---

## Context

The retrieval pipeline (RAG) embeds document chunks during ingestion and query strings during retrieval, matching nearest neighbors within ChromaDB. The selected embedding model dictates retrieval fidelity, latency, and memory footprint.

Options evaluated:

| Model | Size | Quality | Cost | Runtime Requirements |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | ~90 MB | Balanced for English text | Free | Local CPU |
| `text-embedding-ada-002` | Cloud | High | Variable | External API & network |
| `all-mpnet-base-v2` | ~420 MB | Enhanced semantic accuracy | Free | Higher RAM allocation |
| `nomic-embed-text` | ~270 MB | Competitive | Free | Ollama runtime |

---

## Decision

Use **`all-MiniLM-L6-v2`** via the `sentence-transformers` library.

- Compact ~90 MB footprint; operates efficiently on local CPU resources.
- High throughput on CPU architectures (~5,000 sentences/second).
- 512-token context window accommodates target document chunk boundaries (350 words).
- Zero external API dependencies, ensuring reliable offline execution.
- Consistency across pipeline stages: the exact model weights must be used at ingestion (`scripts/ingest.py`), retrieval (`retrieval_server.py`), and evaluation (`mlops/run_eval.py`). Enforced through the shared `EMBEDDING_MODEL_NAME` configuration constant.

---

## Consequences

**Advantages:**
- Zero cloud operational costs and full offline autonomy.
- Guaranteed vector compatibility across ingest, retrieval, and evaluation steps.
- Industry-standard baseline with extensive benchmarks and community validation.

**Limitations:**
- Semantic accuracy trails larger frontier models (such as `text-embedding-3-large`) by 5-15% NDCG on dense retrieval benchmarks.
- 512-token context constraint requires chunk size validation to avoid truncation.
- Purely semantic dense representations may perform suboptimally on exact code identifier matches compared to lexical search algorithms like BM25.

**Production Considerations:**
Implement hybrid search combining dense MiniLM embeddings with sparse BM25 indexing for queries requiring exact syntax or API keyword matches, with reranking applied prior to context generation.
