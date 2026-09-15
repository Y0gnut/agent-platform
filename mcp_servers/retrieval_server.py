"""
mcp_servers/retrieval_server.py — Document Retrieval MCP Server
===============================================================
Exposes one MCP tool: search_documents(query) → top-3 relevant doc chunks.

The server:
  1. Loads the ChromaDB "docs" collection (built by scripts/ingest.py)
  2. Loads the same sentence-transformer model used during ingestion
  3. On each tool call: embeds the query, queries Chroma, returns results

Run standalone (for debugging):
    python mcp_servers/retrieval_server.py

Or let mcp_client.py spawn it as a subprocess (the normal usage pattern).

IMPORTANT: stdout is reserved for MCP JSON-RPC messages.
           All debug/logging output MUST go to stderr.
           Never use print() in a stdio MCP server.
"""

import sys
import logging
import pathlib

# ── Logging must go to stderr — stdout is MCP's wire protocol ─────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] retrieval_server | %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("retrieval_server")

# ── Path constants — must match scripts/ingest.py ─────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
CHROMA_DIR = str(REPO_ROOT / "chroma_db")
COLLECTION_NAME = "docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Top-k results to return per query.
# 3 is the sweet spot for a RAG context window:
#   - Enough to cover multi-part questions or synonym variations
#   - Small enough that the LLM can actually use all the context
#     without being overwhelmed by irrelevant passages
TOP_K = 3

# ── FastMCP server setup ───────────────────────────────────────────────────────
# FastMCP is the high-level interface in the official MCP Python SDK.
# It handles JSON-RPC framing, tool schema generation from type hints,
# and the stdio transport — so we only write application logic.
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="retrieval-server")

# ── Module-level singletons ────────────────────────────────────────────────────
# Load model + DB once at startup, not on every tool call.
# Sentence-transformer model loading takes ~2s; ChromaDB client is near-instant.
_model = None
_collection = None


def _get_model():
    """Lazy-load the embedding model (avoids import cost if not used)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s'…", EMBEDDING_MODEL_NAME)
        from sentence_transformers import SentenceTransformer
        try:
            # Use local cache first to avoid slow/hanging Hugging Face network requests
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model ready.")
    return _model


def _get_collection():
    """Lazy-load the ChromaDB collection."""
    global _collection
    if _collection is None:
        logger.info("Opening ChromaDB at '%s', collection '%s'…", CHROMA_DIR, COLLECTION_NAME)
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
        logger.info("ChromaDB collection ready (%d documents).", _collection.count())
    return _collection


# ── Tool definition ────────────────────────────────────────────────────────────

@mcp.tool()
def search_documents(query: str) -> str:
    """
    Search the FastAPI documentation for content relevant to a query.

    Embeds the query with all-MiniLM-L6-v2 and retrieves the top 3 most
    semantically similar passages from the pre-ingested Chroma collection.

    Args:
        query: A natural-language question or search phrase.

    Returns:
        A formatted string containing up to 3 result passages, each with
        its source filename. Returns an error message if the DB is not ready.
    """
    logger.info("search_documents called | query=%r", query)

    try:
        model = _get_model()
        collection = _get_collection()
    except Exception as e:
        error_msg = (
            f"[ERROR] Could not load retrieval resources: {e}\n"
            "Did you run: python scripts/ingest.py ?"
        )
        logger.error(error_msg)
        return error_msg

    # Embed the query using the same model used during ingestion.
    # Using the same model is CRITICAL — different models produce incompatible
    # embedding spaces, so cross-model queries return garbage results.
    # sentence-transformers v6: encode() returns numpy ndarray; call .tolist()
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB response (it wraps results in a list-of-lists for batch queries)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return "No relevant documents found."

    # Format results for the LLM to consume as context
    parts = []
    for i, (doc_text, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        source = meta.get("source", "unknown")
        # Cosine distance → similarity: similarity = 1 - distance
        similarity = round(1.0 - dist, 3)
        parts.append(
            f"--- Result {i} (source: {source}, similarity: {similarity}) ---\n"
            f"{doc_text}"
        )

    formatted = "\n\n".join(parts)
    logger.info("Returning %d results for query=%r", len(docs), query)
    return formatted


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-warm model and ChromaDB in the main thread before starting the MCP loop.
    # This prevents AnyIO worker thread deadlocks during PyTorch/tokenizers initialization.
    _get_model()
    _get_collection()

    # mcp.run() starts the stdio transport loop.
    # The server reads JSON-RPC from stdin and writes responses to stdout.
    # It blocks until the parent process closes the pipe.
    logger.info("retrieval_server starting (stdio transport)…")
    mcp.run()
