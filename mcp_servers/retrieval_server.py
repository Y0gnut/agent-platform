"""
mcp_servers/retrieval_server.py - Document Retrieval MCP Server
================================================================
Exposes the search_documents(query) tool to embed input queries and
retrieve relevant passages from a local ChromaDB collection.
"""

import logging
import pathlib
import sys

# Standard output is dedicated to MCP JSON-RPC protocol messages; log to stderr.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] retrieval_server | %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("retrieval_server")

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
CHROMA_DIR = str(REPO_ROOT / "chroma_db")
COLLECTION_NAME = "docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="retrieval-server")

# ── Module-level singletons ────────────────────────────────────────────────────
# Load model + DB once at startup, not on every tool call.
# Sentence-transformer model loading takes ~2s; ChromaDB client is near-instant.
_model = None
_collection = None


def _get_model():
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL_NAME)
        from sentence_transformers import SentenceTransformer
        try:
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model ready.")
    return _model


def _get_collection():
    """Lazy-load the ChromaDB collection."""
    global _collection
    if _collection is None:
        logger.info("Opening ChromaDB at '%s', collection '%s'...", CHROMA_DIR, COLLECTION_NAME)
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
        logger.info("ChromaDB collection ready (%d documents).", _collection.count())
    return _collection


@mcp.tool()
def search_documents(query: str) -> str:
    """
    Search indexed documentation for passages relevant to the query.

    Embeds the query with all-MiniLM-L6-v2 and returns the top 3 semantically
    similar passages from the local ChromaDB store.
    """
    logger.info("search_documents called | query=%r", query)

    try:
        model = _get_model()
        collection = _get_collection()
    except Exception as e:
        error_msg = (
            f"[ERROR] Could not load retrieval resources: {e}\n"
            "Ensure the database has been ingested: python scripts/ingest.py"
        )
        logger.error(error_msg)
        return error_msg

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return "No relevant documents found."

    parts = []
    for i, (doc_text, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        source = meta.get("source", "unknown")
        similarity = round(1.0 - dist, 3)
        parts.append(
            f"--- Result {i} (source: {source}, similarity: {similarity}) ---\n"
            f"{doc_text}"
        )

    formatted = "\n\n".join(parts)
    logger.info("Returning %d results for query=%r", len(docs), query)
    return formatted


if __name__ == "__main__":
    _get_model()
    _get_collection()

    logger.info("retrieval_server starting (stdio transport)...")
    mcp.run()
