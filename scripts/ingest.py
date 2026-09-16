"""
scripts/ingest.py — Ingest FastAPI docs into ChromaDB
======================================================
Reads all markdown files from data/raw_docs/, chunks them into passages,
embeds them with a free local model, and persists everything to ./chroma_db/.

Run with:
    python scripts/ingest.py

Design decisions
----------------
Embedding model: all-MiniLM-L6-v2 (sentence-transformers)
  - Runs entirely locally — no API key, no internet call after first download
  - 90 MB model, ~5k sentences/sec on CPU — fast enough for a demo corpus
  - Good semantic quality for English documentation text

Chunk strategy: heading-based first, word-count fallback
  - Split on ## and # Markdown headings preserves semantic coherence:
    each chunk is usually one complete concept or example
  - If a section is very long (> MAX_WORDS), split further with a
    sliding window (STRIDE words overlap) to avoid cutting mid-sentence
  - ~300-400 words per chunk ≈ 400-500 tokens for MiniLM — fits well
    in the 512-token context window

Idempotency: delete-and-rebuild on each run
  - For this stage, rebuilding the collection is the safest approach.
    In production you would use upsert with deterministic IDs instead.
"""

import os
import sys
import re
import logging
import pathlib

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("ingest")

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
RAW_DOCS_DIR = REPO_ROOT / "data" / "raw_docs"
CHROMA_DIR = str(REPO_ROOT / "chroma_db")

# Chunking configuration:
# 350 words fits within the 512-token context window of all-MiniLM-L6-v2.
MAX_WORDS = 350
# 50 words stride ensures sentences spanning chunk boundaries are preserved.
STRIDE = 50

COLLECTION_NAME = "docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def split_into_sections(text: str, source: str) -> list[dict]:
    """
    Split a markdown document on heading boundaries (## or #).
    Returns a list of dicts: {"text": str, "source": str}.

    Why headings? Markdown headings mark semantic section boundaries.
    Splitting here keeps each chunk topically coherent, which improves
    retrieval precision compared to a purely mechanical character/token split.
    """
    # Split on lines that start with # or ## (not deeper headings)
    sections = re.split(r"\n(?=#{1,2} )", text)
    result = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        word_count = len(section.split())
        if word_count <= MAX_WORDS:
            result.append({"text": section, "source": source})
        else:
            # Section is too long — use a sliding window sub-split
            result.extend(sliding_window(section, source))
    return result


def sliding_window(text: str, source: str) -> list[dict]:
    """
    Split `text` into chunks of at most MAX_WORDS words, with STRIDE-word
    overlap between consecutive chunks.

    Overlap rationale: if a relevant sentence falls at the end of chunk N,
    it also appears at the start of chunk N+1. This doubles the chance that
    a semantic search returns a chunk where the sentence is *in context*,
    not dangling at a truncation boundary.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + MAX_WORDS, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({"text": chunk_text, "source": source})
        if end == len(words):
            break
        start += MAX_WORDS - STRIDE  # step forward, keeping STRIDE-word overlap
    return chunks


def load_documents(docs_dir: pathlib.Path) -> list[dict]:
    """Read all .md files and return a list of {text, source} dicts."""
    docs = []
    for md_file in sorted(docs_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8", errors="replace")
        docs.append({"text": text, "source": md_file.name})
    return docs


def main() -> None:
    if not RAW_DOCS_DIR.exists() or not any(RAW_DOCS_DIR.glob("*.md")):
        logger.error(
            "No markdown files found in %s\n"
            "Run: wsl bash scripts/fetch_docs.sh",
            RAW_DOCS_DIR,
        )
        sys.exit(1)

    logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL_NAME)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded.")

    logger.info("Connecting to ChromaDB at '%s'...", CHROMA_DIR)
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Recreate collection to ensure idempotency across ingestion runs
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing '%s' collection.", COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Created collection '%s'.", COLLECTION_NAME)

    raw_docs = load_documents(RAW_DOCS_DIR)
    logger.info("Loaded %d markdown files.", len(raw_docs))

    all_chunks: list[dict] = []
    for doc in raw_docs:
        chunks = split_into_sections(doc["text"], doc["source"])
        all_chunks.extend(chunks)

    logger.info("Generated %d chunks.", len(all_chunks))

    BATCH_SIZE = 64
    texts = [c["text"] for c in all_chunks]
    sources = [c["source"] for c in all_chunks]
    ids = [f"{sources[i]}::chunk_{i}" for i in range(len(all_chunks))]

    logger.info("Generating embeddings for %d chunks (batch_size=%d)...", len(texts), BATCH_SIZE)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    ).tolist()

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(texts))
        collection.add(
            ids=ids[batch_start:batch_end],
            embeddings=embeddings[batch_start:batch_end],
            documents=texts[batch_start:batch_end],
            metadatas=[{"source": s} for s in sources[batch_start:batch_end]],
        )

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Ingestion complete")
    print("=" * 60)
    print(f"  Files processed : {len(raw_docs)}")
    print(f"  Chunks created  : {len(all_chunks)}")
    print(f"  Collection name : {COLLECTION_NAME}")
    print(f"  Persisted to    : {CHROMA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
