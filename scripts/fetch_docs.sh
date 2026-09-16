#!/usr/bin/env bash
# scripts/fetch_docs.sh
#
# Fetches a representative subset of the official FastAPI documentation
# (markdown files only) and copies them into data/raw_docs/.
#
# Usage (from the repo root, via WSL on Windows):
#   wsl bash scripts/fetch_docs.sh
#
# Why shallow clone?
#   We only need the markdown text, not the full git history.
#   A --depth=1 clone is ~10x smaller and much faster.
#
# Why copy only top-level docs/en/docs/ files?
#   The full docs tree has 100+ files across many nested subfolders
#   (advanced/, deployment/, how-to/, reference/). For a demo RAG system,
#   ~20-25 core conceptual docs give representative coverage without
#   bloating the repo or overwhelming the embedding step.

set -euo pipefail

REPO_URL="https://github.com/tiangolo/fastapi.git"
CLONE_DIR="/tmp/fastapi_docs_clone_$$"   # $$ = PID, avoids collisions
OUTPUT_DIR="data/raw_docs"

# ── 1. Resolve the repo root (works whether you cd into scripts/ first or not)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_PATH="${REPO_ROOT}/${OUTPUT_DIR}"

echo "==> FastAPI docs fetcher"
echo "    Target: ${OUTPUT_PATH}"

# ── 2. Clean + recreate the output directory
rm -rf "${OUTPUT_PATH}"
mkdir -p "${OUTPUT_PATH}"

# ── 3. Shallow-clone FastAPI (only latest commit, no history)
echo "==> Cloning ${REPO_URL} (shallow)..."
git clone --depth=1 --quiet "${REPO_URL}" "${CLONE_DIR}"

DOCS_ROOT="${CLONE_DIR}/docs/en/docs"

# ── 4. Copy top-level .md files (index, main concepts)
echo "==> Copying top-level markdown files..."
find "${DOCS_ROOT}" -maxdepth 1 -name "*.md" | while read -r f; do
    cp "${f}" "${OUTPUT_PATH}/"
done

# ── 5. Copy one level of the tutorial/ subfolder (core how-to guides)
#       These are the most query-relevant docs for a RAG demo.
TUTORIAL_DIR="${DOCS_ROOT}/tutorial"
if [[ -d "${TUTORIAL_DIR}" ]]; then
    echo "==> Copying tutorial docs..."
    find "${TUTORIAL_DIR}" -maxdepth 1 -name "*.md" | head -20 | while read -r f; do
        # Prefix filename with "tutorial__" to avoid name collisions
        basename_file="$(basename "${f}")"
        cp "${f}" "${OUTPUT_PATH}/tutorial__${basename_file}"
    done
fi

# ── 6. Remove the clone (we only wanted the markdown text)
echo "==> Removing temporary clone..."
rm -rf "${CLONE_DIR}"

# Summary
FILE_COUNT=$(find "${OUTPUT_PATH}" -name "*.md" | wc -l)
echo ""
echo "Documentation sync complete. ${FILE_COUNT} markdown files written to ${OUTPUT_DIR}/"
echo ""
find "${OUTPUT_PATH}" -name "*.md" | sort | sed "s|${REPO_ROOT}/||"
