"""ChromaDB persistence layer, one collection per company.

Per-company collections (not one big collection with a company filter) keep each
company's chunks physically separate, which is enough isolation for Phase 5's
single-user testing. Phase 10's orchestration/session.py adds per-debate-session
isolation on top of this once concurrent users are a real concern.

chroma_data is a plain on-disk store (no built-in multi-writer protection), so
two processes re-indexing the same (company, filing_type) at once — e.g. the
Streamlit app mid-debate and a test script run from another terminal — could
interleave their delete/upsert calls. index_lock() is a cross-process file lock
callers use to serialize a re-index's delete+upsert critical section.
"""

import contextlib
import os

import chromadb
from filelock import FileLock

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
        _client = chromadb.PersistentClient(path=persist_dir)
    return _client


def _collection_name(company: str) -> str:
    # Chroma collection names must be alnum/underscore/hyphen; company can be a
    # ticker or free-text name, so normalize it.
    safe = "".join(c if c.isalnum() else "_" for c in company.upper())
    return f"filing_{safe}"


def get_or_create_collection(company: str):
    return _get_client().get_or_create_collection(name=_collection_name(company))


def _persist_dir() -> str:
    return os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")


@contextlib.contextmanager
def index_lock(company: str, filing_type: str):
    """Cross-process lock guarding a re-index's delete+upsert critical section for
    one (company, filing_type) — see this module's docstring. A second process
    (or thread) requesting the same lock blocks until the first releases it,
    rather than interleaving writes to the same on-disk collection."""
    persist_dir = _persist_dir()
    os.makedirs(persist_dir, exist_ok=True)
    lock_path = os.path.join(persist_dir, f".lock_{_collection_name(company)}_{filing_type}")
    with FileLock(lock_path):
        yield


def delete_filing_chunks(company: str, filing_type: str, section: str | None = "") -> None:
    """Deletes all previously-indexed chunks for this (company, filing_type), scoped
    to one `section` (default "", the pre-Phase-5-rework main index's blank label),
    or every section when `section` is None — needed since the main index now
    section-splits the filing itself (rag.chunker.chunk_filing), so a re-index's
    section labels/boundaries can differ run to run and old labels have no fixed
    correspondence to new ones to delete individually.

    chunk ids are deterministic (rag.chunker.chunk_filing), so upsert alone handles
    re-indexing an unchanged filing correctly — but if the chunk count or section
    layout changes between indexing runs, upsert only overwrites ids that still
    exist; the old chunk set's tail (or now-renamed sections) would otherwise stay
    behind as orphaned rows. Call this before upsert_chunks for a clean re-index.
    """
    collection = get_or_create_collection(company)
    if collection.count() == 0:
        return
    where = {"filing_type": filing_type} if section is None else {"$and": [{"filing_type": filing_type}, {"section": section}]}
    collection.delete(where=where)


def upsert_chunks(company: str, chunks: list[dict], embeddings: list[list[float]]) -> int:
    """Stores chunk records (as produced by rag.chunker.chunk_filing) with their
    embeddings. Upsert by id, so re-indexing with the same (and same-length) chunk
    ids overwrites rather than duplicates — call delete_filing_chunks first for a
    guaranteed-clean re-index regardless of chunk count changes."""
    if not chunks:
        return 0
    collection = get_or_create_collection(company)
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embeddings,
    )
    return len(chunks)


def query_collection(company: str, query_embedding: list[float], n_results: int = 10, where: dict | None = None) -> list[dict]:
    """Returns [{"id", "text", "metadata", "distance"}, ...] sorted by ascending distance."""
    collection = get_or_create_collection(company)
    if collection.count() == 0:
        return []
    n_results = min(n_results, collection.count())
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
    )
    if not result["ids"][0]:
        return []
    return [
        {"id": result["ids"][0][i], "text": result["documents"][0][i], "metadata": result["metadatas"][0][i], "distance": result["distances"][0][i]}
        for i in range(len(result["ids"][0]))
    ]
