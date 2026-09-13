"""Combines chunker + embedder + chroma_store + reranker into two calls:
index_company_filing() to populate the store, retrieve() to query it.
"""

from mcp_server.tools import get_filing as _get_filing
from rag.chroma_store import delete_filing_chunks, index_lock, query_collection, upsert_chunks
from rag.chunker import chunk_filing
from rag.embedder import embed_query, embed_texts
from rag.reranker import rerank

DEFAULT_CANDIDATE_K = 10
DEFAULT_TOP_K = 3


def index_company_filing(company: str, filing_type: str = "10-K") -> dict:
    """Fetches a company's filing (via the Phase 2 MCP tool), chunks it, embeds the
    chunks, and stores them in that company's Chroma collection.

    Returns {"success": bool, "chunks_indexed": int, "error": str | None}.
    """
    result = _get_filing(company, filing_type)
    if not result["success"]:
        return {"success": False, "chunks_indexed": 0, "error": result["error"]}

    records = chunk_filing(result["data"])
    if not records:
        return {"success": False, "chunks_indexed": 0, "error": "filing produced no chunks"}

    # Embed (slow — runs a local model) before touching the store at all: if this
    # process crashes or is interrupted mid-embed, the old index is untouched and
    # still usable. Only the delete+upsert pair below is the actual crash window,
    # and index_lock keeps a second process from interleaving its own delete/upsert
    # into that same window.
    embeddings = embed_texts([r["text"] for r in records])
    with index_lock(company, filing_type):
        # Clear every section of any prior index for this filing first (section=None) —
        # chunk_filing now section-splits internally, so a re-index's section labels and
        # chunk counts can differ run to run (e.g. after a chunker/splitter fix); old
        # labels have no fixed correspondence to new ones for a scoped delete to target.
        delete_filing_chunks(company, filing_type, section=None)
        count = upsert_chunks(company, records, embeddings)
    return {"success": True, "chunks_indexed": count, "error": None}


def retrieve(
    query: str,
    company: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    where: dict | None = None,
) -> list[dict]:
    """Retrieves the top_k chunks for `query` from `company`'s indexed filing:
    Chroma vector search narrows to candidate_k, then a cross-encoder reranks those
    down to top_k. Returns [] if nothing has been indexed for this company yet.

    `where` is forwarded to Chroma's metadata filter (e.g. {"section": "Item 1A"})
    — Phase 7's deep-dive spawner uses this to scope retrieval to one filing
    section's chunks only.

    Each result dict has: id, text, metadata, distance (vector search), rerank_score.
    """
    query_embedding = embed_query(query)
    candidates = query_collection(company, query_embedding, n_results=candidate_k, where=where)
    return rerank(query, candidates, top_k=top_k)
