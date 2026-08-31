"""Cross-encoder reranking on top of Chroma's vector-similarity candidates.

Bi-encoder (embedder.py) similarity is fast but coarse; a cross-encoder scores the
(query, candidate) pair jointly, which is slower but meaningfully more precise —
run it only over the small candidate set Chroma already narrowed down to.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(MODEL_NAME)


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """Reranks `candidates` (dicts with a "text" key, e.g. from chroma_store.query_collection)
    by cross-encoder relevance to `query`. Returns the top_k, each with a "rerank_score" added,
    sorted descending by that score."""
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    scores = _model().predict(pairs)

    scored = [{**candidate, "rerank_score": float(score)} for candidate, score in zip(candidates, scores)]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]
