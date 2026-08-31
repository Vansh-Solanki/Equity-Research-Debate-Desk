"""Local, free text embeddings via sentence-transformers — no API key, no per-call cost."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # Loading the model is the slow part (~seconds); cache it as a singleton so
    # repeated embed_texts() calls in the same process don't reload it.
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = _model().encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
