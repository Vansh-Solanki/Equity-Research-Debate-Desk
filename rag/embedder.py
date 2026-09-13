"""Local, free text embeddings via sentence-transformers — no API key, no per-call cost."""

from functools import lru_cache

from huggingface_hub.utils import logging as hf_logging
from sentence_transformers import SentenceTransformer

# Downloading all-MiniLM-L6-v2 hits the HF Hub anonymously (no token needed for a
# public model), which makes the Hub respond with an X-HF-Warning nudging you to
# log in — harmless and not actionable here, so silence huggingface_hub's own
# logger rather than leaving it to print on every fresh cache / first run.
hf_logging.set_verbosity_error()

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
