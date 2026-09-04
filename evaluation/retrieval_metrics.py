"""Retrieval-quality metrics: recall@k for a company's already-indexed filing,
computed before vs after cross-encoder reranking.

Question sampling here is the same reproducible-smoke-test approach
scripts/test_phase5_retrieval.py used for Phase 5's acceptance check (moved here,
not duplicated, so Phase 8's dashboard and that script share one implementation):
sample chunks spread across the indexed filing, and for each treat a distinctive
sentence pulled from partway through it as the "question" whose known-correct
chunk id we keep. Distinctiveness filtering (a digit or >= 2 proper-noun-like
capitalized words) avoids landing on genuinely-duplicated 10-K boilerplate
(e.g. risk-factor paragraphs that close with the identical sentence dozens of
times) that no retriever could disambiguate between instances of.
"""

import re

from rag.chroma_store import get_or_create_collection, query_collection
from rag.embedder import embed_query
from rag.reranker import rerank

SNIPPET_WORDS = 20


def _looks_distinctive(snippet: str) -> bool:
    has_digit = bool(re.search(r"\d", snippet))
    words = snippet.split()
    proper_nouns = sum(1 for w in words[1:] if w[:1].isupper())
    return has_digit or proper_nouns >= 2


def sample_test_questions(company: str, n: int) -> list[dict]:
    """Walks chunks spread across `company`'s indexed filing and turns a
    distinctive sentence from partway through each into a "question" whose
    known-correct chunk id we keep, skipping chunks whose sampled span is
    generic boilerplate. Raises RuntimeError if fewer than `n` usable/distinctive
    chunks are found — same as scripts/test_phase5_retrieval.py's original
    behavior."""
    collection = get_or_create_collection(company)
    all_chunks = collection.get(include=["documents"])
    ids, docs = all_chunks["ids"], all_chunks["documents"]

    usable = [(i, d) for i, d in zip(ids, docs) if len(d) > 300]
    if len(usable) < n:
        raise RuntimeError(f"only {len(usable)} usable chunks indexed, need at least {n}")

    bucket_size = len(usable) // n
    questions = []
    for b in range(n):
        bucket = usable[b * bucket_size : (b + 1) * bucket_size] or [usable[-1]]
        for chunk_id, text in bucket:
            words = re.split(r"\s+", text[100:].strip())
            snippet = " ".join(words[:SNIPPET_WORDS])
            if _looks_distinctive(snippet):
                questions.append({"query": snippet, "expected_chunk_id": chunk_id})
                break

    if len(questions) < n:
        raise RuntimeError(f"only found {len(questions)} distinctive candidate questions out of {n} buckets")
    return questions


def recall_at_k(company: str, k: int = 3, n_questions: int = 10, candidate_k: int = 15) -> dict:
    """Measures recall@k for `company`'s indexed filing both before reranking
    (raw Chroma vector-similarity order) and after (cross-encoder reranked) —
    Phase 8's "retrieval recall@k (before vs after reranking)" metric. Reuses
    the same sampled questions for both, so the comparison isolates reranking's
    effect rather than sampling noise between two separate runs.

    Returns {"pre_rerank": float, "post_rerank": float, "n_questions": int}, each
    fraction being the share of questions whose known-correct chunk landed in
    the top k results.
    """
    questions = sample_test_questions(company, n_questions)

    pre_hits = 0
    post_hits = 0
    for q in questions:
        query_embedding = embed_query(q["query"])
        candidates = query_collection(company, query_embedding, n_results=candidate_k)

        pre_ids = [c["id"] for c in candidates[:k]]
        if q["expected_chunk_id"] in pre_ids:
            pre_hits += 1

        reranked = rerank(q["query"], candidates, top_k=k)
        post_ids = [r["id"] for r in reranked]
        if q["expected_chunk_id"] in post_ids:
            post_hits += 1

    n = len(questions)
    return {"pre_rerank": pre_hits / n, "post_rerank": post_hits / n, "n_questions": n}
