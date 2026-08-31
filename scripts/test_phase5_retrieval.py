"""Phase 5 acceptance check: retrieval for a known test question returns the
correct chunk in the top 3 results after reranking, for at least 8/10 test
questions.

No hand-labeled test set exists yet (evaluation/test_set.json is Phase 6's job,
for claim-level entailment). Until then, this builds a reproducible smoke test
from the filing itself: index a real filing, sample chunks spread across it, and
for each treat a distinctive sentence pulled from partway through that chunk as
the "question" — then check whether cross-encoder reranking correctly surfaces
that exact source chunk in the top 3 out of the company's full chunk set.

10-K risk-factor sections in particular repeat boilerplate legal language almost
verbatim across dozens of paragraphs (e.g. Apple's risk factors routinely close
with "...could materially adversely affect the Company's business, reputation,
results of operations, financial condition and stock price."). A random span can
land on that kind of genuinely-duplicated text, which no retriever can disambiguate
between instances — that's a property of the source document, not a retrieval bug.
A real hand-authored question wouldn't ask about undifferentiated boilerplate, so
this filters candidate snippets for "distinctiveness" (contains a number/date/
percentage or specific proper nouns) the same way a human question-writer would
naturally gravitate toward specific facts.

Usage:
    python -m scripts.test_phase5_retrieval [COMPANY]
"""

import re
import sys

from dotenv import load_dotenv

from rag.chroma_store import get_or_create_collection
from rag.retriever import index_company_filing, retrieve

DEFAULT_COMPANY = "AAPL"
NUM_TEST_QUESTIONS = 10
SNIPPET_WORDS = 20


def _looks_distinctive(snippet: str) -> bool:
    has_digit = bool(re.search(r"\d", snippet))
    # Capitalized words after the first (proper nouns, defined terms) — not just
    # ordinary sentence-initial capitalization.
    words = snippet.split()
    proper_nouns = sum(1 for w in words[1:] if w[:1].isupper())
    return has_digit or proper_nouns >= 2


def _sample_test_questions(company: str, n: int) -> list[dict]:
    """Walks chunks spread across the indexed filing and turns a distinctive
    sentence from partway through each into a "question" whose known-correct
    chunk id we keep, skipping chunks whose sampled span is generic boilerplate."""
    collection = get_or_create_collection(company)
    all_chunks = collection.get(include=["documents"])
    ids, docs = all_chunks["ids"], all_chunks["documents"]

    # Only consider chunks with enough text to pull a distinctive mid-chunk sentence
    # from (skips near-empty trailing chunks).
    usable = [(i, d) for i, d in zip(ids, docs) if len(d) > 300]
    if len(usable) < n:
        raise RuntimeError(f"only {len(usable)} usable chunks indexed, need at least {n}")

    # Split the filing into n buckets by position (preserving spread across the
    # whole document) and, within each bucket, take the first chunk whose sampled
    # span is distinctive rather than whatever chunk happens to land on the bucket
    # boundary.
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


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()

    company = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COMPANY

    print(f"Indexing {company}'s 10-K...")
    index_result = index_company_filing(company)
    if not index_result["success"]:
        print(f"FAILED to index: {index_result['error']}")
        return 1
    print(f"Indexed {index_result['chunks_indexed']} chunks.\n")

    questions = _sample_test_questions(company, NUM_TEST_QUESTIONS)

    passed = 0
    for i, q in enumerate(questions, start=1):
        results = retrieve(q["query"], company, top_k=3, candidate_k=15)
        top_ids = [r["id"] for r in results]
        hit = q["expected_chunk_id"] in top_ids
        passed += hit
        status = "PASS" if hit else "FAIL"
        print(f"[{status}] Q{i}: {q['query'][:70]!r}")
        print(f"       expected chunk in top 3: {hit} (top ids: {[i[-8:] for i in top_ids]})")

    print(f"\n{passed}/{len(questions)} passed (need >= 8/10 for acceptance)")
    return 0 if passed >= 8 else 1


if __name__ == "__main__":
    sys.exit(main())
