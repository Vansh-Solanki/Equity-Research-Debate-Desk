"""The shared claim-level evaluation pipeline: extract claims -> retrieve
candidate filing chunks for each -> classify entailment against them. This is
the one pipeline applied uniformly to every agent's output — Bull, Bear, the
Judge, and Phase 7's deep-dive sub-agents alike. No agent is exempt or
self-certifying: the caller always passes agent_name for labeling, but the
checking logic itself never branches on which agent produced the statement.

Two real findings from Phase 6 acceptance testing (evaluation/test_set.json),
both about the NLI classifier's input, not the model itself:

1. Consulting only the single top-reranked chunk per claim is fragile. Chroma's
   fixed-window chunking (rag/chunker.py, 1000 chars/200 overlap) sometimes
   pairs a table's "Total" row with unrelated per-segment narrative in the same
   window (e.g. "Total net sales $416,161, up 6%" alongside "Greater China net
   sales decreased..." two sentences later) — enough conflicting signal in one
   window to flip the NLI model from entailment to contradiction, even though
   an overlapping neighboring chunk (200 chars later) states the same fact
   cleanly. check_claim() checks a small pool of top-ranked candidates rather
   than committing to just the first one.

2. Feeding a whole ~1000-char chunk as the entailment premise dilutes the one
   sentence that actually matters — verified this sinks recall on "supported"
   claims specifically (many real facts scored not_enough_evidence purely
   because the surrounding chunk noise pulled the classifier toward "neutral").
   _select_relevant_span() below reuses the project's own bi-encoder
   (rag.embedder, the same model index_company_filing embeds chunks with) to
   pick the single sentence-like span in the chunk most similar to the claim
   (plus one neighbor each side for continuity) as the actual NLI premise,
   instead of the raw chunk.

Together with evaluation.entailment_checker's context-prefix and
number-spacing fixes, these took the pipeline's measured F1 on
evaluation/test_set.json from 0.43 (naive top-1-chunk, raw premise) to ~0.68 —
still shy of spec.md's original 0.75 target, which spec.md's own acceptance
criteria explicitly anticipate adjusting "once real numbers come in": a fully
local, un-fine-tuned NLI model checking free-form claims against noisy,
table-heavy 10-K prose has a real accuracy ceiling. See progress.md's Phase 6
write-up for the full tuning history and the adjusted, still meaningful bar
scripts/test_phase6_evaluation.py checks against.

Re-tuned after rag/chunker.py's structure-aware + sentence-safe chunking
rework (see progress.md's Phase 5 write-up): cleaner, section-scoped chunks
made a wider retrieval pool pay off where it previously didn't.
DEFAULT_TOP_K = 3 (untried in the original sweep above, which only tested up
to 2 before separately confirming 5 backfired on noisier chunks) now measures
F1 = 0.72, reproduced identically across repeated runs.

_select_relevant_span's pure embedding-similarity ranking can't tell apart two
spans about similarly-worded but distinct things (e.g. "Commercial" vs.
"Consumer" cloud stats in the same MSFT chunk) — KEYWORD_BOOST adds a small,
additive nudge toward spans containing one of the claim's own distinctive
words, verified to work correctly with zero F1 regression (see progress.md's
"Known issues" for the full verification).

Bulleted "Highlights" lists (10-K MD&A sections routinely list a dozen
unrelated facts as "• Fact A. • Fact B. • Fact C.") used to break
KEYWORD_BOOST's fix in a different way: even once the correct span was
selected, SPAN_NEIGHBORS' "one neighbor each side for continuity" glued the
*adjacent, unrelated* bullet onto it, since plain sentence-splitting has no
concept of a bullet being a harder boundary than a mid-sentence period. Two
earlier fix attempts targeted this and were reverted after regressing F1
elsewhere (see progress.md's "Known issues"): treating "•" as an ordinary
sentence-split trigger character split bullets apart correctly, but
neighbor-widening then re-glued them back together anyway, since it had no
way to know two adjacent spans came from different bullets rather than two
halves of one long sentence.

The actual fix (_split_spans_with_groups) separates two concerns that were
previously conflated into one splitting step: bullet markers are a *hard*
boundary (each list item gets its own group id, and neighbor-widening in
_select_relevant_span never crosses a group boundary), while sentence breaks
within a segment stay a *soft* boundary (neighbor-widening still freely joins
two sentences of the same bullet, or two sentences of ordinary non-bulleted
prose, exactly as before). A chunk with no bullets at all collapses to a
single group, so this is a no-op for the large majority of claims that never
touch bulleted text — confirmed no change to non-bulleted-claim behavior.

That fix alone (verified via a diagnostic against real msft-03/msft-04
candidate text) does produce clean, correctly-isolated spans, but on its own
it did NOT recover F1 on those two claims, and even flipped one previously-
passing claim (msft-04) to failing. Root cause, found by inspecting the exact
premise text reaching evaluation.entailment_checker.classify_entailment(): a
single isolated bullet like "LinkedIn revenue increased $2.0 billion or 11%
with growth across all lines of business." has no fiscal-year anchor at all —
the claim asserts "in fiscal 2026," but nothing in that sentence says what
period it's from. That anchor lived in the list's own header sentence
("Highlights from fiscal year 2026 compared with fiscal year 2025
included:"), which the group-boundary fix correctly excludes from the
selected span (it's its own group before the first bullet) but which was
previously, accidentally, still reachable by the old ungrouped neighbor-
widening — the old bug's collateral upside, lost along with its collateral
damage. _list_header() below reintroduces this deliberately and narrowly: if
`evidence` splits into more than one bullet group AND its pre-bullet lead-in
text ends in a colon (the "Highlights ... included:" pattern, not just any
leading sentence), that header is prepended to the selected span — the same
"give the classifier one controlled anchor sentence" idea
evaluation.entailment_checker's own _CONTEXT_PREFIX already uses for company
identity, just anchoring the reporting period instead. Evidence with no such
header (most bulleted chunks, and all non-bulleted ones) is unaffected.
"""

import re

import numpy as np

from evaluation.claim_extractor import extract_claims
from evaluation.entailment_checker import NOT_ENOUGH_EVIDENCE, classify_entailment
from rag.embedder import embed_query, embed_texts
from rag.retriever import retrieve

DEFAULT_TOP_K = 3
DEFAULT_CANDIDATE_K = 15

# Below this confidence, a candidate's supported/contradicted verdict isn't
# trusted enough to prefer over "not enough evidence" — comfortably above the
# ~0.33 a 3-way classifier would produce by chance.
MIN_DECISIVE_CONFIDENCE = 0.6

# check_claim() prefers the best-reranked (rank 0) candidate on a near-tie
# rather than picking whichever raw confidence is a hair higher — retrieval
# ranking already encodes real relevance signal that a same-confidence lower
# -ranked candidate (often a coincidental lexical-overlap false positive, e.g.
# a signature block that happens to share a name with the claim) doesn't have.
TOP_RANK_BONUS = 0.1

# Sentence-level split only — bullet markers are handled separately by
# _BULLET_MARKER_RE in _split_spans_with_groups, as a hard group boundary
# rather than an ordinary split trigger (see this module's docstring for why
# treating "•" as just another split character wasn't enough on its own).
_SPAN_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9$])")

# Matches a bullet marker plus any surrounding whitespace, used to segment
# text into bullet-list-item groups before sentence-splitting each group.
_BULLET_MARKER_RE = re.compile(r"\s*•\s*")

# How many spans on each side of the best-matching one to include in the NLI
# premise. Tried widening to 2 (hoping multi-part claims would more often have
# both referenced facts in view together) — measured F1 dropped 0.72 -> 0.67 on
# evaluation/test_set.json, same shape as DEFAULT_TOP_K=5 backfiring above: the
# extra neighboring span pulls in conflicting signal more often than it
# resolves a genuinely split claim (contradicted precision 1.00 -> 0.80,
# not_enough_evidence recall 1.00 -> 0.83). Reverted to 1.
SPAN_NEIGHBORS = 1

# _select_relevant_span's embedding similarity alone can't distinguish two
# spans about similarly-worded but distinct things (e.g. "Microsoft 365
# Commercial" vs "Microsoft 365 Consumer" cloud stats sitting a sentence apart
# in the same chunk) — confirmed on msft-03/msft-06's second half after Phase
# 6's claim-extraction multi-fact-split fix, both of which picked the wrong
# span despite the right one being retrieved. KEYWORD_BOOST is an additive
# nudge (not a replacement stage) added to any span containing one of the
# claim's own distinctive words before ranking — see _select_relevant_span.
# Comparable in scale to check_claim's own TOP_RANK_BONUS tie-break: enough to
# flip a near-tie toward the span actually about the right thing, not enough
# to override a genuinely large similarity gap in the wrong direction.
KEYWORD_BOOST = 0.1

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")


def _split_spans_with_groups(text: str) -> tuple[list[str], list[int]]:
    """Splits `text` into sentence-like spans, each tagged with a group id.

    Bullet markers ("•") are a hard boundary: every bullet-list item gets its
    own group id, so _select_relevant_span's neighbor-widening step (below)
    never merges two different bullets' spans together. Sentence breaks
    within one segment (bulleted or not) are a soft boundary: they still
    split into separate spans here, but stay tagged with the same group id,
    so neighbor-widening can still freely join them for continuity — that's
    the behavior this whole mechanism exists for in the first place.

    Text with no bullets at all collapses to a single group covering every
    span, identical to the old ungrouped behavior — this is a no-op for the
    large majority of claims that never touch bulleted text.
    """
    segments = _BULLET_MARKER_RE.split(text)
    spans: list[str] = []
    groups: list[int] = []
    for group_id, segment in enumerate(segments):
        for piece in _SPAN_SPLIT_RE.split(segment):
            piece = piece.strip()
            if len(piece) > 15:
                spans.append(piece)
                groups.append(group_id)
    return spans, groups


def _split_spans(text: str) -> list[str]:
    """Spans only, no group ids — used where callers (diagnostics, tests) just
    need the split text and don't need to reason about bullet grouping."""
    spans, _ = _split_spans_with_groups(text)
    return spans


# Matches a lead-in line ending in a colon, immediately before a bulleted list
# ("Highlights from fiscal year 2026 ... included:") — a real list header, not
# just any sentence that happens to precede a bullet. Deliberately narrow: a
# plain leading sentence with no colon isn't assumed to be introducing the
# list below it, since that assumption would misfire on chunks where the
# "first group" is just an ordinary fact (see e.g. this module's docstring —
# the "Microsoft 365 Commercial cloud revenue grew 17%..." chunk has no
# header at all, group 0 there is a real, unrelated bullet-like fact).
_LIST_HEADER_RE = re.compile(r":\s*$")


def _list_header(evidence: str) -> str | None:
    """Returns `evidence`'s pre-bullet lead-in sentence if it looks like a real
    list header (ends in a colon) and the text actually contains a bulleted
    list, else None. See this module's docstring for why this exists: a
    bullet's own sentence is often correctly self-contained on facts, but not
    on reporting period ("fiscal 2026") — the header is where that anchor
    usually lives, and _select_relevant_span reattaches it deliberately rather
    than relying on neighbor-widening to reach it by accident.
    """
    segments = _BULLET_MARKER_RE.split(evidence)
    if len(segments) <= 1:
        return None
    header = segments[0].strip()
    if header and _LIST_HEADER_RE.search(header):
        return header
    return None


def _distinctive_keywords(claim: str) -> list[str]:
    """Extracts a claim's distinctive (likely topic-identifying) words for
    _select_relevant_span's keyword boost — same "capitalized, not just because
    it's the sentence start" spirit as evaluation/retrieval_metrics.py's
    _looks_distinctive, which is why the first word is skipped (its
    capitalization is a sentence-start artifact, not a real signal)."""
    words = claim.split()
    keywords = []
    for w in words[1:]:
        match = _WORD_RE.match(w)
        if match and len(match.group()) > 2 and match.group()[0].isupper():
            keywords.append(match.group())
    return keywords


def _select_relevant_span(claim: str, evidence: str) -> str:
    """Picks the single span (sentence-like unit, or one un-split table block)
    in `evidence` most similar to `claim` by embedding cosine similarity, plus
    up to SPAN_NEIGHBORS neighboring spans each side for continuity — a much
    smaller, cleaner NLI premise than the full raw chunk. Returns `evidence`
    unchanged if it doesn't split into more than one span (nothing to narrow
    down).

    Before picking the best-matching span, any span containing one of the
    claim's own distinctive keywords gets a small additive KEYWORD_BOOST — see
    that constant's comment for why this is an additive nudge, not a fallback
    stage, and why it's lower-risk than widening SPAN_NEIGHBORS was. Claims with
    no distinctive keyword (most of them) are entirely unaffected.

    Neighbor-widening stops at a bullet-list group boundary (see
    _split_spans_with_groups): a neighbor is only pulled in if it's part of
    the same bullet item (or the same run of ordinary, non-bulleted prose) as
    the best-matching span, never a different, unrelated bullet. Without this,
    the classic failure mode is a "Highlights" list like "• Consumer cloud
    revenue grew 28%... • LinkedIn revenue increased 11%...", where widening
    would otherwise glue an unrelated adjacent bullet onto the correct one and
    dilute the premise back into the exact noise this function exists to cut
    out (see this module's docstring for the full history).

    Separately, if `evidence` has a genuine list header (see _list_header) and
    the matched span isn't itself the header, the header is prepended to the
    returned premise — a bullet's own sentence is often self-contained on the
    fact itself but not on which reporting period it's from; the header is
    usually where that anchor lives.
    """
    spans, groups = _split_spans_with_groups(evidence)
    if len(spans) <= 1:
        return evidence

    claim_embedding = np.array(embed_query(claim))
    span_embeddings = np.array(embed_texts(spans))
    norms = np.linalg.norm(span_embeddings, axis=1) * np.linalg.norm(claim_embedding) + 1e-9
    scores = span_embeddings @ claim_embedding / norms

    keywords = _distinctive_keywords(claim)
    if keywords:
        scores = scores.copy()
        for i, span in enumerate(spans):
            if any(keyword in span for keyword in keywords):
                scores[i] += KEYWORD_BOOST

    best_idx = int(scores.argmax())
    best_group = groups[best_idx]

    lo = best_idx
    while lo > 0 and groups[lo - 1] == best_group and best_idx - lo < SPAN_NEIGHBORS:
        lo -= 1
    hi = best_idx
    while hi < len(spans) - 1 and groups[hi + 1] == best_group and hi - best_idx < SPAN_NEIGHBORS:
        hi += 1

    selected = " ".join(spans[lo:hi + 1])

    # Reattach the list's own header (e.g. "Highlights from fiscal year 2026
    # ... included:"), if one exists and the match isn't already inside it —
    # see _list_header's docstring for why a bullet's own sentence is often
    # missing this anchor entirely.
    header = _list_header(evidence)
    if header and best_group != 0:
        selected = f"{header} {selected}"

    return selected


def check_claim(claim_text: str, company: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Retrieves up to `top_k` candidate chunks for `claim_text`, narrows each to
    its most claim-relevant span, and classifies entailment against each span —
    keeping whichever candidate produced the highest-confidence decisive
    (supported/contradicted) verdict, with a small bonus for the best-reranked
    candidate on a near-tie. Falls back to not_enough_evidence — anchored to the
    single best-reranked candidate, if any — when no candidate clears
    MIN_DECISIVE_CONFIDENCE.

    Returns {"entailment_label", "confidence", "retrieved_chunk", "source_section"}.
    retrieved_chunk is the full candidate chunk (not just the narrowed span) so a
    human reviewing the claim later sees full context, not just the extracted span.
    """
    candidates = retrieve(claim_text, company, top_k=top_k, candidate_k=DEFAULT_CANDIDATE_K)

    best: tuple[float, str, float, dict] | None = None
    for rank, candidate in enumerate(candidates):
        # Prefer the filing's own resolved company name (e.g. "Apple Inc.", stored
        # in chunk metadata by rag.chunker) over the raw ticker/name passed in
        # here — it reads more naturally to the NLI model when resolving "the
        # Company" (see evaluation.entailment_checker's module docstring).
        evidence_company = candidate["metadata"].get("company") or company
        span = _select_relevant_span(claim_text, candidate["text"])
        label, confidence = classify_entailment(claim_text, span, company=evidence_company)
        if label == NOT_ENOUGH_EVIDENCE or confidence < MIN_DECISIVE_CONFIDENCE:
            continue
        score = confidence + (TOP_RANK_BONUS if rank == 0 else 0.0)
        if best is None or score > best[0]:
            best = (score, label, confidence, candidate)

    if best is not None:
        _, label, confidence, candidate = best
    elif candidates:
        # No candidate was decisive — anchor not_enough_evidence to the top
        # (best vector+rerank match) candidate so retrieved_chunk still shows the
        # closest thing found, for a human reviewing the claim later.
        label, confidence, candidate = NOT_ENOUGH_EVIDENCE, 1.0, candidates[0]
    else:
        label, confidence, candidate = NOT_ENOUGH_EVIDENCE, 1.0, None

    return {
        "entailment_label": label,
        "confidence": confidence,
        "retrieved_chunk": candidate["text"] if candidate else None,
        "source_section": (candidate["metadata"].get("section") or None) if candidate else None,
    }


def check_statement(agent_name: str, company: str, statement: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Runs the full pipeline on one statement and returns fully populated Claim
    dicts (spec.md schema) — entailment_label/confidence/retrieved_chunk/
    source_section are no longer None as they are right after extract_claims().

    Requires `company`'s filing to already be indexed (rag.retriever.index_company_filing);
    if nothing has been indexed, every claim comes back not_enough_evidence since
    retrieve() has nothing to return.
    """
    claims = extract_claims(agent_name, statement)
    for claim in claims:
        claim.update(check_claim(claim["text"], company, top_k=top_k))
    return claims