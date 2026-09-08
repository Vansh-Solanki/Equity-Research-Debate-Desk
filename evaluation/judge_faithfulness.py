"""Judge-faithfulness framework: is the Judge doing its job correctly?

progress.md's Sept 8 session scoped three separately-measured layers, since a
single groundedness score can't distinguish these failure modes from each
other:

1. Factual groundedness (evaluation/pipeline.py + evaluation/metrics.py,
   already built) — did the Judge state new true facts about the company?
2. Citation accuracy (this module's check_citations, built below) — did the
   Judge accurately describe what each side actually said and its real
   verification status? Checked against all_debate_claims (the real data
   debate_loop.py already computed and handed to the Judge before it wrote
   its memo), not against the filing — the filing can't confirm or deny a
   fact about "what happened in this debate run."
3. Verdict consistency (check_verdict_consistency, built first) — does
   stronger_side actually follow from the real per-side claim data, or does
   the Judge's conclusion contradict its own evidence?

Same "don't trust, verify" principle orchestration/debate_loop.py already
applies to the Judge's self-reported claims_checked/claims_unsupported
(computed programmatically, never taken from the LLM's own output) — both
layers below extend that same logic to the Judge's prose and its final
decision, not just its two summary numbers.

Layer 3 is pure arithmetic (no LLM call at all). Layer 2 needs one small LLM
call per citation sentence to pull out {side, topic, status} — an extraction
step, not a judgment step, same reasoning evaluation/claim_extractor.py uses
for calling litellm directly rather than wrapping a CrewAI Agent/Task/Crew for
a plain utility call. The actual verification (matching the extracted topic
back to a real claim, comparing status) is deterministic embedding similarity
+ string comparison, not a second LLM-as-judge call, consistent with how
Phase 5/6's retrieval and entailment checking and Judge-accuracy F1 are
measured — this project uses LLM-as-judge in exactly one place
(evaluation/engagement_scorer.py's debate-engagement rubric) and deliberately
avoids it everywhere else that checks correctness.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Literal

import litellm
import numpy as np

from agents.llm import GROQ_MODEL, run_with_rate_limit_backoff
from evaluation.metrics import groundedness_pct
from rag.embedder import embed_texts

# ===========================================================================
# Layer 3 — verdict consistency (built first, unchanged from the initial
# implementation; see progress.md's Sept 8 write-up for the synthetic-case
# verification and the live AAPL run that confirmed the threshold's behavior).
# ===========================================================================

# How many percentage points of support-rate gap between sides counts as a real
# disagreement worth flagging, vs. a close call the Judge could reasonably call
# either way. Picked from a real AAPL debate run (2026-09-08) where bull/bear
# support rates were close (within ~10 points) and the Judge's pick tracked the
# higher side anyway — 0.15 leaves room for calls like 55%/50% (a 5-point gap)
# to pass as consistent without flagging every marginal lean as a contradiction.
# Not swept against multiple runs yet; revisit if real debates start producing
# borderline cases right around this line.
DISAGREEMENT_THRESHOLD = 0.15

_SIDES = ("bull", "bear")


def check_verdict_consistency(stronger_side: str, all_debate_claims: list[dict]) -> dict:
    """Compares the Judge's stronger_side call against each side's actual measured
    support rate (fraction of that side's claims labeled "supported" by Phase 6's
    entailment pipeline — the same pipeline the Judge is handed the results of
    before writing its memo, so this checks whether its conclusion matches the
    evidence it was given, not evidence it never saw).

    Returns:
        {
            "judge_said": stronger_side, unchanged from the input,
            "evidence_favors": "bull" | "bear" | "tie",
            "bull_support_rate": float, "bear_support_rate": float,
            "bull_claim_count": int, "bear_claim_count": int,
            "gap": float,  # abs(bull_support_rate - bear_support_rate)
            "consistent": bool,
            "reason": str,  # human-readable explanation, always present
        }

    "tie" from the Judge is treated as consistent whenever the gap is inside
    DISAGREEMENT_THRESHOLD (a tie call on a close debate is reasonable) — the
    only judge_said value that can be inconsistent with a small gap is a
    confident bull/bear pick that contradicts the side with more support, and
    even that is only flagged once the gap clears the threshold, per the
    "55% vs 50% shouldn't be flagged" design note in progress.md.
    """
    by_side = {side: [c for c in all_debate_claims if c["agent"] == side] for side in _SIDES}
    bull_claims, bear_claims = by_side["bull"], by_side["bear"]

    if not bull_claims or not bear_claims:
        # Can't fairly compare support rates when one side has no checked claims
        # at all (e.g. an opening statement retrieved nothing) — nothing to
        # contradict, so this is not a faithfulness failure.
        return {
            "judge_said": stronger_side,
            "evidence_favors": None,
            "bull_support_rate": groundedness_pct(bull_claims),
            "bear_support_rate": groundedness_pct(bear_claims),
            "bull_claim_count": len(bull_claims),
            "bear_claim_count": len(bear_claims),
            "gap": None,
            "consistent": True,
            "reason": "insufficient data: at least one side had zero checked claims",
        }

    bull_rate = groundedness_pct(bull_claims)
    bear_rate = groundedness_pct(bear_claims)
    diff = bull_rate - bear_rate
    gap = abs(diff)

    if gap < DISAGREEMENT_THRESHOLD:
        evidence_favors = "tie"
    else:
        evidence_favors = "bull" if diff > 0 else "bear"

    consistent = evidence_favors == "tie" or stronger_side == evidence_favors

    if consistent:
        reason = (
            f"gap ({gap:.0%}) is within the {DISAGREEMENT_THRESHOLD:.0%} threshold, treated as a close call"
            if evidence_favors == "tie"
            else f"judge's pick ({stronger_side}) matches the better-supported side ({evidence_favors})"
        )
    else:
        reason = (
            f"judge picked {stronger_side}, but {evidence_favors} had the higher support rate "
            f"({bull_rate:.0%} bull vs {bear_rate:.0%} bear, a {gap:.0%} gap)"
        )

    return {
        "judge_said": stronger_side,
        "evidence_favors": evidence_favors,
        "bull_support_rate": bull_rate,
        "bear_support_rate": bear_rate,
        "bull_claim_count": len(bull_claims),
        "bear_claim_count": len(bear_claims),
        "gap": gap,
        "consistent": consistent,
        "reason": reason,
    }


# ===========================================================================
# Layer 2 — citation accuracy.
# ===========================================================================

CitationVerdict = Literal["accurate", "wrong_status", "fabricated", "ambiguous"]

# Cosine similarity floor to count as "a match" at all. Tuned against
# evaluation/judge_citation_test_set.json's real+synthetic examples (see
# scripts/test_judge_citation_accuracy.py) rather than left at an untested
# guess — all-MiniLM-L6-v2 similarity between a short extracted *topic* phrase
# and a full claim *sentence* runs lower than sentence-to-sentence similarity
# (what rag/reranker.py and evaluation/pipeline.py's span selection compare).
# First guess (0.75, a sentence-to-sentence-calibrated number) misclassified
# every real match as fabricated. Real genuine-match similarities measured on
# this test set ranged 0.34-0.82; the one deliberately-fabricated case scored
# 0.10 — a wide, clean gap. 0.25 sits in that gap with margin on both sides.
MATCH_THRESHOLD = 0.25

# If the top-1 and top-2 candidate similarities are within this margin, the
# match is too close to call and gets the "ambiguous" bucket instead of
# picking top-1 blind. Same test set's synthetic-008 case (two near-identical
# "margin pressure" claims) is the calibration anchor for this constant.
AMBIGUITY_MARGIN = 0.05

# Maps a Judge's own vocabulary for describing a citation's status to the set
# of real entailment_label values that count as a match. "unsupported" covers
# BOTH not_enough_evidence and contradicted deliberately — this mirrors
# agents/judge_agent.py's own claims_block prompt text ("X came back
# unsupported or lacking enough evidence"), which already treats those two
# labels as one bucket when describing them to the Judge in the first place.
# A citation naming a specific label ("contradicted", "not_enough_evidence")
# must match that label exactly — only the generic "unsupported" phrasing gets
# the wider bucket. Resolves the labeling-convention question progress.md and
# judge_citation_test_set.json's _readme both flagged as open.
STATUS_EQUIVALENCE: dict[str, set[str]] = {
    "supported": {"supported"},
    "unsupported": {"not_enough_evidence", "contradicted"},
    "contradicted": {"contradicted"},
    "not_enough_evidence": {"not_enough_evidence"},
}

# Sentences that describe the debate itself (what a side said, cited, or was
# found to have gotten right/wrong) rather than a new fact about the company.
# NOT the same pattern as evaluation/claim_extractor.py's meta-commentary
# rule — that rule lives entirely inside an LLM prompt instruction (see its
# _PROMPT's "Also skip any sentence that reports on..." rule), not a compiled
# regex, so there is no shared pattern to import. This regex is Layer 2's own,
# independently calibrated against real Judge memo text seen in this project's
# own test runs (see judge_citation_test_set.json's real, non-synthetic
# entries) — deliberately looser than a strict citation grammar, since a
# missed citation sentence just means one fewer thing checked, not a false
# claim about the debate.
CITATION_SENTENCE_PATTERN = re.compile(
    r"\b(bull|bear)('s)?\b.{0,150}?\b(flagged|unsupported|supported|corroborated|"
    r"contradicted|substantiated|unverified|hinges|cites|cited|argues)\b",
    re.IGNORECASE,
)


@dataclass
class CitationCheck:
    sentence: str
    side_claimed: str | None
    status_claimed: str | None
    matched_claim_id: str | None
    matched_claim_text: str | None
    matched_status: str | None
    similarity: float | None
    verdict: CitationVerdict


@dataclass
class CitationAccuracyResult:
    total_citations: int
    accurate: int
    wrong_status: int
    fabricated: int
    ambiguous: int
    accuracy_pct: float | None  # None means "N/A -- no citations to check"
    details: list[CitationCheck] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total_citations": self.total_citations,
            "accurate": self.accurate,
            "wrong_status": self.wrong_status,
            "fabricated": self.fabricated,
            "ambiguous": self.ambiguous,
            "accuracy_pct": (
                "N/A (no citations to check)" if self.accuracy_pct is None else round(self.accuracy_pct, 1)
            ),
        }


def _split_into_sentences(text: str) -> list[str]:
    """Cheap sentence split, good enough for scanning a short memo — does not
    need to share rag/chunker.py's sentence-boundary edge-case handling
    (abbreviations, etc.) since a missed split here just costs one skipped
    citation, not a corrupted chunk boundary fed to an embedder."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def find_citation_sentences(memo_text: str) -> list[str]:
    """Step 1: pull out every sentence describing the debate (a citation)
    rather than stating a new fact about the company."""
    sentences = _split_into_sentences(memo_text)
    return [s for s in sentences if CITATION_SENTENCE_PATTERN.search(s)]


_EXTRACTION_PROMPT = """Extract structured information from this sentence, which \
describes a claim made during a Bull vs Bear debate.

Sentence: "{sentence}"

Respond with ONLY a JSON object, no preamble, no markdown fences:
{{"side": "bull" or "bear" or null, "topic": "short phrase describing what the claim was about", "status": "supported" or "unsupported" or "contradicted" or "not_enough_evidence" or null}}

If the side or status genuinely cannot be determined from the sentence, use null for that field."""


def _default_llm_call(prompt: str) -> str:
    """Real Groq-backed extraction call, same pattern as
    evaluation/claim_extractor.py's litellm.completion() usage: called
    directly (not via a CrewAI Agent/Task/Crew, since this is a plain
    utility call with no role-play or tool use) so
    agents.llm.run_with_rate_limit_backoff's retry logic applies unchanged,
    with reasoning_effort="low" for the same reason claim_extractor.py needs
    it — GROQ_MODEL is a reasoning model that can otherwise burn its whole
    completion-token budget on hidden reasoning tokens and return empty
    content (see that module's docstring for the confirmed repro)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")

    def _call():
        return litellm.completion(
            model=f"groq/{GROQ_MODEL}",
            api_key=api_key,
            temperature=0.0,
            max_completion_tokens=300,
            reasoning_effort="low",
            messages=[{"role": "user", "content": prompt}],
        )

    response = run_with_rate_limit_backoff(_call, label="citation_extraction")
    return response["choices"][0]["message"]["content"] or ""


def extract_citation_details(sentence: str, llm_call=_default_llm_call) -> dict:
    """Step 2: small LLM call to pull {side, topic, status} out of one citation
    sentence. `llm_call` defaults to the real Groq-backed call above but stays
    injectable (signature: llm_call(prompt: str) -> str) so this can be unit
    tested without hitting the network."""
    raw = llm_call(_EXTRACTION_PROMPT.format(sentence=sentence)).strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"side": None, "topic": sentence, "status": None}
    return parsed


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def match_to_real_claim(
    topic: str,
    side_claimed: str | None,
    all_debate_claims: list[dict],
    embed_texts_fn=embed_texts,
) -> tuple[dict | None, float, bool]:
    """Step 3: embed the extracted topic and every candidate claim (filtered to
    the claimed side when known), return the best match, its similarity, and
    whether the top two candidates were too close to call (ambiguous).

    `embed_texts_fn` defaults to rag/embedder.py's real embed_texts (the same
    already-loaded all-MiniLM-L6-v2 model index_company_filing and
    evaluation/pipeline.py's span selection use) but stays injectable for
    testing.
    """
    candidates = [
        c
        for c in all_debate_claims
        if c.get("agent") in ("bull", "bear") and (side_claimed is None or c.get("agent") == side_claimed)
    ]
    if not candidates:
        return None, 0.0, False

    topic_vec = np.array(embed_texts_fn([topic])[0])
    candidate_vecs = np.array(embed_texts_fn([c["text"] for c in candidates]))

    sims = [_cosine_sim(topic_vec, cv) for cv in candidate_vecs]
    ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)

    best_claim, best_sim = ranked[0]
    if best_sim < MATCH_THRESHOLD:
        return None, best_sim, False

    is_ambiguous = len(ranked) > 1 and (best_sim - ranked[1][1]) < AMBIGUITY_MARGIN
    return best_claim, best_sim, is_ambiguous


def _status_matches(status_claimed: str | None, matched_status: str | None) -> bool:
    if status_claimed is None or matched_status is None:
        return False
    equivalent = STATUS_EQUIVALENCE.get(status_claimed.lower())
    return equivalent is not None and matched_status in equivalent


def check_citations(
    memo_text: str,
    all_debate_claims: list[dict],
    llm_call=_default_llm_call,
    embed_texts_fn=embed_texts,
) -> CitationAccuracyResult:
    """Top-level entry point: run Layer 2 citation-accuracy checking on one
    Judge memo. Mirrors the shape of evaluation/pipeline.py's check_statement,
    but checks against debate ground truth (all_debate_claims), not the
    filing. `llm_call`/`embed_texts_fn` default to this project's real Groq
    call and embedder but stay injectable for testing.
    """
    sentences = find_citation_sentences(memo_text)

    if not sentences:
        return CitationAccuracyResult(
            total_citations=0, accurate=0, wrong_status=0, fabricated=0, ambiguous=0, accuracy_pct=None, details=[]
        )

    details: list[CitationCheck] = []
    counts = {"accurate": 0, "wrong_status": 0, "fabricated": 0, "ambiguous": 0}

    for sentence in sentences:
        extracted = extract_citation_details(sentence, llm_call)
        side_claimed = extracted.get("side")
        status_claimed = extracted.get("status")
        topic = extracted.get("topic") or sentence

        matched_claim, similarity, is_ambiguous = match_to_real_claim(
            topic, side_claimed, all_debate_claims, embed_texts_fn
        )

        if matched_claim is None:
            verdict: CitationVerdict = "fabricated"
            matched_id = matched_text = matched_status = None
        elif is_ambiguous:
            verdict = "ambiguous"
            matched_id = matched_claim["claim_id"]
            matched_text = matched_claim["text"]
            matched_status = matched_claim.get("entailment_label")
        else:
            matched_id = matched_claim["claim_id"]
            matched_text = matched_claim["text"]
            matched_status = matched_claim.get("entailment_label")
            if status_claimed is None:
                # Side/topic matched a real claim, but extraction couldn't
                # tell what status the sentence claimed — can't confirm
                # accuracy either way, so this isn't a confident "accurate"
                # any more than it's a confident "wrong_status".
                verdict = "ambiguous"
            else:
                verdict = "accurate" if _status_matches(status_claimed, matched_status) else "wrong_status"

        counts[verdict] += 1
        details.append(
            CitationCheck(
                sentence=sentence,
                side_claimed=side_claimed,
                status_claimed=status_claimed,
                matched_claim_id=matched_id,
                matched_claim_text=matched_text,
                matched_status=matched_status,
                similarity=similarity,
                verdict=verdict,
            )
        )

    total = len(sentences)
    accuracy_pct = 100.0 * counts["accurate"] / total

    return CitationAccuracyResult(
        total_citations=total,
        accurate=counts["accurate"],
        wrong_status=counts["wrong_status"],
        fabricated=counts["fabricated"],
        ambiguous=counts["ambiguous"],
        accuracy_pct=accuracy_pct,
        details=details,
    )
