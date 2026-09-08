"""Claim extraction: breaks an agent's free-text statement into individual,
self-contained factual claims via one Groq LLM call.

Uses litellm directly rather than a CrewAI Agent/Task/Crew, this is a plain
utility call with no role-play, tools, or multi-turn state, and going through
litellm directly means agents.llm.run_with_rate_limit_backoff's retry logic
(built around litellm.exceptions.RateLimitError/BadRequestError) applies
unchanged without needing a Crew wrapper.

Real bug found during Phase 7 acceptance testing: `openai/gpt-oss-20b` (the
free-tier model this project uses, agents/llm.py's GROQ_MODEL) is a reasoning
model, it can spend its entire completion-token budget on hidden
"reasoning_tokens" before ever emitting the visible answer, especially on a
longer or more structurally complex input (e.g. a multi-sentence deep-dive
summary with several citation markers). Confirmed directly: extracting claims
from one real deep-dive summary returned `content=''` with
`finish_reason="length"` and `reasoning_tokens=798` out of an 800-token budget,
raising the budget to 1500 didn't help either (1498 reasoning tokens, still
empty). This silently produced zero claims for an obviously multi-claim
statement, with no error raised (an empty completion isn't a rate-limit or
malformed-request failure, so `run_with_rate_limit_backoff` had nothing to
retry). Fixed by passing `reasoning_effort="low"`, verified this drops
reasoning token usage to ~200 on the same input and reliably leaves the
requested claims in the visible answer, without needing a much larger (and
more rate-limit-expensive) token budget.

Second real bug, found while diagnosing Phase 8's Judge-groundedness swings
(0% -> 50% -> 0% across runs, see progress.md's Known issues): the Judge's
memo is written to cite specific already-checked claims by design
(judge_agent.py's task explicitly asks it to reference claims "flagged as
unsupported" by name), but this extractor had no notion of that sentence
*type* and treated every sentence the same way. Confirmed via a direct trace
(scripts/diagnose_judge_claims.py) against a real Judge memo: sentences like
"Bull's claim about an 18% EPS lift was flagged as unsupported" were split
into two failure modes, either the status qualifier ("flagged as unsupported")
was dropped entirely, turning a true meta-statement into a bare, false-sounding
standalone claim ("Bull's EPS lift was 18%") that then fails re-verification
for the same reason it failed the first time; or the qualifier survived but
was still extracted as a "claim" ("Bull's claim was flagged as unsupported"),
which is not a fact any filing could ever confirm or deny, so it also comes
back not_enough_evidence. Either way, every claim from a memo that accurately
cites prior verification results ends up scored as ungrounded, penalizing the
Judge for being specific and honest about which claims already failed.

This is not a paraphrase or meaning-matching problem, retrieval and entailment
classification (evaluation/pipeline.py) already handle differently-worded
sentences with the same meaning correctly. The bug is upstream of that: this
extractor was collapsing two genuinely different sentence types (a new
assertion about the company, vs. a report on another statement's already-
determined status) into one. The DEBATE_COMMENTARY rule below tells the
extractor to recognize and skip the second type entirely, rather than
rewriting it into a checkable claim in any form. Skipping (not rephrasing) is
necessary because Phase 8's diagnostic showed rephrased-but-preserved versions
still get extracted and still fail their check for an unrelated reason (the
filing was never going to confirm or deny "a claim was flagged in a debate").

Verified 2026-09-08, both checks from the paragraph above:
- `scripts/test_phase6_evaluation.py`: F1 = 0.72, unchanged (this script feeds
  test_set.json's claim_text straight into check_claim() and never calls
  extract_claims(), so it can't be affected by this change either way — see
  Phase 6's write-up in progress.md for why. Run anyway as a sanity check;
  confirms no accidental regression elsewhere in the pipeline.)
- `scripts/test_phase8_dashboard.py AAPL` (real Groq run): the Judge's memo
  ("The Bull side emphasized ... but these figures were unsupported by the
  filing excerpts. The Bear side highlighted ... also unsupported.") is almost
  entirely debate-commentary — extract_claims("judge", memo) correctly
  returned zero claims, confirmed via scripts/diagnose_judge_claims.py against
  this exact memo text before the dashboard run. compute_groundedness_by_agent
  (evaluation/dashboard.py) only reports agents present in claim_details, so
  the dashboard's groundedness table now shows no "judge" row at all instead
  of a misleading 0%. This resolves the 0% -> 50% -> 0% swing: it was
  hypothesis (1) from progress.md's Known issues (memo is meta-commentary,
  near-zero was the *correct* result for a memo with few/no new facts, not a
  pipeline bug) — the fix makes that explicit (no row) rather than misleading
  (a 0% that looked like a hallucination signal).
"""

import os
import re
import uuid

import litellm

from agents.llm import GROQ_MODEL, run_with_rate_limit_backoff

_CLAIM_LINE_RE = re.compile(r"^CLAIM:\s*(.+)$", re.IGNORECASE)

_PROMPT = """Break the following statement into individual, self-contained factual claims.
Rules:
- One claim per line, each line starting with "CLAIM:".
- Each claim must be a complete, standalone sentence — resolve pronouns like "it" or "the company" to the actual company name so the claim makes sense on its own.
- Only extract claims that assert a specific, checkable fact (a number, date, name, or concrete event) about the company itself. Skip hedges, opinions, and rhetorical framing ("this makes it a strong buy", "we believe", "the outlook is positive").
- Also skip any sentence that reports on another statement's argument, citation, or verification status, rather than asserting a new fact about the company. This includes sentences describing what a side "hinges on," "cites," or "argues," and sentences describing whether a claim was "flagged," "unsupported," "contradicted," "supported," or similar. These describe the debate itself, not the company, and must not be extracted in any form, even if you could technically rephrase them into a standalone sentence. For example, given "Bull's claim about an 18% EPS lift was flagged as unsupported," extract nothing from that sentence: neither "EPS lift was 18%" (which drops the fact that this was already found unsupported) nor "Bull's claim was flagged as unsupported" (which isn't a fact any filing could confirm or deny).
- If a sentence combines two or more distinct factual assertions (e.g., two different numbers, two different time periods, or two different metrics), split it into separate claims, one per fact.
- If the statement contains no extractable factual claims, output nothing.

Example:
Input: "Total revenue was $475.513 billion, with $300.354 billion from the United States."
Output:
CLAIM: Microsoft's total revenue was $475.513 billion in fiscal 2026.
CLAIM: $300.354 billion of Microsoft's fiscal 2026 revenue came from the United States.

Example (debate commentary, skip entirely):
Input: "The Bull side hinges on an 18% EPS lift, but that claim was flagged as unsupported."
Output:
(nothing — this sentence reports on the debate and its verification status, not a new fact about the company)

Statement:
\"\"\"{statement}\"\"\"
"""


def extract_claims(agent_name: str, statement: str) -> list[dict]:
    """Returns a list of Claim dicts (spec.md's schema) for every factual claim in
    `statement`. entailment_label/confidence/retrieved_chunk/source_section are
    left None here, evaluation.pipeline.check_statement fills those in via
    retrieval + entailment classification. Returns [] for a blank statement, one
    with no extractable claims, or one that consists entirely of debate/verification
    commentary (see this module's docstring for why that's now excluded).
    """
    statement = statement.strip()
    if not statement:
        return []

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")

    def _call():
        return litellm.completion(
            model=f"groq/{GROQ_MODEL}",
            api_key=api_key,
            temperature=0.0,
            max_completion_tokens=800,
            # See module docstring's real-bug writeup, without this, gpt-oss-20b's
            # hidden reasoning can consume the entire token budget and leave the
            # visible answer empty.
            reasoning_effort="low",
            messages=[{"role": "user", "content": _PROMPT.format(statement=statement)}],
        )

    response = run_with_rate_limit_backoff(_call, label="claim_extraction")
    raw = response["choices"][0]["message"]["content"] or ""

    claims = []
    for line in raw.splitlines():
        match = _CLAIM_LINE_RE.match(line.strip())
        if not match:
            continue
        text = match.group(1).strip()
        if not text:
            continue
        claims.append(
            {
                "claim_id": str(uuid.uuid4()),
                "agent": agent_name,
                "text": text,
                "source_section": None,
                "retrieved_chunk": None,
                "entailment_label": None,
                "confidence": None,
            }
        )
    return claims
