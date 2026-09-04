"""Debate loop: fixed speaking order, N rounds, then the Judge scores the transcript.

speaking_order is a plain ordered list (per spec.md's known constraints) so
randomizing turn order later is a one-line change — never hardcode agent-by-name
calls. Every transcript entry is labeled by agent name (agents.*.AGENT_NAME),
never by position.

Phase 6 wires in the shared claim-level evaluation pipeline
(evaluation.pipeline.check_statement) here: every statement — Bull's, Bear's, and
the Judge's memo — is run through the same claim extraction -> retrieval ->
entailment check, since no agent is exempt or self-certifying. The Judge's own
claims_checked/claims_unsupported counts (Phase 4: self-reported from its own
reading of the transcript) are replaced with real measured counts from this
pipeline, run over every Bull/Bear claim across the whole debate.

Phase 9 needs the frontend to render each statement as it's produced ("updates
live as a debate runs" — spec.md's acceptance criterion), not wait for the
whole debate to finish. run_debate_stream() is the actual implementation, a
generator yielding one event per step; run_debate() is a thin wrapper that
drains it and returns the final DebateResult, unchanged from before this
refactor — every existing caller (scripts/test_phase4/6/7/8_*.py) keeps working
without modification.
"""

import time
from datetime import datetime, timezone

from agents import bear_agent, bull_agent, judge_agent
from evaluation.pipeline import check_statement
from rag.retriever import index_company_filing

speaking_order = [bull_agent, bear_agent]

# A short pause between turns spreads requests out against Groq's free-tier
# 8000-tokens/minute cap, cutting how often an individual turn needs to fall back
# to agents.llm.run_with_rate_limit_backoff's (much longer) retry wait.
INTER_TURN_PAUSE_SECONDS = 15


def _format_transcript(transcript: list[dict]) -> str:
    return "\n".join(f"[Round {entry['round']}] {entry['agent'].upper()}: {entry['statement']}" for entry in transcript)


def _entry(round_num: int, agent_module, statement: str, company: str, claim_details: dict) -> dict:
    checked = check_statement(agent_module.AGENT_NAME, company, statement)
    for claim in checked:
        claim_details[claim["claim_id"]] = claim
    return {
        "round": round_num,
        "agent": agent_module.AGENT_NAME,
        "statement": statement,
        "claims": [claim["claim_id"] for claim in checked],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_debate_stream(company: str, rounds: int = 3):
    """Generator form of run_debate(): yields one event per step so a caller
    (frontend/app.py) can render the debate as it happens, rather than blocking
    until the whole thing finishes. Event shapes:

    {"type": "indexed", "chunks_indexed": int}
    {"type": "statement", "entry": DebateTranscriptEntry}   -- once per turn
    {"type": "verdict", "judge_verdict": dict}
    {"type": "done", "result": DebateResult}                -- always the last event

    See run_debate()'s docstring for the debate structure itself; this function
    contains the actual logic, run_debate() just drains this generator.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    # Claim-checking retrieves against this company's indexed filing (Phase 5); index
    # it up front so every check_statement call in this debate has something to
    # retrieve against, rather than silently returning not_enough_evidence for lack
    # of an index.
    index_result = index_company_filing(company)
    if not index_result["success"]:
        raise RuntimeError(f"failed to index {company}'s filing for claim-checking: {index_result['error']}")
    yield {"type": "indexed", "chunks_indexed": index_result["chunks_indexed"]}

    transcript: list[dict] = []
    claim_details: dict[str, dict] = {}

    for agent_module in speaking_order:
        if transcript:
            time.sleep(INTER_TURN_PAUSE_SECONDS)
        statement = agent_module.run_opening_statement(company)
        entry = _entry(1, agent_module, statement, company, claim_details)
        transcript.append(entry)
        yield {"type": "statement", "entry": entry}

    for round_num in range(2, rounds + 1):
        for agent_module in speaking_order:
            time.sleep(INTER_TURN_PAUSE_SECONDS)
            statement = agent_module.run_rebuttal(company, _format_transcript(transcript))
            entry = _entry(round_num, agent_module, statement, company, claim_details)
            transcript.append(entry)
            yield {"type": "statement", "entry": entry}

    time.sleep(INTER_TURN_PAUSE_SECONDS)
    judge_verdict = judge_agent.run_verdict(company, _format_transcript(transcript))

    # Fact-check the Judge's own memo too — "no agent is exempt or self-certifying"
    # applies to the Judge as much as to Bull/Bear.
    judge_claims = check_statement(judge_agent.AGENT_NAME, company, judge_verdict["memo"])
    for claim in judge_claims:
        claim_details[claim["claim_id"]] = claim

    # Replace the Judge's self-reported claims_checked/claims_unsupported (Phase 4:
    # its own subjective read of the transcript) with counts measured by the shared
    # pipeline over every Bull/Bear claim in the debate — a real measurement instead
    # of the Judge grading its own homework.
    all_debate_claims = [claim_details[cid] for entry in transcript for cid in entry["claims"]]
    judge_verdict["claims_checked"] = len(all_debate_claims)
    judge_verdict["claims_unsupported"] = sum(
        1 for c in all_debate_claims if c["entailment_label"] != "supported"
    )
    yield {"type": "verdict", "judge_verdict": judge_verdict}

    yield {
        "type": "done",
        "result": {
            "transcript": transcript,
            "judge_verdict": judge_verdict,
            "claim_details": claim_details,
            "deep_dive_available_sections": [],
            "deep_dive_results": {},
        },
    }


def run_debate(company: str, rounds: int = 3) -> dict:
    """Runs a full debate for `company` and returns a dict matching spec.md's
    DebateResult shape (deep_dive fields are left empty — Phase 7 populates those).

    Round 1: each side in speaking_order writes an independent opening statement
    (no visibility into the other's statement). Round 2+: each side reads the full
    transcript so far and must respond to the opponent's specific prior points
    before the Judge reads the finished transcript and produces a verdict.

    Every statement (Bull, Bear, and the Judge's memo) is claim-checked via
    evaluation.pipeline.check_statement — the full Claim records end up in the
    returned dict's "claim_details" (keyed by claim_id; transcript entries and the
    judge_verdict only carry claim_id references, per spec.md's schemas).
    """
    for event in run_debate_stream(company, rounds):
        if event["type"] == "done":
            return event["result"]
    raise AssertionError("unreachable — run_debate_stream always yields a done event")
