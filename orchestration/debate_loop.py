"""Debate loop: fixed speaking order, N rounds, then the Judge scores the transcript.

speaking_order is a plain ordered list (per spec.md's known constraints) so
randomizing turn order later is a one-line change — never hardcode agent-by-name
calls. Every transcript entry is labeled by agent name (agents.*.AGENT_NAME),
never by position.
"""

import time
from datetime import datetime, timezone

from agents import bear_agent, bull_agent, judge_agent

speaking_order = [bull_agent, bear_agent]

# A short pause between turns spreads requests out against Groq's free-tier
# 8000-tokens/minute cap, cutting how often an individual turn needs to fall back
# to agents.llm.run_with_rate_limit_backoff's (much longer) retry wait.
INTER_TURN_PAUSE_SECONDS = 15


def _format_transcript(transcript: list[dict]) -> str:
    return "\n".join(f"[Round {entry['round']}] {entry['agent'].upper()}: {entry['statement']}" for entry in transcript)


def _entry(round_num: int, agent_module, statement: str) -> dict:
    return {
        "round": round_num,
        "agent": agent_module.AGENT_NAME,
        "statement": statement,
        "claims": [],  # populated once Phase 6's claim-extraction pipeline exists
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_debate(company: str, rounds: int = 3) -> dict:
    """Runs a full debate for `company` and returns a dict matching spec.md's
    DebateResult shape (deep_dive fields are left empty — Phase 7 populates those).

    Round 1: each side in speaking_order writes an independent opening statement
    (no visibility into the other's statement). Round 2+: each side reads the full
    transcript so far and must respond to the opponent's specific prior points
    before the Judge reads the finished transcript and produces a verdict.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    transcript: list[dict] = []

    for agent_module in speaking_order:
        if transcript:
            time.sleep(INTER_TURN_PAUSE_SECONDS)
        statement = agent_module.run_opening_statement(company)
        transcript.append(_entry(1, agent_module, statement))

    for round_num in range(2, rounds + 1):
        for agent_module in speaking_order:
            time.sleep(INTER_TURN_PAUSE_SECONDS)
            statement = agent_module.run_rebuttal(company, _format_transcript(transcript))
            transcript.append(_entry(round_num, agent_module, statement))

    time.sleep(INTER_TURN_PAUSE_SECONDS)
    judge_verdict = judge_agent.run_verdict(company, _format_transcript(transcript))

    return {
        "transcript": transcript,
        "judge_verdict": judge_verdict,
        "deep_dive_available_sections": [],
        "deep_dive_results": {},
    }
