"""Debate engagement scoring: an LLM-as-judge rubric measuring whether each
round 2+ statement actually engages the opponent's immediately preceding
specific point, rather than restating its own case in parallel.

One Groq call scores every round-2+ statement in a single pass (plain-text
`STATEMENT <n>: <score>` lines, parsed with a regex) rather than one call per
statement — same reasoning as evaluation/claim_extractor.py and
agents/judge_agent.py: avoids both the extra rate-limit-budget cost of N calls
and output_pydantic's extra, unretried structured-output call.
"""

import os
import re

import litellm

from agents.llm import GROQ_MODEL, run_with_rate_limit_backoff

_SCORE_LINE_RE = re.compile(r"^STATEMENT\s+(\d+):\s*([1-5])\s*$", re.IGNORECASE | re.MULTILINE)

_PROMPT = """You are scoring a Bull vs Bear equity-debate transcript for engagement: does each
numbered statement below directly address a specific point the opponent just made in the
statement immediately before it, or does it just restate its own case in parallel without
engaging?

Score each statement 1-5:
5 = directly names and responds to a specific claim/number from the immediately preceding
    opposing statement before adding anything new
3 = vaguely references the opponent's general position but not a specific prior claim
1 = ignores the opponent entirely, reads as if written independently

Statements to score:
{numbered_statements}

Reply with exactly one line per statement, in this exact format and nothing else:
STATEMENT <n>: <score>
"""


def _format_statements(scoreable: list[dict]) -> str:
    blocks = []
    for i, entry in enumerate(scoreable, start=1):
        blocks.append(
            f"[{i}] Immediately preceding statement ({entry['prev_agent'].upper()}): "
            f"{entry['prev_statement']}\n"
            f"[{i}] Statement to score ({entry['agent'].upper()}): {entry['statement']}"
        )
    return "\n\n".join(blocks)


def compute_engagement_score(transcript: list[dict]) -> dict:
    """Scores every round >= 2 transcript entry (round 1 is independent openings
    by design — spec.md's schema — so there's nothing to engage with yet) against
    the statement immediately before it in speaking order.

    Returns {"per_statement": [{"round", "agent", "score"}, ...], "average": float}.
    Returns {"per_statement": [], "average": 0.0} if the transcript has no round >= 2
    entries (e.g. a 1-round debate) — nothing to score, not a zero/failing score.
    """
    scoreable = [
        {
            "round": entry["round"],
            "agent": entry["agent"],
            "statement": entry["statement"],
            "prev_agent": transcript[i - 1]["agent"],
            "prev_statement": transcript[i - 1]["statement"],
        }
        for i, entry in enumerate(transcript)
        if entry["round"] >= 2
    ]
    if not scoreable:
        return {"per_statement": [], "average": 0.0}

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")

    def _call():
        return litellm.completion(
            model=f"groq/{GROQ_MODEL}",
            api_key=api_key,
            temperature=0.0,
            max_completion_tokens=500,
            reasoning_effort="low",
            messages=[{"role": "user", "content": _PROMPT.format(numbered_statements=_format_statements(scoreable))}],
        )

    response = run_with_rate_limit_backoff(_call, label="engagement_scoring")
    raw = response["choices"][0]["message"]["content"] or ""

    scores: dict[int, int] = {int(n): int(s) for n, s in _SCORE_LINE_RE.findall(raw)}

    per_statement = []
    for i, entry in enumerate(scoreable, start=1):
        # Falls back to the minimum score (not a crash) for a statement the model
        # skipped or mis-formatted — treats "couldn't be judged as engaged" the
        # same as "wasn't engaged" rather than silently dropping it from the average.
        score = scores.get(i, 1)
        per_statement.append({"round": entry["round"], "agent": entry["agent"], "score": score})

    average = sum(s["score"] for s in per_statement) / len(per_statement)
    return {"per_statement": per_statement, "average": average}
