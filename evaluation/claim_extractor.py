"""Claim extraction: breaks an agent's free-text statement into individual,
self-contained factual claims via one Groq LLM call.

Uses litellm directly rather than a CrewAI Agent/Task/Crew — this is a plain
utility call with no role-play, tools, or multi-turn state, and going through
litellm directly means agents.llm.run_with_rate_limit_backoff's retry logic
(built around litellm.exceptions.RateLimitError/BadRequestError) applies
unchanged without needing a Crew wrapper.

Real bug found during Phase 7 acceptance testing: `openai/gpt-oss-20b` (the
free-tier model this project uses, agents/llm.py's GROQ_MODEL) is a reasoning
model — it can spend its entire completion-token budget on hidden
"reasoning_tokens" before ever emitting the visible answer, especially on a
longer or more structurally complex input (e.g. a multi-sentence deep-dive
summary with several citation markers). Confirmed directly: extracting claims
from one real deep-dive summary returned `content=''` with
`finish_reason="length"` and `reasoning_tokens=798` out of an 800-token budget
— raising the budget to 1500 didn't help either (1498 reasoning tokens, still
empty). This silently produced zero claims for an obviously multi-claim
statement, with no error raised (an empty completion isn't a rate-limit or
malformed-request failure, so `run_with_rate_limit_backoff` had nothing to
retry). Fixed by passing `reasoning_effort="low"` — verified this drops
reasoning token usage to ~200 on the same input and reliably leaves the
requested claims in the visible answer, without needing a much larger (and
more rate-limit-expensive) token budget.
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
- Only extract claims that assert a specific, checkable fact (a number, date, name, or concrete event). Skip hedges, opinions, and rhetorical framing ("this makes it a strong buy", "we believe", "the outlook is positive").
- If the statement contains no extractable factual claims, output nothing.

Statement:
\"\"\"{statement}\"\"\"
"""


def extract_claims(agent_name: str, statement: str) -> list[dict]:
    """Returns a list of Claim dicts (spec.md's schema) for every factual claim in
    `statement`. entailment_label/confidence/retrieved_chunk/source_section are
    left None here — evaluation.pipeline.check_statement fills those in via
    retrieval + entailment classification. Returns [] for a blank statement or one
    with no extractable claims.
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
            # See module docstring's real-bug writeup — without this, gpt-oss-20b's
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
