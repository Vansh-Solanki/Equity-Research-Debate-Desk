"""Judge agent: reads the full debate transcript and produces a scored verdict.

Phase 4 scope: the Judge scores the transcript on its own reading (no tool calls,
no retrieval) — it's an LLM-as-judge pass over what Bull and Bear already put on
record. Phase 6 replaces/augments this with the shared claim-extraction ->
retrieval -> entailment pipeline applied uniformly to every agent, including the
Judge itself (no agent is exempt or self-certifying, per spec.md's constraints).
"""

import re

from crewai import Agent, Crew, Process, Task

from agents.llm import build_groq_llm, run_with_rate_limit_backoff
from agents.prompts import JUDGE_SYSTEM_PROMPT
from agents.tools import ALL_TOOLS

AGENT_NAME = "judge"

_VERDICT_FORMAT = """STRONGER_SIDE: <bull, bear, or tie>
CLAIMS_CHECKED: <integer — how many distinct factual claims you evaluated>
CLAIMS_UNSUPPORTED: <integer — how many of those read as unsupported speculation>
MEMO: <2-4 sentence summary of both sides' strongest points and your verdict>"""

_FIELD_RE = re.compile(r"^(STRONGER_SIDE|CLAIMS_CHECKED|CLAIMS_UNSUPPORTED|MEMO):\s*(.*)$", re.IGNORECASE)


def build_judge_agent(company: str) -> Agent:
    return Agent(
        role="Impartial equity research judge",
        goal=f"Fairly score the Bull vs Bear debate about {company} and flag weakly supported claims.",
        backstory=JUDGE_SYSTEM_PROMPT.format(company=company),
        # Tools stay attached (see agents/bull_agent.py's build_bull_agent docstring
        # note): Groq's `openai/gpt-oss-20b` can 400 on a hallucinated tool call when
        # an agent has zero tools declared. The task instructs the Judge not to call
        # them, since Phase 4's verdict is scored from the transcript alone.
        tools=ALL_TOOLS,
        # The verdict covers every claim across a multi-round transcript, so it needs
        # more completion headroom than a single opening statement — otherwise the
        # memo gets cut off mid-sentence at whatever default cap Groq applies.
        llm=build_groq_llm(temperature=0.2, max_completion_tokens=1024),
        max_iter=2,
        verbose=False,
    )


def _parse_verdict(raw: str) -> dict:
    """Parses the Judge's plain-text verdict into the DebateResult.judge_verdict shape.

    Avoids crewai's output_pydantic/instructor converter path, which issues its own
    extra LLM call to coerce free text into a schema — one more call than Groq's
    free-tier rate limit budget reliably supports on top of everything else.
    """
    fields: dict[str, str] = {}
    memo_lines: list[str] = []
    in_memo = False
    for line in raw.splitlines():
        match = _FIELD_RE.match(line.strip())
        if match:
            key, value = match.group(1).upper(), match.group(2).strip()
            fields[key] = value
            in_memo = key == "MEMO"
        elif in_memo and line.strip():
            memo_lines.append(line.strip())

    stronger_side = fields.get("STRONGER_SIDE", "tie").lower()
    if stronger_side not in ("bull", "bear", "tie"):
        stronger_side = "tie"

    def _to_int(value: str | None) -> int:
        if not value:
            return 0
        match = re.search(r"\d+", value)
        return int(match.group()) if match else 0

    memo = " ".join([fields.get("MEMO", "")] + memo_lines).strip() or raw.strip()

    return {
        "stronger_side": stronger_side,
        "memo": memo,
        "claims_checked": _to_int(fields.get("CLAIMS_CHECKED")),
        "claims_unsupported": _to_int(fields.get("CLAIMS_UNSUPPORTED")),
    }


def run_verdict(company: str, transcript_text: str) -> dict:
    """Scores a finished debate transcript. Returns a dict matching spec.md's
    judge_verdict shape: {stronger_side, memo, claims_checked, claims_unsupported}."""
    agent = build_judge_agent(company)
    task = Task(
        description=(
            f"Here is the full Bull vs Bear debate transcript for {company}:\n\n{transcript_text}\n\n"
            f"Do not call any tools — judge from the transcript above alone. Judge which side made "
            f"the stronger, better-supported case. For each factual claim in the transcript, note "
            f"whether it reads as grounded in cited filing/price/news evidence or as unsupported "
            f"speculation. Reply in exactly this format, one field per line:\n\n{_VERDICT_FORMAT}"
        ),
        expected_output=f"Plain text in exactly this format:\n{_VERDICT_FORMAT}",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    raw = str(run_with_rate_limit_backoff(crew.kickoff))
    return _parse_verdict(raw)
