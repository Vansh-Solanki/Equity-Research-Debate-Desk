"""Judge agent: reads the full debate transcript and produces a scored verdict.

The Judge gets two complementary, deliberately non-overlapping sources of
grounding — dropping either one reopens a real gap, not a redundancy:

1. Retrieved filing excerpts (_build_retrieved_context, same pattern as
   agents/bull_agent.py and agents/bear_agent.py's opening statements): raw
   filing language the Judge can quote directly in its memo for texture and
   specificity. Excerpts alone would still leave the Judge to infer
   support/contradiction itself by eyeballing them — exactly the unreliable
   judgment call the next source replaces.
2. Pre-computed entailment verdicts (all_debate_claims/unsupported_claims,
   passed in by orchestration/debate_loop.py): every Bull/Bear claim in this
   debate has already been checked by evaluation.pipeline.check_statement (the
   same measured, F1-tuned pipeline Phase 6 validated) by the time the Judge
   runs — debate_loop.py used to compute this and then only use it to silently
   overwrite the Judge's self-reported claims_checked/claims_unsupported
   *after* run_verdict() returned, meaning the Judge wrote its memo blind to
   the very data those two numbers came from. The Judge now sees the actual
   labels before writing, so the memo's prose and the programmatic counts
   describe the same underlying reality instead of two independently-guessed
   ones. claims_checked/claims_unsupported are still set from this same data
   programmatically after the call, never trusted from the LLM's own output.

Separately, debate_loop.py also runs check_statement over the Judge's own memo
after it's written — no agent is exempt or self-certifying, per spec.md's
constraints — but that's a post-hoc check of the Judge's output, not a
substitute for the Judge having this evidence while forming its verdict.
"""

import re

from crewai import Agent, Crew, Process, Task

from agents.bear_agent import BEAR_OPENING_QUERIES
from agents.bull_agent import BULL_OPENING_QUERIES
from agents.llm import build_groq_llm, run_with_rate_limit_backoff
from agents.prompts import JUDGE_SYSTEM_PROMPT
from agents.tools import ALL_TOOLS
from rag.retriever import retrieve

AGENT_NAME = "judge"

MAX_CHUNKS_PER_QUERY = 3

# Combines both sides' query angles so the Judge has real filing text to check
# transcript claims against, instead of eyeballing groundedness from prose alone.
JUDGE_EVIDENCE_QUERIES = BULL_OPENING_QUERIES + BEAR_OPENING_QUERIES


def _build_retrieved_context(company: str, queries: list[str]) -> str:
    """Pre-fetches excerpts via Phase 5's RAG retriever, formatted for the verdict
    task description. Mirrors agents/bull_agent.py's helper of the same name —
    returns "" (not [] or a placeholder sentence) if every query comes back empty,
    so the caller can branch the task instruction cleanly."""
    blocks = []
    for query in queries:
        for r in retrieve(query, company, top_k=MAX_CHUNKS_PER_QUERY):
            section = r["metadata"].get("section") or "filing"
            blocks.append(f"[{section}] {r['text']}")
    return "\n\n".join(blocks) if blocks else ""


# Caps the unsupported-claims list shown to the Judge so a heavily-unsupported
# debate (one test run had 25) doesn't blow the token budget — see this module's
# docstring point 2 for why this data matters enough to include at all.
MAX_UNSUPPORTED_CLAIMS_SHOWN = 10


def _format_unsupported_claims(unsupported_claims: list[dict]) -> str:
    """Formats up to MAX_UNSUPPORTED_CLAIMS_SHOWN unsupported/contradicted claims as
    `[AGENT] "claim text" -> label` lines — lets the memo cite specifics ("Bull's
    claim about X was unsupported") instead of just restating a count. Sorted by
    ascending confidence (least confident first) so a cap still shows the most
    anomalous claims rather than an arbitrary slice.

    Note: evaluation.pipeline.check_claim anchors a claim to confidence=1.0 when no
    retrieved candidate was decisive at all (not an actual classifier confidence,
    just "nothing to weigh"), so this ordering is a useful heuristic across mixed
    labels, not a strict severity ranking.
    """
    ranked = sorted(unsupported_claims, key=lambda c: c["confidence"])[:MAX_UNSUPPORTED_CLAIMS_SHOWN]
    return "\n".join(f'[{c["agent"].upper()}] "{c["text"]}" -> {c["entailment_label"]}' for c in ranked)


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


def run_verdict(
    company: str,
    transcript_text: str,
    all_debate_claims: list[dict],
    unsupported_claims: list[dict],
) -> dict:
    """Scores a finished debate transcript. Returns a dict matching spec.md's
    judge_verdict shape: {stronger_side, memo, claims_checked, claims_unsupported}.

    `all_debate_claims`/`unsupported_claims` are every Bull/Bear Claim dict
    (evaluation.pipeline.check_statement's output) from this debate, already
    entailment-checked — see this module's docstring point 2 for why the Judge
    needs these, not just the raw retrieved excerpts.
    """
    agent = build_judge_agent(company)
    context = _build_retrieved_context(company, JUDGE_EVIDENCE_QUERIES)

    if context:
        evidence_block = (
            f"Below are retrieved excerpts from {company}'s actual SEC filing, for quoting "
            f"specific filing language in your memo (do not call any tools):\n\n{context}\n\n"
        )
    else:
        # retrieve() came back empty for every query — nothing indexed, or nothing
        # matched. Fall back to judging from the transcript's own text alone, same
        # as this agent did before retrieval was wired in.
        evidence_block = (
            "No retrieved filing excerpts were available to quote (the index may be empty or "
            "nothing matched) — judge from the transcript's own text and the pre-checked claims "
            "below alone. Do not call any tools.\n\n"
        )

    if unsupported_claims:
        shown = _format_unsupported_claims(unsupported_claims)
        capped_note = (
            f" (showing the {min(len(unsupported_claims), MAX_UNSUPPORTED_CLAIMS_SHOWN)} least confident)"
            if len(unsupported_claims) > MAX_UNSUPPORTED_CLAIMS_SHOWN
            else ""
        )
        claims_block = (
            f"This debate's claims have already been fact-checked against retrieved filing "
            f"evidence by a separate, measured verification pipeline — trust these verdicts over "
            f"your own read of the transcript. {len(all_debate_claims)} claims were checked in "
            f"total; {len(unsupported_claims)} came back unsupported or lacking enough "
            f"evidence{capped_note}:\n\n{shown}\n\n"
        )
    else:
        claims_block = (
            f"This debate's claims have already been fact-checked against retrieved filing "
            f"evidence: all {len(all_debate_claims)} checked claims came back supported.\n\n"
        )

    task = Task(
        description=(
            f"{evidence_block}"
            f"{claims_block}"
            f"Here is the full Bull vs Bear debate transcript for {company}:\n\n{transcript_text}\n\n"
            f"Judge which side made the stronger, better-supported case, citing specific "
            f"unsupported claims listed above where relevant (e.g. by name, "
            f'"Bull\'s claim about X was unsupported"). Reply in exactly this format, one field '
            f"per line:\n\n{_VERDICT_FORMAT}"
        ),
        expected_output=f"Plain text in exactly this format:\n{_VERDICT_FORMAT}",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    raw = str(run_with_rate_limit_backoff(crew.kickoff, label="judge_verdict"))
    return _parse_verdict(raw)
