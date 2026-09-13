"""Bull agent: argues the optimistic case for a company, backed by retrieved evidence."""

from crewai import Agent, Crew, Process, Task

from agents.llm import build_groq_llm, run_with_rate_limit_backoff
from agents.prompts import BULL_SYSTEM_PROMPT
from agents.tools import ALL_TOOLS
from rag.retriever import retrieve

# Used to label every transcript entry/log line — never by position ("speaker 1").
AGENT_NAME = "bull"

MAX_CHUNKS_PER_QUERY = 3

# Role-relevant queries: Bull digs for strength/growth signal.
BULL_OPENING_QUERIES = [
    "revenue growth and financial performance highlights",
    "competitive strengths and growth strategy",
]


def _build_retrieved_context(company: str, queries: list[str]) -> str:
    """Pre-fetches role-relevant excerpts via Phase 5's RAG retriever, formatted for
    the task description. retrieve() returns [] gracefully if nothing's indexed yet
    (debate_loop.run_debate() indexes the filing before any agent runs), or if a
    query just doesn't match anything — returns "" in either case so the caller can
    branch the task instruction instead of citing facts that don't exist."""
    blocks = []
    for query in queries:
        for r in retrieve(query, company, top_k=MAX_CHUNKS_PER_QUERY):
            section = r["metadata"].get("section") or "filing"
            blocks.append(f"[{section}] {r['text']}")
    return "\n\n".join(blocks) if blocks else ""


def build_bull_agent(company: str) -> Agent:
    return Agent(
        role="Bull equity analyst",
        goal=f"Build the strongest, evidence-backed optimistic case for {company}.",
        backstory=BULL_SYSTEM_PROMPT.format(company=company),
        # Tools stay attached (avoids a Groq zero-tools crash) but are inert decoys
        # that never fetch real data — see agents/tools.py's module docstring.
        tools=ALL_TOOLS,
        llm=build_groq_llm(),
        # Kept tight for Groq's free-tier 8000 tokens/minute cap: 1 tool-call turn +
        # 1 final-answer turn is the most this budget reliably supports right now.
        max_iter=2,
        verbose=False,
    )


def run_opening_statement(company: str, on_wait=None) -> str:
    """Runs the Bull agent's independent research and opening statement for `company`.

    Independent: takes no Bear transcript, so this cannot see or reference Bear's output.

    `on_wait` — see agents.llm.run_with_rate_limit_backoff's docstring; forwarded
    unchanged so a UI caller can show a live rate-limit-wait status.
    """
    agent = build_bull_agent(company)
    context = _build_retrieved_context(company, BULL_OPENING_QUERIES)

    if context:
        description = (
            f"Below are retrieved excerpts from {company}'s actual SEC filing — do not "
            f"call any tools, everything you need is here:\n\n{context}\n\n"
            f"Write an opening statement making the bull case for {company}, citing at "
            f"least 2 specific facts from the excerpts above."
        )
        expected_output = (
            "A 3-6 sentence opening statement making the optimistic case for the company, "
            "referencing at least 2 specific retrieved facts."
        )
    else:
        # retrieve() came back empty for every query — nothing indexed, or nothing
        # matched. Don't ask for citations that can't exist; say so plainly instead
        # of inviting the model to invent facts to satisfy the instruction.
        description = (
            f"No retrieved filing excerpts were available for {company} (the index may be "
            f"empty or nothing matched). Do not call any tools. State plainly that you could "
            f"not retrieve supporting evidence for {company}, and do not assert any specific "
            f"facts, figures, or claims about the company."
        )
        expected_output = (
            "A short statement explicitly noting that no retrieved evidence was available, "
            "with no specific factual claims about the company."
        )

    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(run_with_rate_limit_backoff(crew.kickoff, label="bull_opening", on_wait=on_wait))


def run_rebuttal(company: str, transcript_text: str, on_wait=None) -> str:
    """Runs the Bull agent's next statement, reading the full transcript so far.

    Instructed not to call any tools: this round is about engaging with what's
    already on the record (its own prior facts and Bear's specific points), not
    fetching new evidence — also keeps token usage low for Groq's free-tier limit.

    `on_wait` — see run_opening_statement's docstring.
    """
    agent = build_bull_agent(company)
    task = Task(
        description=(
            f"Here is the debate transcript so far for {company}:\n\n{transcript_text}\n\n"
            f"Do not call any tools — everything you need is in the transcript above. Write your "
            f"next statement as the Bull. Directly address Bear's most recent specific point(s) "
            f"before adding anything new. Build on facts already established in the transcript "
            f"rather than inventing new ones."
        ),
        expected_output=(
            "A 3-6 sentence rebuttal that explicitly references at least one specific claim Bear "
            "made in their most recent statement before responding to it."
        ),
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(run_with_rate_limit_backoff(crew.kickoff, label="bull_rebuttal", on_wait=on_wait))