"""Bull agent: argues the optimistic case for a company, backed by retrieved evidence."""

from crewai import Agent, Crew, Process, Task

from agents.llm import build_groq_llm, run_with_rate_limit_backoff
from agents.prompts import BULL_SYSTEM_PROMPT
from agents.tools import ALL_TOOLS

# Used to label every transcript entry/log line — never by position ("speaker 1").
AGENT_NAME = "bull"


def build_bull_agent(company: str) -> Agent:
    return Agent(
        role="Bull equity analyst",
        goal=f"Build the strongest, evidence-backed optimistic case for {company}.",
        backstory=BULL_SYSTEM_PROMPT.format(company=company),
        # Tools stay attached even on turns that shouldn't call them (see run_rebuttal):
        # Groq's `openai/gpt-oss-20b` will occasionally emit a hallucinated tool call
        # (a phantom "browser.get_filing") when an agent has zero tools declared, and
        # Groq's strict tool_choice="none" validation then 400s the whole request.
        # Keeping real tools attached and instructing "don't call them" in the task
        # avoids that path entirely.
        tools=ALL_TOOLS,
        llm=build_groq_llm(),
        # Kept tight for Groq's free-tier 8000 tokens/minute cap: 1 tool-call turn +
        # 1 final-answer turn is the most this budget reliably supports right now.
        max_iter=2,
        verbose=False,
    )


def run_opening_statement(company: str) -> str:
    """Runs the Bull agent's independent research and opening statement for `company`.

    Independent: takes no Bear transcript, so this cannot see or reference Bear's output.
    """
    agent = build_bull_agent(company)
    task = Task(
        description=(
            f"Call get_filing for {company} exactly once (do not call get_price or get_news — "
            f"the free-tier token budget is tight) and write an opening statement making the bull "
            f"case, citing at least 2 specific facts from the filing you retrieved."
        ),
        expected_output=(
            "A 3-6 sentence opening statement making the optimistic case for the company, "
            "referencing at least 2 specific retrieved facts (e.g. filing language, price trend, news event)."
        ),
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(run_with_rate_limit_backoff(crew.kickoff))


def run_rebuttal(company: str, transcript_text: str) -> str:
    """Runs the Bull agent's next statement, reading the full transcript so far.

    Instructed not to call any tools: this round is about engaging with what's
    already on the record (its own prior facts and Bear's specific points), not
    fetching new evidence — also keeps token usage low for Groq's free-tier limit.
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
    return str(run_with_rate_limit_backoff(crew.kickoff))
