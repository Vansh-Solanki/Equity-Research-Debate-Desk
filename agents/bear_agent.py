"""Bear agent: argues the cautious case for a company, backed by retrieved evidence."""

from crewai import Agent, Crew, Process, Task

from agents.llm import build_groq_llm, run_with_rate_limit_backoff
from agents.prompts import BEAR_SYSTEM_PROMPT
from agents.tools import ALL_TOOLS


def build_bear_agent(company: str) -> Agent:
    return Agent(
        role="Bear equity analyst",
        goal=f"Build the strongest, evidence-backed cautious case for {company}.",
        backstory=BEAR_SYSTEM_PROMPT.format(company=company),
        tools=ALL_TOOLS,
        llm=build_groq_llm(),
        # Kept tight for Groq's free-tier 8000 tokens/minute cap: 1 tool-call turn +
        # 1 final-answer turn is the most this budget reliably supports right now.
        max_iter=2,
        verbose=False,
    )


def run_opening_statement(company: str) -> str:
    """Runs the Bear agent's independent research and opening statement for `company`.

    Independent: takes no Bull transcript, so this cannot see or reference Bull's output.
    """
    agent = build_bear_agent(company)
    task = Task(
        description=(
            f"Call get_filing for {company} exactly once (do not call get_price or get_news — "
            f"the free-tier token budget is tight) and write an opening statement making the case "
            f"for caution, citing at least 2 specific facts from the filing you retrieved."
        ),
        expected_output=(
            "A 3-6 sentence opening statement making the cautious case for the company, "
            "referencing at least 2 specific retrieved facts (e.g. filing language, price trend, news event)."
        ),
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(run_with_rate_limit_backoff(crew.kickoff))
