"""Deep-dive spawner: on demand, summarizes one filing section via a
section-scoped sub-agent, retrieval filtered to that section's indexed chunks
only (deep_dive.section_splitter.index_filing_sections). Caches per
(company, filing_type, section) so a second click on the same section doesn't
reprocess — spec.md's Phase 7 acceptance criterion.

The sub-agent never calls get_filing directly (unlike Bull/Bear/Judge in
earlier phases): it's handed only the retrieved excerpts for its one section,
consistent with the project's established rate-limit-conscious pattern
(agents/bull_agent.py's run_rebuttal, agents/judge_agent.py's run_verdict) of
feeding pre-gathered context into the task description instead of letting the
agent fetch more on its own.

Per spec.md's Claim schema (agent: "bull | bear | judge | deep_dive_sub_agent")
and the roadmap's "apply this pipeline uniformly ... deep-dive sub-agents"
requirement, every summary is claim-checked through the exact same
evaluation.pipeline.check_statement Bull/Bear/Judge go through — this sub-agent
is not exempt or self-certifying either.
"""

from crewai import Agent, Crew, Process, Task

from agents.llm import build_groq_llm, run_with_rate_limit_backoff
from agents.tools import ALL_TOOLS
from deep_dive.section_splitter import index_filing_sections
from evaluation.pipeline import check_statement
from rag.retriever import retrieve

AGENT_NAME = "deep_dive_sub_agent"

DEFAULT_RETRIEVE_TOP_K = 5
DEFAULT_RETRIEVE_CANDIDATE_K = 15

# In-memory only — no persistent job/session store exists yet (Phase 10's
# orchestration/session.py is where that belongs). Cleared on process restart;
# fine for this phase's scope, since the acceptance criterion is "does not
# reprocess if clicked twice" within a running process, not across restarts.
_CACHE: dict[tuple[str, str, str], dict] = {}


def get_available_sections(company: str, filing_type: str = "10-K") -> list[str]:
    """Indexes the filing's sections (idempotent — re-running just re-upserts the
    same chunk ids) and returns their labels in document order. This is the list
    Phase 9's UI generates one deep-dive button per."""
    result = index_filing_sections(company, filing_type)
    if not result["success"]:
        raise RuntimeError(f"failed to index {company}'s filing sections: {result['error']}")
    return result["sections"]


def _build_sub_agent(company: str, section: str) -> Agent:
    return Agent(
        role="Equity research analyst — section deep dive",
        goal=f"Write a focused, evidence-backed summary of {company}'s '{section}' filing section.",
        backstory=(
            f"You are producing a concise deep-dive summary of one specific section of "
            f"{company}'s SEC filing for an equity research debate. Only state facts that "
            f"appear in the retrieved excerpts you are given — never state a fact you were "
            f"not given."
        ),
        # Tools stay attached even though this task's instructions say not to call
        # them — see agents/bull_agent.py's build_bull_agent docstring: a
        # zero-tools agent can still trigger a Groq 400 on a hallucinated tool call.
        tools=ALL_TOOLS,
        llm=build_groq_llm(temperature=0.3, max_completion_tokens=512),
        max_iter=2,
        verbose=False,
    )


def _run_summary(company: str, section: str, excerpts: list[dict]) -> str:
    context = "\n\n".join(f"[Excerpt {i + 1}]\n{excerpt['text']}" for i, excerpt in enumerate(excerpts))
    agent = _build_sub_agent(company, section)
    task = Task(
        description=(
            f"Here are retrieved excerpts from {company}'s '{section}' filing section:\n\n"
            f"{context}\n\n"
            f"Do not call any tools — everything you need is in the excerpts above. Write a "
            f"concise 3-5 sentence summary of this section, citing at least 2 specific facts "
            f"from the excerpts."
        ),
        expected_output="A 3-5 sentence summary citing at least 2 specific facts from the excerpts.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(run_with_rate_limit_backoff(crew.kickoff))


def spawn_deep_dive(
    company: str,
    section: str,
    filing_type: str = "10-K",
    debate_result: dict | None = None,
) -> dict:
    """Spawns (or returns the cached result of) a deep-dive sub-agent scoped to
    one filing section. Section must already be indexed — call
    get_available_sections(company, filing_type) first.

    Returns a DeepDiveResult: {"section", "summary", "claims" (claim_id list),
    "claim_details" ({claim_id: Claim}, a schema extension mirroring
    orchestration.debate_loop's DebateResult so a caller can dereference the
    claim ids without a second lookup), "cached" (bool, True only when this call
    hit the cache instead of running the sub-agent)}.

    If `debate_result` (a DebateResult dict, e.g. from
    orchestration.debate_loop.run_debate) is passed, merges this result into it —
    "merge sub-agent findings back into the main debate context" per the
    roadmap: adds `section` to deep_dive_available_sections if not already
    listed, sets deep_dive_results[section], and folds the claim records into
    claim_details. This merge always runs, even on a cache hit, since a second
    debate_result for the same section still needs the merge applied to it.
    """
    cache_key = (company, filing_type, section)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        result = {**cached, "cached": True}
    else:
        excerpts = retrieve(
            f"key facts in {section}",
            company,
            top_k=DEFAULT_RETRIEVE_TOP_K,
            candidate_k=DEFAULT_RETRIEVE_CANDIDATE_K,
            where={"section": section},
        )
        if not excerpts:
            raise ValueError(f"no indexed chunks for {company}/{section} — call get_available_sections() first")

        summary = _run_summary(company, section, excerpts)
        claims = check_statement(AGENT_NAME, company, summary)
        result = {
            "section": section,
            "summary": summary,
            "claims": [claim["claim_id"] for claim in claims],
            "claim_details": {claim["claim_id"]: claim for claim in claims},
            "cached": False,
        }
        _CACHE[cache_key] = {k: v for k, v in result.items() if k != "cached"}

    if debate_result is not None:
        available = debate_result.setdefault("deep_dive_available_sections", [])
        if section not in available:
            available.append(section)
        debate_result.setdefault("deep_dive_results", {})[section] = {
            k: v for k, v in result.items() if k != "claim_details"
        }
        debate_result.setdefault("claim_details", {}).update(result["claim_details"])

    return result
