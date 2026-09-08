"""Phase 8 evaluation dashboard: assembles the six metrics roadmap.md's Phase 8
lists, each pulled from a module that already exists for a different phase
rather than reimplemented here — this module is thin glue, not new measurement
logic (only debate engagement scoring is new, since nothing scored that before).

1. Groundedness % per agent       -> evaluation.metrics.groundedness_pct, grouped
                                      by claim["agent"] from a DebateResult's claim_details.
2. Judge accuracy (F1)            -> evaluation.pipeline.check_claim run over
                                      evaluation/test_set.json (spec.md's project
                                      structure comment already scopes that file
                                      as "for Judge F1 validation" — Phase 6's
                                      entailment pipeline is what backs the
                                      Judge's claims_checked/claims_unsupported
                                      counts since debate_loop.py's Phase 6 wiring,
                                      so this is that same F1, reported here as
                                      the dashboard's "Judge accuracy" figure).
3. Debate engagement score        -> evaluation.engagement_scorer (new this phase).
4. Retrieval recall@k             -> evaluation.retrieval_metrics.recall_at_k,
                                      before vs after cross-encoder reranking.
5. Latency / LLM calls per debate -> agents.llm's call log (agents/llm.py's
                                      run_with_rate_limit_backoff records one
                                      entry per successful logical call), summarized
                                      here. Caller must bracket the debate run with
                                      agents.llm.reset_call_log() and its own
                                      time.time() timing — see
                                      scripts/test_phase8_dashboard.py for the
                                      pattern — since this module has no hook into
                                      when a debate starts/ends.
6. Deep-dive lift                 -> groundedness_pct over claim_details before
                                      vs after merging one deep_dive.spawner
                                      result in, measuring whether the targeted
                                      deep dive actually raised measured
                                      groundedness for that debate.
7. Verdict consistency (new, Sept 8 session, Layer 3 of the Judge-faithfulness
                                      framework) -> evaluation.judge_faithfulness.
                                      check_verdict_consistency, comparing the
                                      Judge's stronger_side against each side's
                                      actual measured support rate. Not one of
                                      roadmap.md's original six — added after
                                      the Sept 8 claim-extractor fix (see
                                      evaluation/claim_extractor.py's docstring)
                                      exposed that metric 1's "judge" entry can
                                      go empty on a well-behaved memo, which
                                      means groundedness alone can't confirm the
                                      Judge's final call is trustworthy.
8. Citation accuracy (new, Sept 8 session, Layer 2 of the same framework) ->
                                      evaluation.judge_faithfulness.check_citations,
                                      checking whether the Judge's memo
                                      accurately describes what each side
                                      actually said and its real verification
                                      status, against all_debate_claims (not
                                      the filing). Deliberately kept separate
                                      from both groundedness (Layer 1: new
                                      stated facts) and verdict consistency
                                      (Layer 3: the final conclusion) — this
                                      one checks the memo's prose about the
                                      debate itself.

Layers 2 and 3 are dashboard-only, not wired into orchestration/debate_loop.py's
live run — same reasoning judge_accuracy already follows (measured against a
separate hand-labeled set, not computed inline during every debate): each
citation check costs one extra Groq call per detected citation sentence, and
adding that to the live per-debate rate-limit budget (already tight — see
progress.md's Phase 3/4 write-ups) isn't worth it for a metric meant to be
read after the fact, not acted on mid-debate.
"""

import json

from deep_dive.spawner import spawn_deep_dive
from evaluation.engagement_scorer import compute_engagement_score
from evaluation.judge_faithfulness import check_citations, check_verdict_consistency
from evaluation.metrics import groundedness_pct, score_predictions
from evaluation.pipeline import check_claim
from evaluation.retrieval_metrics import recall_at_k
from rag.retriever import index_company_filing

DEFAULT_TEST_SET_PATH = "evaluation/test_set.json"


EXPECTED_AGENTS = ("bull", "bear", "judge")


def compute_groundedness_by_agent(
    claim_details: dict, expected_agents: tuple[str, ...] = EXPECTED_AGENTS
) -> dict[str, float | None]:
    """Splits a DebateResult's claim_details ({claim_id: Claim}) by claim["agent"]
    and runs groundedness_pct over each group — "bull"/"bear"/"judge" from a normal
    debate, plus "deep_dive_sub_agent" if any deep-dive results were merged in.

    Every name in `expected_agents` is always a key in the result, even if that
    agent contributed zero checkable claims — value is None in that case, not a
    missing key. Found necessary after the Sept 8 claim-extractor fix (skips
    debate-commentary sentences, see evaluation/claim_extractor.py): a Judge memo
    that's entirely commentary on already-checked claims (exactly what it's now
    designed to write) legitimately produces zero extractable claims, and a
    silently-missing "judge" key read as a bug/crash rather than the expected
    "nothing new to check" outcome. None is deliberately distinct from 0.0 — 0.0
    would mean "checked N>0 claims, none supported" (a real hallucination
    signal), not "checked nothing."
    """
    by_agent: dict[str, list[dict]] = {}
    for claim in claim_details.values():
        by_agent.setdefault(claim["agent"], []).append(claim)
    result = {agent: groundedness_pct(claims) for agent, claims in by_agent.items()}
    for agent in expected_agents:
        result.setdefault(agent, None)
    return result


def compute_judge_accuracy(test_set_path: str = DEFAULT_TEST_SET_PATH) -> dict:
    """Runs evaluation.pipeline.check_claim over the hand-labeled test set and
    scores it with evaluation.metrics.score_predictions — the exact measurement
    scripts/test_phase6_evaluation.py already validates against (F1 = 0.68,
    bar = 0.65), reported here as the dashboard's Judge-accuracy figure.
    Indexes every test-set company's filing first, in case this is the first
    thing to touch that company's Chroma collection in this run."""
    with open(test_set_path, encoding="utf-8") as f:
        test_set = json.load(f)

    for company in sorted({case["company"] for case in test_set}):
        result = index_company_filing(company)
        if not result["success"]:
            raise RuntimeError(f"failed to index {company} for judge-accuracy scoring: {result['error']}")

    predicted, expected = [], []
    for case in test_set:
        result = check_claim(case["claim_text"], case["company"])
        predicted.append(result["entailment_label"])
        expected.append(case["expected_label"])

    return score_predictions(predicted, expected)


def compute_engagement(transcript: list[dict]) -> dict:
    return compute_engagement_score(transcript)


def compute_retrieval_recall(company: str, k: int = 3, n_questions: int = 10) -> dict:
    return recall_at_k(company, k=k, n_questions=n_questions)


def summarize_latency(elapsed_seconds: float, call_log: list[dict]) -> dict:
    """Summarizes a call log captured via agents.llm.get_call_log() (reset with
    reset_call_log() before the measured run) alongside the run's total wall-clock
    time. calls_by_label breaks the count down per logical step (bull_opening,
    judge_verdict, claim_extraction, deep_dive_summary, etc.) so a deep dive's
    added calls are visible separately from the base debate's."""
    calls_by_label: dict[str, int] = {}
    for call in call_log:
        calls_by_label[call["label"]] = calls_by_label.get(call["label"], 0) + 1

    total_calls = len(call_log)
    total_call_seconds = sum(c["duration_seconds"] for c in call_log)
    return {
        "wall_clock_seconds": elapsed_seconds,
        "llm_call_count": total_calls,
        "calls_by_label": calls_by_label,
        "avg_call_seconds": (total_call_seconds / total_calls) if total_calls else 0.0,
    }


def compute_deep_dive_lift(debate_result: dict, company: str, section: str) -> dict:
    """Measures whether spawning a deep dive on `section` raises the debate's
    overall measured groundedness: groundedness_pct over all of claim_details
    before vs after merging that section's deep-dive claims in.

    Mutates a shallow copy of debate_result's claim_details (via spawn_deep_dive's
    debate_result= merge, per deep_dive/spawner.py) — does not touch the caller's
    original dict.
    """
    before_claims = list(debate_result["claim_details"].values())
    groundedness_before = groundedness_pct(before_claims)

    result_copy = {**debate_result, "claim_details": dict(debate_result["claim_details"])}
    spawn_deep_dive(company, section, debate_result=result_copy)

    after_claims = list(result_copy["claim_details"].values())
    groundedness_after = groundedness_pct(after_claims)

    return {
        "groundedness_before": groundedness_before,
        "groundedness_after": groundedness_after,
        "lift": groundedness_after - groundedness_before,
    }


def assemble_dashboard(
    company: str,
    debate_result: dict,
    elapsed_seconds: float,
    call_log: list[dict],
    deep_dive_section: str | None = None,
) -> dict:
    """Computes all six Phase 8 metrics for one completed debate run. Pass the
    same `company`, the DebateResult from orchestration.debate_loop.run_debate,
    and the wall-clock time + agents.llm.get_call_log() captured around that run
    (see this module's docstring and scripts/test_phase8_dashboard.py).

    `deep_dive_section` is optional — pass one of debate_result's
    deep_dive_available_sections to also compute deep-dive lift; omitted (None)
    when the caller hasn't run a deep dive for this debate, since there's nothing
    to measure lift against yet.
    """
    bull_bear_claims = [c for c in debate_result["claim_details"].values() if c["agent"] in ("bull", "bear")]
    dashboard = {
        "groundedness_by_agent": compute_groundedness_by_agent(debate_result["claim_details"]),
        "judge_accuracy": compute_judge_accuracy(),
        "engagement": compute_engagement(debate_result["transcript"]),
        "retrieval_recall": compute_retrieval_recall(company),
        "latency": summarize_latency(elapsed_seconds, call_log),
        "deep_dive_lift": None,
        # Layer 3 of the Sept 8 Judge-faithfulness framework (see
        # evaluation/judge_faithfulness.py): does stronger_side actually follow
        # from the same claim data the Judge was handed before writing its
        # memo? Separate from groundedness_by_agent's "judge" entry (Layer 1,
        # which checks new facts the Judge stated) — this checks the Judge's
        # final conclusion instead, so it's its own dashboard key, not blended in.
        "verdict_consistency": check_verdict_consistency(
            debate_result["judge_verdict"]["stronger_side"], bull_bear_claims
        ),
        # Layer 2: does the Judge's memo prose accurately describe what each
        # side actually said (per all_debate_claims), not just whether the
        # memo's own new factual claims are true (Layer 1) or whether its
        # final verdict follows from the evidence (Layer 3).
        "citation_accuracy": check_citations(debate_result["judge_verdict"]["memo"], bull_bear_claims).as_dict(),
    }
    if deep_dive_section is not None:
        dashboard["deep_dive_lift"] = compute_deep_dive_lift(debate_result, company, deep_dive_section)
    return dashboard
