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
"""

import json

from deep_dive.spawner import spawn_deep_dive
from evaluation.engagement_scorer import compute_engagement_score
from evaluation.metrics import groundedness_pct, score_predictions
from evaluation.pipeline import check_claim
from evaluation.retrieval_metrics import recall_at_k
from rag.retriever import index_company_filing

DEFAULT_TEST_SET_PATH = "evaluation/test_set.json"


def compute_groundedness_by_agent(claim_details: dict) -> dict[str, float]:
    """Splits a DebateResult's claim_details ({claim_id: Claim}) by claim["agent"]
    and runs groundedness_pct over each group — "bull"/"bear"/"judge" from a normal
    debate, plus "deep_dive_sub_agent" if any deep-dive results were merged in."""
    by_agent: dict[str, list[dict]] = {}
    for claim in claim_details.values():
        by_agent.setdefault(claim["agent"], []).append(claim)
    return {agent: groundedness_pct(claims) for agent, claims in by_agent.items()}


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
    dashboard = {
        "groundedness_by_agent": compute_groundedness_by_agent(debate_result["claim_details"]),
        "judge_accuracy": compute_judge_accuracy(),
        "engagement": compute_engagement(debate_result["transcript"]),
        "retrieval_recall": compute_retrieval_recall(company),
        "latency": summarize_latency(elapsed_seconds, call_log),
        "deep_dive_lift": None,
    }
    if deep_dive_section is not None:
        dashboard["deep_dive_lift"] = compute_deep_dive_lift(debate_result, company, deep_dive_section)
    return dashboard
