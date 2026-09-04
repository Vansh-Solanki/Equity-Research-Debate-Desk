"""Phase 8 acceptance check (spec.md): all six metrics from the roadmap's
evaluation dashboard are computed and displayed for at least one full debate
run — groundedness % per agent, Judge accuracy (F1), debate engagement score,
retrieval recall@k (pre vs post rerank), latency/LLM-calls per debate, and
deep-dive lift.

Runs one real debate, brackets it with agents.llm's call-log tracking and wall
-clock timing (dashboard.summarize_latency's expected inputs — see
evaluation/dashboard.py's module docstring), spawns one deep dive on top of it
so deep-dive lift has something to measure, then hands all of that to
evaluation.dashboard.assemble_dashboard and prints every metric.

Usage:
    python -m scripts.test_phase8_dashboard [COMPANY]
"""

import sys
import time

from dotenv import load_dotenv

import agents.llm as llm
from deep_dive.spawner import get_available_sections
from evaluation.dashboard import assemble_dashboard
from orchestration.debate_loop import run_debate

DEFAULT_COMPANY = "AAPL"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()

    company = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COMPANY

    llm.reset_call_log()
    start = time.monotonic()
    print(f"Running a full debate for {company}...")
    result = run_debate(company, rounds=3)
    elapsed = time.monotonic() - start
    call_log = llm.get_call_log()
    print(f"Debate finished in {elapsed:.1f}s over {len(call_log)} LLM calls.\n")

    sections = get_available_sections(company)
    deep_dive_section = next((s for s in sections if "risk factors" in s.lower()), sections[0]) if sections else None
    if deep_dive_section:
        print(f"Spawning a deep dive on {deep_dive_section!r} to measure lift...\n")

    dashboard = assemble_dashboard(
        company,
        result,
        elapsed_seconds=elapsed,
        call_log=call_log,
        deep_dive_section=deep_dive_section,
    )

    print("=== 1. Groundedness % per agent ===")
    for agent, pct in dashboard["groundedness_by_agent"].items():
        print(f"  {agent}: {pct:.0%}")

    print("\n=== 2. Judge accuracy (entailment pipeline F1 vs hand-labeled test set) ===")
    ja = dashboard["judge_accuracy"]
    print(f"  accuracy={ja['accuracy']:.2f} precision={ja['precision']:.2f} recall={ja['recall']:.2f} f1={ja['f1']:.2f}")

    print("\n=== 3. Debate engagement score ===")
    eng = dashboard["engagement"]
    print(f"  average: {eng['average']:.2f}/5 over {len(eng['per_statement'])} scored statements")
    for s in eng["per_statement"]:
        print(f"    round {s['round']} {s['agent']}: {s['score']}/5")

    print("\n=== 4. Retrieval recall@k (pre vs post rerank) ===")
    rr = dashboard["retrieval_recall"]
    print(f"  pre_rerank={rr['pre_rerank']:.0%} post_rerank={rr['post_rerank']:.0%} (n={rr['n_questions']})")

    print("\n=== 5. Latency / LLM calls per debate ===")
    lat = dashboard["latency"]
    print(f"  wall_clock={lat['wall_clock_seconds']:.1f}s llm_calls={lat['llm_call_count']} avg_call={lat['avg_call_seconds']:.1f}s")
    for label, count in lat["calls_by_label"].items():
        print(f"    {label}: {count}")

    print("\n=== 6. Deep-dive lift ===")
    lift = dashboard["deep_dive_lift"]
    if lift is None:
        print("  no sections available to deep-dive on — skipped")
    else:
        print(
            f"  groundedness before={lift['groundedness_before']:.0%} "
            f"after={lift['groundedness_after']:.0%} lift={lift['lift']:+.0%}"
        )

    print("\nPASSED: all six dashboard metrics computed for one full debate run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
