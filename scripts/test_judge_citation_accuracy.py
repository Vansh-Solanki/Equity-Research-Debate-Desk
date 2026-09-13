"""Judge-faithfulness Layer 2 acceptance check: evaluation.judge_faithfulness.
check_citations against the hand-labeled test set (evaluation/judge_citation_test_set.json).

Same shape as scripts/test_phase6_evaluation.py: run each test-set memo
sentence's citation check through the real pipeline (one small LLM extraction
call per sentence, plus the real local embedder for matching) and compare the
resulting verdict bucket (accurate/wrong_status/fabricated/ambiguous) against
the hand-assigned correct_verdict.

Unlike Phase 6's test set (30 independent claims), each entry here is a single
memo_sentence + its own all_debate_claims context, so this drives
find_citation_sentences -> extract_citation_details -> match_to_real_claim
directly per entry rather than indexing a company's filing first — there's no
filing lookup in Layer 2 at all, everything is checked against the supplied
all_debate_claims list.

MATCH_THRESHOLD/AMBIGUITY_MARGIN in evaluation/judge_faithfulness.py were
tuned against this script's output — see that module's comments for the
values this run settled on and why.

Usage:
    python -m scripts.test_judge_citation_accuracy
"""

import json
import sys

from dotenv import load_dotenv

from evaluation.judge_faithfulness import check_citations
from utils.console import fix_windows_console

TEST_SET_PATH = "evaluation/judge_citation_test_set.json"


def main() -> int:
    fix_windows_console()
    load_dotenv()

    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)

    entries = test_set["entries"]
    correct = 0
    per_label_seen: dict[str, int] = {}
    per_label_correct: dict[str, int] = {}

    for entry in entries:
        result = check_citations(entry["memo_sentence"], entry["all_debate_claims"])
        expected = entry["correct_verdict"]

        if not result.details:
            got = "NO_CITATION_FOUND"
        else:
            # judge_citation_test_set.json's entries are single-sentence, so
            # there's exactly one detail to compare against correct_verdict.
            got = result.details[0].verdict

        is_correct = got == expected
        correct += is_correct
        per_label_seen[expected] = per_label_seen.get(expected, 0) + 1
        per_label_correct[expected] = per_label_correct.get(expected, 0) + is_correct

        status = "PASS" if is_correct else "FAIL"
        print(f"[{status}] {entry['id']}: expected={expected} got={got}")
        print(f"       sentence: {entry['memo_sentence'][:100]!r}")
        if result.details:
            d = result.details[0]
            sim = f"{d.similarity:.3f}" if d.similarity is not None else "n/a"
            print(
                f"       extracted: side={d.side_claimed} status={d.status_claimed} "
                f"-> matched={d.matched_claim_id} (status={d.matched_status}, sim={sim})"
            )

    accuracy = correct / len(entries) if entries else 0.0
    print(f"\naccuracy={accuracy:.2f} ({correct}/{len(entries)})")
    for label in sorted(per_label_seen):
        seen, right = per_label_seen[label], per_label_correct[label]
        print(f"  {label}: {right}/{seen} correct")

    # Small hand-built test set (8 entries, 2 of them synthetic) — this is a
    # sanity floor, not a tuned acceptance bar like Phase 6's MIN_F1. Revisit
    # once the set grows past a first pass (see judge_citation_test_set.json's
    # _readme: aiming for 20-30 like Phase 6's).
    MIN_ACCURACY = 0.75
    print(f"(need >= {MIN_ACCURACY:.0%} on this small first-pass set)")
    return 0 if accuracy >= MIN_ACCURACY else 1


if __name__ == "__main__":
    sys.exit(main())
