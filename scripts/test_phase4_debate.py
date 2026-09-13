"""Phase 4 acceptance check: a full 3-round debate transcript is produced where
round 2+ statements demonstrably reference the opponent's prior specific claim,
and the Judge produces a scored verdict.

Usage:
    python -m scripts.test_phase4_debate [COMPANY]
"""

import sys

from dotenv import load_dotenv

from orchestration.debate_loop import run_debate
from utils.console import fix_windows_console

DEFAULT_COMPANY = "AAPL"


def main() -> int:
    fix_windows_console()
    load_dotenv()

    company = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COMPANY

    result = run_debate(company, rounds=3)

    print(f"=== Debate transcript ({company}) ===\n")
    for entry in result["transcript"]:
        print(f"[Round {entry['round']}] {entry['agent'].upper()} ({entry['timestamp']}):")
        print(entry["statement"])
        print()

    verdict = result["judge_verdict"]
    print("=== Judge verdict ===")
    print(f"stronger_side: {verdict['stronger_side']}")
    print(f"claims_checked: {verdict['claims_checked']}")
    print(f"claims_unsupported: {verdict['claims_unsupported']}")
    print(f"memo: {verdict['memo']}")

    print(
        "\nManually verify above: pick 3 round-2+ statements and confirm each explicitly "
        "references a specific claim the opponent made in an earlier round, not just a "
        "generic restatement of its own position."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
