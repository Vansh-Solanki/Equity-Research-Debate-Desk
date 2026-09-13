"""Phase 3 acceptance check: Bull and Bear each independently research a test
company and produce an opening statement backed by at least 2 retrieved facts,
without seeing each other's output.

Usage:
    python scripts/test_phase3_agents.py [COMPANY]
"""

import sys

from dotenv import load_dotenv

from agents import bear_agent, bull_agent
from utils.console import fix_windows_console

DEFAULT_COMPANY = "AAPL"


def main() -> int:
    fix_windows_console()
    load_dotenv()

    company = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COMPANY

    print(f"=== Bull opening statement ({company}) ===")
    bull_statement = bull_agent.run_opening_statement(company)
    print(bull_statement)

    print(f"\n=== Bear opening statement ({company}) ===")
    bear_statement = bear_agent.run_opening_statement(company)
    print(bear_statement)

    print(
        "\nManually verify above: each statement cites >= 2 specific retrieved facts, "
        "and neither statement references or acknowledges the other side's statement "
        "(they were generated from separate, non-shared crew runs)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
