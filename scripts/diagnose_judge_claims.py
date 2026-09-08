"""Diagnostic: shows exactly what claim_extractor.extract_claims() produces from
a Judge memo, so we can confirm (or rule out) the meta-commentary-stripping bug
discussed in progress.md's Known issues.

This does NOT run a full debate. It takes the Judge memo text directly (either
the hardcoded example below, from the AAPL test_phase8_dashboard run, or one
you paste in) and runs it through the exact same extraction step
debate_loop.py uses, so you can see the claims side by side with the original
memo sentence they came from.

Usage:
    python -m scripts.diagnose_judge_claims
    python -m scripts.diagnose_judge_claims "some other memo text to test"
"""

import sys

from dotenv import load_dotenv

from evaluation.claim_extractor import extract_claims

# The exact memo text from the AAPL test_phase8_dashboard run you posted.
EXAMPLE_MEMO = (
    "The Bull side hinges on an 18% EPS lift and a 66% liquid deferred-revenue "
    "buffer, but each of those claims was flagged as unsupported. The Bear side "
    "cites documented risk-factor language and margin compression, also flagged "
    "as unsupported, yet it presents fewer unsupported claims overall. "
    "Consequently, the Bear side presents the stronger, better-supported argument."
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()

    memo = sys.argv[1] if len(sys.argv) > 1 else EXAMPLE_MEMO

    print("=== Source memo text ===")
    print(memo)
    print()

    claims = extract_claims("judge", memo)

    print(f"=== Extracted {len(claims)} claim(s) ===\n")
    if not claims:
        print("(no claims extracted)")
        return 0

    for i, claim in enumerate(claims, start=1):
        print(f"{i}. \"{claim['text']}\"")

    print()
    print("=== What to look for ===")
    print("If any claim above reads as a bare factual assertion (e.g. \"Bull's")
    print("EPS lift was 18%\") with NO mention that it was already flagged as")
    print("unsupported, that confirms the extractor is stripping the epistemic")
    print("status and turning a true meta-statement into a false standalone claim.")

    return 0


if __name__ == "__main__":
    sys.exit(main())