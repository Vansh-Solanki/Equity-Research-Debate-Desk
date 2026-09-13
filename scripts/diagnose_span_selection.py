"""Throwaway diagnostic for bug 6 — unlike diagnose_span_bug.py (which only
prints retrieve()'s raw chunks), this calls _select_relevant_span() directly,
so we can see the ACTUAL narrowed premise text that gets fed to the NLI
model, after both splitting AND group-aware neighbor-widening are applied.

Also prints each span's bullet-group id (from _split_spans_with_groups) next
to it, so it's visible at a glance whether neighbor-widening is staying
inside one bullet/group (expected, correct) or crossing into a different one
(the bug this fix targets).

Not part of the permanent scripts/ suite — delete after use.
"""

import sys

from dotenv import load_dotenv

from evaluation.pipeline import _select_relevant_span, _split_spans_with_groups
from rag.retriever import index_company_filing, retrieve
from utils.console import fix_windows_console

CASES = [
    ("msft-03", "Microsoft 365 Consumer cloud revenue grew 28% in fiscal 2026, with subscriber growth of 7%."),
    ("msft-04", "Microsoft's LinkedIn revenue increased $2.0 billion, or 11%, in fiscal 2026."),
]


def main() -> int:
    fix_windows_console()
    load_dotenv()

    index_company_filing("MSFT")

    for claim_id, claim_text in CASES:
        print(f"=== {claim_id} ===")
        print(f"claim: {claim_text}\n")
        candidates = retrieve(claim_text, "MSFT", top_k=3, candidate_k=15)
        for i, c in enumerate(candidates):
            print(f"--- candidate {i} (section={c['metadata'].get('section')}) ---")

            spans, groups = _split_spans_with_groups(c["text"])
            print(f"split into {len(spans)} span(s):")
            for j, (s, g) in enumerate(zip(spans, groups)):
                print(f"  [{j}] (group={g}) {s!r}")

            selected = _select_relevant_span(claim_text, c["text"])
            print(f"\n>>> FINAL SELECTED SPAN (fed to NLI model):\n{selected!r}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())