"""Phase 7 acceptance check (spec.md): clicking a deep-dive button for a specific
section returns a section-scoped summary within a reasonable time, and does not
reprocess if clicked twice.

Simulates the not-yet-built (Phase 9) UI buttons: lists a filing's available
sections, "clicks" one (spawn_deep_dive) and times it, then "clicks" the same
one again and confirms the second call is near-instant (cache hit, no new LLM
call) rather than reprocessing.

Usage:
    python -m scripts.test_phase7_deepdive [COMPANY]
"""

import sys
import time

from dotenv import load_dotenv

from deep_dive.spawner import get_available_sections, spawn_deep_dive

DEFAULT_COMPANY = "AAPL"
MIN_SECTIONS = 3
# A cache hit should be near-instant (dict lookup only); a re-run would cost at
# least several seconds of real Groq + local-model latency. 2s is a generous
# margin above dict-lookup time while still clearly separating "cached" from
# "reprocessed."
MAX_CACHED_SECONDS = 2.0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()

    company = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COMPANY

    print(f"Indexing {company}'s filing sections...")
    sections = get_available_sections(company)
    print(f"Found {len(sections)} sections (need >= {MIN_SECTIONS}):")
    for label in sections:
        print(f"  - {label}")
    if len(sections) < MIN_SECTIONS:
        print("FAILED: fewer than the minimum expected sections.")
        return 1

    # Prefer a substantial, content-rich section (Risk Factors) over a
    # boilerplate one so the summary is a meaningful acceptance check.
    target = next((s for s in sections if "risk factors" in s.lower()), sections[0])
    print(f"\n'Clicking' deep-dive button for: {target!r}")

    start = time.monotonic()
    first = spawn_deep_dive(company, target)
    first_elapsed = time.monotonic() - start
    print(f"First call: {first_elapsed:.1f}s, cached={first['cached']}")
    print(f"Summary: {first['summary']}")
    print(f"Claims checked: {len(first['claims'])}")

    if first["cached"]:
        print("FAILED: first call should not be a cache hit.")
        return 1
    if not first["summary"].strip():
        print("FAILED: summary is empty.")
        return 1

    print(f"\n'Clicking' the same deep-dive button again: {target!r}")
    start = time.monotonic()
    second = spawn_deep_dive(company, target)
    second_elapsed = time.monotonic() - start
    print(f"Second call: {second_elapsed:.2f}s, cached={second['cached']}")

    if not second["cached"]:
        print("FAILED: second call for the same section should hit the cache.")
        return 1
    if second_elapsed > MAX_CACHED_SECONDS:
        print(f"FAILED: cached call took {second_elapsed:.2f}s, expected < {MAX_CACHED_SECONDS}s (looks reprocessed).")
        return 1
    if second["summary"] != first["summary"]:
        print("FAILED: cached call returned a different summary than the first call.")
        return 1

    print("\nPASSED: section-scoped summary produced, second click served from cache without reprocessing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
