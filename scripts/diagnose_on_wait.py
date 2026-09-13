"""Throwaway diagnostic for bug 8's on_wait fix — simulates a RateLimitError
without needing to actually hit Groq's real limit, so we can confirm the
countdown callback fires correctly. Not part of the permanent scripts/ suite.
"""

import sys

from litellm.exceptions import RateLimitError

from agents.llm import run_with_rate_limit_backoff
from utils.console import fix_windows_console

_attempt = {"count": 0}


def _flaky_call():
    _attempt["count"] += 1
    if _attempt["count"] == 1:
        # Simulate Groq's real error message shape so _RETRY_AFTER_RE parses it.
        raise RateLimitError(
            message="Rate limit reached. Please try again in 3.5s.",
            llm_provider="groq",
            model="openai/gpt-oss-20b",
        )
    return "success on retry"


def main() -> int:
    fix_windows_console()

    def on_wait(remaining, reason):
        print(f"  [on_wait] {reason}: {remaining:.1f}s remaining...")

    print("Calling a function that fails once with a simulated rate limit...")
    result = run_with_rate_limit_backoff(_flaky_call, label="test_call", on_wait=on_wait)
    print(f"Final result: {result!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())