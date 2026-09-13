"""Phase 1 acceptance check: confirm a completion can be pulled from Groq's free tier
and surface the rate-limit headers so later phases (debate loop, deep-dive spawner)
know what they need to respect.

Usage:
    python scripts/test_groq_setup.py
"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq

from utils.console import fix_windows_console

MODEL = "openai/gpt-oss-20b"

RATE_LIMIT_HEADERS = [
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
]


def main() -> int:
    fix_windows_console()
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 1

    client = Groq(api_key=api_key)

    completion = client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Reply with exactly: Groq connection OK."}
        ],
    )

    parsed = completion.parse()
    message = parsed.choices[0].message.content
    print(f"Model: {MODEL}")
    print(f"Response: {message}")

    print("\nRate limit headers:")
    for header in RATE_LIMIT_HEADERS:
        value = completion.headers.get(header)
        print(f"  {header}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
