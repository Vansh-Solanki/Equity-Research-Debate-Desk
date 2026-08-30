"""Shared Groq-backed LLM config for all CrewAI agents."""

import os
import re
import time

import crewai.llms.cache as _crewai_cache
from crewai import LLM
from litellm.exceptions import RateLimitError

# crewai 1.15.18's generic (litellm-fallback) LLM class stamps every message with
# a "cache_breakpoint" field meant for prompt-caching providers (Anthropic, etc.)
# but never strips it for providers accessed via the litellm fallback path. Groq's
# API strictly rejects unknown message fields, so every call 400s without this.
# Both crewai executors re-import mark_cache_breakpoint from this module on each
# call, so patching the attribute here is enough to neutralize it everywhere.
_crewai_cache.mark_cache_breakpoint = lambda message: dict(message)

# Matches scripts/test_groq_setup.py — llama-3.1-8b-instant is no longer available
# on the free tier as of Aug 2026, using openai/gpt-oss-20b instead.
GROQ_MODEL = "openai/gpt-oss-20b"


def build_groq_llm(temperature: float = 0.4) -> LLM:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
    # litellm (which CrewAI's LLM wraps) routes Groq-hosted models via the "groq/" prefix.
    return LLM(model=f"groq/{GROQ_MODEL}", api_key=api_key, temperature=temperature)


_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s")


def run_with_rate_limit_backoff(fn, max_attempts: int = 6):
    """Run `fn()`, retrying with backoff on Groq's free-tier TPM rate limit.

    litellm's own built-in retry doesn't wait long enough for Groq's per-minute
    token window to reset, so this parses the "try again in Ns" hint Groq returns
    and sleeps that long (plus a margin) before retrying.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except RateLimitError as exc:
            if attempt == max_attempts - 1:
                raise
            match = _RETRY_AFTER_RE.search(str(exc))
            wait_seconds = float(match.group(1)) + 2 if match else 15.0
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")
