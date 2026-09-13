"""Shared Groq-backed LLM config for all CrewAI agents."""

import os
import re
import time

import crewai.llms.cache as _crewai_cache
from crewai import LLM
from litellm.exceptions import BadRequestError, RateLimitError

# Phase 8's latency/LLM-calls-per-debate metric needs a call count and timing
# per logical step (one opening statement, one rebuttal, one verdict, one claim
# extraction, one deep-dive summary) — run_with_rate_limit_backoff already wraps
# every one of those (see each call site's `label=` argument), so it's the one
# place that can record this without touching every agent's internals. Retries
# are not logged separately: a label's count reflects successful logical calls,
# not raw HTTP attempts.
_CALL_LOG: list[dict] = []


def reset_call_log() -> None:
    """Clears the call log — call before a debate run you want to measure, since
    the log is process-global and otherwise accumulates across runs."""
    _CALL_LOG.clear()


def get_call_log() -> list[dict]:
    """Returns the call log so far: [{"label", "duration_seconds", "timestamp"}, ...]."""
    return list(_CALL_LOG)

# crewai 1.15.18's generic (litellm-fallback) LLM class stamps every message with
# a "cache_breakpoint" field meant for prompt-caching providers (Anthropic, etc.)
# but never strips it for providers accessed via the litellm fallback path. Groq's
# API strictly rejects unknown message fields, so every call 400s without this.
# Both crewai executors re-import mark_cache_breakpoint from this module on each
# call, so patching the attribute here is enough to neutralize it everywhere.
if not hasattr(_crewai_cache, "mark_cache_breakpoint"):
    raise RuntimeError(
        "crewai.llms.cache.mark_cache_breakpoint not found — this project's Groq-"
        "compatibility patch (pinned to crewai==1.15.18) may need updating for "
        "your installed crewai version."
    )
_crewai_cache.mark_cache_breakpoint = lambda message: dict(message)

# Matches scripts/test_groq_setup.py — llama-3.1-8b-instant is no longer available
# on the free tier as of Aug 2026, using openai/gpt-oss-20b instead.
GROQ_MODEL = "openai/gpt-oss-20b"


def build_groq_llm(temperature: float = 0.4, max_completion_tokens: int | None = None) -> LLM:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
    # litellm (which CrewAI's LLM wraps) routes Groq-hosted models via the "groq/" prefix.
    return LLM(
        model=f"groq/{GROQ_MODEL}",
        api_key=api_key,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        # GROQ_MODEL (openai/gpt-oss-20b) is a reasoning model that can spend its
        # entire completion-token budget on hidden reasoning tokens before ever
        # emitting the visible answer, leaving the response empty/truncated on
        # longer or more complex tasks — found and root-caused in Phase 7 (see
        # evaluation/claim_extractor.py's module docstring for the confirmed
        # repro: 798/800 tokens spent reasoning, content=""). "low" reliably
        # leaves most of the budget for the actual answer; likely also explains
        # Phase 4's previously-unresolved Judge memo truncation.
        reasoning_effort="low",
    )


# Groq's rate-limit message reports the wait as "Xm Y.Zs" or just "Y.Zs" depending on
# whether it's the per-minute (TPM) or per-day (TPD) limit that tripped.
_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")

# A wait this long almost always means the free tier's daily token cap (200,000
# tokens/day), not the per-minute cap — retrying won't help within a normal script
# run, so fail fast with a clear message instead of blocking for tens of minutes.
MAX_SENSIBLE_WAIT_SECONDS = 90.0


def run_with_rate_limit_backoff(fn, max_attempts: int = 6, label: str = "unlabeled", on_wait=None):
    """Run `fn()`, retrying with backoff on a few observed Groq/crewai flakes:

    - RateLimitError: Groq's free-tier rate limit (per-minute or per-day token cap).
      litellm's own built-in retry doesn't wait long enough for the window to reset,
      so this parses the "try again in [Xm]Y.Zs" hint Groq returns and sleeps that
      long (plus a margin) before retrying — unless the wait is long enough to
      indicate the daily cap, in which case it fails fast instead (see
      MAX_SENSIBLE_WAIT_SECONDS).
    - ValueError("... None or empty"): crewai raises this when Groq returns a blank
      completion body — an observed, infrequent flake unrelated to rate limiting.
    - BadRequestError("Tool choice is none, but model called a tool"): `openai/gpt-oss-20b`
      non-deterministically attempts a tool call on turns crewai has told it not to
      (e.g. after forcing a final answer past max_iter, or on a no-tool agent it still
      sees tool schemas for). Groq validates this strictly and 400s instead of ignoring
      it; retrying resamples the completion and usually avoids the same attempt.

    `label` identifies the logical call site (e.g. "bull_opening", "judge_verdict")
    for Phase 8's evaluation.dashboard latency/call-count metric — recorded once per
    successful call, not per retry attempt.

    `on_wait`, if given, is called roughly once per second during any wait this
    function does (`on_wait(remaining_seconds, label)`), instead of the wait being
    a single silent time.sleep(). This exists so a UI caller (frontend/app.py) can
    show a live "waiting Ns for Groq's rate limit" status rather than the app just
    appearing frozen for anywhere up to MAX_SENSIBLE_WAIT_SECONDS. None (default)
    preserves the old silent-wait behavior exactly — every existing caller (the
    agents/*.py functions, when called without on_wait) is unaffected.
    """

    def _wait(wait_seconds: float, reason: str) -> None:
        if on_wait is None:
            time.sleep(wait_seconds)
            return
        remaining = wait_seconds
        while remaining > 0:
            step = min(1.0, remaining)
            on_wait(remaining, reason)
            time.sleep(step)
            remaining -= step

    start = time.time()
    for attempt in range(max_attempts):
        try:
            result = fn()
            _CALL_LOG.append({"label": label, "duration_seconds": time.time() - start, "timestamp": start})
            return result
        except RateLimitError as exc:
            match = _RETRY_AFTER_RE.search(str(exc))
            if match:
                minutes, seconds = match.groups()
                wait_seconds = (float(minutes) * 60 if minutes else 0.0) + float(seconds) + 2
            else:
                wait_seconds = 15.0
            if wait_seconds > MAX_SENSIBLE_WAIT_SECONDS:
                raise RuntimeError(
                    f"Groq rate limit wait ({wait_seconds:.0f}s) looks like the free-tier "
                    "daily token cap (200,000 tokens/day), not a transient per-minute limit — "
                    "not retrying. Wait for the daily reset or use a different API key."
                ) from exc
            if attempt == max_attempts - 1:
                raise
            _wait(wait_seconds, label)
        except ValueError as exc:
            if attempt == max_attempts - 1 or "None or empty" not in str(exc):
                raise
            _wait(10.0, label)
        except BadRequestError as exc:
            if attempt == max_attempts - 1 or "Tool choice is none" not in str(exc):
                raise
            _wait(5.0, label)
    raise AssertionError("unreachable")