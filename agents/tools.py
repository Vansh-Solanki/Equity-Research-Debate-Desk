"""Decoy tools attached to Bull, Bear, Judge, and the deep-dive sub-agent.

None of these agents are ever supposed to call a tool for real: every task
description explicitly says "do not call any tools" and hands pre-retrieved
context (rag/retriever.py) straight into the prompt instead. Real data
fetching always goes through mcp_server.tools directly — in-process, via
rag/retriever.py and deep_dive/section_splitter.py — never through an agent
tool call.

Tools stay attached anyway to dodge a separate Groq bug: `openai/gpt-oss-20b`
will occasionally hallucinate a tool call (a phantom "browser.get_filing")
when an agent has zero tools declared, and Groq's strict tool_choice="none"
validation then 400s the whole request. A task instruction ("don't call
these") is not enforcement, though — nothing stopped the model from actually
reaching SEC/Yahoo Finance/Google News mid-turn if it ignored that. So these
tools keep the real names (Groq still sees "this agent has tools", avoiding
the zero-tools crash) but never perform the real fetch: any call just logs a
warning and returns a clear "disabled" result, at the cost of one of the
agent's two turns rather than an unchecked, unbudgeted live network call.
"""

import logging

from crewai.tools import tool

logger = logging.getLogger(__name__)

# Counts calls the model made despite being told not to — read via
# get_disabled_tool_call_count() to track how often this actually happens.
_DISABLED_CALL_COUNT = {"get_filing": 0, "get_price": 0, "get_news": 0}


def _disabled_result(name: str) -> dict:
    _DISABLED_CALL_COUNT[name] += 1
    logger.warning(
        "agent called disabled tool %r (call #%d this process) — task instructions "
        "say not to; returning a disabled result instead of fetching real data",
        name, _DISABLED_CALL_COUNT[name],
    )
    return {
        "success": False,
        "data": None,
        "error": f"{name} is disabled for this turn — use only the context already provided in the task.",
    }


@tool("get_filing")
def get_filing_tool(company: str, filing_type: str = "10-K") -> dict:
    """Disabled for this agent — do not call. Filing excerpts you need are already
    provided in the task description; calling this always returns an error."""
    return _disabled_result("get_filing")


@tool("get_price")
def get_price_tool(company: str, period: str = "1y") -> dict:
    """Disabled for this agent — do not call. Use only the context already
    provided in the task description; calling this always returns an error."""
    return _disabled_result("get_price")


@tool("get_news")
def get_news_tool(company: str, limit: int = 5) -> dict:
    """Disabled for this agent — do not call. Use only the context already
    provided in the task description; calling this always returns an error."""
    return _disabled_result("get_news")


ALL_TOOLS = [get_filing_tool, get_price_tool, get_news_tool]


def get_disabled_tool_call_count() -> dict:
    """Returns a copy of the per-tool disabled-call counters, for monitoring how
    often the model attempts a call despite the tool-choice workaround."""
    return dict(_DISABLED_CALL_COUNT)
