"""CrewAI tool wrappers around the Phase 2 MCP tool functions.

These call the same get_filing/get_price/get_news functions registered on the
MCP server (mcp_server/server.py) directly, in-process, rather than round-tripping
through the MCP stdio transport — CrewAI agents just need callable tools with the
same contract.
"""

from crewai.tools import tool

from mcp_server.tools import get_filing as _get_filing
from mcp_server.tools import get_news as _get_news
from mcp_server.tools import get_price as _get_price

# Groq's free tier caps at 8000 tokens/minute, and a ReAct-style agent stacks every
# prior tool result into its context on each turn, so oversized tool output blows
# the budget within one or two turns. Until Phase 5's chunked RAG retrieval exists,
# cap what reaches the agent's context well below what the raw MCP tools return.
MAX_FILING_CHARS_FOR_AGENT = 2500
MAX_PRICE_POINTS_FOR_AGENT = 5
MAX_NEWS_SUMMARY_CHARS = 200


@tool("get_filing")
def get_filing_tool(company: str, filing_type: str = "10-K") -> dict:
    """Fetch a company's most recent SEC filing of the given type. Returns
    {success, data: {company, filing_type, filed_date, raw_text, source_url}, error}.
    raw_text is truncated to the filing's opening section to fit free-tier LLM limits."""
    result = _get_filing(company, filing_type)
    if result.get("success") and result["data"].get("raw_text"):
        # mcp_server/tools/get_filing.py's _strip_html already strips the non-rendered
        # inline-XBRL tag-data block, so raw_text starts with real cover-page prose —
        # a plain prefix truncation is enough here, no need to search past anything.
        result["data"]["raw_text"] = result["data"]["raw_text"][:MAX_FILING_CHARS_FOR_AGENT]
    return result


@tool("get_price")
def get_price_tool(company: str, period: str = "1y") -> dict:
    """Fetch a company's price summary stats and a few recent closing prices for the
    given lookback period (e.g. '1mo', '6mo', '1y'). Returns
    {success, data: {ticker, prices, summary_stats}, error}, where prices is the
    most recent points only (full history is a summary_stats concern, not per-day)."""
    result = _get_price(company, period)
    if result.get("success") and result["data"].get("prices"):
        result["data"]["prices"] = result["data"]["prices"][-MAX_PRICE_POINTS_FOR_AGENT:]
    return result


@tool("get_news")
def get_news_tool(company: str, limit: int = 5) -> dict:
    """Fetch recent news articles about a company. Returns
    {success, data: {articles: [{title, published, source, summary}]}, error}.
    Article summaries are truncated to fit free-tier LLM limits."""
    result = _get_news(company, limit)
    if result.get("success"):
        for article in result["data"]["articles"]:
            article["summary"] = article["summary"][:MAX_NEWS_SUMMARY_CHARS]
    return result


ALL_TOOLS = [get_filing_tool, get_price_tool, get_news_tool]
