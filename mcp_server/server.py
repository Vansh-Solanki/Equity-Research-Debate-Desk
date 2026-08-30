"""MCP server exposing get_filing, get_price, and get_news as callable tools.

Run directly to serve over stdio (the transport CrewAI/Claude Desktop-style
MCP clients expect):

    python -m mcp_server.server
"""

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from mcp_server.tools import get_filing as _get_filing
from mcp_server.tools import get_news as _get_news
from mcp_server.tools import get_price as _get_price

load_dotenv()

mcp = FastMCP("equity-debate-desk")


@mcp.tool()
def get_filing(company: str, filing_type: str = "10-K") -> dict:
    """Fetch a company's most recent SEC filing of the given type and return its plain text."""
    return _get_filing(company, filing_type)


@mcp.tool()
def get_price(company: str, period: str = "1y") -> dict:
    """Fetch a company's price history and summary stats for the given lookback period."""
    return _get_price(company, period)


@mcp.tool()
def get_news(company: str, limit: int = 5) -> dict:
    """Fetch recent news articles about a company."""
    return _get_news(company, limit)


if __name__ == "__main__":
    mcp.run()
