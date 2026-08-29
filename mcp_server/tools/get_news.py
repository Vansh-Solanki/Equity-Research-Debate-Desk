"""Recent news tool, backed by Google News RSS (free, no API key required)."""

import re
from urllib.parse import quote

import feedparser

NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def _clean_summary(raw_summary: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw_summary or "")
    return re.sub(r"\s+", " ", text).strip()


def get_news(company: str, limit: int = 5) -> dict:
    try:
        query = quote(f"{company} stock")
        feed = feedparser.parse(NEWS_RSS_URL.format(query=query))

        if getattr(feed, "bozo", False) and not feed.entries:
            return {"success": False, "data": None, "error": f"failed to fetch news feed: {feed.bozo_exception}"}

        articles = []
        for entry in feed.entries[:limit]:
            source = getattr(entry, "source", None)
            source_title = source.get("title") if isinstance(source, dict) else getattr(source, "title", None)

            articles.append(
                {
                    "title": entry.get("title", ""),
                    "published": entry.get("published", ""),
                    "source": source_title or "Google News",
                    "summary": _clean_summary(entry.get("summary", "")),
                }
            )

        return {"success": True, "data": {"articles": articles}, "error": None}
    except Exception as exc:  # noqa: BLE001 - surface any unexpected failure through the tool contract
        return {"success": False, "data": None, "error": str(exc)}
