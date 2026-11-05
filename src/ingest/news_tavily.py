"""Recent news via Tavily. Used ONLY in the written summary, never as a model input."""
from __future__ import annotations

import json

from config import RAW, TAVILY_API_KEY
from src.utils.log import get_logger

log = get_logger(__name__)


def fetch_news(query: str = "Bitcoin price news today", days: int = 3, max_results: int = 8) -> list[dict]:
    if not TAVILY_API_KEY:
        log.warning("TAVILY_API_KEY not set; summary will have no headlines")
        return []
    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)
    res = client.search(query=query, topic="news", days=days, max_results=max_results)
    return res.get("results", [])


def main() -> None:
    news = fetch_news()
    out = RAW / "news.json"
    out.write_text(json.dumps(news, indent=2, default=str), encoding="utf-8")
    log.info("news saved: %d headlines -> %s", len(news), out)


if __name__ == "__main__":
    main()
