"""News search service for finding and caching stock-related news articles."""

import os
import json
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo
from pydantic import BaseModel


class SourceItem(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    published: Optional[str] = None


# Optional on-disk cache for news lookups to reduce DDG usage across restarts
# Cache files include date in filename, so they naturally expire at midnight ET
NEWS_CACHE_DIR = os.getenv("NEWS_CACHE_DIR") or os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "news")


def _search_news_for_symbol(symbol: str, days: int, max_results: int) -> List[SourceItem]:
    """Query DuckDuckGo News with tiny on-disk cache to limit rate usage"""
    try:
        if NEWS_CACHE_DIR:
            os.makedirs(NEWS_CACHE_DIR, exist_ok=True)
            # Include date in cache key so cache naturally expires each day
            today = datetime.now(ZoneInfo("America/New_York")).date()
            cache_key = f"{symbol.upper()}_{days}_{max_results}_{today.isoformat()}.json"
            cache_path = os.path.join(NEWS_CACHE_DIR, cache_key)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    return [SourceItem(**it) for it in cached]
                except Exception:
                    pass
    except Exception:
        pass
    
    try:
        # Import here to avoid hard dependency at module import time
        from duckduckgo_search import DDGS  # type: ignore
    except Exception as import_err:
        print(f"DuckDuckGo search unavailable: {import_err}")
        return []

    timelimit = f"d{max(1, min(days, 30))}"
    items: List[SourceItem] = []
    
    # Try different query patterns if one fails
    query_patterns = [
        f"{symbol} stock",      # Original pattern
        f"{symbol}",            # Just the ticker
        f"{symbol} news",       # Ticker + news
    ]

    for query in query_patterns:
        if len(items) >= max_results:
            break
            
        attempts = 0
        while attempts < 2 and len(items) == 0:  # Reduced to 2 attempts per query
            attempts += 1
            try:
                with DDGS() as ddgs:
                    for n in ddgs.news(
                        query,
                        region="us-en",
                        safesearch="moderate",
                        timelimit=timelimit,
                        max_results=max_results,
                    ):
                        items.append(
                            SourceItem(
                                title=(n.get("title") or ""),
                                url=(n.get("url") or n.get("link") or ""),
                                snippet=(n.get("excerpt") or n.get("body")),
                                published=n.get("date"),
                            )
                        )
                        if len(items) >= max_results:
                            break
            except Exception as e:
                if attempts == 1:
                    print(f"DDG news search failed for '{query}': {e}")
                import time
                time.sleep(0.5 * attempts)
        
        if items:
            # Found results, no need to try other patterns
            break

    cleaned = [i for i in items if i.url]
    
    # Save to cache
    try:
        if cleaned and NEWS_CACHE_DIR:
            os.makedirs(NEWS_CACHE_DIR, exist_ok=True)
            today = datetime.now(ZoneInfo("America/New_York")).date()
            cache_key = f"{symbol.upper()}_{days}_{max_results}_{today.isoformat()}.json"
            cache_path = os.path.join(NEWS_CACHE_DIR, cache_key)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump([c.model_dump() for c in cleaned], f)
    except Exception:
        pass
    
    return cleaned


def cleanup_old_news_cache():
    """Remove news cache files older than 2 days to prevent disk bloat."""
    if not NEWS_CACHE_DIR or not os.path.exists(NEWS_CACHE_DIR):
        return
    
    try:
        import time
        cutoff = time.time() - (2 * 86400)  # 2 days
        for filename in os.listdir(NEWS_CACHE_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(NEWS_CACHE_DIR, filename)
                try:
                    if os.path.getmtime(filepath) < cutoff:
                        os.remove(filepath)
                except Exception:
                    pass
    except Exception as e:
        print(f"News cache cleanup error: {e}")
