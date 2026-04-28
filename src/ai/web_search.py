"""Lightweight web search tool for the brain (DuckDuckGo HTML, free).

Operator: "it should also have access to MCP so it can search web if
needed."

ACT already has news_digest + knowledge_graph + economic_intelligence
(structured news pipelines). This module adds free-form web search
for cases the brain hits a knowledge gap none of those cover —
e.g., "what does <obscure-token-name> do?", "is there a known
exchange outage?", "what are the rules for X?"

Uses DuckDuckGo's HTML endpoint (no API key, no rate-limit auth).
Daily call cap (default 50/day) prevents the brain from over-using
it instead of the structured tools. Cached 30 minutes per query.

Anti-overfit / anti-noise:
  * Daily cap (ACT_WEB_SEARCH_DAILY_CAP, default 50)
  * Per-query cache (30 min TTL) — same query = same result
  * Result count capped at 5 per call
  * Snippet length capped at 200 chars per result
  * Brain prompt instructed to use this AFTER trying news_digest /
    knowledge_graph (those are higher-signal)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DAILY_CAP = 50
CACHE_TTL_S = 1800  # 30 min
DAILY_COUNTER_PATH = "data/web_search_daily_counter.json"

_query_cache: Dict[str, Dict[str, Any]] = {}  # query → {ts, results}


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title[:120],
            "url": self.url[:200],
            "snippet": self.snippet[:200],
        }


@dataclass
class WebSearchResponse:
    query: str
    n_results: int
    results: List[WebSearchResult] = field(default_factory=list)
    daily_cap_remaining: int = 0
    cache_hit: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:200],
            "n_results": int(self.n_results),
            "results": [r.to_dict() for r in self.results[:5]],
            "daily_cap_remaining": int(self.daily_cap_remaining),
            "cache_hit": self.cache_hit,
            "error": self.error,
        }


def _read_daily_counter() -> Dict[str, Any]:
    """Returns {date: 'YYYY-MM-DD', count: int}."""
    try:
        from datetime import datetime, timezone
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        p = Path(DAILY_COUNTER_PATH)
        if not p.exists():
            return {"date": today, "count": 0}
        data = json.loads(p.read_text())
        if data.get("date") != today:
            return {"date": today, "count": 0}
        return {"date": today, "count": int(data.get("count", 0))}
    except Exception:
        return {"date": "?", "count": 0}


def _write_daily_counter(state: Dict[str, Any]) -> None:
    try:
        p = Path(DAILY_COUNTER_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state))
    except Exception as e:
        logger.debug("web_search counter write failed: %s", e)


def _daily_cap() -> int:
    try:
        return max(0, int(os.environ.get("ACT_WEB_SEARCH_DAILY_CAP")
                          or DEFAULT_DAILY_CAP))
    except Exception:
        return DEFAULT_DAILY_CAP


def _ddg_html_search(query: str, max_results: int = 5) -> List[WebSearchResult]:
    """Fetch DDG HTML endpoint, parse results. Returns [] on any failure."""
    try:
        import requests
        url = "https://html.duckduckgo.com/html/"
        r = requests.post(
            url,
            data={"q": query},
            timeout=6,
            headers={
                "User-Agent": "Mozilla/5.0 ACT-bot",
            },
        )
        if r.status_code != 200:
            return []
        html = r.text
        # Simple regex pull — DDG HTML wraps results in known classes
        # title links: class="result__a" href="..." > title
        results = []
        title_pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        snippet_pattern = re.compile(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        title_matches = list(title_pattern.finditer(html))
        snippet_matches = list(snippet_pattern.finditer(html))
        for i, tm in enumerate(title_matches[:max_results]):
            url_raw = tm.group(1)
            # DDG wraps real URL in /l/?uddg=<encoded>
            real_url = url_raw
            if "uddg=" in url_raw:
                m = re.search(r"uddg=([^&]+)", url_raw)
                if m:
                    try:
                        from urllib.parse import unquote
                        real_url = unquote(m.group(1))
                    except Exception:
                        pass
            title_html = tm.group(2)
            title = re.sub(r"<[^>]+>", "", title_html).strip()[:120]
            snippet = ""
            if i < len(snippet_matches):
                snippet_html = snippet_matches[i].group(1)
                snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()[:200]
            results.append(WebSearchResult(
                title=title, url=real_url, snippet=snippet,
            ))
        return results
    except Exception as e:
        logger.debug("ddg search failed: %s", e)
        return []


def search(query: str, max_results: int = 5) -> WebSearchResponse:
    """Search web with daily cap + per-query cache."""
    query = str(query or "").strip()
    max_results = max(1, min(5, int(max_results)))
    if not query:
        return WebSearchResponse(query="", n_results=0,
                                   error="empty_query",
                                   daily_cap_remaining=_daily_cap())

    # Cache check
    cached = _query_cache.get(query)
    if cached and time.time() - cached["ts"] < CACHE_TTL_S:
        return WebSearchResponse(
            query=query, n_results=len(cached["results"]),
            results=cached["results"][:max_results],
            cache_hit=True,
            daily_cap_remaining=_daily_cap() - _read_daily_counter()["count"],
        )

    # Daily cap check
    counter = _read_daily_counter()
    cap = _daily_cap()
    if counter["count"] >= cap:
        return WebSearchResponse(
            query=query, n_results=0,
            error=f"daily_cap_hit ({cap}/day)",
            daily_cap_remaining=0,
        )

    # Search
    results = _ddg_html_search(query, max_results=max_results)
    if not results:
        return WebSearchResponse(
            query=query, n_results=0,
            error="no_results_or_search_failed",
            daily_cap_remaining=cap - counter["count"],
        )

    # Cache + counter
    _query_cache[query] = {"ts": time.time(), "results": results}
    counter["count"] += 1
    _write_daily_counter(counter)

    return WebSearchResponse(
        query=query, n_results=len(results),
        results=results, cache_hit=False,
        daily_cap_remaining=cap - counter["count"],
    )
