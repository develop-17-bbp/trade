"""
RSS News Aggregator — Multi-source Crypto News
================================================
Replaces the dead CryptoPanic free API with a pool of free RSS feeds.

Why RSS:
  - No API keys required
  - No rate limits
  - Survived the 2025 crypto-news API consolidation
  - Every major outlet still publishes RSS

Sources (all free, no auth):
  - CoinDesk, Cointelegraph, Decrypt, Bitcoin Magazine,
    CryptoSlate, Bitcoinist, U.Today, NewsBTC, CryptoPotato

Design:
  - Fetch all feeds in parallel (ThreadPoolExecutor)
  - Filter headlines by asset keywords (btc, bitcoin, eth, ethereum, etc.)
  - Dedupe by normalized title
  - Time-decay weighting handled by caller (SentimentPipeline)
  - Graceful degradation — any feed can fail without affecting the rest
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)


# ── Feed registry ───────────────────────────────────────────────────
# Format: (name, url, weight) — weight controls relative source trust
RSS_FEEDS: List[Tuple[str, str, float]] = [
    ('coindesk',       'https://www.coindesk.com/arc/outboundfeeds/rss/',          1.0),
    ('cointelegraph',  'https://cointelegraph.com/rss',                            1.0),
    ('decrypt',        'https://decrypt.co/feed',                                  0.9),
    ('bitcoin_mag',    'https://bitcoinmagazine.com/feed',                         0.9),
    ('cryptoslate',    'https://cryptoslate.com/feed/',                            0.8),
    ('bitcoinist',     'https://bitcoinist.com/feed/',                             0.7),
    ('utoday',         'https://u.today/rss',                                      0.7),
    ('newsbtc',        'https://www.newsbtc.com/feed/',                            0.7),
    ('cryptopotato',   'https://cryptopotato.com/feed/',                           0.7),
]

# Asset keyword map for headline filtering
ASSET_KEYWORDS: Dict[str, List[str]] = {
    'BTC':  ['bitcoin', 'btc', '$btc'],
    'ETH':  ['ethereum', 'eth ', '$eth', 'ether '],  # avoid matching "other"
    'SOL':  ['solana', 'sol ', '$sol'],
    'AVAX': ['avalanche', 'avax', '$avax'],
    'AAVE': ['aave', '$aave'],
}

# Generic crypto keywords — always relevant regardless of asset
MARKET_KEYWORDS = [
    'crypto', 'cryptocurrency', 'blockchain', 'etf', 'sec',
    'regulation', 'hack', 'exploit', 'defi', 'stablecoin',
]


@dataclass
class Headline:
    text: str
    timestamp: float
    source: str
    url: str = ''
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'timestamp': self.timestamp,
            'source': self.source,
            'url': self.url,
            'weight': self.weight,
        }


_RFC822_MONTHS = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}


def _parse_rfc822(date_str: str) -> float:
    """Parse an RFC822 date (RSS standard) to a unix timestamp. Fallback to now()."""
    if not date_str:
        return time.time()
    try:
        # "Wed, 17 Apr 2025 08:30:00 +0000"
        parts = date_str.strip().split()
        if len(parts) < 5:
            return time.time()
        day = int(parts[1])
        month = _RFC822_MONTHS.get(parts[2][:3], 1)
        year = int(parts[3])
        hms = parts[4].split(':')
        hour = int(hms[0]); minute = int(hms[1]); second = int(hms[2]) if len(hms) > 2 else 0
        import calendar
        return calendar.timegm((year, month, day, hour, minute, second, 0, 0, 0))
    except Exception:
        return time.time()


def _strip_html(s: str) -> str:
    """Remove HTML tags and CDATA wrappers from a string."""
    if not s:
        return ''
    s = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', s, flags=re.DOTALL)
    s = re.sub(r'<[^>]+>', '', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _fetch_feed(source: str, url: str, weight: float, timeout: int = 8) -> List[Headline]:
    """Fetch a single RSS feed and parse into Headline objects."""
    headlines: List[Headline] = []
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; ACT-News/1.0)'},
        )
        if r.status_code != 200:
            logger.debug(f"[RSS] {source} returned HTTP {r.status_code}")
            return headlines

        # Parse XML — strip BOM if present
        content = r.content
        if content.startswith(b'\xef\xbb\xbf'):
            content = content[3:]
        root = ET.fromstring(content)

        # RSS 2.0: channel/item, Atom: feed/entry
        items = root.findall('.//item')
        if not items:
            items = root.findall('.//{http://www.w3.org/2005/Atom}entry')

        # NOTE: ElementTree Element has a truthiness based on child-count, not
        # presence, so "a or b" chains will silently drop leaf elements that
        # have text but no children. Must compare against None explicitly.
        atom_ns = '{http://www.w3.org/2005/Atom}'
        for item in items[:30]:  # cap per feed
            title_el = item.find('title')
            if title_el is None:
                title_el = item.find(atom_ns + 'title')

            link_el = item.find('link')
            if link_el is None:
                link_el = item.find(atom_ns + 'link')

            date_el = item.find('pubDate')
            if date_el is None:
                date_el = item.find(atom_ns + 'published')
            if date_el is None:
                date_el = item.find(atom_ns + 'updated')

            title = _strip_html(title_el.text if title_el is not None and title_el.text else '')
            link = ''
            if link_el is not None:
                link = (link_el.text or link_el.get('href') or '').strip()
            date_str = date_el.text if date_el is not None and date_el.text else ''
            ts = _parse_rfc822(date_str)

            if title:
                headlines.append(Headline(text=title, timestamp=ts,
                                          source=source, url=link, weight=weight))

    except ET.ParseError as e:
        logger.debug(f"[RSS] {source} XML parse failed: {e}")
    except Exception as e:
        logger.debug(f"[RSS] {source} fetch failed: {e}")

    return headlines


def _matches_asset(text: str, asset: str) -> bool:
    """Return True if the headline text mentions the asset or generic crypto."""
    if not text:
        return False
    low = text.lower()
    keywords = ASSET_KEYWORDS.get(asset.upper(), [])
    if any(k in low for k in keywords):
        return True
    # Fall back to generic market keywords — these move all majors
    return any(k in low for k in MARKET_KEYWORDS)


def _normalize_for_dedupe(text: str) -> str:
    """Collapse headlines to a dedup key: lowercased, alphanumeric-only, first 80 chars."""
    low = text.lower()
    low = re.sub(r'[^a-z0-9 ]', '', low)
    low = re.sub(r'\s+', ' ', low).strip()
    return low[:80]


class RSSNewsAggregator:
    """Multi-source RSS news aggregator with parallel fetching and dedupe."""

    def __init__(
        self,
        feeds: Optional[List[Tuple[str, str, float]]] = None,
        fetch_interval: int = 300,       # 5 minutes between full refreshes
        max_workers: int = 6,
        max_age_hours: float = 48.0,     # drop headlines older than 48h
    ):
        self.feeds = feeds or RSS_FEEDS
        self.fetch_interval = fetch_interval
        self.max_workers = max_workers
        self.max_age_seconds = max_age_hours * 3600

        self._buffer: List[Headline] = []
        self._last_fetch: float = 0.0
        self._lock = threading.Lock()

    def _refresh(self) -> None:
        """Fetch all feeds in parallel and merge into the buffer."""
        all_headlines: List[Headline] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(_fetch_feed, name, url, weight): name
                for name, url, weight in self.feeds
            }
            for fut in as_completed(futures, timeout=30):
                try:
                    all_headlines.extend(fut.result(timeout=12))
                except Exception as e:
                    logger.debug(f"[RSS] future failed for {futures[fut]}: {e}")

        # Dedupe by normalized title, keep highest-weight copy
        deduped: Dict[str, Headline] = {}
        for h in all_headlines:
            key = _normalize_for_dedupe(h.text)
            if not key:
                continue
            existing = deduped.get(key)
            if existing is None or h.weight > existing.weight:
                deduped[key] = h

        # Drop stale items
        cutoff = time.time() - self.max_age_seconds
        fresh = [h for h in deduped.values() if h.timestamp >= cutoff]
        fresh.sort(key=lambda h: h.timestamp, reverse=True)

        with self._lock:
            self._buffer = fresh
            self._last_fetch = time.time()

        logger.info(
            f"[RSS] Refresh complete: {len(fresh)} unique headlines from "
            f"{len(set(h.source for h in fresh))} sources"
        )

    def get_headlines(
        self,
        asset: str = 'BTC',
        limit: int = 30,
        force_refresh: bool = False,
    ) -> List[Headline]:
        """Return the latest asset-relevant headlines. Auto-refreshes on interval."""
        now = time.time()
        if force_refresh or (now - self._last_fetch) > self.fetch_interval:
            try:
                self._refresh()
            except Exception as e:
                logger.warning(f"[RSS] refresh failed: {e}")

        with self._lock:
            buf = list(self._buffer)

        filtered = [h for h in buf if _matches_asset(h.text, asset)]
        return filtered[:limit]

    def get_headline_texts(self, asset: str = 'BTC', limit: int = 30) -> List[str]:
        """Convenience accessor returning just the title strings."""
        return [h.text for h in self.get_headlines(asset=asset, limit=limit)]

    def summary(self) -> dict:
        """Lightweight status for diagnostics/dashboard."""
        with self._lock:
            by_source: Dict[str, int] = {}
            for h in self._buffer:
                by_source[h.source] = by_source.get(h.source, 0) + 1
            return {
                'total_headlines': len(self._buffer),
                'last_fetch_ts': self._last_fetch,
                'seconds_since_fetch': int(time.time() - self._last_fetch) if self._last_fetch else None,
                'by_source': by_source,
                'feeds_configured': len(self.feeds),
            }
