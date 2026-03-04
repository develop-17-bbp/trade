"""
L2 Sentiment Layer — Multi-Source News Fetcher
================================================
Fetches real-time news and social data from multiple free sources:
  1. Reddit crypto subreddits (r/cryptocurrency, r/Bitcoin, r/ethereum)
  2. CryptoPanic RSS (free tier)
  3. CoinGecko trending / news
  4. NewsAPI (if API key provided)

Each source returns normalized headlines with timestamps and source metadata.
"""

import time
import requests
from typing import List, Dict, Optional
from datetime import datetime, timezone


class NewsItem:
    """Normalized news item with metadata."""
    __slots__ = ('title', 'source', 'timestamp', 'url', 'tickers', 'event_type')

    def __init__(self, title: str, source: str, timestamp: float,
                 url: str = '', tickers: Optional[List[str]] = None,
                 event_type: str = 'general'):
        self.title = title
        self.source = source
        self.timestamp = timestamp  # unix epoch
        self.url = url
        self.tickers = tickers or []
        self.event_type = event_type

    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'source': self.source,
            'timestamp': self.timestamp,
            'url': self.url,
            'tickers': self.tickers,
            'event_type': self.event_type,
        }

    def age_seconds(self) -> float:
        return time.time() - self.timestamp


class NewsFetcher:
    """
    Multi-source news aggregator for crypto markets.
    Fetches from all available sources and returns deduplicated, normalized items.
    """

    # Event keywords for classification
    EVENT_KEYWORDS = {
        'regulatory': ['sec', 'regulation', 'ban', 'law', 'legal', 'compliance',
                        'senate', 'congress', 'government', 'policy', 'license'],
        'hack': ['hack', 'exploit', 'breach', 'stolen', 'vulnerability', 'attack',
                  'compromised', 'drained'],
        'etf': ['etf', 'spot etf', 'blackrock', 'fidelity', 'grayscale',
                'ishares', 'approval', 'filing'],
        'macro': ['fed', 'interest rate', 'inflation', 'gdp', 'employment',
                   'treasury', 'fomc', 'recession', 'cpi', 'ppi'],
        'exchange': ['coinbase', 'binance', 'kraken', 'exchange', 'listing',
                      'delisting', 'outage', 'maintenance'],
        'adoption': ['adoption', 'partnership', 'integration', 'payment',
                      'institution', 'tesla', 'microstrategy', 'el salvador'],
    }

    # Ticker keywords
    TICKER_MAP = {
        'BTC': ['bitcoin', 'btc', 'satoshi'],
        'ETH': ['ethereum', 'eth', 'ether', 'vitalik'],
        'SOL': ['solana', 'sol'],
        'XRP': ['ripple', 'xrp'],
        'ADA': ['cardano', 'ada'],
        'DOGE': ['dogecoin', 'doge'],
        'DOT': ['polkadot', 'dot'],
        'AVAX': ['avalanche', 'avax'],
        'LINK': ['chainlink', 'link'],
        'MATIC': ['polygon', 'matic'],
    }

    def __init__(self, user_agent: str = 'CryptoTradeBot/2.0',
                 newsapi_key: Optional[str] = None,
                 cryptopanic_token: Optional[str] = None):
        self.headers = {'User-Agent': user_agent}
        self.newsapi_key = newsapi_key
        self.cryptopanic_token = cryptopanic_token
        self.reddit_api = 'https://www.reddit.com/r/{}/hot.json'
        self._cache: Dict[str, tuple] = {}  # source -> (timestamp, items)
        self._cache_ttl = 120  # seconds

    def fetch_all(self, query: str = 'crypto', limit: int = 100) -> List[NewsItem]:
        """Fetch from all available sources, dedupe, and return sorted by recency.

        The `limit` parameter controls the maximum number of headlines returned
        after deduplication; internally each source is queried with the same
        limit.  Defaults to a generous 100 to avoid starving the sentiment layer.
        """
        items: List[NewsItem] = []

        # Reddit (always available, no API key needed)
        items.extend(self._fetch_reddit(query, limit))

        # CryptoPanic (free tier available)
        if self.cryptopanic_token:
            items.extend(self._fetch_cryptopanic(limit))

        # NewsAPI (requires key)
        if self.newsapi_key:
            items.extend(self._fetch_newsapi(query, limit))

        # CoinGecko trending (free, no key)
        items.extend(self._fetch_coingecko_trending())

        # Deduplicate by title similarity
        items = self._dedupe(items)

        # Classify events and extract tickers
        for item in items:
            item.event_type = self._classify_event(item.title)
            item.tickers = self._extract_tickers(item.title)

        # Sort by timestamp (most recent first)
        items.sort(key=lambda x: x.timestamp, reverse=True)
        return items[:limit]

    def fetch_headlines(self, query: str, limit: int = 50) -> List[str]:
        """Backward-compatible: return just headline strings.

        `limit` default increased so callers without explicit bounds receive more
        than the previous 10 headlines.
        """
        items = self.fetch_all(query, limit)
        return [item.title for item in items]

    # -----------------------------------------------------------------------
    # Source-specific fetchers
    # -----------------------------------------------------------------------
    def _fetch_reddit(self, query: str, limit: int = 10) -> List[NewsItem]:
        """Fetch from crypto subreddits."""
        cache_key = f'reddit_{query}'
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        items: List[NewsItem] = []
        subreddits = ['cryptocurrency', 'Bitcoin', 'ethereum', 'CryptoMarkets']

        for sub in subreddits:
            try:
                url = self.reddit_api.format(sub)
                resp = requests.get(url, headers=self.headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    for post in data.get('data', {}).get('children', [])[:limit]:
                        pd = post.get('data', {})
                        title = pd.get('title', '')
                        created = pd.get('created_utc', time.time())
                        permalink = pd.get('permalink', '')
                        if title:
                            items.append(NewsItem(
                                title=title,
                                source=f'reddit/r/{sub}',
                                timestamp=created,
                                url=f'https://reddit.com{permalink}',
                            ))
            except Exception:
                pass

        self._set_cache(cache_key, items)
        return items

    def _fetch_cryptopanic(self, limit: int = 10) -> List[NewsItem]:
        """Fetch from CryptoPanic API (free tier)."""
        cache_key = 'cryptopanic'
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        items: List[NewsItem] = []
        try:
            url = f'https://cryptopanic.com/api/v1/posts/?auth_token={self.cryptopanic_token}&public=true'
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for post in data.get('results', [])[:limit]:
                    title = post.get('title', '')
                    pub_at = post.get('published_at', '')
                    try:
                        ts = datetime.fromisoformat(pub_at.replace('Z', '+00:00')).timestamp()
                    except Exception:
                        ts = time.time()
                    if title:
                        items.append(NewsItem(
                            title=title,
                            source='cryptopanic',
                            timestamp=ts,
                            url=post.get('url', ''),
                        ))
        except Exception:
            pass

        self._set_cache(cache_key, items)
        return items

    def _fetch_newsapi(self, query: str, limit: int = 10) -> List[NewsItem]:
        """Fetch from NewsAPI.org (requires API key)."""
        cache_key = f'newsapi_{query}'
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        items: List[NewsItem] = []
        try:
            url = (f'https://newsapi.org/v2/everything?q={query}'
                   f'&sortBy=publishedAt&pageSize={limit}'
                   f'&apiKey={self.newsapi_key}')
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for article in data.get('articles', [])[:limit]:
                    title = article.get('title', '')
                    pub_at = article.get('publishedAt', '')
                    try:
                        ts = datetime.fromisoformat(pub_at.replace('Z', '+00:00')).timestamp()
                    except Exception:
                        ts = time.time()
                    if title:
                        items.append(NewsItem(
                            title=title,
                            source=f"newsapi/{article.get('source', {}).get('name', 'unknown')}",
                            timestamp=ts,
                            url=article.get('url', ''),
                        ))
        except Exception:
            pass

        self._set_cache(cache_key, items)
        return items

    def _fetch_coingecko_trending(self) -> List[NewsItem]:
        """Fetch trending coins from CoinGecko (free, no key)."""
        cache_key = 'coingecko_trending'
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        items: List[NewsItem] = []
        try:
            url = 'https://api.coingecko.com/api/v3/search/trending'
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for coin in data.get('coins', [])[:7]:
                    item = coin.get('item', {})
                    name = item.get('name', '')
                    symbol = item.get('symbol', '')
                    rank = item.get('market_cap_rank', '?')
                    if name:
                        items.append(NewsItem(
                            title=f"🔥 Trending: {name} ({symbol}) — Rank #{rank}",
                            source='coingecko/trending',
                            timestamp=time.time(),
                            tickers=[symbol.upper()] if symbol else [],
                        ))
        except Exception:
            pass

        self._set_cache(cache_key, items)
        return items

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------
    def _classify_event(self, text: str) -> str:
        """Classify headline into event type."""
        text_lower = text.lower()
        for event_type, keywords in self.EVENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return event_type
        return 'general'

    def _extract_tickers(self, text: str) -> List[str]:
        """Extract mentioned cryptocurrency tickers from headline."""
        text_lower = text.lower()
        found: List[str] = []
        for ticker, keywords in self.TICKER_MAP.items():
            for kw in keywords:
                if kw in text_lower:
                    found.append(ticker)
                    break
        return found

    def _dedupe(self, items: List[NewsItem]) -> List[NewsItem]:
        """Remove near-duplicate headlines using simple title matching."""
        seen: set = set()
        unique: List[NewsItem] = []
        for item in items:
            # Normalize for comparison
            key = item.title.lower().strip()[:80]
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _check_cache(self, key: str) -> Optional[List[NewsItem]]:
        if key in self._cache:
            ts, items = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return items
        return None

    def _set_cache(self, key: str, items: List[NewsItem]):
        self._cache[key] = (time.time(), items)
