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
import os
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

    # Ticker keywords. Crypto + top-50 US large-cap stocks so the
    # FinBERT sentiment layer + agents see news for stock symbols too.
    # Without stock entries, get_news_digest('NVDA') returned zero
    # headlines → sentiment_decoder returned conf=0.2 baseline →
    # 13-agent voter consensus stayed FLAT every tick on stocks.
    # 2026-04-30 fix: stocks now get the full sentiment alpha pipeline.
    TICKER_MAP = {
        # Crypto (existing)
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
        # Mega-caps (top 20)
        'NVDA':  ['nvidia', 'nvda', 'jensen huang', 'h100', 'b200', 'blackwell', 'cuda'],
        'MSFT':  ['microsoft', 'msft', 'satya nadella', 'azure', 'copilot'],
        'AAPL':  ['apple', 'aapl', 'tim cook', 'iphone', 'ipad', 'mac', 'app store', 'cupertino'],
        'AMZN':  ['amazon', 'amzn', 'andy jassy', 'aws', 'prime', 'bezos'],
        'GOOGL': ['google', 'googl', 'alphabet', 'sundar pichai', 'gemini', 'youtube', 'android'],
        'GOOG':  ['google', 'goog', 'alphabet'],
        'META':  ['meta', 'facebook', 'instagram', 'zuckerberg', 'reels', 'whatsapp'],
        'TSLA':  ['tesla', 'tsla', 'elon musk', 'cybertruck', 'optimus', 'fsd', 'robotaxi'],
        'AVGO':  ['broadcom', 'avgo'],
        'BRK-B': ['berkshire', 'buffett', 'munger', 'brk'],
        'JPM':   ['jpmorgan', 'jpm', 'jamie dimon', 'chase'],
        'LLY':   ['eli lilly', 'lly', 'mounjaro', 'zepbound', 'tirzepatide'],
        'UNH':   ['unitedhealth', 'unh'],
        'V':     ['visa', 'visa inc'],
        'XOM':   ['exxon', 'xom', 'exxonmobil'],
        'MA':    ['mastercard'],
        'JNJ':   ['johnson & johnson', 'jnj'],
        'COST':  ['costco'],
        'HD':    ['home depot'],
        'PG':    ['procter', 'gamble', 'p&g'],
        'WMT':   ['walmart', 'wmt'],
        # Tier 2
        'ABBV':  ['abbvie'],
        'BAC':   ['bank of america', 'bofa'],
        'NFLX':  ['netflix', 'nflx', 'reed hastings'],
        'CVX':   ['chevron'],
        'KO':    ['coca-cola', 'coca cola', 'coke'],
        'MRK':   ['merck'],
        'AMD':   ['amd', 'lisa su', 'advanced micro', 'mi300', 'mi325'],
        'CRM':   ['salesforce', 'benioff'],
        'ORCL':  ['oracle', 'larry ellison'],
        'PEP':   ['pepsi', 'pepsico'],
        'COIN':  ['coinbase', 'coin', 'brian armstrong'],
        'MSTR':  ['microstrategy', 'mstr', 'saylor'],
        'INTC':  ['intel', 'intc', 'pat gelsinger'],
        'IBM':   ['ibm', 'international business machines'],
        'DIS':   ['disney', 'dis', 'iger'],
        # ETFs
        'SPY':   ['s&p 500', 'sp500', 's&p500', 'spx', 'spy'],
        'QQQ':   ['nasdaq 100', 'nasdaq-100', 'qqq'],
        'IWM':   ['russell 2000', 'iwm'],
        'TQQQ':  ['tqqq', '3x leveraged nasdaq'],
        'SOXL':  ['soxl', '3x semiconductor'],
        'SOXS':  ['soxs', '3x inverse semiconductor'],
        'SQQQ':  ['sqqq', '3x inverse nasdaq'],
    }

    def __init__(self, user_agent: str = 'CryptoTradeBot/2.0',
                 newsapi_key: Optional[str] = None,
                 cryptopanic_token: Optional[str] = None):
        import os as _os
        self.headers = {'User-Agent': user_agent}
        # Fall back to env vars when not explicitly passed. Most callers
        # (web_context, executor, etc.) instantiate NewsFetcher() with no
        # args and rely on env-var loading from .env / setx; previously
        # that meant NewsAPI + CryptoPanic were silently disabled.
        self.newsapi_key = newsapi_key or _os.environ.get('NEWSAPI_KEY', '').strip() or None
        self.cryptopanic_token = (
            cryptopanic_token
            or _os.environ.get('CRYPTOPANIC_TOKEN', '').strip()
            or _os.environ.get('CRYPTOPANIC_KEY', '').strip()
            or None
        )
        self.reddit_api = 'https://www.reddit.com/r/{}/hot.json'
        self._cache: Dict[str, tuple] = {}  # source -> (timestamp, items)
        self._cache_ttl = 120  # seconds
        
        # Log which news sources are enabled
        print("\n[NewsAPI] Source Configuration:")
        print(f"  - NewsAPI:      {'ENABLED' if self.newsapi_key else 'DISABLED (no key in NEWSAPI_KEY)'}")
        print(f"  - CryptoPanic:  {'ENABLED' if self.cryptopanic_token else 'DISABLED (no token in CRYPTOPANIC_TOKEN)'}")
        print(f"  - Reddit:       ALWAYS ENABLED (no key needed)")
        print(f"  - CoinGecko:    FALLBACK ONLY (used if other sources return <50 items)")
        print(f"  - Priority: NewsAPI -> CryptoPanic -> Reddit -> CoinGecko fallback\n")

    def fetch_all(self, query: str = 'crypto', limit: int = 100) -> List[NewsItem]:
        """Fetch from all available sources, dedupe, and return sorted by recency.

        Source priority depends on the asset class detected from `query`:

        STOCKS (NVDA, SPY, AAPL, ...):
          1. Alpaca News API — symbol-tagged headlines, the proper feed
          2. NewsAPI — generic web news with stock keyword
          3. Reddit r/investing / r/stocks fallback
          (Skips CryptoPanic + CoinGecko + r/cryptocurrency — wrong universe.)

        CRYPTO (BTC, ETH, ...) and unknown queries:
          1. NewsAPI (general news + crypto keyword)
          2. CryptoPanic (real-time crypto)
          3. Reddit r/cryptocurrency / r/Bitcoin
          4. CoinGecko trending (fallback)

        Asset class detection: query matches a stock TICKER_MAP entry,
        OR caller passed a known stock ticker. Defaults to crypto path
        for backwards compatibility.
        """
        items: List[NewsItem] = []
        sources_used: List[str] = []

        # Asset-class routing: is this a stock symbol query?
        q_upper = (query or "").strip().upper()
        # Stocks set: any TICKER_MAP key not in the crypto whitelist.
        _CRYPTO_TICKERS = {'BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE', 'DOT',
                            'AVAX', 'LINK', 'MATIC'}
        _STOCK_TICKERS = {k for k in self.TICKER_MAP.keys() if k not in _CRYPTO_TICKERS}
        is_stock_query = q_upper in _STOCK_TICKERS

        if is_stock_query:
            # PRIMARY: Alpaca News API — symbol-tagged
            alp_items = self._fetch_alpaca_news(symbols=[q_upper], limit=limit)
            items.extend(alp_items)
            sources_used.append(f"Alpaca news ({len(alp_items)} items)")

            # SECONDARY: NewsAPI with stock keyword (e.g. "Nvidia stock")
            if self.newsapi_key:
                # Use full-name keyword for better recall
                kws = self.TICKER_MAP.get(q_upper, [q_upper.lower()])
                primary_kw = kws[0] if kws else q_upper.lower()
                na_items = self._fetch_newsapi(f"{primary_kw} stock earnings", limit)
                items.extend(na_items)
                sources_used.append(f"NewsAPI ({len(na_items)} items)")
            else:
                sources_used.append("NewsAPI (no key)")

            # Note: skipping crypto-only sources for stock queries — that
            # was the bug: NewsAPI was queried with "{stock} crypto"
            # which returned nothing relevant; CryptoPanic + CoinGecko
            # never have stock data; r/cryptocurrency is wrong universe.

        else:
            # PRIMARY: General news via NewsAPI (highest quality when available)
            if self.newsapi_key:
                newsapi_items = self._fetch_newsapi(f"{query} crypto", limit)
                newsapi_items.extend(self._fetch_newsapi(f"Binance {query}", limit // 2))
                items.extend(newsapi_items)
                sources_used.append(f"NewsAPI ({len(newsapi_items)} items)")
            else:
                sources_used.append("NewsAPI (no key configured)")

            # SECONDARY: CryptoPanic (real-time crypto specific)
            if self.cryptopanic_token:
                cp_items = self._fetch_cryptopanic(limit)
                items.extend(cp_items)
                sources_used.append(f"CryptoPanic ({len(cp_items)} items)")
            else:
                sources_used.append("CryptoPanic (no token configured)")

            # TERTIARY: Reddit (community sentiment)
            reddit_items = self._fetch_reddit(query, limit)
            items.extend(reddit_items)
            sources_used.append(f"Reddit ({len(reddit_items)} items)")

            # FALLBACK: CoinGecko trending
            if len(items) < limit // 2:
                cg_items = self._fetch_coingecko_trending()
                items.extend(cg_items)
                sources_used.append(f"CoinGecko FALLBACK ({len(cg_items)} items)")
            else:
                sources_used.append("CoinGecko (skipped - sufficient items)")

        # Deduplicate by title similarity
        items = self._dedupe(items)

        # Classify events and extract tickers
        for item in items:
            item.event_type = self._classify_event(item.title)
            item.tickers = self._extract_tickers(item.title)

        # Sort by timestamp (most recent first)
        items.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Log sources used and items fetched
        print(f"\n[NewsAPI] Fetch Summary:")
        for source in sources_used:
            print(f"  • {source}")
        print(f"  └─ Total: {len(items)} unique items (after dedup)")
        
        # Show breakdown of sources in returned items
        returned_items = items[:limit]
        source_breakdown = {}
        for item in returned_items:
            source = item.source
            source_breakdown[source] = source_breakdown.get(source, 0) + 1
        
        print(f"\n[NewsAPI] Returned Items Breakdown ({len(returned_items)} items):")
        for source, count in sorted(source_breakdown.items(), key=lambda x: -x[1]):
            pct = (count / len(returned_items)) * 100 if returned_items else 0
            print(f"  • {source}: {count} items ({pct:.1f}%)")
        print()
        
        return returned_items

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

    def _fetch_alpaca_news(self, symbols: Optional[List[str]] = None,
                           limit: int = 50) -> List[NewsItem]:
        """Fetch news from Alpaca's /v1beta1/news endpoint — proper source
        for stock news (and increasingly crypto too).

        Each headline is symbol-tagged by Alpaca, so we get the precise
        per-stock news the FinBERT sentiment layer + 13-agent voter need
        instead of relying on keyword-matching against generic NewsAPI
        crypto queries (which never returned stock results).

        Free tier (paper): full access. Auth: APCA_API_KEY_ID +
        APCA_API_SECRET_KEY (same as paper trading auth).

        Symbol filter is optional. Without it, returns the broadest
        cross-symbol feed which is useful for general market sentiment.
        """
        import os as _os
        key = _os.environ.get('APCA_API_KEY_ID', '').strip()
        secret = _os.environ.get('APCA_API_SECRET_KEY', '').strip()
        if not key or not secret:
            return []
        # Cache by symbol-set so we don't re-hit Alpaca for every agent
        # asking the same question within the 120s window.
        sym_key = ",".join(sorted(symbols or [])) if symbols else "ALL"
        cache_key = f'alpaca_news_{sym_key}_{limit}'
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        items: List[NewsItem] = []
        try:
            url = "https://data.alpaca.markets/v1beta1/news"
            params: Dict[str, Any] = {
                "limit": int(min(max(limit, 1), 50)),
                "sort":  "desc",
                "include_content": "false",   # headlines + summary only
            }
            if symbols:
                params["symbols"] = ",".join(s.upper() for s in symbols)
            headers = {
                "APCA-API-KEY-ID":     key,
                "APCA-API-SECRET-KEY": secret,
                **self.headers,
            }
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code != 200:
                return []
            data = resp.json() or {}
            for article in (data.get("news") or [])[:limit]:
                headline = article.get("headline") or ""
                summary = article.get("summary") or ""
                title = headline or summary
                if not title:
                    continue
                # Alpaca returns ISO 8601 created_at
                created_at = article.get("created_at") or ""
                try:
                    ts = datetime.fromisoformat(
                        str(created_at).replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    ts = time.time()
                # Symbols field is a list — turn it into a comma string
                # so the matcher in fetch_all can extract tickers cleanly.
                syms = article.get("symbols") or []
                items.append(NewsItem(
                    title=title,
                    source=f"alpaca/{article.get('source') or 'newsapi'}",
                    timestamp=ts,
                    url=article.get("url", ""),
                    tickers=[s.upper() for s in syms if isinstance(s, str)],
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
