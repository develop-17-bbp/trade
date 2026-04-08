"""
News Aggregator — Online Sentiment Aggregation with Auto-Fetch
================================================================
Aggregates headlines from CryptoPanic (free API) with time-decay weighting.
Falls back to empty sentiment if no API key configured.
"""

import math
import time
import logging
import threading
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from src.ai.sentiment import SentimentPipeline
    _SENTIMENT_OK = True
except Exception:
    _SENTIMENT_OK = False

# CryptoPanic free API (no key needed for public posts)
_CRYPTOPANIC_PUBLIC_URL = "https://cryptopanic.com/api/free/v1/posts/"


class NewsAggregator:
    """Online news aggregator with auto-fetch from CryptoPanic (free tier)."""

    def __init__(self, sentiment: Optional[object] = None, decay_gamma: float = 0.5,
                 fetch_interval: int = 300, max_headlines: int = 30):
        self.sentiment = sentiment
        self.gamma = decay_gamma
        self.fetch_interval = fetch_interval  # seconds between fetches
        self.max_headlines = max_headlines

        # Online headline buffer (auto-populated by background fetcher)
        self._headline_buffer: List[Dict] = []  # [{text, timestamp, source}]
        self._last_fetch: float = 0
        self._fetch_lock = threading.Lock()

    def fetch_headlines(self, asset: str = 'BTC') -> List[str]:
        """Fetch latest crypto headlines from CryptoPanic free API."""
        now = time.time()
        if now - self._last_fetch < self.fetch_interval:
            return [h['text'] for h in self._headline_buffer[-self.max_headlines:]]

        try:
            import requests
            # CryptoPanic free public endpoint (no API key needed)
            params = {'currencies': asset.lower(), 'filter': 'important', 'public': 'true'}
            resp = requests.get(_CRYPTOPANIC_PUBLIC_URL, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get('results', [])
                with self._fetch_lock:
                    new_headlines = []
                    for r in results[:self.max_headlines]:
                        title = r.get('title', '')
                        ts = time.time()  # Use current time as proxy
                        new_headlines.append({'text': title, 'timestamp': ts, 'source': 'cryptopanic'})
                    self._headline_buffer = new_headlines + self._headline_buffer
                    # Keep buffer bounded
                    self._headline_buffer = self._headline_buffer[:self.max_headlines * 2]
                    self._last_fetch = now
                    logger.info(f"[NEWS] Fetched {len(new_headlines)} headlines for {asset}")
            else:
                logger.debug(f"[NEWS] CryptoPanic returned {resp.status_code}")
        except Exception as e:
            logger.debug(f"[NEWS] Fetch failed ({e}) — using cached headlines")

        return [h['text'] for h in self._headline_buffer[-self.max_headlines:]]

    def aggregate(self, headlines: Optional[List[str]] = None, asset: str = 'BTC') -> Dict:
        """Return aggregated sentiment score with time-decay weighting.

        If headlines not provided, auto-fetches from CryptoPanic.
        """
        if headlines is None:
            headlines = self.fetch_headlines(asset)

        if not headlines:
            return {'S_t': 0.0, 'count': 0, 'details': [], 'source': 'none'}

        # Score headlines using sentiment pipeline or simple rule-based
        scored = []
        if self.sentiment and hasattr(self.sentiment, 'analyze'):
            try:
                scored = self.sentiment.analyze(headlines)
            except Exception:
                scored = []

        weighted_sum = 0.0
        weight_total = 0.0
        details = []
        for i, h in enumerate(headlines):
            dt = i  # index as proxy for age (most recent = 0)
            w = math.exp(-self.gamma * dt)

            val = 0.0
            if i < len(scored):
                s = scored[i]
                if isinstance(s, dict):
                    label = s.get('label', '').upper()
                    score = s.get('score', 0.5)
                    if label.startswith('POS') or label.startswith('BULL'):
                        val = score
                    elif label.startswith('NEG') or label.startswith('BEAR'):
                        val = -score
                else:
                    try:
                        val = float(s)
                    except Exception:
                        val = 0.0
            else:
                # Simple rule-based fallback
                val = self._quick_score(h)

            weighted_sum += val * w
            weight_total += w
            details.append({'headline': h[:80], 'val': round(val, 3), 'weight': round(w, 3)})

        S_t = weighted_sum / weight_total if weight_total > 0 else 0.0
        return {'S_t': round(S_t, 4), 'count': len(headlines), 'details': details, 'source': 'cryptopanic'}

    @staticmethod
    def _quick_score(text: str) -> float:
        """Ultra-fast keyword scoring (< 0.01ms per headline)."""
        t = text.lower()
        score = 0.0
        pos = ['bullish', 'surge', 'rally', 'breakout', 'ath', 'approved', 'adoption', 'inflow', 'pump', 'moon']
        neg = ['bearish', 'crash', 'plunge', 'dump', 'hack', 'ban', 'scam', 'liquidation', 'panic', 'selloff']
        for w in pos:
            if w in t:
                score += 0.3
        for w in neg:
            if w in t:
                score -= 0.3
        return max(-1.0, min(1.0, score))
