"""Layer 9 — Crypto Regulatory & Legal Monitor"""
import time, logging, os, requests
logger = logging.getLogger(__name__)

KEYWORDS = ["SEC crypto", "bitcoin ETF", "crypto regulation", "crypto ban", "exchange hack", "stablecoin law"]

class Regulatory:
    def __init__(self):
        self._last_result = None
        self._last_fetch = 0
        self._api_key = os.environ.get('NEWS_API_KEY', '')

    def fetch(self) -> dict:
        if not self._api_key:
            self._last_result = {'value': 0, 'change_pct': 0, 'signal': 'NEUTRAL', 'confidence': 0,
                                 'source': 'newsapi', 'stale': True, 'regulatory_risk_score': 0}
            return self._last_result
        try:
            query = ' OR '.join(f'"{kw}"' for kw in KEYWORDS[:3])
            r = requests.get(f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=20",
                             headers={"X-Api-Key": self._api_key}, timeout=15)
            data = r.json()
            articles = data.get('articles', [])
            negative_kw = ['ban', 'hack', 'enforcement', 'lawsuit', 'fraud', 'ponzi']
            positive_kw = ['ETF approved', 'legal clarity', 'favorable', 'adoption']
            neg_count = sum(1 for a in articles if any(kw in (a.get('title', '') + a.get('description', '')).lower() for kw in negative_kw))
            pos_count = sum(1 for a in articles if any(kw in (a.get('title', '') + a.get('description', '')).lower() for kw in positive_kw))
            risk_score = min(100, int(neg_count / max(1, len(articles)) * 100))
            if neg_count > pos_count + 2:
                signal = 'BEARISH'
            elif pos_count > neg_count + 2:
                signal = 'BULLISH'
            else:
                signal = 'NEUTRAL'
            self._last_result = {'value': len(articles), 'change_pct': 0, 'signal': signal,
                                 'confidence': round(min(1.0, len(articles) / 20), 2), 'source': 'newsapi',
                                 'stale': False, 'regulatory_risk_score': risk_score}
            self._last_fetch = time.time()
        except Exception as e:
            logger.warning(f"[REGULATORY] fetch failed: {e}")
            if self._last_result:
                self._last_result['stale'] = True
            else:
                self._last_result = {'value': 0, 'change_pct': 0, 'signal': 'NEUTRAL', 'confidence': 0, 'source': 'newsapi', 'stale': True}
        return self._last_result

    def get_cached(self):
        return self._last_result or {'signal': 'NEUTRAL', 'confidence': 0, 'stale': True}
