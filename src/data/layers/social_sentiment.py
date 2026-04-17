"""Layer 6 — Fear & Greed + Social Sentiment"""
import time, logging, requests
logger = logging.getLogger(__name__)

class SocialSentiment:
    def __init__(self):
        self._last_result = None
        self._last_fetch = 0

    def fetch(self) -> dict:
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=7", timeout=10)
            data = r.json().get('data', [])
            if data:
                current = int(data[0].get('value', 50))
                prev = int(data[1].get('value', 50)) if len(data) > 1 else 50
                change = current - prev
                if current < 20:
                    signal = 'BULLISH'  # extreme fear = contrarian buy
                elif current > 80:
                    signal = 'BEARISH'  # extreme greed = top signal
                else:
                    signal = 'NEUTRAL'
                conf = abs(current - 50) / 50.0
            else:
                current, change, signal, conf = 50, 0, 'NEUTRAL', 0.3
            self._last_result = {'value': current, 'change_pct': change, 'signal': signal,
                                 'confidence': round(conf, 2), 'source': 'alternative.me', 'stale': False,
                                 'label': data[0].get('value_classification', '') if data else ''}
            self._last_fetch = time.time()
        except Exception as e:
            logger.warning(f"[SOCIAL] fetch failed: {e}")
            if self._last_result:
                self._last_result['stale'] = True
            else:
                self._last_result = {'value': 50, 'change_pct': 0, 'signal': 'NEUTRAL', 'confidence': 0, 'source': 'alternative.me', 'stale': True}
        return self._last_result

    def get_cached(self):
        return self._last_result or {'signal': 'NEUTRAL', 'confidence': 0, 'stale': True}

    def get_retail_vs_institutional_signal(self):
        cached = self.get_cached()
        val = cached.get('value', 50)
        if val > 75:
            return 'RETAIL_FOMO'
        elif val < 25:
            return 'INSTITUTIONAL_ACCUMULATION'
        return 'NEUTRAL'
