"""Layer 5 — Crypto On-Chain Intelligence"""
import time, logging, requests
logger = logging.getLogger(__name__)

class OnChain:
    def __init__(self):
        self._last_result = None
        self._last_fetch = 0

    def fetch(self) -> dict:
        try:
            r = requests.get("https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan=7days&format=json", timeout=10)
            data = r.json()
            values = [p['y'] for p in data.get('values', [])[-7:]]
            if len(values) >= 2:
                change = (values[-1] - values[-2]) / max(values[-2], 1) * 100
                signal = 'BULLISH' if change < -10 else ('BEARISH' if change > 15 else 'NEUTRAL')
                conf = min(1.0, abs(change) / 30)
            else:
                change, signal, conf = 0, 'NEUTRAL', 0.3
            self._last_result = {'value': values[-1] if values else 0, 'change_pct': round(change, 2),
                                 'signal': signal, 'confidence': round(conf, 2), 'source': 'blockchain.info', 'stale': False}
            self._last_fetch = time.time()
        except Exception as e:
            logger.warning(f"[ONCHAIN] fetch failed: {e}")
            if self._last_result:
                self._last_result['stale'] = True
            else:
                self._last_result = {'value': 0, 'change_pct': 0, 'signal': 'NEUTRAL', 'confidence': 0, 'source': 'blockchain.info', 'stale': True}
        return self._last_result

    def get_cached(self):
        return self._last_result or {'signal': 'NEUTRAL', 'confidence': 0, 'stale': True}

    def get_liquidity_walls(self):
        return {}
