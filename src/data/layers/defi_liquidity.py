"""Layer 12 — Stablecoin & DeFi Liquidity"""
import time, logging, requests
logger = logging.getLogger(__name__)

class DeFiLiquidity:
    def __init__(self):
        self._last_result = None
        self._last_fetch = 0

    def fetch(self) -> dict:
        tvl_change = 0
        total_tvl = 0
        try:
            r = requests.get("https://api.llama.fi/v2/historicalChainTvl", timeout=15)
            if r.status_code == 200:
                data = r.json()
                if len(data) >= 8:
                    current = data[-1].get('tvl', 0)
                    week_ago = data[-8].get('tvl', 0)
                    total_tvl = current
                    tvl_change = (current - week_ago) / max(week_ago, 1) * 100
        except Exception as e:
            logger.warning(f"[DEFI] DefiLlama failed: {e}")

        if tvl_change < -5:
            signal = 'BEARISH'
        elif tvl_change > 5:
            signal = 'BULLISH'
        else:
            signal = 'NEUTRAL'
        conf = min(1.0, abs(tvl_change) / 10)
        self._last_result = {'value': round(total_tvl / 1e9, 2), 'change_pct': round(tvl_change, 2),
                             'signal': signal, 'confidence': round(conf, 2), 'source': 'defillama',
                             'stale': False, 'tvl_billions': round(total_tvl / 1e9, 2)}
        self._last_fetch = time.time()
        return self._last_result

    def get_cached(self):
        return self._last_result or {'signal': 'NEUTRAL', 'confidence': 0, 'stale': True}

    def get_institutional_flow_score(self) -> int:
        cached = self.get_cached()
        tvl_chg = cached.get('change_pct', 0)
        score = int(tvl_chg * 10)
        return max(-100, min(100, score))
