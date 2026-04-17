"""Layer 10 — Energy & Mining Economics"""
import time, logging
logger = logging.getLogger(__name__)

class MiningEconomics:
    def __init__(self):
        self._last_result = None
        self._last_fetch = 0

    def fetch(self) -> dict:
        try:
            import yfinance as yf
            oil = yf.Ticker("CL=F").history(period="5d")
            if len(oil) >= 2:
                oil_price = float(oil['Close'].iloc[-1])
                oil_prev = float(oil['Close'].iloc[-2])
                oil_chg = (oil_price - oil_prev) / oil_prev * 100
            else:
                oil_price, oil_chg = 70, 0
            if oil_chg > 5:
                signal = 'BEARISH'  # energy spike = mining cost shock
            elif oil_chg < -3:
                signal = 'BULLISH'  # cheaper energy = more mining profit
            else:
                signal = 'NEUTRAL'
            conf = min(1.0, abs(oil_chg) / 10)
            self._last_result = {'value': round(oil_price, 2), 'change_pct': round(oil_chg, 2),
                                 'signal': signal, 'confidence': round(conf, 2), 'source': 'yfinance',
                                 'stale': False, 'eth_staking_yield': 3.5}
            self._last_fetch = time.time()
        except Exception as e:
            logger.warning(f"[MINING] fetch failed: {e}")
            if self._last_result:
                self._last_result['stale'] = True
            else:
                self._last_result = {'value': 70, 'change_pct': 0, 'signal': 'NEUTRAL', 'confidence': 0, 'source': 'yfinance', 'stale': True}
        return self._last_result

    def get_cached(self):
        return self._last_result or {'signal': 'NEUTRAL', 'confidence': 0, 'stale': True}
