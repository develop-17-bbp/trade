"""Layer 7 — Global Equity Market Correlation"""
import time, logging
logger = logging.getLogger(__name__)

class EquityCorrelation:
    def __init__(self):
        self._last_result = None
        self._last_fetch = 0

    def fetch(self) -> dict:
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX").history(period="5d")
            spy = yf.Ticker("SPY").history(period="5d")
            qqq = yf.Ticker("QQQ").history(period="5d")
            vix_val = float(vix['Close'].iloc[-1]) if len(vix) > 0 else 20
            spy_chg = float((spy['Close'].iloc[-1] / spy['Close'].iloc[-2] - 1) * 100) if len(spy) > 1 else 0
            qqq_chg = float((qqq['Close'].iloc[-1] / qqq['Close'].iloc[-2] - 1) * 100) if len(qqq) > 1 else 0
            if vix_val > 30:
                signal = 'BEARISH'
            elif vix_val < 15:
                signal = 'BULLISH'
            else:
                signal = 'NEUTRAL'
            conf = min(1.0, abs(vix_val - 20) / 20)
            self._last_result = {'value': round(vix_val, 1), 'change_pct': round(spy_chg, 2),
                                 'signal': signal, 'confidence': round(conf, 2), 'source': 'yfinance',
                                 'stale': False, 'vix': vix_val, 'spy_chg': spy_chg, 'qqq_chg': qqq_chg}
            self._last_fetch = time.time()
        except Exception as e:
            logger.warning(f"[EQUITY] fetch failed: {e}")
            if self._last_result:
                self._last_result['stale'] = True
            else:
                self._last_result = {'value': 20, 'change_pct': 0, 'signal': 'NEUTRAL', 'confidence': 0, 'source': 'yfinance', 'stale': True}
        return self._last_result

    def get_cached(self):
        return self._last_result or {'signal': 'NEUTRAL', 'confidence': 0, 'stale': True}

    def get_risk_on_off_score(self) -> int:
        cached = self.get_cached()
        vix = cached.get('vix', 20)
        spy = cached.get('spy_chg', 0)
        score = int(-((vix - 20) * 3) + spy * 10)
        return max(-100, min(100, score))

    def is_correlated_selloff(self, threshold=-0.02) -> bool:
        cached = self.get_cached()
        spy = cached.get('spy_chg', 0) / 100
        qqq = cached.get('qqq_chg', 0) / 100
        return spy < threshold and qqq < threshold
