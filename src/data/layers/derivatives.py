"""Layer 11 — Derivatives & Options Flow"""
import time, logging, requests
logger = logging.getLogger(__name__)

class Derivatives:
    def __init__(self):
        self._last_result = None
        self._last_fetch = 0

    def fetch(self) -> dict:
        put_call = 1.0
        funding_rate = 0.0
        try:
            # Deribit options P/C ratio
            r = requests.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option", timeout=10)
            if r.status_code == 200:
                options = r.json().get('result', [])
                puts = sum(o.get('open_interest', 0) for o in options if '-P' in o.get('instrument_name', ''))
                calls = sum(o.get('open_interest', 0) for o in options if '-C' in o.get('instrument_name', ''))
                put_call = round(puts / max(calls, 1), 3)
        except Exception as e:
            logger.debug(f"[DERIV] Deribit failed: {e}")

        try:
            # Bybit funding rate
            r2 = requests.get("https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT", timeout=10)
            if r2.status_code == 200:
                tickers = r2.json().get('result', {}).get('list', [])
                if tickers:
                    funding_rate = float(tickers[0].get('fundingRate', 0)) * 100
        except Exception as e:
            logger.debug(f"[DERIV] Bybit funding failed: {e}")

        if put_call > 1.2 or funding_rate > 0.1:
            signal = 'BEARISH'
        elif put_call < 0.7 or funding_rate < -0.05:
            signal = 'BULLISH'
        else:
            signal = 'NEUTRAL'
        conf = min(1.0, abs(put_call - 1.0) + abs(funding_rate) * 5)
        self._last_result = {'value': put_call, 'change_pct': round(funding_rate, 4), 'signal': signal,
                             'confidence': round(conf, 2), 'source': 'deribit+bybit', 'stale': False,
                             'put_call_ratio': put_call, 'funding_rate_pct': funding_rate}
        self._last_fetch = time.time()
        return self._last_result

    def get_cached(self):
        return self._last_result or {'signal': 'NEUTRAL', 'confidence': 0, 'stale': True}
