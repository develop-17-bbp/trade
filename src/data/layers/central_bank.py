"""ACT v8.0 — Central Bank / interest-rate economic data layer."""

import logging
import time

import requests
import yfinance as yf

logger = logging.getLogger(__name__)


class CentralBank:
    """Tracks Fed rates, Treasury yields, and yield-curve dynamics."""

    def __init__(self, fred_api_key: str | None = None):
        self._fred_api_key = fred_api_key
        self._last_result: dict | None = None
        self._last_fetch_time: float = 0.0

    # ── internal helpers ──────────────────────────────────────────

    def _fetch_fred_series(self, series_id: str) -> float | None:
        """Fetch latest value from FRED API."""
        if not self._fred_api_key:
            return None
        try:
            from fredapi import Fred

            fred = Fred(api_key=self._fred_api_key)
            s = fred.get_series(series_id)
            if s is not None and len(s) > 0:
                return float(s.dropna().iloc[-1])
        except Exception as e:
            logger.debug("FRED fetch %s failed: %s", series_id, e)
        return None

    def _fetch_yfinance_yield(self, ticker_sym: str) -> float | None:
        try:
            hist = yf.Ticker(ticker_sym).history(period="5d")
            if hist is not None and len(hist) >= 1:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return None

    # ── public API ────────────────────────────────────────────────

    def fetch(self) -> dict:
        try:
            # 10-Year Treasury
            y10 = self._fetch_fred_series("DGS10") or self._fetch_yfinance_yield("^TNX")
            # 2-Year Treasury
            y2 = self._fetch_fred_series("DGS2") or self._fetch_yfinance_yield("^TYX")
            # Fed Funds Rate
            fed_rate = self._fetch_fred_series("FEDFUNDS")

            spread = round(y10 - y2, 4) if y10 is not None and y2 is not None else None

            # Signal logic
            if spread is not None and spread < 0:
                signal = "CRISIS"
                confidence = min(0.95, 0.6 + abs(spread) * 0.1)
            elif fed_rate is not None:
                # Simple heuristic: compare to a neutral-ish 3 %
                if fed_rate < 2.5:
                    signal = "BULLISH"
                    confidence = 0.6
                elif fed_rate > 5.0:
                    signal = "BEARISH"
                    confidence = 0.7
                else:
                    signal = "NEUTRAL"
                    confidence = 0.4
            else:
                signal = "NEUTRAL"
                confidence = 0.3

            result = {
                "value": {
                    "10y": y10,
                    "2y": y2,
                    "fed_rate": fed_rate,
                    "spread_10y_2y": spread,
                },
                "change_pct": None,
                "signal": signal,
                "confidence": round(confidence, 3),
                "source": "fred" if self._fred_api_key else "yfinance",
                "stale": False,
            }
            self._last_result = result
            self._last_fetch_time = time.time()
            return result

        except Exception as e:
            logger.error("CentralBank.fetch failed: %s", e)
            if self._last_result is not None:
                return {**self._last_result, "stale": True}
            return {
                "value": None,
                "change_pct": None,
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "source": "error",
                "stale": True,
            }

    def get_cached(self) -> dict | None:
        return self._last_result

    def get_liquidity_regime(self) -> str:
        """Return 'EXPANDING', 'CONTRACTING', or 'NEUTRAL'."""
        r = self._last_result
        if r is None:
            return "NEUTRAL"
        v = r.get("value") or {}
        spread = v.get("spread_10y_2y")
        fed_rate = v.get("fed_rate")

        if spread is not None and spread < -0.5:
            return "CONTRACTING"
        if fed_rate is not None and fed_rate < 2.0:
            return "EXPANDING"
        if fed_rate is not None and fed_rate > 5.0:
            return "CONTRACTING"
        return "NEUTRAL"
