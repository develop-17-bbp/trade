"""ACT v8.0 — Macro-economic indicators data layer."""

import logging
import time
from datetime import datetime, timedelta

import requests
import yfinance as yf

logger = logging.getLogger(__name__)


class MacroIndicators:
    """Tracks CPI, unemployment, GDP and derives macro regime."""

    def __init__(self, fred_api_key: str | None = None):
        self._fred_api_key = fred_api_key
        self._last_result: dict | None = None
        self._last_fetch_time: float = 0.0

    # ── helpers ───────────────────────────────────────────────────

    def _fred(self, series_id: str) -> float | None:
        if not self._fred_api_key:
            return None
        try:
            from fredapi import Fred

            s = Fred(api_key=self._fred_api_key).get_series(series_id)
            if s is not None and len(s) > 0:
                return float(s.dropna().iloc[-1])
        except Exception as e:
            logger.debug("FRED %s: %s", series_id, e)
        return None

    @staticmethod
    def _yf_proxy(ticker_sym: str) -> float | None:
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
            cpi = self._fred("CPIAUCSL")
            unemployment = self._fred("UNRATE")
            gdp = self._fred("GDP")

            # yfinance proxies if FRED unavailable
            source = "fred" if self._fred_api_key else "yfinance"
            if cpi is None:
                # TIP ETF as inflation proxy (price rises with inflation expectations)
                cpi = self._yf_proxy("TIP")
                if cpi is not None:
                    source = "yfinance:TIP"
            if unemployment is None:
                # No direct proxy; leave None
                pass

            # Signal logic
            high_cpi = cpi is not None and cpi > 280  # rough CPI index level threshold
            rising_unemp = unemployment is not None and unemployment > 5.0

            if high_cpi and rising_unemp:
                signal = "CRISIS"
                confidence = 0.8
            elif high_cpi:
                signal = "BEARISH"
                confidence = 0.6
            elif cpi is not None and not high_cpi:
                signal = "BULLISH"
                confidence = 0.55
            else:
                signal = "NEUTRAL"
                confidence = 0.3

            result = {
                "value": {"cpi": cpi, "unemployment": unemployment, "gdp": gdp},
                "change_pct": None,
                "signal": signal,
                "confidence": round(confidence, 3),
                "source": source,
                "stale": False,
            }
            self._last_result = result
            self._last_fetch_time = time.time()
            return result

        except Exception as e:
            logger.error("MacroIndicators.fetch failed: %s", e)
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

    def get_macro_regime(self) -> str:
        """Classify current macro regime."""
        r = self._last_result
        if r is None:
            return "NEUTRAL"
        v = r.get("value") or {}
        cpi = v.get("cpi")
        unemp = v.get("unemployment")

        high_cpi = cpi is not None and cpi > 280
        high_unemp = unemp is not None and unemp > 5.0

        if high_cpi and high_unemp:
            return "STAGFLATION"
        if high_cpi and not high_unemp:
            return "REFLATION"
        if not high_cpi and high_unemp:
            return "DEFLATION"
        return "GOLDILOCKS"

    @staticmethod
    def get_next_cpi_release_days() -> int:
        """Approximate days until next CPI release (~13th of each month)."""
        today = datetime.utcnow().date()
        # CPI is typically released around the 13th
        release_day = 13
        year, month = today.year, today.month

        candidate = today.replace(day=release_day)
        if today.day > release_day:
            # Next month
            if month == 12:
                candidate = today.replace(year=year + 1, month=1, day=release_day)
            else:
                candidate = today.replace(month=month + 1, day=release_day)

        return (candidate - today).days
