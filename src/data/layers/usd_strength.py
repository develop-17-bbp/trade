"""ACT v8.0 — USD Strength economic data layer."""

import logging
import time
import json

import requests
import yfinance as yf

logger = logging.getLogger(__name__)


class USDStrength:
    """Tracks DXY / USD strength and its impact on crypto."""

    def __init__(self):
        self._last_result: dict | None = None
        self._last_fetch_time: float = 0.0

    def fetch(self) -> dict:
        """Fetch DXY data and USD exchange rates. Returns standardised dict."""
        try:
            # Try DXY first, fall back to UUP ETF
            dxy_price = None
            dxy_change_pct = 0.0
            source = "yfinance"

            for ticker_sym in ("DX-Y.NYB", "UUP"):
                try:
                    ticker = yf.Ticker(ticker_sym)
                    hist = ticker.history(period="5d")
                    if hist is not None and len(hist) >= 2:
                        dxy_price = float(hist["Close"].iloc[-1])
                        prev = float(hist["Close"].iloc[-2])
                        dxy_change_pct = ((dxy_price - prev) / prev) * 100 if prev else 0.0
                        source = f"yfinance:{ticker_sym}"
                        break
                except Exception:
                    continue

            # Supplementary: free exchange-rate API
            fx_rates = {}
            try:
                resp = requests.get(
                    "https://api.exchangerate-api.com/v4/latest/USD", timeout=10
                )
                if resp.status_code == 200:
                    fx_rates = resp.json().get("rates", {})
            except Exception:
                pass

            # Compute signal
            if dxy_change_pct > 1.0:
                signal = "BEARISH"
                confidence = min(0.9, 0.5 + abs(dxy_change_pct) * 0.1)
            elif dxy_change_pct < -0.5:
                signal = "BULLISH"
                confidence = min(0.9, 0.5 + abs(dxy_change_pct) * 0.1)
            else:
                signal = "NEUTRAL"
                confidence = 0.4

            result = {
                "value": dxy_price,
                "change_pct": round(dxy_change_pct, 4),
                "signal": signal,
                "confidence": round(confidence, 3),
                "source": source,
                "stale": False,
                "fx_rates_sample": {
                    k: fx_rates.get(k) for k in ("EUR", "GBP", "JPY", "CNY") if k in fx_rates
                },
            }
            self._last_result = result
            self._last_fetch_time = time.time()
            return result

        except Exception as e:
            logger.error("USDStrength.fetch failed: %s", e)
            if self._last_result is not None:
                stale = {**self._last_result, "stale": True}
                return stale
            return {
                "value": None,
                "change_pct": 0.0,
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "source": "error",
                "stale": True,
            }

    def get_cached(self) -> dict | None:
        """Return last fetch result."""
        return self._last_result

    def get_btc_usd_correlation_impact(self) -> dict:
        """Estimate BTC-USD correlation regime from recent data."""
        try:
            btc = yf.Ticker("BTC-USD").history(period="30d")["Close"]
            dxy = yf.Ticker("DX-Y.NYB").history(period="30d")["Close"]

            if btc is None or dxy is None or len(btc) < 5 or len(dxy) < 5:
                raise ValueError("Insufficient data")

            # Align on common dates
            import pandas as pd

            df = pd.DataFrame({"btc": btc, "dxy": dxy}).dropna()
            if len(df) < 5:
                raise ValueError("Insufficient aligned data")

            corr = float(df["btc"].pct_change().corr(df["dxy"].pct_change()))

            if corr < -0.3:
                regime = "inverse"
            elif corr > 0.3:
                regime = "correlated"
            else:
                regime = "decoupled"

            return {
                "correlation_coefficient": round(corr, 4),
                "regime": regime,
            }
        except Exception as e:
            logger.error("get_btc_usd_correlation_impact failed: %s", e)
            return {
                "correlation_coefficient": None,
                "regime": "unknown",
            }
