"""Alpaca paper-stocks bar fetcher — provides the same `fetch_ohlcv()`
interface as `src/data/fetcher.py::PriceFetcher` so the agentic-loop
quant tools, scanner, and analyst all work unchanged on stocks.

Free-tier note (Phase L of the plan):
    Alpaca's free tier serves IEX market data only — ~2-8% of total
    consolidated US equity volume. Bid/ask + fill prices the bot sees
    in paper soak are NOT what live SIP would show. Set
    ACT_ALPACA_DATA_FEED=sip ONCE the operator pays for AlgoTrader
    Plus ($99/mo) or migrates to Polygon. Until then, IEX is fine for
    relative-feature training and direction prediction; absolute
    fill-cost models will be optimistic.

Single-name vs leveraged-ETF caveats:
    Module is venue-only — it doesn't enforce the SPY/QQQ/TQQQ/SOXL
    basket. That filter lives in `config.yaml:exchanges[name=alpaca].assets`
    and is enforced by the executor before the order leaves.

Usage:
    from src.data.alpaca_fetcher import AlpacaFetcher
    f = AlpacaFetcher()
    bars = f.fetch_ohlcv('SPY', timeframe='1Min', limit=200)
    # → list of [ts_ns, open, high, low, close, volume]
    snap = f.fetch_ticker('SPY')        # last + bid + ask + ts
    health = f.health()
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Map ACT timeframe strings to Alpaca's `timeframe` parameter format.
_TIMEFRAME_MAP = {
    "1m":  "1Min", "1Min":  "1Min",
    "5m":  "5Min", "5Min":  "5Min",
    "15m": "15Min", "15Min": "15Min",
    "30m": "30Min", "30Min": "30Min",
    "1h":  "1Hour", "1Hour": "1Hour",
    "1d":  "1Day",  "1Day":  "1Day",
}


def _ts_iso_to_ns(iso: str) -> int:
    """Alpaca returns ISO 8601 with Z; convert to UTC nanos."""
    from datetime import datetime, timezone
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


class AlpacaFetcher:
    """Bars + ticker for US equities via Alpaca Market Data v2.

    Reuses the existing AlpacaClient for auth/session so we don't
    duplicate credential plumbing. If `APCA_API_KEY_ID` /
    `APCA_API_SECRET_KEY` aren't set, all methods return empty results
    + log debug — the live executor can detect via `available` and
    fall back to a different fetcher.
    """

    DATA_BASE = "https://data.alpaca.markets"

    def __init__(self, paper: bool = True):
        self.feed = (os.getenv("ACT_ALPACA_DATA_FEED") or "iex").lower()
        if self.feed not in ("iex", "sip"):
            logger.warning(
                "AlpacaFetcher: unknown feed %r, defaulting to 'iex'", self.feed,
            )
            self.feed = "iex"
        self._session = None
        self.available = False
        try:
            from src.data.fetcher import AlpacaClient
            self._client = AlpacaClient(paper=paper)
            if self._client.available:
                self._session = self._client._session
                self.available = True
        except Exception as e:
            logger.debug("AlpacaFetcher init failed: %s", e)
            self._client = None

    # ── Public API ──────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Subsystem traffic light for /status."""
        if not self.available:
            return {"available": False, "reason": "no_credentials"}
        try:
            acct = self._client.get_account()
            return {
                "available":     True,
                "feed":          self.feed,
                "paper":         self._client.paper,
                "equity":        float(acct.get("equity", 0) or 0),
                "buying_power":  float(acct.get("buying_power", 0) or 0),
                "status":        acct.get("status"),
            }
        except Exception as e:
            return {"available": False, "reason": f"probe_failed: {e}"}

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1Min",
                    limit: int = 200) -> List[List[float]]:
        """Return the most recent `limit` bars as [ts_ns, o, h, l, c, v].

        Default timeframe is `1Min` because the agentic loop runs at
        ~60-180s tick cadence — minute bars are the right resolution.

        Routes by symbol shape:
          * Stocks (e.g. 'NVDA', 'SPY')  -> /v2/stocks/{symbol}/bars
          * Crypto (e.g. 'BTC/USD')       -> /v1beta3/crypto/us/bars
            (different endpoint, response keyed by symbol map)

        Implementation note (2026-04-30 fix): Alpaca's bars endpoint
        without an explicit `start` parameter returns nothing when the
        market is closed — the default lookback window doesn't span
        overnight gaps. Outside RTH (e.g. operator's 4060 booting at
        02:25 ET), every stock symbol returned 0 candles. Forcing a
        7-day lookback always captures at least the last trading
        session, weekends + holidays included. Crypto trades 24/7 so
        the start parameter is harmless there but kept for consistency.
        """
        if not self.available or self._session is None:
            return []
        tf = _TIMEFRAME_MAP.get(timeframe) or _TIMEFRAME_MAP.get(timeframe.lower())
        if tf is None:
            logger.debug("AlpacaFetcher: unknown timeframe %r", timeframe)
            return []
        from datetime import datetime, timedelta, timezone
        start_iso = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        sym = symbol.upper()
        is_crypto = "/" in sym

        if is_crypto:
            # /v1beta3/crypto/us/bars takes `symbols` (plural) as a query
            # param, not in the path. Response shape is:
            #   {"bars": {"BTC/USD": [{"t": ..., ...}, ...]}}
            # so we have to dig into the per-symbol map. Crypto endpoint
            # ignores the `feed` parameter (no IEX/SIP distinction for
            # 24/7 crypto markets).
            url = f"{self.DATA_BASE}/v1beta3/crypto/us/bars"
            params: Dict[str, Any] = {
                "symbols":    sym,
                "timeframe":  tf,
                "start":      start_iso,
                "limit":      int(min(max(limit, 1), 10000)),
                "sort":       "desc",
            }
        else:
            url = f"{self.DATA_BASE}/v2/stocks/{sym}/bars"
            params = {
                "timeframe":  tf,
                "start":      start_iso,
                "limit":      int(min(max(limit, 1), 10000)),
                "feed":       self.feed,
                "adjustment": "raw",
                "sort":       "desc",
            }

        try:
            resp = self._session.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                logger.debug(
                    "AlpacaFetcher.fetch_ohlcv(%s, %s) HTTP %s: %s",
                    symbol, tf, resp.status_code, resp.text[:200],
                )
                return []
            data = resp.json() or {}
        except Exception as e:
            logger.debug("AlpacaFetcher.fetch_ohlcv: %s", e)
            return []

        if is_crypto:
            bars = (data.get("bars") or {}).get(sym) or []
        else:
            bars = data.get("bars") or []
        # Reverse so newest is last (matches CCXT convention).
        bars.reverse()
        rows: List[List[float]] = []
        for b in bars:
            try:
                rows.append([
                    _ts_iso_to_ns(b["t"]),
                    float(b["o"]),
                    float(b["h"]),
                    float(b["l"]),
                    float(b["c"]),
                    float(b.get("v") or 0.0),
                ])
            except Exception:
                continue
        return rows

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Latest quote (best bid + ask + last trade).

        Routes by symbol shape (same as fetch_ohlcv):
          * Stocks  -> /v2/stocks/{sym}/quotes/latest + /trades/latest
          * Crypto  -> /v1beta3/crypto/us/latest/quotes?symbols=BTC/USD
                       /v1beta3/crypto/us/latest/trades?symbols=BTC/USD
        """
        if not self.available or self._session is None:
            return {}
        sym = symbol.upper()
        is_crypto = "/" in sym
        out: Dict[str, Any] = {"symbol": sym, "ts_ns": time.time_ns()}
        try:
            if is_crypto:
                quote_resp = self._session.get(
                    f"{self.DATA_BASE}/v1beta3/crypto/us/latest/quotes",
                    params={"symbols": sym},
                    timeout=8,
                )
                if quote_resp.status_code == 200:
                    q = (quote_resp.json() or {}).get("quotes", {}).get(sym) or {}
                    out["bid"] = float(q.get("bp", 0) or 0)
                    out["ask"] = float(q.get("ap", 0) or 0)
                    out["bid_size"] = int(q.get("bs", 0) or 0)
                    out["ask_size"] = int(q.get("as", 0) or 0)
                    out["quote_ts"] = q.get("t")
                trade_resp = self._session.get(
                    f"{self.DATA_BASE}/v1beta3/crypto/us/latest/trades",
                    params={"symbols": sym},
                    timeout=8,
                )
                if trade_resp.status_code == 200:
                    t = (trade_resp.json() or {}).get("trades", {}).get(sym) or {}
                    out["last"] = float(t.get("p", 0) or 0)
                    out["last_size"] = int(t.get("s", 0) or 0)
                    out["trade_ts"] = t.get("t")
            else:
                quote_resp = self._session.get(
                    f"{self.DATA_BASE}/v2/stocks/{sym}/quotes/latest",
                    params={"feed": self.feed},
                    timeout=8,
                )
                if quote_resp.status_code == 200:
                    q = (quote_resp.json() or {}).get("quote") or {}
                    out["bid"] = float(q.get("bp", 0) or 0)
                    out["ask"] = float(q.get("ap", 0) or 0)
                    out["bid_size"] = int(q.get("bs", 0) or 0)
                    out["ask_size"] = int(q.get("as", 0) or 0)
                    out["quote_ts"] = q.get("t")
                trade_resp = self._session.get(
                    f"{self.DATA_BASE}/v2/stocks/{sym}/trades/latest",
                    params={"feed": self.feed},
                    timeout=8,
                )
                if trade_resp.status_code == 200:
                    t = (trade_resp.json() or {}).get("trade") or {}
                    out["last"] = float(t.get("p", 0) or 0)
                    out["last_size"] = int(t.get("s", 0) or 0)
                    out["trade_ts"] = t.get("t")
        except Exception as e:
            logger.debug("AlpacaFetcher.fetch_ticker: %s", e)
        return out

    def fetch_account(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False}
        return self._client.get_account()
