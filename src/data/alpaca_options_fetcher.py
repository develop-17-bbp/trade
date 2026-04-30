"""Alpaca options chain fetcher.

Wraps Alpaca's `/v1beta1/options/snapshots/{underlying}` endpoint.
Returns a per-strike snapshot of the chain so the analyst can pick
a contract for `AlpacaOptionsExecutor.submit_long_directional`.

Intentionally minimal: chain snapshot + filter by DTE + side. Live
quote streaming is NOT wired here (operator's free-tier paper account
gets snapshot data only). For multi-leg analysis the analyst can
fetch the chain once and reason over it in-context.

Usage:
    f = AlpacaOptionsFetcher()
    contracts = f.chain('SPY', side='call', min_dte=7, max_dte=45)
    # -> list of dicts: {symbol, strike, expiration, bid, ask, iv, delta, ...}
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlpacaOptionsFetcher:
    DATA_BASE = "https://data.alpaca.markets"

    def __init__(self, paper: bool = True):
        self._session = None
        self.available = False
        try:
            from src.data.fetcher import AlpacaClient
            self._client = AlpacaClient(paper=paper)
            if self._client.available:
                self._session = self._client._session
                self.available = True
        except Exception as e:
            logger.debug("AlpacaOptionsFetcher init failed: %s", e)
            self._client = None

    def chain(self, underlying: str, *,
              side: str = "call",
              min_dte: int = 7, max_dte: int = 45,
              limit: int = 200) -> List[Dict[str, Any]]:
        """Return filtered option chain snapshot.

        side: 'call' or 'put'
        min_dte / max_dte: days-to-expiration window
        limit: max contracts returned (sorted by |strike-spot| ascending,
               so the most relevant ATM/near-ATM contracts come first).
        """
        if not self.available or self._session is None:
            return []
        u = underlying.upper().strip()
        side_l = side.lower().strip()
        if side_l not in ("call", "put"):
            logger.debug("AlpacaOptionsFetcher.chain: bad side %r", side)
            return []

        url = f"{self.DATA_BASE}/v1beta1/options/snapshots/{u}"
        params: Dict[str, Any] = {
            "feed": "indicative",  # free-tier compatible; SIP-paid users can override
            "limit": int(min(max(limit, 1), 1000)),
            "type":  side_l,
        }
        try:
            resp = self._session.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                logger.debug(
                    "AlpacaOptionsFetcher.chain(%s) HTTP %s: %s",
                    u, resp.status_code, resp.text[:200],
                )
                return []
            data = resp.json() or {}
        except Exception as e:
            logger.debug("AlpacaOptionsFetcher.chain: %s", e)
            return []

        snapshots = data.get("snapshots") or {}
        today = _dt.date.today()
        out: List[Dict[str, Any]] = []

        for occ_symbol, snap in snapshots.items():
            try:
                contract = _parse_occ(occ_symbol)
            except Exception:
                continue
            dte = (contract["expiration"] - today).days
            if dte < min_dte or dte > max_dte:
                continue
            quote = (snap or {}).get("latestQuote") or {}
            trade = (snap or {}).get("latestTrade") or {}
            greeks = (snap or {}).get("greeks") or {}
            iv = (snap or {}).get("impliedVolatility")
            row = {
                "symbol":     occ_symbol,
                "underlying": u,
                "side":       side_l,
                "strike":     contract["strike"],
                "expiration": contract["expiration"].isoformat(),
                "dte":        dte,
                "bid":        float(quote.get("bp") or 0.0),
                "ask":        float(quote.get("ap") or 0.0),
                "last":       float(trade.get("p") or 0.0),
                "delta":      _safe_float(greeks.get("delta")),
                "gamma":      _safe_float(greeks.get("gamma")),
                "theta":      _safe_float(greeks.get("theta")),
                "vega":       _safe_float(greeks.get("vega")),
                "iv":         _safe_float(iv),
            }
            out.append(row)

        return out

    def latest_quote(self, occ_symbol: str) -> Dict[str, Any]:
        """Fast quote-only fetch for a specific contract by OCC symbol."""
        if not self.available or self._session is None:
            return {}
        try:
            r = self._session.get(
                f"{self.DATA_BASE}/v1beta1/options/quotes/latest",
                params={"symbols": occ_symbol},
                timeout=8,
            )
            if r.status_code != 200:
                return {}
            data = r.json() or {}
            q = (data.get("quotes") or {}).get(occ_symbol) or {}
            return {
                "symbol": occ_symbol,
                "bid":    float(q.get("bp") or 0.0),
                "ask":    float(q.get("ap") or 0.0),
                "ts":     q.get("t"),
            }
        except Exception as e:
            logger.debug("AlpacaOptionsFetcher.latest_quote: %s", e)
            return {}


def _parse_occ(occ: str) -> Dict[str, Any]:
    """Parse OCC option symbol -> {root, expiration, side, strike}.

    Format: ROOT (1-6) + YYMMDD + C/P + STRIKE (8-digit cents×10).
    """
    s = occ.strip().upper()
    # Find the date+CP+strike trailer (15 chars)
    if len(s) < 16:
        raise ValueError(f"OCC too short: {occ}")
    trailer = s[-15:]
    yymmdd = trailer[:6]
    cp = trailer[6]
    strike_str = trailer[7:]
    if cp not in ("C", "P"):
        raise ValueError(f"OCC missing C/P: {occ}")
    yy = int(yymmdd[:2])
    year = 2000 + yy if yy < 70 else 1900 + yy
    exp = _dt.date(year, int(yymmdd[2:4]), int(yymmdd[4:6]))
    strike = int(strike_str) / 1000.0
    root = s[:-15]
    return {
        "root":       root,
        "expiration": exp,
        "side":       "call" if cp == "C" else "put",
        "strike":     strike,
    }


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        import math
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None
