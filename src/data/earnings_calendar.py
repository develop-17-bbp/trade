"""Earnings calendar fetcher — equity equivalent of crypto's halving cycle.

The brain needs to know: "Is this stock about to print earnings?" Earnings
windows are the single biggest source of overnight gap risk in single-name
equities. The conviction gate uses `next_earnings_in_days(asset)` to:

  - Refuse swing entries within 3 trading days of an earnings print.
  - Tighten size on intraday plans within the same window.
  - Skip the post-earnings 1-2 sessions (IV crush + algo whipsaw).

For ETFs (SPY/QQQ/TQQQ/SOXL) earnings_in_days returns +inf — they don't
print earnings, so the guard is a no-op.

Source priority (graceful fallback chain):
  1. yfinance Ticker.calendar (free, no key) — most reliable.
  2. Alpaca v2 calendar (no earnings, only trading days) — used as the
     fallback for ETFs and for the trading-day grid the conviction gate
     uses to count "trading days until earnings".
  3. Hard fallback: return +inf so the gate behaves as if no earnings
     are imminent — fail-open is safer than fail-closed at this layer
     (the bot would never trade if every fetcher is down).

Cache: 6h TTL — earnings dates don't move intraday and the SEC pre-files
within 30 days of the print, so a 6-hour staleness window is fine.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

from src.data.base_fetcher import CachedFetcher, CACHE_TTL_LONG

logger = logging.getLogger(__name__)

# How long earnings-date lookups are cached. 6 hours — earnings dates are
# announced 4+ weeks ahead and rarely move; intraday churn would just
# burn API budget.
CACHE_TTL_EARNINGS = 6 * 3600

# ETFs in the active basket — they don't have earnings prints.
_ETFS = frozenset({"SPY", "QQQ", "TQQQ", "SOXL", "IWM", "DIA", "VOO", "VTI",
                   "UPRO", "SQQQ", "SPXU", "SOXS"})


class EarningsCalendar(CachedFetcher):
    """Next-earnings lookups for the active equity basket.

    Public API:
        next_earnings_dt(symbol) -> datetime | None
        next_earnings_in_days(symbol) -> float   (+inf if none / ETF)
        is_earnings_window(symbol, days=3) -> bool
    """

    def __init__(self) -> None:
        super().__init__(timeout=8)

    # ── Core lookup ──────────────────────────────────────────────────────

    def next_earnings_dt(self, symbol: str) -> Optional[datetime]:
        """UTC datetime of the next earnings print, or None if unknown."""
        sym = (symbol or "").upper().strip()
        if not sym or sym in _ETFS:
            return None

        cache_key = f"earnings:{sym}"
        cached = self._get_cached(cache_key)
        if cached:
            # Cache stores epoch seconds; convert back to datetime.
            return datetime.fromtimestamp(cached["ts"], tz=timezone.utc) if cached.get("ts") else None

        dt = self._fetch_yfinance(sym)
        # Cache the result either way (None caches as no-earnings) so we
        # don't hammer the API for tickers without earnings data.
        self._set_cached(
            cache_key,
            {"ts": dt.timestamp() if dt else None},
            ttl=CACHE_TTL_EARNINGS,
        )
        return dt

    def next_earnings_in_days(self, symbol: str) -> float:
        """Trading-days-ish until the next earnings print. +inf if unknown
        or ETF. Used by conviction gate as a soft filter — see
        `is_earnings_window` for the boolean form."""
        dt = self.next_earnings_dt(symbol)
        if dt is None:
            return math.inf
        delta = dt - datetime.now(tz=timezone.utc)
        # Calendar days (good enough — the gate is intentionally fuzzy on
        # weekends; an earnings print 3 calendar days out covers any
        # weekend-adjacent print without a full trading-day calendar).
        return max(0.0, delta.total_seconds() / 86400.0)

    def is_earnings_window(self, symbol: str, days: int = 3) -> bool:
        """True if `symbol` prints earnings within `days` calendar days."""
        return self.next_earnings_in_days(symbol) <= float(days)

    # ── Source: yfinance ─────────────────────────────────────────────────

    def _fetch_yfinance(self, sym: str) -> Optional[datetime]:
        """Try yfinance first — free, no key, but the optional dep may
        not be installed on every box. Returns None on any failure."""
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            logger.debug("earnings_calendar: yfinance not installed; skipping %s", sym)
            return None

        try:
            ticker = yf.Ticker(sym)
            # `.calendar` returns a dict like {'Earnings Date': [datetime, ...]}
            # in newer yfinance versions, or a DataFrame in older ones. Handle both.
            cal = getattr(ticker, "calendar", None)
            if cal is None:
                return None

            dates = None
            if isinstance(cal, dict):
                dates = cal.get("Earnings Date") or cal.get("earnings_date")
            else:
                # DataFrame path
                try:
                    if "Earnings Date" in getattr(cal, "index", []):
                        dates = cal.loc["Earnings Date"].dropna().tolist()
                except Exception:
                    pass

            if not dates:
                return None

            # Pick the earliest future date.
            now = datetime.now(tz=timezone.utc)
            future = []
            for d in (dates if isinstance(dates, (list, tuple)) else [dates]):
                if d is None:
                    continue
                try:
                    if isinstance(d, datetime):
                        dt = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
                    else:
                        # Try string parse, then pandas Timestamp.
                        dt = datetime.fromisoformat(str(d))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                if dt > now:
                    future.append(dt)

            return min(future) if future else None
        except Exception as e:
            logger.debug("earnings_calendar: yfinance lookup failed for %s: %s", sym, e)
            return None


# Module-level singleton — fetchers are designed to be process-global so
# the cache is shared across callers.
_singleton: Optional[EarningsCalendar] = None


def get_earnings_calendar() -> EarningsCalendar:
    global _singleton
    if _singleton is None:
        _singleton = EarningsCalendar()
    return _singleton


def next_earnings_in_days(symbol: str) -> float:
    """Convenience wrapper used by conviction_gate / factor_synthesis."""
    return get_earnings_calendar().next_earnings_in_days(symbol)


def is_earnings_window(symbol: str, days: int = 3) -> bool:
    return get_earnings_calendar().is_earnings_window(symbol, days=days)
