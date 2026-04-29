"""US market-hours helper — `is_us_market_open(now_utc)`, `next_open()`, `next_close()`.

Authoritative source: `pandas_market_calendars.get_calendar('NYSE')` when
the optional dep is installed. When it's not, we fall back to a hardcoded
2026 calendar (full closures + early-close days) so the bot doesn't
silently mis-trade on holidays.

This module is the single source of truth for "can I send a stocks
order right now?" — the stocks executor + conviction gate + finetune
router all call into it.
"""
from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Hardcoded 2026 fallback ────────────────────────────────────────────
# Source: NYSE 2026 yearly trading calendar PDF + ICE press release.
# Full-closure dates (UTC date when the closure starts).
_NYSE_FULL_CLOSURES_2026 = frozenset({
    _dt.date(2026,  1,  1),  # New Year's Day
    _dt.date(2026,  1, 19),  # MLK Day
    _dt.date(2026,  2, 16),  # Washington's Birthday
    _dt.date(2026,  4,  3),  # Good Friday
    _dt.date(2026,  5, 25),  # Memorial Day
    _dt.date(2026,  6, 19),  # Juneteenth
    _dt.date(2026,  7,  3),  # Independence Day observed (July 4 = Saturday)
    _dt.date(2026,  9,  7),  # Labor Day
    _dt.date(2026, 11, 26),  # Thanksgiving
    _dt.date(2026, 12, 25),  # Christmas
})

# Early-close days: market closes at 13:00 ET (17:00 UTC during EDT,
# 18:00 UTC during EST). 2026: Nov 27 is post-Thanksgiving + Dec 24
# is Christmas Eve.
_NYSE_EARLY_CLOSE_2026 = frozenset({
    _dt.date(2026, 11, 27),
    _dt.date(2026, 12, 24),
})

# Regular trading hours, in ET. NYSE: 09:30 → 16:00.
_RTH_OPEN_ET  = _dt.time(9, 30)
_RTH_CLOSE_ET = _dt.time(16, 0)
_EARLY_CLOSE_ET = _dt.time(13, 0)


# Try the live calendar first; if missing, every helper falls back to the
# hardcoded list above. Operator can force fallback by setting
# ACT_MARKET_HOURS_USE_FALLBACK=1.
def _live_calendar():
    import os
    if os.getenv("ACT_MARKET_HOURS_USE_FALLBACK") == "1":
        return None
    try:
        import pandas_market_calendars as mcal  # type: ignore
        return mcal.get_calendar("NYSE")
    except Exception as e:
        logger.debug("market_hours: pandas_market_calendars unavailable (%s); using fallback", e)
        return None


# ── ET ↔ UTC conversion (DST-aware) ──────────────────────────────────────


def _to_et(now_utc: _dt.datetime) -> _dt.datetime:
    """Convert UTC datetime to America/New_York. Falls back to a fixed
    -5h offset if zoneinfo isn't installed (rare on modern Pythons)."""
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=_dt.timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        return now_utc.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        # Fixed -5 fallback: imprecise across DST but better than crashing.
        return now_utc.astimezone(_dt.timezone(_dt.timedelta(hours=-5)))


def _et_to_utc(et: _dt.datetime) -> _dt.datetime:
    """Convert America/New_York datetime to UTC."""
    if et.tzinfo is None:
        try:
            from zoneinfo import ZoneInfo
            et = et.replace(tzinfo=ZoneInfo("America/New_York"))
        except Exception:
            et = et.replace(tzinfo=_dt.timezone(_dt.timedelta(hours=-5)))
    return et.astimezone(_dt.timezone.utc)


# ── Public API ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MarketSession:
    """Resolved session metadata for one date — holiday awareness baked in."""
    is_trading_day:   bool
    open_et:          Optional[_dt.time]
    close_et:         Optional[_dt.time]
    is_early_close:   bool
    note:             str


def session_for_date_et(d: _dt.date) -> MarketSession:
    """Return today's NYSE session (or 'closed' marker if holiday/weekend)."""
    cal = _live_calendar()
    if cal is not None:
        try:
            sched = cal.schedule(start_date=d, end_date=d)
            if sched.empty:
                return MarketSession(False, None, None, False, "non_trading_day")
            row = sched.iloc[0]
            open_et  = row["market_open"].tz_convert("America/New_York").time()
            close_et = row["market_close"].tz_convert("America/New_York").time()
            early = (close_et < _RTH_CLOSE_ET)
            return MarketSession(True, open_et, close_et, early,
                                 "early_close" if early else "regular")
        except Exception as e:
            logger.debug("market_hours: live calendar lookup failed for %s (%s); falling back", d, e)
            # fall through to hardcoded path

    # Hardcoded 2026 fallback.
    if d.weekday() >= 5:
        return MarketSession(False, None, None, False, "weekend")
    if d in _NYSE_FULL_CLOSURES_2026:
        return MarketSession(False, None, None, False, "holiday")
    if d in _NYSE_EARLY_CLOSE_2026:
        return MarketSession(True, _RTH_OPEN_ET, _EARLY_CLOSE_ET, True, "early_close")
    return MarketSession(True, _RTH_OPEN_ET, _RTH_CLOSE_ET, False, "regular")


def is_us_market_open(now_utc: Optional[_dt.datetime] = None) -> bool:
    """True iff NYSE is in regular (or early) cash-equity session right now."""
    now_utc = now_utc or _dt.datetime.now(_dt.timezone.utc)
    et = _to_et(now_utc)
    sess = session_for_date_et(et.date())
    if not sess.is_trading_day:
        return False
    return sess.open_et <= et.time() < sess.close_et


def next_open(now_utc: Optional[_dt.datetime] = None) -> _dt.datetime:
    """Next NYSE open (UTC). Skips weekends + holidays."""
    now_utc = now_utc or _dt.datetime.now(_dt.timezone.utc)
    et = _to_et(now_utc)
    for delta in range(0, 14):
        d = et.date() + _dt.timedelta(days=delta)
        sess = session_for_date_et(d)
        if not sess.is_trading_day:
            continue
        candidate_et = _dt.datetime.combine(d, sess.open_et)
        candidate_utc = _et_to_utc(candidate_et)
        if candidate_utc > now_utc:
            return candidate_utc
    # Beyond two weeks of misses — should never happen.
    raise RuntimeError("no NYSE trading day found within 14 days")


def next_close(now_utc: Optional[_dt.datetime] = None) -> _dt.datetime:
    """Next NYSE close (UTC). Returns today's close if currently open;
    else next trading day's close."""
    now_utc = now_utc or _dt.datetime.now(_dt.timezone.utc)
    et = _to_et(now_utc)
    for delta in range(0, 14):
        d = et.date() + _dt.timedelta(days=delta)
        sess = session_for_date_et(d)
        if not sess.is_trading_day:
            continue
        candidate_et = _dt.datetime.combine(d, sess.close_et)
        candidate_utc = _et_to_utc(candidate_et)
        if candidate_utc > now_utc:
            return candidate_utc
    raise RuntimeError("no NYSE trading day found within 14 days")


def minutes_to_close(now_utc: Optional[_dt.datetime] = None) -> Optional[float]:
    """Minutes until next close, or None if market closed right now."""
    now_utc = now_utc or _dt.datetime.now(_dt.timezone.utc)
    if not is_us_market_open(now_utc):
        return None
    return (next_close(now_utc) - now_utc).total_seconds() / 60.0


def is_pre_close_blackout(now_utc: Optional[_dt.datetime] = None,
                          minutes: float = 5.0) -> bool:
    """True if we're inside the last `minutes` before close.

    Used by the stocks authority overlay to refuse new entries once
    we're inside the blackout window — leveraged ETFs especially must
    not open positions they can't manage if liquidity tightens at the
    close auction."""
    m = minutes_to_close(now_utc)
    return m is not None and m <= minutes


def is_pre_close_leveraged_blackout(now_utc: Optional[_dt.datetime] = None,
                                    minutes: float = 30.0) -> bool:
    """True if we're inside the last `minutes` before close — used to
    refuse leveraged-ETF entries within 30 min of close to avoid
    overnight-decay / next-day-gap risk."""
    return is_pre_close_blackout(now_utc, minutes=minutes)
