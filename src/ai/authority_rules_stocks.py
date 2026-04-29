"""US-stocks authority overlay.

Composes on top of `src/ai/authority_rules.py` (the 7 universal
operator rules from the authority PDF) — overlays add stocks-specific
guardrails that don't apply to crypto on Robinhood:

  1. RTH-only — no orders outside NYSE regular session.
  2. NYSE holiday calendar — explicit refusal on full closures.
  3. Pre-close blackout — last 5 min for all symbols, last 30 min for
     leveraged ETFs (TQQQ/SOXL daily-reset decay risk).
  4. Leveraged ETFs cannot be held overnight (forced flat-by-EOD via
     pre_exit hook at 15:55 ET on regular days, 12:55 ET on early
     close).
  5. Daily margin exposure cap — replaces PDT given 2026-04-14 SEC
     rule change. Default 200% (matches Alpaca 4× intraday × 50%
     safety). Configurable via ACT_STOCKS_MAX_MARGIN_EXPOSURE_PCT.
  6. Fractional-only longs — Alpaca limitation; refuse fractional
     short orders (operator must size to whole shares for shorts).
  7. ETB requirement — short-sell blocked unless symbol is on the
     Easy-To-Borrow list. Inverse ETFs (SOXS/SQQQ) are the safe
     alternative for downside exposure.

Returns AuthorityViolation list — empty list = passed. Same shape
as crypto authority_rules.evaluate so the executor's downstream
pipeline doesn't branch on asset class.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuthorityViolation:
    rule:    str
    severity: str = "hard"          # 'hard' (block) | 'soft' (advisory)
    detail:  str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule, "severity": self.severity, "detail": self.detail}


# Tunables — env-overridable so ops can adjust without restart.
def _max_margin_exposure_pct() -> float:
    return float(os.getenv("ACT_STOCKS_MAX_MARGIN_EXPOSURE_PCT", "200.0"))


def _etb_symbols() -> set:
    """Easy-To-Borrow symbols. Operator's basket only contains long-
    biased instruments (SPY, QQQ) + leveraged 3× longs (TQQQ, SOXL).
    For inverse exposure use SOXS / SQQQ instead of shorting.
    """
    raw = os.getenv("ACT_STOCKS_ETB_SYMBOLS", "SPY,QQQ,IWM,AAPL,MSFT,NVDA,TSLA,AMD")
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


# ── Public API ──────────────────────────────────────────────────────

def evaluate(
    *,
    symbol: str,
    side: str,                                # 'buy' | 'sell' | 'long' | 'short'
    qty: float,
    is_overnight: bool = False,
    fractional: bool = False,
    intent: str = "open",                     # 'open' | 'close'
    current_margin_exposure_pct: Optional[float] = None,
    base_authority_violations: Optional[List[AuthorityViolation]] = None,
) -> List[AuthorityViolation]:
    """Return list of violations. Empty list = pass.

    Composes with the base authority_rules — caller passes
    `base_authority_violations` from `src/ai/authority_rules.evaluate(...)`
    and we extend it. Stocks-only rules append; base rules pass through
    unchanged.
    """
    from src.models.asset_class import classify
    from src.utils.market_hours import (
        is_us_market_open, session_for_date_et,
        is_pre_close_blackout, is_pre_close_leveraged_blackout,
    )
    import datetime as _dt

    out: List[AuthorityViolation] = list(base_authority_violations or [])

    side_l = (side or "").strip().lower()
    is_short = side_l in ("sell", "short") and intent == "open"

    meta = classify(symbol, venue_hint="alpaca")

    # Rule 1 + 2 — RTH + NYSE calendar (intent=close trims trading-day
    # tail; rejects close-orders outside RTH because Alpaca refuses
    # them anyway, but the failure mode is uglier without an explicit
    # gate).
    if not is_us_market_open():
        sess = session_for_date_et(_dt.datetime.now(_dt.timezone.utc).date())
        if not sess.is_trading_day:
            out.append(AuthorityViolation(
                rule="stocks.rth_only", severity="hard",
                detail=f"NYSE not in session: {sess.note}",
            ))
        else:
            out.append(AuthorityViolation(
                rule="stocks.rth_only", severity="hard",
                detail="Outside regular trading hours",
            ))

    # Rule 3 — pre-close blackout
    if intent == "open" and is_pre_close_blackout(minutes=5.0):
        out.append(AuthorityViolation(
            rule="stocks.pre_close_blackout_5min", severity="hard",
            detail="Within 5 min of close — refuse new entries",
        ))
    if (intent == "open" and meta.is_leveraged_etf
            and is_pre_close_leveraged_blackout(minutes=30.0)):
        out.append(AuthorityViolation(
            rule="stocks.leveraged_etf_30min_blackout", severity="hard",
            detail=f"{symbol}: leveraged ETF inside 30-min pre-close window",
        ))

    # Rule 4 — leveraged ETFs must close by EOD
    if meta.is_leveraged_etf and is_overnight and intent == "open":
        out.append(AuthorityViolation(
            rule="stocks.leveraged_etf_no_overnight", severity="hard",
            detail=f"{symbol}: leveraged ETFs forced flat-by-EOD (daily-reset decay)",
        ))

    # Rule 5 — margin exposure cap
    cap = _max_margin_exposure_pct()
    if current_margin_exposure_pct is not None and current_margin_exposure_pct > cap:
        out.append(AuthorityViolation(
            rule="stocks.daily_margin_exposure_cap", severity="hard",
            detail=f"current exposure {current_margin_exposure_pct:.1f}% > cap {cap:.1f}%",
        ))

    # Rule 6 — fractional shorts not supported
    if fractional and is_short:
        out.append(AuthorityViolation(
            rule="stocks.fractional_short_unsupported", severity="hard",
            detail="Alpaca does not support fractional short selling",
        ))

    # Rule 7 — ETB only for shorts
    if is_short and symbol.upper() not in _etb_symbols():
        out.append(AuthorityViolation(
            rule="stocks.short_requires_etb", severity="hard",
            detail=f"{symbol}: not on Easy-To-Borrow list. Use inverse ETF instead.",
        ))

    return out


def passed(violations: List[AuthorityViolation]) -> bool:
    """True iff no `hard` violations present (soft = advisory only)."""
    return all(v.severity != "hard" for v in violations)
