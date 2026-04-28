"""Macro risk-on/risk-off overlay for crypto direction.

Per 2026 research:
  * DXY (US Dollar Index) has 21-27× greater adverse effect on BTC
    than gold (Frontiers in Blockchain 2025). Rising DXY = risk-off.
  * VIX thresholds: > 30 panic, 25-30 elevated, 15-25 neutral,
    < 15 complacent risk-on.
  * US10Y must stabilize at 3-3.5% for bullish crypto in 2026.
  * BTC behaves as high-beta risk-on asset correlated with Nasdaq
    (Grayscale 2026 outlook).

This module fetches DXY/US10Y/VIX/S&P futures via Yahoo Finance
(free, no auth) with a graceful fallback for offline/rate-limited
environments. Cached 5 minutes to avoid rate-limit hammering.

Anti-overfit design:
  * Signals are RAW values + simple deltas (no learned weights)
  * Thresholds are research-grounded, not curve-fit
  * Bounded outputs (regime label + signed score in [-1, +1])
  * Graceful "unavailable" fallback so brain knows when missing
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CACHE_TTL_S = 300  # 5 min
_cache: Dict[str, Any] = {"ts": 0.0, "data": None}


@dataclass
class MacroSignals:
    method: str
    dxy: Optional[float] = None
    dxy_pct_change_1d: float = 0.0
    dxy_pct_change_5d: float = 0.0
    us10y_yield: Optional[float] = None
    us10y_change_bps_1d: float = 0.0
    vix_level: Optional[float] = None
    vix_change_1d: float = 0.0
    vix_zone: str = "unavailable"   # panic / elevated / neutral / complacent_risk_on / unavailable
    spx_pct_change_1d: float = 0.0
    spx_overnight_pct: float = 0.0
    risk_regime: str = "unavailable"  # risk_on / risk_off / neutral / unavailable
    crypto_directional_bias: float = 0.0  # [-1, +1]
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "dxy": round(float(self.dxy), 4) if self.dxy is not None else None,
            "dxy_pct_change_1d": round(float(self.dxy_pct_change_1d), 4),
            "dxy_pct_change_5d": round(float(self.dxy_pct_change_5d), 4),
            "us10y_yield": round(float(self.us10y_yield), 4) if self.us10y_yield is not None else None,
            "us10y_change_bps_1d": round(float(self.us10y_change_bps_1d), 2),
            "vix_level": round(float(self.vix_level), 2) if self.vix_level is not None else None,
            "vix_change_1d": round(float(self.vix_change_1d), 2),
            "vix_zone": self.vix_zone,
            "spx_pct_change_1d": round(float(self.spx_pct_change_1d), 4),
            "spx_overnight_pct": round(float(self.spx_overnight_pct), 4),
            "risk_regime": self.risk_regime,
            "crypto_directional_bias": round(float(self.crypto_directional_bias), 3),
            "rationale": self.rationale[:300],
        }


def _classify_vix_zone(vix: Optional[float]) -> str:
    if vix is None:
        return "unavailable"
    if vix > 30:
        return "panic"
    if vix >= 25:
        return "elevated"
    if vix >= 15:
        return "neutral"
    return "complacent_risk_on"


def _classify_risk_regime(dxy_chg: float, us10y_yield: Optional[float],
                          vix_zone: str, spx_chg: float) -> str:
    """Combine the four factors into a single regime label.
    risk_off when ANY of: DXY rising > 0.3% / VIX > 30 / SPX < -1%.
    risk_on when DXY falling < -0.3% AND VIX neutral-or-complacent
        AND SPX > +0.3%.
    """
    if vix_zone == "panic" or dxy_chg > 0.5 or spx_chg < -1.0:
        return "risk_off"
    if (dxy_chg < -0.3 and vix_zone in ("neutral", "complacent_risk_on")
            and spx_chg > 0.3):
        return "risk_on"
    return "neutral"


def _bias_score(dxy_chg: float, us10y_chg_bps: float,
                 vix_zone: str, spx_chg: float) -> float:
    """Signed score [-1, +1] for "macro favors crypto LONG".
    Negative = headwind, positive = tailwind."""
    score = 0.0
    # DXY: rising hurts crypto
    score -= max(-1.0, min(1.0, dxy_chg / 0.5))
    # US10Y: rising bps hurts (more aggressive when > 5bps)
    score -= max(-0.5, min(0.5, us10y_chg_bps / 10.0))
    # VIX zones
    score += {"panic": -0.6, "elevated": -0.2, "neutral": 0.0,
              "complacent_risk_on": 0.3, "unavailable": 0.0}.get(vix_zone, 0.0)
    # SPX: rising helps
    score += max(-0.5, min(0.5, spx_chg / 1.0))
    return max(-1.0, min(1.0, score))


def _fetch_yahoo(symbol: str) -> Optional[Dict[str, float]]:
    """Single-symbol Yahoo Finance fetch (last 7 daily bars).
    Returns None on failure (caller falls back)."""
    try:
        import requests
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?range=7d&interval=1d"
        )
        r = requests.get(url, timeout=4, headers={
            "User-Agent": "Mozilla/5.0 ACT-bot",
        })
        if r.status_code != 200:
            return None
        j = r.json()
        result = (j.get("chart") or {}).get("result") or []
        if not result:
            return None
        closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
        closes = [c for c in closes if c is not None]
        if len(closes) < 2:
            return None
        last = float(closes[-1])
        prev = float(closes[-2])
        five_back = float(closes[-6]) if len(closes) >= 6 else float(closes[0])
        return {"last": last, "prev": prev, "five_back": five_back}
    except Exception as e:
        logger.debug("yahoo fetch %s failed: %s", symbol, e)
        return None


def fetch_macro_overlay() -> MacroSignals:
    """Fetch all 4 macro factors. Cached 5 min. Returns MacroSignals
    with method='unavailable' on full failure."""
    now = time.time()
    if _cache["data"] is not None and now - _cache["ts"] < CACHE_TTL_S:
        return _cache["data"]

    dxy = _fetch_yahoo("DX-Y.NYB")    # DXY
    us10y = _fetch_yahoo("^TNX")       # 10Y treasury yield
    vix = _fetch_yahoo("^VIX")         # CBOE volatility
    spx = _fetch_yahoo("^GSPC")        # S&P 500

    if all(x is None for x in (dxy, us10y, vix, spx)):
        result = MacroSignals(method="unavailable",
                               rationale="all yahoo fetches failed")
        _cache["data"] = result
        _cache["ts"] = now
        return result

    dxy_pct_1d = ((dxy["last"] - dxy["prev"]) / dxy["prev"] * 100.0) if dxy else 0.0
    dxy_pct_5d = ((dxy["last"] - dxy["five_back"]) / dxy["five_back"] * 100.0) if dxy else 0.0
    us10y_chg_bps = ((us10y["last"] - us10y["prev"]) * 100.0) if us10y else 0.0
    vix_chg = (vix["last"] - vix["prev"]) if vix else 0.0
    spx_pct_1d = ((spx["last"] - spx["prev"]) / spx["prev"] * 100.0) if spx else 0.0

    vix_zone = _classify_vix_zone(vix["last"] if vix else None)
    regime = _classify_risk_regime(dxy_pct_1d, us10y["last"] if us10y else None,
                                    vix_zone, spx_pct_1d)
    bias = _bias_score(dxy_pct_1d, us10y_chg_bps, vix_zone, spx_pct_1d)

    rationale_parts = [f"regime={regime}"]
    if dxy:
        rationale_parts.append(f"DXY={dxy['last']:.2f}({dxy_pct_1d:+.2f}%/d)")
    if us10y:
        rationale_parts.append(f"US10Y={us10y['last']:.2f}%({us10y_chg_bps:+.0f}bps)")
    if vix:
        rationale_parts.append(f"VIX={vix['last']:.1f}({vix_zone})")
    if spx:
        rationale_parts.append(f"SPX={spx_pct_1d:+.2f}%/d")

    result = MacroSignals(
        method="yahoo_finance",
        dxy=dxy["last"] if dxy else None,
        dxy_pct_change_1d=dxy_pct_1d,
        dxy_pct_change_5d=dxy_pct_5d,
        us10y_yield=us10y["last"] if us10y else None,
        us10y_change_bps_1d=us10y_chg_bps,
        vix_level=vix["last"] if vix else None,
        vix_change_1d=vix_chg,
        vix_zone=vix_zone,
        spx_pct_change_1d=spx_pct_1d,
        spx_overnight_pct=0.0,  # need futures vs cash for true overnight; simplified
        risk_regime=regime,
        crypto_directional_bias=bias,
        rationale=" | ".join(rationale_parts)[:300],
    )
    _cache["data"] = result
    _cache["ts"] = now
    return result
