"""Cost-awareness gate — C19 (WebCryptoAgent-inspired).

WebCryptoAgent (arXiv 2601.04687) makes frictional cost explicit: a
trade only fires if `expected_return > liquidity_fee + impact + gas +
spread + MEV`. ACT's conviction gate already bakes Robinhood's 1.69%
spread into the "min expected move" threshold, but the arithmetic is
implicit and the per-component breakdown isn't visible to the analyst
or the audit log.

This module makes that arithmetic a first-class gate:

    cost_gate.evaluate(
        expected_return_pct=2.8,       # analyst's expected move
        spread_pct=1.69,               # venue-specific round-trip
        fees_pct=0.0,                  # Robinhood
        atr_pct=0.35,                  # recent ATR as % of price
        size_pct=1.0,                  # position size as % of equity
        venue="robinhood",
    ) -> CostGateResult(
        passed=True|False,
        margin_pct=expected - frictional,
        breakdown=CostBreakdown(...),
        reason="...",
    )

Venue presets live in VENUE_COSTS so callers don't have to remember
the number; per-call overrides still win. Pure Python, no deps.

Related:
  * src/trading/conviction_gate.py — tier classification (still runs)
  * src/trading/macro_bias.py — crisis gate
  * src/learning/reward.py (C18a) — risk-adjusted post-hoc reward

Not a replacement for the conviction gate — it's an ADDITIONAL check.
Callers wire it in before submit_trade_plan so a plan that passes
every other gate but has negative cost margin still gets rejected.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# Venue-specific cost presets. Each entry is the AMORTIZED round-trip
# cost in percent (entry + exit combined). Overridable per-call.
#
# Sources:
#   robinhood    — historical preset 1.69% round-trip; public retail
#                  data (bitget academy, 2026-Q1) shows real BTC/ETH
#                  spread is 0.35-0.85% per side + 0-0.6% commission,
#                  putting typical round-trip closer to ~1.0%. Operator
#                  should measure with scripts/measure_spread.py and
#                  override via ACT_ROBINHOOD_SPREAD_PCT env if needed.
#   bybit_spot   — 0.055% taker, maker rebates possible
#   bybit_perp   — 0.02% maker / 0.06% taker per side; funding separate
#   polymarket   — ~1% spread + 2% protocol fee on binary markets
#   kraken_spot  — 0.26% taker (high-tier volume ladder cheaper)
VENUE_COSTS: Dict[str, Dict[str, float]] = {
    "robinhood":  {"spread_pct": 1.69, "fees_pct": 0.00},
    "bybit_spot": {"spread_pct": 0.05, "fees_pct": 0.10},
    "bybit_perp": {"spread_pct": 0.04, "fees_pct": 0.12},
    "polymarket": {"spread_pct": 1.00, "fees_pct": 2.00},
    "kraken":     {"spread_pct": 0.20, "fees_pct": 0.26},
}

# Per-venue env overrides. When set, override the preset above without
# requiring a code change. Format: percent (1.0 == 1.0%). The override
# applies to ALL evaluate() calls for that venue unless the caller
# passes an explicit `spread_pct` / `fees_pct` argument.
#
#   ACT_ROBINHOOD_SPREAD_PCT  ACT_ROBINHOOD_FEES_PCT
#   ACT_BYBIT_SPOT_SPREAD_PCT ACT_BYBIT_SPOT_FEES_PCT
#   ACT_BYBIT_PERP_SPREAD_PCT ACT_BYBIT_PERP_FEES_PCT
#   ACT_POLYMARKET_SPREAD_PCT ACT_POLYMARKET_FEES_PCT
#   ACT_KRAKEN_SPREAD_PCT     ACT_KRAKEN_FEES_PCT
_VENUE_OVERRIDE_PREFIX_BY_KEY = {
    "robinhood":  "ACT_ROBINHOOD",
    "bybit_spot": "ACT_BYBIT_SPOT",
    "bybit_perp": "ACT_BYBIT_PERP",
    "polymarket": "ACT_POLYMARKET",
    "kraken":     "ACT_KRAKEN",
}

# Default minimum margin (expected - frictional). 1.0% means every
# trade must clear at least a 1% buffer above break-even. Overridable
# via ACT_COST_MIN_MARGIN_PCT env or per-call arg.
DEFAULT_MIN_MARGIN_PCT = 1.0

# Slippage model: a fraction of recent ATR (round-trip). 0.3 is
# conservative for BTC/ETH on RH/Bybit. Binary markets get a flat
# bump since ATR doesn't apply.
ATR_SLIPPAGE_FRACTION_RT = 0.30

# Price-impact model — linear in size for small trades, quadratic past
# a knee. For BTC/ETH at 1% equity this is negligible (<0.01%); only
# matters at the tail end of large positions.
IMPACT_LINEAR_BPS_PER_PCT = 0.5         # 0.005% per 1% equity
IMPACT_KNEE_PCT = 5.0                    # quadratic above 5% equity


@dataclass
class CostBreakdown:
    """Per-component frictional cost, all in percent, round-trip.

    All positive-magnitude components except `usd_drift_pct`, which is
    signed: positive = expected headwind (USD strengthens while we
    hold BTC/USD long), negative = tailwind (USD weakens while long).
    """
    spread_pct: float = 0.0
    fees_pct: float = 0.0
    slippage_pct: float = 0.0
    impact_pct: float = 0.0
    extra_pct: float = 0.0          # gas / MEV / misc, opt-in
    usd_drift_pct: float = 0.0      # SIGNED — DXY drift × hold × direction

    @property
    def total_pct(self) -> float:
        return round(
            self.spread_pct + self.fees_pct + self.slippage_pct
            + self.impact_pct + self.extra_pct + self.usd_drift_pct,
            4,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "spread_pct": round(self.spread_pct, 4),
            "fees_pct": round(self.fees_pct, 4),
            "slippage_pct": round(self.slippage_pct, 4),
            "impact_pct": round(self.impact_pct, 4),
            "extra_pct": round(self.extra_pct, 4),
            "usd_drift_pct": round(self.usd_drift_pct, 4),
            "total_pct": self.total_pct,
        }


@dataclass
class CostGateResult:
    """Outcome of cost_gate.evaluate. Audit-friendly."""
    passed: bool
    expected_return_pct: float
    frictional_cost_pct: float
    margin_pct: float               # expected - frictional
    min_margin_pct: float
    breakdown: CostBreakdown = field(default_factory=CostBreakdown)
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "expected_return_pct": round(self.expected_return_pct, 3),
            "frictional_cost_pct": round(self.frictional_cost_pct, 3),
            "margin_pct": round(self.margin_pct, 3),
            "min_margin_pct": round(self.min_margin_pct, 3),
            "breakdown": self.breakdown.to_dict(),
            "reason": self.reason,
        }


def get_spread_pct(venue: str = "robinhood") -> float:
    """Single source of truth for the active round-trip spread (percent).

    Reads the venue preset including ACT_<VENUE>_SPREAD_PCT env override.
    Falls back to 1.69% only if cost_gate itself cannot answer (which
    cannot actually happen for known venues, but defensive default
    keeps callers safe).

    Used by prompt_constraints, dual_brain.PERFORMANCE_TARGET, executor
    spread inits, and any future caller that wants the live cost
    figure without re-implementing the lookup chain.
    """
    try:
        return float(_resolve_venue_costs(venue).get("spread_pct", 1.69))
    except Exception:
        return 1.69


def _resolve_venue_costs(venue: Optional[str]) -> Dict[str, float]:
    key = (venue or "robinhood").strip().lower()
    base = dict(VENUE_COSTS.get(key, VENUE_COSTS["robinhood"]))
    prefix = _VENUE_OVERRIDE_PREFIX_BY_KEY.get(key)
    if prefix:
        for field_name, env_suffix in (("spread_pct", "_SPREAD_PCT"),
                                       ("fees_pct", "_FEES_PCT")):
            raw = os.environ.get(prefix + env_suffix, "").strip()
            if raw:
                try:
                    base[field_name] = max(0.0, float(raw))
                except ValueError:
                    logger.warning(
                        "cost_gate: ignoring non-numeric %s%s=%r",
                        prefix, env_suffix, raw,
                    )
    return base


def _impact_pct(size_pct: float) -> float:
    """Linear below the knee, quadratic above. Round-trip."""
    if size_pct <= 0:
        return 0.0
    linear = (IMPACT_LINEAR_BPS_PER_PCT / 100.0) * size_pct
    if size_pct <= IMPACT_KNEE_PCT:
        return linear
    excess = size_pct - IMPACT_KNEE_PCT
    # Quadratic term scaled so the two piecewise expressions match at
    # the knee — doubles cost by ~2x by 10% equity.
    return linear + (IMPACT_LINEAR_BPS_PER_PCT / 100.0) * (excess * excess) / IMPACT_KNEE_PCT


def _min_margin_default() -> float:
    # Paper-soak loose overlay (C22) takes precedence when paper mode
    # and operator has enabled loose gates.
    try:
        from skills.paper_soak_loose.action import get_paper_soak_overlay
        overlay = get_paper_soak_overlay()
        if overlay:
            ov = (overlay.get("cost_gate") or {}).get("min_margin_pct")
            if isinstance(ov, (int, float)):
                return max(0.0, float(ov))
    except Exception:
        pass
    env = (os.environ.get("ACT_COST_MIN_MARGIN_PCT") or "").strip()
    if env:
        try:
            return max(0.0, float(env))
        except ValueError:
            pass
    return DEFAULT_MIN_MARGIN_PCT


def _resolve_usd_drift_from_econ() -> float:
    """Best-effort DXY drift read from economic_intelligence.usd_strength.

    Returns signed daily drift in percent (positive = USD strengthening).
    Returns 0.0 on any failure — cost gate should never crash the tick.
    """
    try:
        from src.data.economic_intelligence import get_intelligence
        ei = get_intelligence()
        snap = ei.get_layer_snapshot("usd_strength") if ei else None
        if not snap:
            return 0.0
        # Layer payload is loose across revisions; support a few keys.
        for k in ("daily_drift_pct", "dxy_daily_change_pct", "signed_bias_pct"):
            if k in snap:
                return float(snap.get(k) or 0.0)
        # Fallback — map signal + confidence to a crude drift estimate.
        # NEUTRAL → 0; STRONG/CRISIS → ±0.25%/day.
        signal = str(snap.get("signal") or "").upper()
        conf = float(snap.get("confidence") or snap.get("conf") or 0.0)
        if signal in ("STRONG", "BULLISH"):
            return 0.25 * max(0.0, min(1.0, conf))
        if signal in ("WEAK", "BEARISH", "CRISIS"):
            return -0.25 * max(0.0, min(1.0, conf))
    except Exception:
        pass
    return 0.0


def evaluate(
    expected_return_pct: float,
    *,
    venue: Optional[str] = "robinhood",
    spread_pct: Optional[float] = None,
    fees_pct: Optional[float] = None,
    atr_pct: float = 0.30,
    size_pct: float = 1.0,
    extra_cost_pct: float = 0.0,
    min_margin_pct: Optional[float] = None,
    direction: str = "LONG",
    usd_drift_pct_per_day: Optional[float] = None,
    expected_hold_days: float = 1.0,
) -> CostGateResult:
    """Reject trades whose expected return can't clear round-trip friction.

    Arguments:
      expected_return_pct: Analyst's predicted move (percent, not
        fraction). A 2.8% expected move → 2.8, not 0.028.
      venue: preset lookup key (robinhood/bybit_spot/bybit_perp/
        polymarket/kraken). Overridable by explicit spread/fees args.
      spread_pct: round-trip spread in percent. Overrides venue preset.
      fees_pct: round-trip fees in percent. Overrides venue preset.
      atr_pct: recent ATR as percent of price. Used for slippage model.
      size_pct: position size as percent of equity. Used for impact.
      extra_cost_pct: opt-in for gas/MEV/misc frictional costs (DEX use).
      min_margin_pct: minimum margin above break-even. None → env or default.
      direction: "LONG" or "SHORT". Determines USD drift sign.
      usd_drift_pct_per_day: expected daily DXY drift (signed, %). If
        None, auto-read from economic_intelligence.usd_strength. Positive
        = USD strengthening → headwind for LONG BTC/USD, tailwind for SHORT.
      expected_hold_days: hold duration in days. Scales USD drift.

    Returns:
      CostGateResult with passed, margin, and full breakdown including
      signed USD drift. Soft-fails to REJECT on malformed input.
    """
    try:
        exp = max(0.0, float(expected_return_pct))
    except (TypeError, ValueError):
        return CostGateResult(
            passed=False, expected_return_pct=0.0,
            frictional_cost_pct=0.0, margin_pct=0.0,
            min_margin_pct=_min_margin_default(),
            reason="invalid_expected_return",
        )

    preset = _resolve_venue_costs(venue)
    spread = spread_pct if spread_pct is not None else preset["spread_pct"]
    fees = fees_pct if fees_pct is not None else preset["fees_pct"]
    slippage = max(0.0, atr_pct) * ATR_SLIPPAGE_FRACTION_RT
    impact = _impact_pct(max(0.0, size_pct))
    extra = max(0.0, extra_cost_pct)

    # USD drift — BTC/USD and ETH/USD are dollar-denominated, so DXY
    # movement mechanically shifts the quote. Direction-adjusted so a
    # strengthening USD is a headwind for LONGs and a tailwind for SHORTs.
    if usd_drift_pct_per_day is None:
        usd_daily = _resolve_usd_drift_from_econ()
    else:
        try:
            usd_daily = float(usd_drift_pct_per_day)
        except (TypeError, ValueError):
            usd_daily = 0.0
    direction_sign = 1.0 if str(direction).upper() in ("LONG", "BUY") else -1.0
    usd_drift = usd_daily * max(0.0, float(expected_hold_days)) * direction_sign

    breakdown = CostBreakdown(
        spread_pct=spread, fees_pct=fees, slippage_pct=slippage,
        impact_pct=impact, extra_pct=extra, usd_drift_pct=usd_drift,
    )
    total = breakdown.total_pct
    margin = exp - total

    threshold = _min_margin_default() if min_margin_pct is None else max(0.0, float(min_margin_pct))
    passed = margin >= threshold

    usd_note = f" usd={usd_drift:+.2f}%" if abs(usd_drift) > 0.01 else ""
    if passed:
        reason = f"cost_ok:{margin:+.2f}%>={threshold:.2f}%{usd_note}"
    else:
        reason = (
            f"cost_reject:margin={margin:+.2f}% < {threshold:.2f}% "
            f"(exp={exp:.2f}% frictional={total:.2f}%{usd_note})"
        )

    result = CostGateResult(
        passed=passed,
        expected_return_pct=exp,
        frictional_cost_pct=total,
        margin_pct=margin,
        min_margin_pct=threshold,
        breakdown=breakdown,
        reason=reason,
    )
    logger.debug("cost_gate: %s", result.reason)
    return result


def is_enabled() -> bool:
    """Gated by ACT_COST_GATE env, on by default. Set to '0' to bypass."""
    v = (os.environ.get("ACT_COST_GATE") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")
