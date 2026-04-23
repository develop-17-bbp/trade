"""
Funding-rate carry / cash-and-carry arbitrage strategy.

Direction-neutral income strategy. When the perpetual-futures funding rate
is strongly positive or negative, a delta-neutral position captures the funding
payment regardless of price direction:

    Funding > 0  (longs pay shorts)   -> SHORT perp + LONG spot  (receive funding)
    Funding < 0  (shorts pay longs)   -> LONG perp  + SHORT spot (receive funding)

Earns ~|funding_rate| × hours_held × notional per 8h funding window, minus
trading fees and cross-venue spread. On Bybit BTCUSDT historically runs
0.01% per 8h (≈10.95% annualised when consistent) but can spike to
0.1-0.5% per 8h in trending markets.

WHY THIS MODULE EXISTS BUT IS DORMANT ON ROBINHOOD:
  Robinhood crypto doesn't offer perpetual futures. Funding-rate arbitrage
  requires BOTH a spot venue AND a perp-futures venue with authenticated
  order placement. When the operator migrates to a venue pair like
  (Kraken spot + Bybit perp) or (Binance spot + Bybit perp) or just
  Bybit spot + Bybit perp, this module becomes trade-producing.

Contract:
    evaluate_funding_opportunity(funding_pct_per_8h, spot_fee_bps, perp_fee_bps,
                                  spread_cost_pct, min_annualised_return=10.0,
                                  notional_usd=2000.0) -> FundingOpportunity

    Returns {tier: 'enter' | 'hold' | 'skip', direction, expected_return_annual_pct, ...}

Integration steps (when a perp venue is wired):
    1. Config: add `funding_arb: { enabled: true, min_annualised_pct: 12.0 }`
    2. Executor: in the top-of-cycle loop for each asset with a perp venue:
         funding = self._economic_intelligence._layers['derivatives'].get_cached().get('funding_rate')
         opp = evaluate_funding_opportunity(funding, ...)
         if opp.tier == 'enter': place_paired_trade(opp)
    3. Position tracker: handle the two-leg position as a single delta-neutral
       unit; track P&L from funding accruals + leg-basis drift.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# Thresholds — moved out of magic numbers so operators can tune via config.
DEFAULT_MIN_ANNUALISED_PCT = 10.0      # floor for trade; below this not worth the operational cost
DEFAULT_SPOT_FEE_BPS = 10.0            # 0.10% e.g. Kraken taker
DEFAULT_PERP_FEE_BPS = 5.5             # 0.055% Bybit taker
DEFAULT_SPREAD_COST_PCT = 0.02         # cross-venue spread ~ 0.02% on BTC
FUNDING_WINDOWS_PER_DAY = 3            # 8h funding cycle → 3 payments/day


@dataclass
class FundingOpportunity:
    tier: str                                  # 'enter' | 'hold' | 'skip'
    direction: str                             # 'LONG_PERP_SHORT_SPOT' | 'SHORT_PERP_LONG_SPOT' | ''
    funding_pct_per_8h: float
    annualised_pct: float
    expected_net_annual_pct: float             # after fees + spread
    notional_usd: float
    reason: str = ""
    legs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "direction": self.direction,
            "funding_pct_per_8h": round(self.funding_pct_per_8h, 5),
            "annualised_pct": round(self.annualised_pct, 2),
            "expected_net_annual_pct": round(self.expected_net_annual_pct, 2),
            "notional_usd": round(self.notional_usd, 2),
            "reason": self.reason,
            "legs": dict(self.legs),
        }


def evaluate_funding_opportunity(
    funding_pct_per_8h: float,
    spot_fee_bps: float = DEFAULT_SPOT_FEE_BPS,
    perp_fee_bps: float = DEFAULT_PERP_FEE_BPS,
    spread_cost_pct: float = DEFAULT_SPREAD_COST_PCT,
    min_annualised_return: float = DEFAULT_MIN_ANNUALISED_PCT,
    notional_usd: float = 2000.0,
    expected_hold_days: float = 7.0,
) -> FundingOpportunity:
    """Evaluate whether a funding-rate arbitrage is worth entering right now.

    Args:
        funding_pct_per_8h: current funding rate in %/8h.
            e.g. 0.01 means perp longs pay 0.01% to shorts every 8h.
        spot_fee_bps / perp_fee_bps: taker fees as basis-points
            (bps = 0.01%). Used for round-trip cost per leg.
        spread_cost_pct: bid-ask slippage on entry (% round-trip).
        min_annualised_return: skip if projected net annual < this.
        notional_usd: position size per leg in USD (leg sizes matched).
        expected_hold_days: used to estimate fees amortisation.
    """
    annualised_pct = funding_pct_per_8h * FUNDING_WINDOWS_PER_DAY * 365

    # Round-trip fee cost (entry + exit, both legs). bps → %.
    # Two legs × two sides × fee_bps → total fee cost as % of notional per side.
    total_fee_pct = (2 * spot_fee_bps + 2 * perp_fee_bps) / 100.0
    # Cross-venue spread eats once per round trip.
    entry_cost_pct = total_fee_pct + spread_cost_pct

    # Funding accrual over the hold
    hold_windows = expected_hold_days * FUNDING_WINDOWS_PER_DAY
    gross_accrual_pct = abs(funding_pct_per_8h) * hold_windows

    net_pct = gross_accrual_pct - entry_cost_pct
    if expected_hold_days > 0:
        # Annualise the net return
        expected_net_annual = (net_pct / expected_hold_days) * 365
    else:
        expected_net_annual = 0.0

    # Direction
    if funding_pct_per_8h > 0:
        direction = "SHORT_PERP_LONG_SPOT"   # longs are paying; we want to be short
    elif funding_pct_per_8h < 0:
        direction = "LONG_PERP_SHORT_SPOT"
    else:
        direction = ""

    # Tiering
    if abs(funding_pct_per_8h) < 1e-6 or direction == "":
        return FundingOpportunity(
            tier="skip", direction="",
            funding_pct_per_8h=funding_pct_per_8h,
            annualised_pct=annualised_pct,
            expected_net_annual_pct=expected_net_annual,
            notional_usd=notional_usd,
            reason="funding_neutral",
        )
    if expected_net_annual < min_annualised_return:
        return FundingOpportunity(
            tier="skip", direction=direction,
            funding_pct_per_8h=funding_pct_per_8h,
            annualised_pct=annualised_pct,
            expected_net_annual_pct=expected_net_annual,
            notional_usd=notional_usd,
            reason=(f"net_{expected_net_annual:.1f}pct_below_min_{min_annualised_return:.1f}pct"),
        )

    # Entry
    legs = {
        "perp": {
            "side": "SHORT" if direction == "SHORT_PERP_LONG_SPOT" else "LONG",
            "notional_usd": notional_usd,
        },
        "spot": {
            "side": "LONG" if direction == "SHORT_PERP_LONG_SPOT" else "SHORT",
            "notional_usd": notional_usd,
        },
    }
    return FundingOpportunity(
        tier="enter", direction=direction,
        funding_pct_per_8h=funding_pct_per_8h,
        annualised_pct=annualised_pct,
        expected_net_annual_pct=expected_net_annual,
        notional_usd=notional_usd,
        reason=f"funding_{funding_pct_per_8h:+.4f}pct_8h_net_{expected_net_annual:.1f}pct_annual",
        legs=legs,
    )


def is_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Funding arb requires BOTH a configured perp venue AND explicit enable.

    It's dormant unless:
      1. ACT_FUNDING_ARB=1 OR config['funding_arb']['enabled']=True
      2. AND the venue config includes a perp-futures exchange

    Robinhood-only configs always return False — it's structurally
    impossible without a futures venue.
    """
    env = (os.environ.get("ACT_FUNDING_ARB") or "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    cfg_enabled = False
    if isinstance(config, dict):
        cfg_enabled = bool(config.get("funding_arb", {}).get("enabled", False))
    if env in ("1", "true", "yes", "on") or cfg_enabled:
        # Check for a perp venue in the exchange list
        exchanges = (config or {}).get("exchanges", []) or []
        perp_venues = {"bybit", "delta", "binance_futures", "okx_perp"}
        has_perp = any(
            isinstance(ex, dict) and str(ex.get("name", "")).lower() in perp_venues
            for ex in exchanges
        )
        return has_perp
    return False


def describe_dormancy_reason(config: Optional[Dict[str, Any]] = None) -> str:
    """Explain WHY funding arb is off. For the evaluator / dashboard."""
    env = (os.environ.get("ACT_FUNDING_ARB") or "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return "disabled_by_env:ACT_FUNDING_ARB=0"
    exchanges = (config or {}).get("exchanges", []) or []
    perp_venues = {"bybit", "delta", "binance_futures", "okx_perp"}
    has_perp = any(
        isinstance(ex, dict) and str(ex.get("name", "")).lower() in perp_venues
        for ex in exchanges
    )
    if not has_perp:
        return "no_perp_venue_configured:robinhood_only_setup_cannot_arb_funding"
    cfg_enabled = isinstance(config, dict) and config.get("funding_arb", {}).get("enabled", False)
    if not cfg_enabled and env not in ("1", "true", "yes", "on"):
        return "disabled_by_default:set_ACT_FUNDING_ARB=1_or_config.funding_arb.enabled=true"
    return "active"
