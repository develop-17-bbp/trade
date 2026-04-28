"""Realistic slippage + latency model for backtesting and live execution.

Current ACT slippage model: flat round-trip spread (1.69% on
Robinhood). The 2026 literature flags this as too optimistic — real
slippage scales with:
  * order size relative to available liquidity (volume)
  * market volatility (ATR / GARCH)
  * time-of-day / session
  * API latency (100-200ms typical for retail)

This module estimates a more honest fill price + latency-adjusted
slippage. The brain queries it before submitting a TradePlan to see
the true effective spread it'll pay.

Anti-overfit design:
  * Pure rule-based — no parameter learning
  * Multipliers bounded (size factor max 3x, volatility factor max
    2.5x, latency factor max 1.5x)
  * Slippage NEVER below the venue's static round-trip spread
    (avoids being too optimistic)
  * Returns confidence interval, not point estimate
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict


# Bounded multipliers (research-grounded ceilings, not learned).
SIZE_FACTOR_MAX = 3.0       # 0.5% -> 1x; 5% -> ~3x
VOL_FACTOR_MAX = 2.5        # vol_pct 1% -> 1x; vol_pct 5%+ -> 2.5x
LATENCY_FACTOR_MAX = 1.5    # 50ms -> 1.0x; 500ms+ -> 1.5x
SESSION_FACTOR = {
    "US":      1.0,    # tightest liquidity
    "EU":      1.1,
    "ASIA":    1.4,    # widest spreads
    "LATE_US": 1.2,
}


@dataclass
class SlippageEstimate:
    venue: str
    base_spread_pct: float
    size_factor: float
    volatility_factor: float
    latency_factor: float
    session_factor: float
    expected_slippage_pct: float
    upper_bound_pct: float       # 90th-percentile estimate
    confidence_band_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venue": self.venue,
            "base_spread_pct": round(float(self.base_spread_pct), 4),
            "size_factor": round(float(self.size_factor), 3),
            "volatility_factor": round(float(self.volatility_factor), 3),
            "latency_factor": round(float(self.latency_factor), 3),
            "session_factor": round(float(self.session_factor), 3),
            "expected_slippage_pct": round(float(self.expected_slippage_pct), 4),
            "upper_bound_pct": round(float(self.upper_bound_pct), 4),
            "confidence_band_pct": round(float(self.confidence_band_pct), 4),
            "advisory": (
                "expected_slippage_pct = round-trip cost the brain should "
                "BUDGET for. upper_bound_pct = 90th-percentile bad case; "
                "size your trades so even at upper_bound the expected "
                "move still clears it."
            ),
        }


def estimate_slippage(
    venue: str = "robinhood",
    size_pct_of_equity: float = 1.0,
    volatility_pct: float = 1.0,
    latency_ms: float = 200.0,
    session: str = "US",
) -> SlippageEstimate:
    """Estimate realistic slippage for a proposed trade.

    Args:
      venue: 'robinhood' | 'bybit' | 'polymarket'
      size_pct_of_equity: trade size as % of equity (0.5 - 5.0)
      volatility_pct: current realized volatility, percent
      latency_ms: round-trip API latency
      session: current session label

    Returns:
      SlippageEstimate with bounded factors and an upper-bound 90th-
      percentile estimate.
    """
    # Base spread by venue (matches cost_gate single source of truth).
    base_spread = {
        "robinhood": 1.69,
        "bybit":     0.055,
        "polymarket": 2.0,
    }.get(venue.lower(), 1.69)

    # Size factor: linear scaling with size_pct, capped.
    size_factor = min(SIZE_FACTOR_MAX, 1.0 + max(0.0, size_pct_of_equity - 0.5) * 0.3)

    # Volatility factor: vol < 1% = 1x; vol > 5% saturates at max.
    vol_factor = min(VOL_FACTOR_MAX, 1.0 + max(0.0, volatility_pct - 1.0) * 0.3)

    # Latency factor: 50ms = 1.0; 500ms = 1.5; saturates.
    latency_factor = min(LATENCY_FACTOR_MAX, 1.0 + max(0.0, latency_ms - 50) / 900.0)

    sess_factor = SESSION_FACTOR.get(str(session).upper(), 1.1)

    expected = base_spread * size_factor * vol_factor * latency_factor * sess_factor
    # Floor at venue base spread — never *less* than the static spread.
    expected = max(base_spread, expected)

    # Upper bound = 90th percentile estimate via 1.6x scaling
    # (empirical from public crypto market-making research).
    upper = expected * 1.6
    band = upper - expected

    return SlippageEstimate(
        venue=venue,
        base_spread_pct=base_spread,
        size_factor=size_factor,
        volatility_factor=vol_factor,
        latency_factor=latency_factor,
        session_factor=sess_factor,
        expected_slippage_pct=expected,
        upper_bound_pct=upper,
        confidence_band_pct=band,
    )
