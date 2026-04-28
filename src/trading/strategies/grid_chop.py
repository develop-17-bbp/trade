"""Grid trading for ranging / CHOP regimes.

When the market is ranging (HMM regime CHOP, Hurst < 0.45), trend
strategies underperform but grid strategies excel — small bracketed
buy-low / sell-high orders that profit from oscillation around a
median.

This module computes grid levels but does NOT auto-fire trades. It
returns structured grid suggestions; the LLM brain reviews and
chooses whether to submit them as small TradePlans, respecting the
concentration cap (3 per asset).

Anti-overfit design:
  * Levels derived from current ATR and median-price — no fitting
    on past data
  * Grid spacing must clear round-trip spread by ≥ 50% (each rung's
    profit target > 1.5 × spread to be +EV)
  * Levels capped at N=5 to prevent over-stacking
  * Returns empty list when regime is NOT ranging
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class GridLevel:
    rung: int                     # 1, 2, 3...
    side: str                     # "BUY" | "SELL"
    price: float
    size_pct: float               # of equity per rung
    target_price: float           # exit target for this rung
    expected_pnl_pct: float       # gross gain at target
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rung": int(self.rung),
            "side": self.side,
            "price": round(float(self.price), 2),
            "size_pct": round(float(self.size_pct), 3),
            "target_price": round(float(self.target_price), 2),
            "expected_pnl_pct": round(float(self.expected_pnl_pct), 3),
            "rationale": self.rationale[:200],
        }


def compute_grid_levels(
    asset: str,
    current_price: float,
    atr: float,
    spread_pct: float = 1.69,
    n_levels: int = 5,
    base_size_pct: float = 1.0,
    is_ranging: bool = True,
) -> List[Dict[str, Any]]:
    """Generate up to N grid rungs spaced around current price.

    Each rung places a BUY at (price - k × ATR) with target back at
    current_price ± a margin that clears the spread. Returns an
    empty list when:
      * regime not ranging (caller passes is_ranging=False)
      * ATR too small to clear spread × 1.5
      * inputs degenerate (price <= 0)
    """
    if not is_ranging or current_price <= 0 or atr <= 0:
        return []

    # Each rung's expected gross gain target — must clear spread by 50%
    min_profitable_move_pct = max(2.0, spread_pct * 1.5)
    rung_spacing_atr = max(0.3, min_profitable_move_pct / 100.0
                           * current_price / max(0.01, atr))
    n_levels = max(1, min(5, n_levels))

    out: List[Dict[str, Any]] = []
    for k in range(1, n_levels + 1):
        buy_price = current_price - k * atr * rung_spacing_atr
        if buy_price <= 0:
            break
        # Target is one ATR-step back toward current price
        target = buy_price + atr * rung_spacing_atr
        expected_pct = ((target - buy_price) / buy_price * 100.0)
        if expected_pct < min_profitable_move_pct:
            continue
        out.append(GridLevel(
            rung=k, side="BUY",
            price=buy_price,
            size_pct=base_size_pct / n_levels,  # split base size across rungs
            target_price=target,
            expected_pnl_pct=expected_pct,
            rationale=(
                f"rung {k}: buy ${buy_price:.2f} target ${target:.2f} "
                f"({expected_pct:+.2f}% gross, clears {spread_pct:.2f}% spread)"
            ),
        ).to_dict())

    return out


def grid_advisory(asset: str, current_price: float, atr: float,
                  regime: str = "unknown",
                  hurst_value: float = 0.5,
                  spread_pct: float = 1.69) -> Dict[str, Any]:
    """Top-level advisory: should the brain consider grid right now,
    and what would the rungs look like?"""
    is_ranging = (
        str(regime).upper() == "CHOP"
        or float(hurst_value) < 0.45
    )
    levels = compute_grid_levels(
        asset=asset, current_price=current_price, atr=atr,
        spread_pct=spread_pct, is_ranging=is_ranging,
    )
    return {
        "asset": asset,
        "is_ranging": is_ranging,
        "regime": str(regime),
        "hurst_value": round(float(hurst_value), 3),
        "n_levels_proposed": len(levels),
        "levels": levels,
        "advisory": (
            "Grid rungs are SUGGESTIONS only. Submit each as a small "
            "TradePlan via submit_trade_plan; concentration cap (3/asset) "
            "limits how many fire. NOT used in trending regimes."
            if is_ranging else
            "Regime is not ranging — grid not advised. Use trend "
            "strategies instead."
        ),
    }
