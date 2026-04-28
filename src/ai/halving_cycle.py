"""Bitcoin halving cycle position factor.

The Bitcoin halving (every ~210,000 blocks, ~4 years) is historically
the dominant macro driver of BTC cycles. Each halving has triggered
a bull cycle 12-18 months later (2012 → 2013, 2016 → 2017, 2020 →
2021, 2024 → 2025-26).

ACT operates in 2026 — ~1 year past the April 2024 halving (block
840,000). We are in the "post-halving bull cycle" phase historically.

This module is calendar/block-deterministic — pure compute, no
external API. Returns:
  * days_since_last_halving
  * cycle_position_pct (0-100% through the 4-year cycle)
  * cycle_phase label based on historical cycle structure
  * historical_comparable (which prior cycle most resembles current
    position)

Anti-overfit:
  * Pure deterministic calendar — no learned weights
  * Phase labels research-grounded (matches Glassnode / on-chain
    cycle research)
  * Reads only the halving date constants below
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# Confirmed Bitcoin halving dates (UTC)
HALVING_DATES = (
    datetime(2012, 11, 28, tzinfo=timezone.utc),  # 1st halving
    datetime(2016, 7,  9, tzinfo=timezone.utc),    # 2nd halving
    datetime(2020, 5, 11, tzinfo=timezone.utc),   # 3rd halving
    datetime(2024, 4, 19, tzinfo=timezone.utc),   # 4th halving (Apr 2024)
)

# Estimated next halving — ~4 years after 4th
ESTIMATED_NEXT_HALVING = datetime(2028, 4, 18, tzinfo=timezone.utc)

# Historical cycle structure (rough peak timing post-halving)
#   Months 0-12:  accumulation / early markup
#   Months 12-18: peak / blowoff top historically
#   Months 18-30: distribution / bear
#   Months 30-48: capitulation / accumulation for next halving
CYCLE_PHASES = (
    (0, 12, "post_halving_markup"),
    (12, 18, "blowoff_top_zone"),
    (18, 30, "distribution_bear"),
    (30, 48, "capitulation_accumulation"),
)


@dataclass
class HalvingCycleSignals:
    method: str
    days_since_last_halving: int
    months_since_last_halving: float
    cycle_position_pct: float                  # 0-100
    cycle_phase: str
    historical_comparable: str                 # "2017 cycle peak", etc.
    days_to_next_halving: int
    bullish_phase: bool                         # markup or blowoff
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "days_since_last_halving": int(self.days_since_last_halving),
            "months_since_last_halving": round(float(self.months_since_last_halving), 1),
            "cycle_position_pct": round(float(self.cycle_position_pct), 2),
            "cycle_phase": self.cycle_phase,
            "historical_comparable": self.historical_comparable,
            "days_to_next_halving": int(self.days_to_next_halving),
            "bullish_phase": bool(self.bullish_phase),
            "rationale": self.rationale[:300],
        }


def _classify_phase(months: float) -> str:
    for lo, hi, name in CYCLE_PHASES:
        if lo <= months < hi:
            return name
    return "post_cycle"


def _comparable_cycle(months: float) -> str:
    """Pick the most-similar prior cycle phase for narrative context."""
    if 6 <= months < 12:
        return "2017_pre_peak (~Jul 2017 BTC ran from $2.5K to $20K)"
    if 12 <= months < 18:
        return "2017_blowoff_top OR 2021_double_top zone"
    if 18 <= months < 30:
        return "2018_bear OR 2022_bear (deep drawdown phase)"
    return "transition_phase"


def get_halving_cycle(now_utc: Optional[datetime] = None) -> HalvingCycleSignals:
    """Compute current halving cycle position. Pure deterministic."""
    if now_utc is None:
        now_utc = datetime.now(tz=timezone.utc)
    last_halving = max(d for d in HALVING_DATES if d <= now_utc)
    days_since = (now_utc - last_halving).days
    months_since = days_since / 30.4375
    cycle_pct = min(100.0, days_since / (4 * 365.25) * 100.0)
    phase = _classify_phase(months_since)
    days_to_next = max(0, (ESTIMATED_NEXT_HALVING - now_utc).days)
    bullish = phase in ("post_halving_markup", "blowoff_top_zone")
    comparable = _comparable_cycle(months_since)

    rationale = (
        f"phase={phase} months_since_halving={months_since:.1f} "
        f"cycle_pos={cycle_pct:.1f}% "
        f"comparable={comparable[:50]} "
        f"bullish={bullish}"
    )

    return HalvingCycleSignals(
        method="halving_cycle",
        days_since_last_halving=days_since,
        months_since_last_halving=months_since,
        cycle_position_pct=cycle_pct,
        cycle_phase=phase,
        historical_comparable=comparable,
        days_to_next_halving=days_to_next,
        bullish_phase=bullish,
        rationale=rationale[:300],
    )
