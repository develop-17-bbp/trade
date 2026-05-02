"""Gann angles + Gann squares strategy.

W.D. Gann's theory: price moves in geometric progressions related to
time. Key angles measured from a significant pivot (high or low):

    1x8  = 82.5° (steep — strong trend)
    1x4  = 75°
    1x3  = 71.25°
    1x2  = 63.75°
    1x1  = 45°  ← THE CARDINAL ANGLE (price moves 1 unit per 1 unit of time)
    2x1  = 26.25°
    3x1  = 18.75°
    4x1  = 15°
    8x1  = 7.5° (shallow — weak trend / consolidation)

Gann angles act as dynamic support/resistance:
  * Price respecting 1x1 from a swing low = healthy uptrend
  * Price falling below 1x1 = trend weakening; targets next angle below
  * Price breaking above 2x1 from a swing high = strong reversal up

Strategy logic:
  1. Identify recent significant pivot (highest high / lowest low in window)
  2. Project Gann angles forward in time from that pivot
  3. If current price is bouncing off 1x1 angle → directional signal
  4. If current price has crossed 1x1 → trend-change signal in the
     direction of the cross

Anti-overfit:
  * Pivot detection uses fixed window (operator can tune via env)
  * Angle projection uses fixed Gann ratios — no parameter learning
  * Confidence dampens when |distance to angle| > 0.5 * ATR
  * Returns 'unclear' rather than forced signal when no angle is near
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Gann angle definitions: (rise/run ratio, name, weight)
# Rise/run is bars-per-price-unit OR price-per-bar depending on direction.
# The 1x1 angle is the cardinal one — most respected.
GANN_ANGLES: List[Tuple[float, str, float]] = [
    (8.0, "8x1",  0.5),
    (4.0, "4x1",  0.7),
    (3.0, "3x1",  0.8),
    (2.0, "2x1",  0.9),
    (1.0, "1x1",  1.0),   # cardinal
    (0.5, "1x2",  0.9),
    (0.333, "1x3", 0.8),
    (0.25, "1x4", 0.7),
    (0.125, "1x8", 0.5),
]


@dataclass
class GannSignal:
    direction: str          # "LONG" / "SHORT" / "FLAT"
    confidence: float       # 0.0 - 1.0
    pivot_type: str         # "high" or "low"
    pivot_price: float
    pivot_bar_offset: int   # bars-ago of the pivot
    nearest_angle: str      # e.g. "1x1"
    angle_value: float      # current y-value of the angle line
    distance_pct: float     # signed distance current price - angle, % of price
    rationale: str = ""
    factors: List[str] = field(default_factory=list)


def _find_pivot(
    highs: List[float],
    lows: List[float],
    window: int = 50,
) -> Tuple[str, float, int]:
    """Find the most-significant pivot in the last `window` bars.

    Returns ('high', price, bars_ago) or ('low', price, bars_ago).
    """
    if len(highs) < 2 or len(lows) < 2:
        return ("high", float('nan'), 0)
    n = min(window, len(highs))
    recent_highs = highs[-n:]
    recent_lows = lows[-n:]
    max_high = max(recent_highs)
    min_low = min(recent_lows)
    high_idx = recent_highs.index(max_high)  # 0 = oldest in window
    low_idx = recent_lows.index(min_low)
    high_age = n - 1 - high_idx              # bars-ago
    low_age = n - 1 - low_idx
    range_pct = (max_high - min_low) / max(min_low, 1e-9)

    # Pick the pivot most recent + most extreme
    # Heuristic: more-recent pivot is more relevant; weight by age
    high_score = (1.0 / (1 + high_age)) * range_pct
    low_score = (1.0 / (1 + low_age)) * range_pct

    if high_score > low_score:
        return ("high", float(max_high), high_age)
    return ("low", float(min_low), low_age)


def _project_angle(
    pivot_price: float,
    pivot_bar_offset: int,
    current_bar: int,
    rise_per_run: float,
    pivot_is_low: bool,
    atr_unit: float,
) -> float:
    """Project a Gann angle forward from the pivot to the current bar.

    rise_per_run = price-units per bar when rise>1 (bars per unit when <1).
    pivot_is_low → angle goes UP from pivot (LONG support).
    pivot_is_high → angle goes DOWN from pivot (SHORT resistance).
    """
    bars_since_pivot = current_bar - (current_bar - pivot_bar_offset)
    bars_since_pivot = pivot_bar_offset
    # Convert rise/run ratio to price-per-bar using ATR as the natural unit
    if rise_per_run >= 1.0:
        price_delta = rise_per_run * atr_unit * bars_since_pivot
    else:
        # 1x2, 1x3 etc. — shallow angle: smaller price move per bar
        price_delta = rise_per_run * atr_unit * bars_since_pivot
    if pivot_is_low:
        return pivot_price + price_delta
    return pivot_price - price_delta


def evaluate(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    atr_value: Optional[float] = None,
    pivot_window: int = 50,
) -> GannSignal:
    """Compute Gann angle signal from OHLC bars.

    Returns GannSignal with direction + confidence + rationale.
    """
    if len(closes) < 10:
        return GannSignal(
            direction="FLAT", confidence=0.0,
            pivot_type="high", pivot_price=float('nan'),
            pivot_bar_offset=0, nearest_angle="n/a",
            angle_value=float('nan'), distance_pct=0.0,
            rationale="insufficient bars (<10)",
            factors=["low_sample"],
        )
    current = closes[-1]
    if atr_value is None or atr_value <= 0:
        # Fallback: simple 14-bar ATR estimate
        ranges = [highs[i] - lows[i] for i in range(max(0, len(highs) - 14), len(highs))]
        atr_value = sum(ranges) / max(len(ranges), 1)
    if atr_value <= 0:
        return GannSignal(
            direction="FLAT", confidence=0.0,
            pivot_type="high", pivot_price=float('nan'),
            pivot_bar_offset=0, nearest_angle="n/a",
            angle_value=float('nan'), distance_pct=0.0,
            rationale="zero ATR — flat market or bad data",
            factors=["zero_atr"],
        )

    pivot_type, pivot_price, pivot_age = _find_pivot(highs, lows, pivot_window)
    if math.isnan(pivot_price):
        return GannSignal(
            direction="FLAT", confidence=0.0,
            pivot_type="high", pivot_price=float('nan'),
            pivot_bar_offset=0, nearest_angle="n/a",
            angle_value=float('nan'), distance_pct=0.0,
            rationale="pivot detection failed",
            factors=["no_pivot"],
        )
    pivot_is_low = pivot_type == "low"

    # Project all 9 Gann angles forward to current bar
    candidates: List[Tuple[str, float, float, float]] = []
    for ratio, name, weight in GANN_ANGLES:
        angle_y = _project_angle(
            pivot_price, pivot_age, len(closes) - 1,
            ratio, pivot_is_low, atr_value,
        )
        if math.isnan(angle_y) or angle_y <= 0:
            continue
        distance = current - angle_y
        distance_pct = distance / current * 100.0
        candidates.append((name, angle_y, distance, weight))

    if not candidates:
        return GannSignal(
            direction="FLAT", confidence=0.0,
            pivot_type=pivot_type, pivot_price=pivot_price,
            pivot_bar_offset=pivot_age, nearest_angle="n/a",
            angle_value=float('nan'), distance_pct=0.0,
            rationale="no projectable angles",
            factors=["projection_failed"],
        )

    # Pick the nearest angle by absolute distance
    candidates.sort(key=lambda t: abs(t[2]))
    nearest_name, nearest_y, nearest_dist, nearest_weight = candidates[0]
    distance_pct = (current - nearest_y) / current * 100.0
    abs_dist_atr = abs(nearest_dist) / atr_value

    # Direction logic:
    # pivot_is_low  + price ABOVE 1x1 → LONG continuation
    # pivot_is_low  + price BELOW 1x1 → trend weakening → FLAT or SHORT
    # pivot_is_high + price BELOW 1x1 → SHORT continuation
    # pivot_is_high + price ABOVE 1x1 → reversal → FLAT or LONG
    cardinal = next((c for c in candidates if c[0] == "1x1"), None)
    if cardinal is None:
        return GannSignal(
            direction="FLAT", confidence=0.5,
            pivot_type=pivot_type, pivot_price=pivot_price,
            pivot_bar_offset=pivot_age, nearest_angle=nearest_name,
            angle_value=nearest_y, distance_pct=distance_pct,
            rationale="cardinal 1x1 unprojectable",
            factors=["no_cardinal"],
        )
    cardinal_dist_pct = (current - cardinal[1]) / current * 100.0

    factors: List[str] = []
    if pivot_is_low and cardinal_dist_pct > 0:
        direction = "LONG"
        factors.append("pivot_low_above_1x1")
    elif pivot_is_low and cardinal_dist_pct <= 0:
        direction = "FLAT"
        factors.append("pivot_low_below_1x1_trend_weakening")
    elif (not pivot_is_low) and cardinal_dist_pct < 0:
        direction = "SHORT"
        factors.append("pivot_high_below_1x1")
    else:
        direction = "FLAT"
        factors.append("pivot_high_above_1x1_potential_reversal")

    # Confidence: high when price is within 0.3 ATR of an angle, weighted by
    # angle importance, dampened by pivot age (older = less reliable)
    proximity_score = max(0.0, 1.0 - abs_dist_atr / 0.5)
    age_decay = math.exp(-pivot_age / 100.0)
    confidence = float(min(1.0, max(0.0, proximity_score * nearest_weight * age_decay)))
    if abs_dist_atr > 1.0:
        factors.append("price_far_from_any_angle")
        confidence = min(confidence, 0.3)
    if pivot_age > 80:
        factors.append("stale_pivot")

    rationale = (
        f"pivot_{pivot_type}@{pivot_price:.2f} ({pivot_age}b ago); "
        f"current={current:.2f} cardinal_1x1@{cardinal[1]:.2f} "
        f"({cardinal_dist_pct:+.2f}%); nearest={nearest_name}@{nearest_y:.2f} "
        f"({distance_pct:+.2f}%, {abs_dist_atr:.2f} ATR); dir={direction}"
    )
    return GannSignal(
        direction=direction,
        confidence=confidence,
        pivot_type=pivot_type,
        pivot_price=pivot_price,
        pivot_bar_offset=pivot_age,
        nearest_angle=nearest_name,
        angle_value=nearest_y,
        distance_pct=distance_pct,
        rationale=rationale,
        factors=factors,
    )


def evaluate_dict(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    atr_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Dict form for tool registry. Truncated to <500 chars."""
    sig = evaluate(highs, lows, closes, atr_value)
    return {
        "strategy": "gann_angles",
        "direction": sig.direction,
        "confidence": round(sig.confidence, 3),
        "pivot": {
            "type": sig.pivot_type,
            "price": round(sig.pivot_price, 4),
            "bars_ago": sig.pivot_bar_offset,
        },
        "nearest_angle": sig.nearest_angle,
        "angle_value": round(sig.angle_value, 4),
        "distance_pct": round(sig.distance_pct, 3),
        "factors": sig.factors[:4],
        "rationale": sig.rationale[:200],
    }
