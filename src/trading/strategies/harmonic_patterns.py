"""Harmonic pattern detector — Gartley / Bat / Butterfly / Crab.

Harmonic patterns are 5-pivot price structures (X-A-B-C-D) where the
ratios between legs match Fibonacci proportions. Each named pattern
has fixed ratio bands:

  GARTLEY:  AB = 0.618 × XA, BC = 0.382-0.886 × AB,
            CD = 1.272-1.618 × BC, AD = 0.786 × XA
  BAT:      AB = 0.382-0.50 × XA, BC = 0.382-0.886 × AB,
            CD = 1.618-2.618 × BC, AD = 0.886 × XA
  BUTTERFLY: AB = 0.786 × XA, BC = 0.382-0.886 × AB,
             CD = 1.618-2.24 × BC, AD = 1.272-1.618 × XA
  CRAB:     AB = 0.382-0.618 × XA, BC = 0.382-0.886 × AB,
            CD = 2.24-3.618 × BC, AD = 1.618 × XA

Bullish pattern: X=high, A=low, B=high, C=low, D=low (potential reversal up at D)
Bearish pattern: X=low,  A=high, B=low, C=high, D=high (potential reversal down at D)

The "Potential Reversal Zone" (PRZ) at point D is the entry area.
Detection uses tight Fibonacci tolerance bands; partial matches return
lower-confidence "incomplete" signals.

Anti-overfit:
  * Hardcoded Fibonacci ratios — no learning
  * Each pattern has explicit tolerance windows (±5% on the key ratios)
  * Confidence dampens by tolerance breach distance
  * Returns 'unclear' rather than forcing a pattern when ratios don't fit
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from src.trading.strategies.elliott_wave import _zigzag_pivots, Pivot


# Pattern definitions: key fib ratios (XA, AB/XA, BC/AB, CD/BC, AD/XA)
# Tolerance ±5% on key ratios.
PATTERN_DEFS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "gartley": {
        "ab_xa":  (0.618 - 0.05, 0.618 + 0.05),
        "bc_ab":  (0.382, 0.886),
        "cd_bc":  (1.272, 1.618),
        "ad_xa":  (0.786 - 0.05, 0.786 + 0.05),
    },
    "bat": {
        "ab_xa":  (0.382, 0.50),
        "bc_ab":  (0.382, 0.886),
        "cd_bc":  (1.618, 2.618),
        "ad_xa":  (0.886 - 0.05, 0.886 + 0.05),
    },
    "butterfly": {
        "ab_xa":  (0.786 - 0.05, 0.786 + 0.05),
        "bc_ab":  (0.382, 0.886),
        "cd_bc":  (1.618, 2.24),
        "ad_xa":  (1.272, 1.618),
    },
    "crab": {
        "ab_xa":  (0.382, 0.618),
        "bc_ab":  (0.382, 0.886),
        "cd_bc":  (2.24, 3.618),
        "ad_xa":  (1.618 - 0.05, 1.618 + 0.05),
    },
}


@dataclass
class HarmonicMatch:
    pattern: str          # gartley/bat/butterfly/crab/none
    bias: str             # bullish/bearish/none
    direction: str        # LONG/SHORT/FLAT
    confidence: float
    points: Dict[str, float] = field(default_factory=dict)   # X/A/B/C/D prices
    ratios: Dict[str, float] = field(default_factory=dict)
    tolerance_breaches: List[str] = field(default_factory=list)
    prz_low: Optional[float] = None
    prz_high: Optional[float] = None
    rationale: str = ""


def _ratio_in_band(ratio: float, band: Tuple[float, float]) -> bool:
    return band[0] <= ratio <= band[1]


def _tolerance_distance(ratio: float, band: Tuple[float, float]) -> float:
    """How far outside the band (0 if inside). Used for confidence dampening."""
    if band[0] <= ratio <= band[1]:
        return 0.0
    if ratio < band[0]:
        return (band[0] - ratio) / band[0]
    return (ratio - band[1]) / band[1]


def _evaluate_xabcd(
    x: Pivot, a: Pivot, b: Pivot, c: Pivot, d: Pivot,
) -> Tuple[Optional[str], Optional[str], Dict[str, float], List[str]]:
    """Test XABCD configuration against all 4 patterns. Return best match."""
    # Verify alternating pivot kinds
    bullish_kinds = ["high", "low", "high", "low", "low"]
    bearish_kinds = ["low", "high", "low", "high", "high"]
    actual = [x.kind, a.kind, b.kind, c.kind, d.kind]
    if actual == bullish_kinds:
        bias = "bearish"  # bullish-shaped XABCD where D is LOW = potential LONG entry
        # Wait — naming: structure with X=high A=low gives bullish setup at D
        bias = "bullish"
    elif actual == bearish_kinds:
        bias = "bearish"
    else:
        return None, None, {}, ["pivot_sequence_mismatch"]

    xa = abs(a.price - x.price)
    ab = abs(b.price - a.price)
    bc = abs(c.price - b.price)
    cd = abs(d.price - c.price)
    ad = abs(d.price - a.price)
    if xa <= 0 or ab <= 0 or bc <= 0 or cd <= 0:
        return None, None, {}, ["zero_leg_length"]

    ratios = {
        "ab_xa": ab / xa,
        "bc_ab": bc / ab,
        "cd_bc": cd / bc,
        "ad_xa": ad / xa,
    }

    best_match: Optional[Tuple[str, float, List[str]]] = None
    for pattern_name, bands in PATTERN_DEFS.items():
        breaches: List[str] = []
        total_breach = 0.0
        for ratio_name, band in bands.items():
            r = ratios[ratio_name]
            if not _ratio_in_band(r, band):
                breaches.append(f"{ratio_name}={r:.3f}_outside_{band[0]:.2f}-{band[1]:.2f}")
                total_breach += _tolerance_distance(r, band)
        score = max(0.0, 1.0 - total_breach)
        if best_match is None or score > best_match[1]:
            best_match = (pattern_name, score, breaches)

    if best_match is None or best_match[1] < 0.3:
        return None, bias, ratios, best_match[2] if best_match else ["no_match"]
    return best_match[0], bias, ratios, best_match[2]


def evaluate(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    atr_value: Optional[float] = None,
    threshold_atr: float = 1.5,
) -> HarmonicMatch:
    """Scan recent pivots for harmonic XABCD patterns."""
    if len(closes) < 30:
        return HarmonicMatch(
            pattern="none", bias="none", direction="FLAT",
            confidence=0.0, rationale="insufficient bars (<30)",
        )
    if atr_value is None or atr_value <= 0:
        ranges = [highs[i] - lows[i] for i in range(max(0, len(highs) - 14), len(highs))]
        atr_value = sum(ranges) / max(len(ranges), 1)
    if atr_value <= 0:
        return HarmonicMatch(
            pattern="none", bias="none", direction="FLAT",
            confidence=0.0, rationale="zero ATR",
        )

    pivots = _zigzag_pivots(highs, lows, atr_value, threshold_atr)
    if len(pivots) < 5:
        return HarmonicMatch(
            pattern="none", bias="none", direction="FLAT",
            confidence=0.0,
            rationale=f"only {len(pivots)} zigzag pivots — need >=5 for XABCD",
        )

    # Try the most-recent 5 pivots as XABCD
    best: Optional[HarmonicMatch] = None
    for start in range(max(0, len(pivots) - 5), len(pivots) - 4):
        x, a, b, c, d = pivots[start:start + 5]
        pattern, bias, ratios, breaches = _evaluate_xabcd(x, a, b, c, d)
        if pattern is None:
            continue
        # Confidence based on breach count
        confidence = max(0.2, 1.0 - 0.15 * len(breaches))
        # Direction: bullish bias at point D = LONG; bearish = SHORT
        direction = "LONG" if bias == "bullish" else "SHORT" if bias == "bearish" else "FLAT"
        prz_buffer = atr_value * 0.5
        match = HarmonicMatch(
            pattern=pattern,
            bias=bias or "none",
            direction=direction,
            confidence=float(confidence),
            points={"X": x.price, "A": a.price, "B": b.price, "C": c.price, "D": d.price},
            ratios={k: round(v, 3) for k, v in ratios.items()},
            tolerance_breaches=breaches,
            prz_low=float(d.price - prz_buffer),
            prz_high=float(d.price + prz_buffer),
            rationale=f"{bias} {pattern} at D={d.price:.2f}; "
                      f"AB/XA={ratios['ab_xa']:.3f}, AD/XA={ratios['ad_xa']:.3f}; "
                      f"{len(breaches)} tolerance breaches",
        )
        if best is None or match.confidence > best.confidence:
            best = match

    if best is None:
        return HarmonicMatch(
            pattern="none", bias="none", direction="FLAT",
            confidence=0.1,
            rationale=f"{len(pivots)} pivots present but no XABCD matches",
        )
    return best


def evaluate_dict(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    atr_value: Optional[float] = None,
) -> Dict[str, Any]:
    m = evaluate(highs, lows, closes, atr_value)
    return {
        "strategy": "harmonic_patterns",
        "pattern": m.pattern,
        "bias": m.bias,
        "direction": m.direction,
        "confidence": round(m.confidence, 3),
        "points": {k: round(v, 4) for k, v in m.points.items()},
        "ratios": m.ratios,
        "prz_low": round(m.prz_low, 4) if m.prz_low else None,
        "prz_high": round(m.prz_high, 4) if m.prz_high else None,
        "tolerance_breaches": m.tolerance_breaches[:3],
        "rationale": m.rationale[:200],
    }
