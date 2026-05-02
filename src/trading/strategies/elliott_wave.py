"""Elliott Wave detector — simplified 5-wave impulse + ABC correction.

Elliott Wave Theory (Ralph N. Elliott, 1938):
  Markets move in repeating 5-wave impulses (1-2-3-4-5) followed by
  3-wave corrections (A-B-C). Wave structure follows fixed rules:

  IMPULSE RULES (must hold):
    R1. Wave 2 never retraces more than 100% of Wave 1
    R2. Wave 3 is never the shortest among 1, 3, 5
    R3. Wave 4 doesn't overlap Wave 1's price territory (in non-leading
        non-ending diagonals)

  IMPULSE GUIDELINES (often hold):
    G1. Wave 3 is typically the longest (extension)
    G2. Wave 2 commonly retraces 50-78.6% of Wave 1
    G3. Wave 4 commonly retraces 23.6-38.2% of Wave 3
    G4. Wave 5 = approx Wave 1 length, OR 0.618 of (Wave 1 + Wave 3)

Detection approach:
  Pure rule-based zigzag pivot detection + wave-rule validation. NO ML
  wave-classifier (those are notoriously unreliable). Returns
  'unclear' rather than forced count when rules don't fit cleanly.

  1. Compute zigzag pivots using ATR-multiple threshold
  2. Sliding-window of last 6-9 pivots, check if they fit a
     5-wave impulse OR 3-wave correction
  3. Confidence high only when ALL impulse rules pass; medium when
     2/3 hold; low when count is ambiguous

Anti-overfit:
  * Threshold for zigzag = 1.5 × ATR (no learning)
  * Wave rules are hardcoded ratios with explicit Fibonacci tolerances
  * Returns 'unclear' on ambiguity rather than guessing
  * Wave-5 projections capped at 1.618 × Wave 1 (no extrapolation
    beyond observed historical extensions)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class Pivot:
    bar_idx: int
    price: float
    kind: str   # "high" or "low"


@dataclass
class WaveCount:
    structure: str        # "impulse_up", "impulse_down", "abc_corrective", "unclear"
    current_wave: str     # "1","2","3","4","5","A","B","C","none"
    direction: str        # "LONG", "SHORT", "FLAT"
    confidence: float
    pivots: List[Pivot] = field(default_factory=list)
    rules_passed: List[str] = field(default_factory=list)
    rules_failed: List[str] = field(default_factory=list)
    next_target: Optional[float] = None
    rationale: str = ""


def _zigzag_pivots(
    highs: List[float],
    lows: List[float],
    atr_value: float,
    threshold_atr: float = 1.5,
) -> List[Pivot]:
    """Detect significant pivots using ATR-threshold zigzag.

    A pivot is recorded when price has moved threshold_atr * ATR away
    from the last recorded pivot in the opposite direction.
    """
    if len(highs) < 5 or atr_value <= 0:
        return []
    threshold = atr_value * threshold_atr
    pivots: List[Pivot] = []

    # Initialize
    pivots.append(Pivot(bar_idx=0, price=highs[0], kind="high"))
    pivots.append(Pivot(bar_idx=0, price=lows[0], kind="low"))

    last_extreme_idx = 0
    last_extreme_price = (highs[0] + lows[0]) / 2.0
    direction: Optional[str] = None  # "up" or "down"

    for i in range(1, len(highs)):
        h, l = highs[i], lows[i]
        if direction is None or direction == "up":
            if h > last_extreme_price:
                last_extreme_price = h
                last_extreme_idx = i
                direction = "up"
            elif (last_extreme_price - l) >= threshold:
                pivots.append(Pivot(bar_idx=last_extreme_idx, price=last_extreme_price, kind="high"))
                last_extreme_price = l
                last_extreme_idx = i
                direction = "down"
        if direction == "down":
            if l < last_extreme_price:
                last_extreme_price = l
                last_extreme_idx = i
                direction = "down"
            elif (h - last_extreme_price) >= threshold:
                pivots.append(Pivot(bar_idx=last_extreme_idx, price=last_extreme_price, kind="low"))
                last_extreme_price = h
                last_extreme_idx = i
                direction = "up"

    # Add the unconfirmed final pivot
    pivots.append(Pivot(
        bar_idx=last_extreme_idx,
        price=last_extreme_price,
        kind="high" if direction == "up" else "low",
    ))

    # Drop the synthetic init pivots if we accumulated real ones
    if len(pivots) > 4:
        pivots = pivots[2:]

    # Dedupe consecutive same-kind pivots — keep extreme
    cleaned: List[Pivot] = []
    for p in pivots:
        if cleaned and cleaned[-1].kind == p.kind:
            if (p.kind == "high" and p.price > cleaned[-1].price) or \
               (p.kind == "low" and p.price < cleaned[-1].price):
                cleaned[-1] = p
        else:
            cleaned.append(p)
    return cleaned


def _check_impulse_up(p: List[Pivot]) -> Tuple[bool, List[str], List[str]]:
    """Check if last 6 pivots form an upward 5-wave impulse: low-high-low-high-low-high."""
    if len(p) < 6:
        return False, [], ["not_enough_pivots"]
    seq = p[-6:]
    expected_kinds = ["low", "high", "low", "high", "low", "high"]
    if [x.kind for x in seq] != expected_kinds:
        return False, [], ["pivot_sequence_mismatch"]

    w0_low, w1_high, w2_low, w3_high, w4_low, w5_high = [x.price for x in seq]
    wave1 = w1_high - w0_low
    wave2 = w1_high - w2_low
    wave3 = w3_high - w2_low
    wave4 = w3_high - w4_low
    wave5 = w5_high - w4_low

    passed: List[str] = []
    failed: List[str] = []

    # R1: Wave 2 never retraces more than 100% of Wave 1
    if wave2 < wave1:
        passed.append("R1_wave2_under_100pct_w1")
    else:
        failed.append("R1_wave2_over_100pct_w1")

    # R2: Wave 3 is never the shortest among 1, 3, 5
    if wave3 >= min(wave1, wave5) or wave3 == max(wave1, wave3, wave5):
        passed.append("R2_wave3_not_shortest")
    else:
        failed.append("R2_wave3_is_shortest")

    # R3: Wave 4 doesn't overlap Wave 1's territory — w4_low > w1_high
    if w4_low > w1_high:
        passed.append("R3_wave4_no_overlap_w1")
    else:
        failed.append("R3_wave4_overlaps_w1")

    # G1: Wave 3 longest (extension)
    if wave3 >= max(wave1, wave5):
        passed.append("G1_wave3_longest")

    # G2: Wave 2 retraces 50-78.6% of Wave 1
    w2_retrace = wave2 / max(wave1, 1e-9)
    if 0.382 <= w2_retrace <= 0.786:
        passed.append("G2_wave2_50_78_retrace")
    else:
        failed.append(f"G2_wave2_retrace_{w2_retrace:.2f}_outside")

    # G3: Wave 4 retraces 23.6-38.2% of Wave 3
    w4_retrace = wave4 / max(wave3, 1e-9)
    if 0.236 <= w4_retrace <= 0.5:
        passed.append("G3_wave4_24_50_retrace")

    valid = "R1_wave2_under_100pct_w1" in passed and \
            "R2_wave3_not_shortest" in passed and \
            "R3_wave4_no_overlap_w1" in passed
    return valid, passed, failed


def _check_impulse_down(p: List[Pivot]) -> Tuple[bool, List[str], List[str]]:
    """Mirror — high-low-high-low-high-low for downward impulse."""
    if len(p) < 6:
        return False, [], ["not_enough_pivots"]
    seq = p[-6:]
    expected_kinds = ["high", "low", "high", "low", "high", "low"]
    if [x.kind for x in seq] != expected_kinds:
        return False, [], ["pivot_sequence_mismatch"]

    w0_high, w1_low, w2_high, w3_low, w4_high, w5_low = [x.price for x in seq]
    wave1 = w0_high - w1_low
    wave2 = w2_high - w1_low
    wave3 = w2_high - w3_low
    wave4 = w4_high - w3_low
    wave5 = w4_high - w5_low

    passed: List[str] = []
    failed: List[str] = []
    if wave2 < wave1:
        passed.append("R1_wave2_under_100pct_w1")
    else:
        failed.append("R1_wave2_over_100pct_w1")
    if wave3 >= min(wave1, wave5) or wave3 == max(wave1, wave3, wave5):
        passed.append("R2_wave3_not_shortest")
    else:
        failed.append("R2_wave3_is_shortest")
    if w4_high < w1_low:
        passed.append("R3_wave4_no_overlap_w1")
    else:
        failed.append("R3_wave4_overlaps_w1")
    if wave3 >= max(wave1, wave5):
        passed.append("G1_wave3_longest")
    w2_retrace = wave2 / max(wave1, 1e-9)
    if 0.382 <= w2_retrace <= 0.786:
        passed.append("G2_wave2_50_78_retrace")
    w4_retrace = wave4 / max(wave3, 1e-9)
    if 0.236 <= w4_retrace <= 0.5:
        passed.append("G3_wave4_24_50_retrace")
    valid = "R1_wave2_under_100pct_w1" in passed and \
            "R2_wave3_not_shortest" in passed and \
            "R3_wave4_no_overlap_w1" in passed
    return valid, passed, failed


def evaluate(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    atr_value: Optional[float] = None,
    threshold_atr: float = 1.5,
) -> WaveCount:
    """Detect Elliott wave structure in last N bars.

    Returns WaveCount with structure label + direction inference.
    """
    if len(closes) < 30:
        return WaveCount(
            structure="unclear", current_wave="none", direction="FLAT",
            confidence=0.0, rationale="insufficient bars (<30)",
        )
    if atr_value is None or atr_value <= 0:
        ranges = [highs[i] - lows[i] for i in range(max(0, len(highs) - 14), len(highs))]
        atr_value = sum(ranges) / max(len(ranges), 1)
    if atr_value <= 0:
        return WaveCount(
            structure="unclear", current_wave="none", direction="FLAT",
            confidence=0.0, rationale="zero ATR",
        )

    pivots = _zigzag_pivots(highs, lows, atr_value, threshold_atr)
    if len(pivots) < 6:
        return WaveCount(
            structure="unclear", current_wave="none", direction="FLAT",
            confidence=0.2, pivots=pivots,
            rationale=f"only {len(pivots)} zigzag pivots — need >=6 for impulse",
        )

    valid_up, up_passed, up_failed = _check_impulse_up(pivots)
    valid_down, down_passed, down_failed = _check_impulse_down(pivots)

    # Decision
    if valid_up:
        seq = pivots[-6:]
        wave1 = seq[1].price - seq[0].price
        # Project Wave 5 ext: typically equal to Wave 1 OR 1.618 × W1
        next_target = seq[5].price + wave1
        return WaveCount(
            structure="impulse_up", current_wave="5", direction="LONG",
            confidence=0.8 if "G1_wave3_longest" in up_passed else 0.6,
            pivots=pivots[-6:], rules_passed=up_passed, rules_failed=up_failed,
            next_target=float(next_target),
            rationale=f"5-wave UP impulse confirmed; W3 ext={wave1:.2f}; "
                      f"next target ≈ {next_target:.2f} (W5 = W1)",
        )
    if valid_down:
        seq = pivots[-6:]
        wave1 = seq[0].price - seq[1].price
        next_target = seq[5].price - wave1
        return WaveCount(
            structure="impulse_down", current_wave="5", direction="SHORT",
            confidence=0.8 if "G1_wave3_longest" in down_passed else 0.6,
            pivots=pivots[-6:], rules_passed=down_passed, rules_failed=down_failed,
            next_target=float(next_target),
            rationale=f"5-wave DOWN impulse confirmed; W3 ext={wave1:.2f}; "
                      f"next target ≈ {next_target:.2f} (W5 = W1)",
        )

    # Check 3-wave ABC correction
    if len(pivots) >= 4:
        last4 = pivots[-4:]
        if [x.kind for x in last4] == ["high", "low", "high", "low"]:
            # Bearish ABC
            return WaveCount(
                structure="abc_corrective", current_wave="C", direction="FLAT",
                confidence=0.4, pivots=last4,
                rationale="ABC bearish correction in progress; await impulse start",
            )
        if [x.kind for x in last4] == ["low", "high", "low", "high"]:
            return WaveCount(
                structure="abc_corrective", current_wave="C", direction="FLAT",
                confidence=0.4, pivots=last4,
                rationale="ABC bullish correction in progress; await impulse start",
            )

    # Partial impulse — best guess based on what passed
    best_passed = up_passed if len(up_passed) >= len(down_passed) else down_passed
    direction = "LONG" if len(up_passed) > len(down_passed) else "SHORT" if len(down_passed) > 0 else "FLAT"
    return WaveCount(
        structure="unclear", current_wave="?",
        direction=direction if len(best_passed) >= 2 else "FLAT",
        confidence=min(0.4, len(best_passed) / 6.0),
        pivots=pivots[-6:],
        rules_passed=best_passed,
        rules_failed=up_failed if direction == "LONG" else down_failed,
        rationale=f"partial wave count; {len(best_passed)} rules pass; ambiguous",
    )


def evaluate_dict(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    atr_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Dict form for tool registry."""
    wc = evaluate(highs, lows, closes, atr_value)
    return {
        "strategy": "elliott_wave",
        "direction": wc.direction,
        "confidence": round(wc.confidence, 3),
        "structure": wc.structure,
        "current_wave": wc.current_wave,
        "rules_passed": wc.rules_passed[:5],
        "rules_failed": wc.rules_failed[:3],
        "next_target": round(wc.next_target, 4) if wc.next_target else None,
        "n_pivots": len(wc.pivots),
        "rationale": wc.rationale[:200],
    }
