"""Liquidity-sweep / stop-hunt reversal detector (ICT pattern).

The 2026 SMC literature converges on liquidity sweeps as the
highest-EV reversal pattern: price spikes past a recent swing high
(sweeps buy-stops resting above it), reverses sharply within 3-5
candles, leaves a Fair Value Gap behind, and re-enters the prior
range.

Symmetric for swing lows (sell-stops below).

This module is a pattern *detector* — it returns structured info
about whether a sweep occurred, with what confluence. It does not
emit trades by itself; the LLM brain reads the signal and decides
whether to act.

Anti-overfit / anti-noise design:
  * Requires multiple confluence factors (sweep + reversal candle +
    FVG presence) — single-factor detections are flagged "weak"
  * Bounded output (one structured dict per call)
  * Confidence is rule-based, not over-tuned to backtest
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SweepDetection:
    detected: bool
    direction: str = "NONE"      # "LONG" (low-sweep reversal) | "SHORT" (high-sweep reversal)
    swept_level: float = 0.0     # the swing high/low that was swept
    reversal_strength: float = 0.0  # 0.0 - 1.0
    fvg_present: bool = False
    confluence_count: int = 0    # 0-5 factors aligned
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "direction": self.direction,
            "swept_level": round(float(self.swept_level), 2),
            "reversal_strength": round(float(self.reversal_strength), 3),
            "fvg_present": self.fvg_present,
            "confluence_count": int(self.confluence_count),
            "rationale": self.rationale[:300],
        }


def _find_swing_high(highs: List[float], lookback: int = 20) -> Optional[float]:
    """Highest high over `lookback` bars excluding the most recent 5
    (so we have separation between sweep target and current action)."""
    if len(highs) < lookback + 5:
        return None
    return max(highs[-lookback - 5:-5])


def _find_swing_low(lows: List[float], lookback: int = 20) -> Optional[float]:
    if len(lows) < lookback + 5:
        return None
    return min(lows[-lookback - 5:-5])


def _has_fvg_after_sweep(highs: List[float], lows: List[float],
                         direction: str) -> bool:
    """Check if a 3-candle FVG formed in the most recent 5 bars in the
    direction of the reversal. For a LONG reversal (low-sweep), we
    look for a bullish FVG (gap above)."""
    if len(highs) < 5 or len(lows) < 5:
        return False
    # Walk last 5 bars; check candles[-5..-3] for 3-candle imbalance.
    for i in range(-5, -2):
        try:
            c1_high = highs[i]
            c1_low = lows[i]
            c3_high = highs[i + 2]
            c3_low = lows[i + 2]
            if direction == "LONG":
                # Bullish FVG: c3_low > c1_high (gap up)
                if c3_low > c1_high:
                    return True
            elif direction == "SHORT":
                # Bearish FVG: c3_high < c1_low (gap down)
                if c3_high < c1_low:
                    return True
        except (IndexError, TypeError):
            continue
    return False


def detect_liquidity_sweep(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    lookback: int = 20,
) -> SweepDetection:
    """Scan for a recent liquidity sweep + reversal pattern.

    A sweep is detected when:
      1. Recent N bars include a wick that exceeded a prior swing
         high/low (the stop hunt)
      2. The current candle closed BACK INSIDE the prior range
         (the reversal)
      3. Optional confluence: FVG formed in the reversal direction,
         volume spike on the sweep candle, candle-body strength

    Returns SweepDetection.detected=False (with rationale) when no
    pattern is found. No exceptions raised on bad inputs.
    """
    if len(closes) < lookback + 5 or len(highs) != len(closes):
        return SweepDetection(detected=False, rationale="insufficient_bars")

    swing_high = _find_swing_high(highs, lookback=lookback)
    swing_low = _find_swing_low(lows, lookback=lookback)
    if swing_high is None or swing_low is None:
        return SweepDetection(detected=False, rationale="no_swing_pivots")

    recent_highs = highs[-5:]
    recent_lows = lows[-5:]
    last_close = closes[-1]
    avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 0.0
    last_vol = volumes[-1] if volumes else 0.0

    # SHORT setup — high-sweep reversal: a wick poked above swing_high,
    # then price closed back below it.
    high_sweep = max(recent_highs) > swing_high and last_close < swing_high
    # LONG setup — low-sweep reversal
    low_sweep = min(recent_lows) < swing_low and last_close > swing_low

    if not (high_sweep or low_sweep):
        return SweepDetection(detected=False,
                              rationale="no_sweep_in_recent_5_bars")

    direction = "SHORT" if high_sweep else "LONG"
    swept_level = swing_high if high_sweep else swing_low

    # Reversal strength = how far back inside the range
    if direction == "SHORT":
        reversal_strength = max(0.0, min(1.0,
            (swing_high - last_close) / max(0.001, swing_high - swing_low)))
    else:
        reversal_strength = max(0.0, min(1.0,
            (last_close - swing_low) / max(0.001, swing_high - swing_low)))

    fvg_present = _has_fvg_after_sweep(highs, lows, direction)

    # Confluence factors:
    #   1. sweep occurred (always 1 if we're here)
    #   2. reversal_strength >= 0.3 (close back >= 30% into range)
    #   3. FVG present in reversal direction
    #   4. volume spike on sweep candle (>= 1.5x avg)
    #   5. closing candle body >= 50% of range
    factors = 1
    if reversal_strength >= 0.3:
        factors += 1
    if fvg_present:
        factors += 1
    if last_vol >= avg_vol * 1.5:
        factors += 1
    last_range = recent_highs[-1] - recent_lows[-1]
    if last_range > 0:
        body_pct = abs(last_close - closes[-2]) / last_range if len(closes) >= 2 else 0
        if body_pct >= 0.5:
            factors += 1

    parts = [f"swept {direction}-side liquidity at ${swept_level:.2f}"]
    parts.append(f"reversal_strength={reversal_strength:.2f}")
    if fvg_present:
        parts.append("FVG_present")
    if last_vol >= avg_vol * 1.5:
        parts.append(f"volume_spike={last_vol/max(1, avg_vol):.1f}x")
    parts.append(f"confluence={factors}/5")

    return SweepDetection(
        detected=factors >= 2,
        direction=direction,
        swept_level=swept_level,
        reversal_strength=reversal_strength,
        fvg_present=fvg_present,
        confluence_count=factors,
        rationale=" | ".join(parts)[:300],
    )
