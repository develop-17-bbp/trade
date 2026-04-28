"""Wyckoff phase detector (accumulation / markup / distribution / markdown).

Wyckoff method partitions market cycles into four phases:
  1. ACCUMULATION   — sideways consolidation at lows after a downtrend;
                       smart money buying; volume drying up except
                       on bottoming spikes
  2. MARKUP         — uptrend out of accumulation; rising prices on
                       sustained volume; higher highs and higher lows
  3. DISTRIBUTION   — sideways consolidation at highs after an uptrend;
                       smart money selling; lower highs forming
  4. MARKDOWN       — downtrend out of distribution; falling prices

Detector heuristic (no ML — purely structural):
  * Accumulation: price is in lower 30% of recent range, volatility
    (ATR/price) declining, recent lows are higher than older lows
  * Markup: price > 60% of recent range, EMA rising, higher highs
  * Distribution: price in upper 30% of recent range, volatility
    declining at highs, recent highs lower than older highs
  * Markdown: price < 40% of recent range, EMA falling, lower lows

Anti-overfit design:
  * Pure structural heuristic — no parameter learning
  * Confidence is rule-based with sample-size warning when
    bars < 50 (low-sample = low-confidence)
  * Output bounded to one phase + one confidence float
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class WyckoffVerdict:
    phase: str                    # accumulation/markup/distribution/markdown/unclear
    confidence: float             # 0.0 - 1.0
    factors: List[str]
    sample_size: int
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "confidence": round(float(self.confidence), 3),
            "factors": self.factors[:8],
            "sample_size": int(self.sample_size),
            "rationale": self.rationale[:300],
        }


def _ema(values: List[float], period: int) -> List[float]:
    """Quick EMA without numpy."""
    if not values or period <= 0:
        return []
    k = 2 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def detect_phase(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float],
    lookback: int = 50,
) -> WyckoffVerdict:
    """Identify the current Wyckoff phase from price + volume structure.

    Returns "unclear" with confidence 0 when bars < 30.
    """
    n = min(len(closes), len(highs), len(lows), len(volumes))
    if n < 30:
        return WyckoffVerdict(
            phase="unclear", confidence=0.0, factors=["insufficient_bars"],
            sample_size=n, rationale="need >= 30 bars for phase detection",
        )

    bars = min(lookback, n)
    closes = closes[-bars:]
    highs = highs[-bars:]
    lows = lows[-bars:]
    volumes = volumes[-bars:]

    period_high = max(highs)
    period_low = min(lows)
    range_size = max(0.0001, period_high - period_low)
    last_close = closes[-1]
    pct_in_range = (last_close - period_low) / range_size

    # EMA slope direction
    ema_20 = _ema(closes, period=20)
    ema_slope = (ema_20[-1] - ema_20[-5]) / max(0.0001, ema_20[-5]) if len(ema_20) >= 5 else 0.0

    # Volatility trend (ATR proxy = avg range, last-half vs first-half)
    half = bars // 2
    atr_first = sum(h - l for h, l in zip(highs[:half], lows[:half])) / max(1, half)
    atr_second = sum(h - l for h, l in zip(highs[half:], lows[half:])) / max(1, bars - half)
    vol_declining = atr_second < atr_first * 0.85

    # Higher-lows / lower-highs structure
    third = bars // 3
    older_low = min(lows[:third])
    recent_low = min(lows[-third:])
    older_high = max(highs[:third])
    recent_high = max(highs[-third:])
    higher_lows = recent_low > older_low * 1.005
    lower_highs = recent_high < older_high * 0.995

    factors: List[str] = []
    phase = "unclear"
    confidence = 0.3

    # Accumulation
    if (pct_in_range <= 0.35 and vol_declining and higher_lows and abs(ema_slope) < 0.005):
        phase = "accumulation"
        factors = [
            f"price_in_lower_{int(pct_in_range*100)}%_of_range",
            "volatility_declining",
            "higher_lows_forming",
            f"ema_slope_flat={ema_slope:+.4f}",
        ]
        confidence = 0.7

    # Markup
    elif pct_in_range >= 0.55 and ema_slope > 0.005 and not lower_highs:
        phase = "markup"
        factors = [
            f"price_in_upper_{int(pct_in_range*100)}%_of_range",
            f"ema_rising_slope={ema_slope:+.4f}",
            "higher_highs",
        ]
        confidence = 0.65 + min(0.2, ema_slope * 20)

    # Distribution
    elif (pct_in_range >= 0.65 and vol_declining and lower_highs and abs(ema_slope) < 0.005):
        phase = "distribution"
        factors = [
            f"price_in_upper_{int(pct_in_range*100)}%_of_range",
            "volatility_declining",
            "lower_highs_forming",
            f"ema_slope_flat={ema_slope:+.4f}",
        ]
        confidence = 0.7

    # Markdown
    elif pct_in_range <= 0.45 and ema_slope < -0.005 and not higher_lows:
        phase = "markdown"
        factors = [
            f"price_in_lower_{int(pct_in_range*100)}%_of_range",
            f"ema_falling_slope={ema_slope:+.4f}",
            "lower_lows",
        ]
        confidence = 0.65 + min(0.2, abs(ema_slope) * 20)

    # Sample-size penalty: low-confidence flag below 50 bars
    if n < 50:
        confidence *= 0.7
        factors.append("low_sample_warning")

    confidence = max(0.0, min(1.0, confidence))
    rationale = (
        f"phase={phase} pct_in_range={pct_in_range:.2f} "
        f"ema_slope={ema_slope:+.4f} vol_declining={vol_declining} "
        f"factors={len(factors)}"
    )

    return WyckoffVerdict(
        phase=phase, confidence=confidence,
        factors=factors, sample_size=n,
        rationale=rationale,
    )
