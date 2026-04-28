"""Cumulative Volume Delta (CVD) — order flow imbalance from OHLCV.

CVD is foundational order-flow trading practice (Bookmap/footprint
charts). It estimates buying vs selling pressure from per-bar volume
+ close-position-within-range, accumulated over a window.

Algorithm (close-relative-to-range estimator, no tick data needed):
  buy_volume_estimate  = volume × (close - low) / (high - low + ε)
  sell_volume_estimate = volume × (high - close) / (high - low + ε)
  cvd_delta            = buy_volume - sell_volume
  cvd_cumulative       = running sum over last N bars

Predictive value:
  * cvd_slope rising → accumulation phase (buyers in control)
  * cvd_slope falling → distribution phase (sellers in control)
  * cvd_divergence (price up + CVD down or price down + CVD up) often
    precedes reversals — leading indicator

Anti-noise:
  * Divergence requires N-bar persistence (default 5 bars), not a
    single-bar flip
  * Bounded outputs (slope sign, divergence bool, momentum_score in [-1, +1])
  * Pure compute — no external API, no learned weights
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_LOOKBACK = 50
DEFAULT_DIVERGENCE_BARS = 5


@dataclass
class CVDSignals:
    method: str
    cvd_recent: float = 0.0
    cvd_slope_sign: int = 0           # +1 rising, -1 falling, 0 flat
    cvd_divergence: bool = False
    cvd_divergence_kind: str = "none"  # bearish / bullish / none
    cvd_momentum_score: float = 0.0   # [-1, +1]
    n_bars_used: int = 0
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "cvd_recent": round(float(self.cvd_recent), 4),
            "cvd_slope_sign": int(self.cvd_slope_sign),
            "cvd_divergence": bool(self.cvd_divergence),
            "cvd_divergence_kind": self.cvd_divergence_kind,
            "cvd_momentum_score": round(float(self.cvd_momentum_score), 3),
            "n_bars_used": int(self.n_bars_used),
            "rationale": self.rationale[:300],
        }


def _per_bar_delta(open_p: float, high: float, low: float, close: float,
                    volume: float) -> float:
    """Close-relative-to-range CVD estimator."""
    rng = max(high - low, 1e-9)
    buy = volume * (close - low) / rng
    sell = volume * (high - close) / rng
    return buy - sell


def compute_cvd(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float],
    opens: Optional[List[float]] = None,
    lookback: int = DEFAULT_LOOKBACK,
    divergence_bars: int = DEFAULT_DIVERGENCE_BARS,
) -> CVDSignals:
    """Compute CVD signals over the last `lookback` bars."""
    n = min(len(closes), len(highs), len(lows), len(volumes))
    if n < max(20, divergence_bars + 5):
        return CVDSignals(method="cvd",
                           rationale="insufficient_bars",
                           n_bars_used=n)
    if opens is None or len(opens) < n:
        opens = [closes[i - 1] if i > 0 else closes[0] for i in range(n)]

    bars = min(lookback, n)
    closes = closes[-bars:]
    highs = highs[-bars:]
    lows = lows[-bars:]
    volumes = volumes[-bars:]
    opens = opens[-bars:]

    deltas = []
    for o, h, l, c, v in zip(opens, highs, lows, closes, volumes):
        deltas.append(_per_bar_delta(o, h, l, c, v))
    cumulative = []
    s = 0.0
    for d in deltas:
        s += d
        cumulative.append(s)

    # Slope sign over last divergence_bars window
    if len(cumulative) >= divergence_bars + 1:
        recent = cumulative[-divergence_bars - 1:]
        slope = recent[-1] - recent[0]
        slope_sign = 1 if slope > 0 else (-1 if slope < 0 else 0)
    else:
        slope_sign = 0

    # Divergence: compare price direction vs CVD direction over divergence_bars
    cvd_divergence = False
    cvd_div_kind = "none"
    if len(closes) >= divergence_bars + 1:
        price_change = closes[-1] - closes[-divergence_bars - 1]
        cvd_change = cumulative[-1] - cumulative[-divergence_bars - 1]
        if price_change > 0 and cvd_change < 0:
            cvd_divergence = True
            cvd_div_kind = "bearish"  # price rose but CVD fell
        elif price_change < 0 and cvd_change > 0:
            cvd_divergence = True
            cvd_div_kind = "bullish"  # price fell but CVD rose

    # Momentum score [-1, +1]
    abs_max = max(abs(min(cumulative)), abs(max(cumulative)), 1e-9)
    momentum = cumulative[-1] / abs_max
    momentum = max(-1.0, min(1.0, momentum))

    parts = [f"slope={slope_sign:+d}",
             f"momentum={momentum:+.2f}",
             f"divergence={cvd_div_kind}"]

    return CVDSignals(
        method="cvd",
        cvd_recent=cumulative[-1],
        cvd_slope_sign=slope_sign,
        cvd_divergence=cvd_divergence,
        cvd_divergence_kind=cvd_div_kind,
        cvd_momentum_score=momentum,
        n_bars_used=bars,
        rationale=" | ".join(parts)[:300],
    )
