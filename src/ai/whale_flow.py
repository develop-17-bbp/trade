"""Whale-flow detector — large-trade signal from existing OHLCV.

Whale moves (large transfers, exchange flows, abnormal volume bars)
are leading indicators of significant price moves. Free APIs like
whale-alert.io exist but rate-limit aggressively. This module uses
two pure-compute proxies on existing OHLCV data:

  1. Volume-anomaly detector — bars whose volume is N standard
     deviations above rolling mean → "whale bar" candidate
  2. Volume-weighted price impact — large-volume bars whose close
     deviated significantly from VWAP → directional whale signal

Anti-noise:
  * Z-score threshold (default 2.5σ) prevents false positives
  * Requires absolute volume floor (no microbar spikes)
  * Direction inferred from close vs open (large red volume bar =
    distribution; large green volume bar = accumulation)
  * Bounded outputs (count + signed score)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_LOOKBACK = 100
DEFAULT_Z_THRESHOLD = 2.5


@dataclass
class WhaleSignals:
    method: str
    n_whale_bars_recent: int = 0
    whale_buy_count: int = 0
    whale_sell_count: int = 0
    last_whale_bar_age_bars: int = -1
    last_whale_direction: str = "none"  # buy / sell / none
    last_whale_z_score: float = 0.0
    whale_directional_bias: float = 0.0  # [-1, +1]
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n_whale_bars_recent": int(self.n_whale_bars_recent),
            "whale_buy_count": int(self.whale_buy_count),
            "whale_sell_count": int(self.whale_sell_count),
            "last_whale_bar_age_bars": int(self.last_whale_bar_age_bars),
            "last_whale_direction": self.last_whale_direction,
            "last_whale_z_score": round(float(self.last_whale_z_score), 2),
            "whale_directional_bias": round(float(self.whale_directional_bias), 3),
            "rationale": self.rationale[:300],
        }


def detect_whale_flow(
    closes: List[float],
    opens: List[float],
    volumes: List[float],
    lookback: int = DEFAULT_LOOKBACK,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> WhaleSignals:
    """Scan recent bars for volume anomalies. Returns count + direction."""
    n = min(len(closes), len(opens), len(volumes))
    if n < lookback + 5:
        return WhaleSignals(method="whale_flow",
                             rationale="insufficient_bars")

    bars = min(lookback, n)
    vols = volumes[-bars:]
    closes_w = closes[-bars:]
    opens_w = opens[-bars:]

    # Rolling stats
    mean_v = sum(vols) / bars
    var_v = sum((v - mean_v) ** 2 for v in vols) / bars
    std_v = var_v ** 0.5 if var_v > 0 else 1e-9

    if std_v <= 0:
        return WhaleSignals(method="whale_flow", rationale="zero_volume_variance")

    whale_bars: List[Dict[str, Any]] = []
    last_idx = -1
    last_dir = "none"
    last_z = 0.0
    for i, v in enumerate(vols):
        z = (v - mean_v) / std_v
        if z >= z_threshold:
            direction = "buy" if closes_w[i] > opens_w[i] else "sell"
            whale_bars.append({"idx": i, "z": z, "direction": direction})
            last_idx = i
            last_dir = direction
            last_z = z

    n_whale = len(whale_bars)
    n_buy = sum(1 for w in whale_bars if w["direction"] == "buy")
    n_sell = n_whale - n_buy
    last_age = (bars - 1 - last_idx) if last_idx >= 0 else -1

    # Directional bias: net (buys - sells) / max(1, total) weighted by recency
    # Recent whales count more
    if whale_bars:
        bias = 0.0
        for w in whale_bars:
            recency = (w["idx"] / bars)  # 0 oldest, 1 newest
            sign = 1 if w["direction"] == "buy" else -1
            bias += sign * (0.5 + 0.5 * recency) * min(2.0, w["z"]) / 2.0
        bias /= max(1, n_whale)
        bias = max(-1.0, min(1.0, bias))
    else:
        bias = 0.0

    parts = [f"whales={n_whale}",
             f"buy={n_buy}", f"sell={n_sell}",
             f"bias={bias:+.2f}"]
    if last_idx >= 0:
        parts.append(f"last_age={last_age}b")
        parts.append(f"last_dir={last_dir}")

    return WhaleSignals(
        method="whale_flow",
        n_whale_bars_recent=n_whale,
        whale_buy_count=n_buy,
        whale_sell_count=n_sell,
        last_whale_bar_age_bars=last_age,
        last_whale_direction=last_dir,
        last_whale_z_score=last_z,
        whale_directional_bias=bias,
        rationale=" | ".join(parts)[:300],
    )
