"""
Probability calibration + data-driven score-delta mapping for LightGBM binary models.

The binary SKIP/TRADE model trained with is_unbalance=True plus a 1.5x TRADE sample
weight produces probabilities that are systematically overconfident at the extremes.
The executor converts those probabilities into entry_score deltas with hand-picked
thresholds (+2 at >0.60, +1 at >0.45, etc). That hand mapping is the wrong shape
for an uncalibrated model.

This module provides:
  1. Isotonic calibration fitted on a held-out slice
  2. Data-driven entry_score deltas derived from each calibrated-probability bucket's
     actual win rate on the same slice
  3. A {"abstain" zone} around 0.5 so indifferent predictions add no noise

Persisted alongside the model as `models/lgbm_{asset}_calibration.json`. The executor
loads it at startup and falls back to the hand-tuned deltas if the file is missing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# Default bucket boundaries over [0, 1] on *calibrated* probability.
# The middle bucket [0.45, 0.55] is the abstain zone — delta forced to 0 regardless
# of observed win rate in that slice, because a coin-flip signal amplifies noise.
DEFAULT_BUCKETS: Tuple[Tuple[float, float], ...] = (
    (0.00, 0.30),
    (0.30, 0.45),
    (0.45, 0.55),
    (0.55, 0.70),
    (0.70, 1.01),  # 1.01 so prob==1.0 lands in the top bucket
)

# Fallback deltas used when no calibration file is on disk. Matches the historical
# hand-tuned values at executor.py:4832-4850 so we degrade gracefully.
DEFAULT_FALLBACK_DELTAS: Tuple[int, ...] = (-3, -1, 0, 1, 2)

# Hard bounds on deltas so a miscalibrated bucket can't contribute a wild multiplier.
MIN_DELTA, MAX_DELTA = -3, 2

# Minimum samples per bucket before we trust its observed win rate.
MIN_SAMPLES_PER_BUCKET = 30


@dataclass
class CalibrationBundle:
    """Isotonic calibrator + score-delta lookup, round-trippable to JSON."""

    isotonic_x: List[float] = field(default_factory=list)
    isotonic_y: List[float] = field(default_factory=list)
    buckets: List[List[float]] = field(default_factory=list)
    deltas: List[int] = field(default_factory=list)
    fit_timestamp: str = ""
    fit_n_samples: int = 0
    baseline_win_rate: float = 0.5
    asset: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, blob: str) -> "CalibrationBundle":
        d = json.loads(blob)
        return cls(**d)

    def is_usable(self) -> bool:
        """True if the bundle has a working isotonic curve and bucket/delta lists."""
        return (
            len(self.isotonic_x) >= 2
            and len(self.isotonic_x) == len(self.isotonic_y)
            and len(self.buckets) == len(self.deltas)
            and len(self.buckets) > 0
        )

    def apply(self, raw_prob: float) -> float:
        """Monotone piecewise-linear interpolation between stored (x, y) knots."""
        xs = self.isotonic_x
        ys = self.isotonic_y
        if not xs:
            return float(raw_prob)
        p = max(0.0, min(1.0, float(raw_prob)))
        return float(np.interp(p, xs, ys))

    def delta_for(self, calibrated_prob: float) -> int:
        """Return the integer entry_score delta for a given calibrated probability."""
        if not self.buckets:
            return 0
        for (lo, hi), d in zip(self.buckets, self.deltas):
            if lo <= calibrated_prob < hi:
                return int(d)
        return int(self.deltas[-1])


def _reduce_isotonic(
    iso: "IsotonicRegression", x_ref: np.ndarray, max_knots: int = 64
) -> Tuple[List[float], List[float]]:
    """Downsample the isotonic curve to a compact (x, y) knot list for JSON storage."""
    xs = np.linspace(0.0, 1.0, max_knots)
    # Blend explicit grid with observed x range so we stay on the fitted manifold.
    if len(x_ref) > 0:
        xs = np.unique(np.concatenate([xs, np.quantile(x_ref, np.linspace(0, 1, 16))]))
    xs = np.clip(xs, 0.0, 1.0)
    ys = iso.predict(xs)
    # Enforce monotone non-decreasing (IsotonicRegression guarantees it, but rounding
    # to float64->JSON->float64 can tick down by 1 ulp).
    ys = np.maximum.accumulate(ys)
    return xs.tolist(), ys.tolist()


def fit_calibration(
    raw_probs: Sequence[float],
    y_true: Sequence[int],
    *,
    asset: str = "",
    buckets: Sequence[Tuple[float, float]] = DEFAULT_BUCKETS,
    baseline_win_rate: Optional[float] = None,
) -> Optional[CalibrationBundle]:
    """
    Fit isotonic calibration on held-out (raw_prob, y_true) pairs and derive data-driven
    score deltas from each bucket's observed win rate.

    Returns None if sklearn is unavailable, or the slice is degenerate (< 50 samples,
    or only one class present — isotonic can't fit those).

    Scaling: delta ≈ round((bucket_win_rate − baseline_win_rate) × 10), clipped to
    [MIN_DELTA, MAX_DELTA]. The abstain bucket (center straddling 0.5) is forced to 0.
    """
    if not HAS_SKLEARN:
        return None

    p = np.asarray(raw_probs, dtype=float)
    y = np.asarray(y_true, dtype=int)

    if len(p) < 50:
        return None
    if len(np.unique(y)) < 2:
        return None

    # Isotonic fit.
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p, y)
    iso_x, iso_y = _reduce_isotonic(iso, p)

    # Bucket deltas on calibrated probabilities.
    calibrated = iso.predict(p)
    base = float(baseline_win_rate) if baseline_win_rate is not None else float(np.mean(y))

    delta_list: List[int] = []
    bucket_list: List[List[float]] = []
    for lo, hi in buckets:
        mask = (calibrated >= lo) & (calibrated < hi)
        abstain = lo <= 0.5 < hi  # any bucket that straddles 0.5 is the abstain zone
        if abstain:
            d = 0
        elif mask.sum() < MIN_SAMPLES_PER_BUCKET:
            # Too few samples to trust — fall back to the default delta at this position.
            idx = len(delta_list)
            d = int(DEFAULT_FALLBACK_DELTAS[idx]) if idx < len(DEFAULT_FALLBACK_DELTAS) else 0
        else:
            bucket_win = float(np.mean(y[mask]))
            raw_d = int(round((bucket_win - base) * 10))
            d = max(MIN_DELTA, min(MAX_DELTA, raw_d))
        delta_list.append(d)
        bucket_list.append([float(lo), float(hi)])

    return CalibrationBundle(
        isotonic_x=iso_x,
        isotonic_y=iso_y,
        buckets=bucket_list,
        deltas=delta_list,
        fit_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        fit_n_samples=int(len(p)),
        baseline_win_rate=base,
        asset=asset,
    )


def load_calibration(path: str) -> Optional[CalibrationBundle]:
    """Load a calibration bundle from disk. Returns None if missing or malformed."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            bundle = CalibrationBundle.from_json(f.read())
        if not bundle.is_usable():
            return None
        return bundle
    except Exception:
        return None


def apply_calibration(bundle: Optional[CalibrationBundle], raw_prob: float) -> float:
    """Return calibrated probability, or raw_prob if no bundle is available."""
    if bundle is None or not bundle.is_usable():
        return float(raw_prob)
    return bundle.apply(raw_prob)


def score_delta_for(
    bundle: Optional[CalibrationBundle],
    raw_prob: float,
    *,
    fallback_deltas: Sequence[int] = DEFAULT_FALLBACK_DELTAS,
    fallback_buckets: Sequence[Tuple[float, float]] = DEFAULT_BUCKETS,
) -> Tuple[int, float]:
    """
    Return (entry_score_delta, calibrated_probability).

    If a bundle is present, uses calibrated_prob and bundle-derived deltas. Otherwise
    maps raw_prob through fallback_buckets/fallback_deltas.
    """
    if bundle is not None and bundle.is_usable():
        cp = bundle.apply(raw_prob)
        return bundle.delta_for(cp), cp

    cp = float(raw_prob)
    for (lo, hi), d in zip(fallback_buckets, fallback_deltas):
        if lo <= cp < hi:
            return int(d), cp
    return int(fallback_deltas[-1]), cp


def calibration_path_for(models_dir: str, asset: str) -> str:
    """Canonical path where the bundle for a given asset lives."""
    return os.path.join(models_dir, f"lgbm_{asset.lower()}_calibration.json")


def save_calibration(bundle: CalibrationBundle, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(bundle.to_json())
