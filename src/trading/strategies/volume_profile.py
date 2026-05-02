"""Volume Profile (VPVR) — point of control + value area + HVN/LVN.

Volume Profile distributes traded volume across price levels (bins)
rather than time. Key levels:

    POC   — Point of Control: price level with highest traded volume
            (the most-accepted "fair value" by recent traders)
    VAH   — Value Area High: top of the 70% volume zone
    VAL   — Value Area Low: bottom of the 70% volume zone
    HVN   — High Volume Node: any local volume peak (acts as magnet)
    LVN   — Low Volume Node: any local volume trough (acts as fast-traverse zone)

Trading interpretation:
    Price ABOVE VAH + extending = breakout in progress (LONG bias)
    Price BELOW VAL + extending = breakdown in progress (SHORT bias)
    Price reverting from edge → POC = mean-reversion trade
    Price entering LVN = fast move expected (no resistance)
    Price entering HVN = consolidation expected (acts as magnet)

Anti-overfit:
    * Bin count auto-computed via Sturges rule based on bar count
    * 70% value-area threshold is the standard (no learning)
    * Volume must be present and non-zero — returns 'unclear' otherwise
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class VolumeProfile:
    poc_price: float
    vah_price: float                  # value area high
    val_price: float                  # value area low
    bins: List[Tuple[float, float, float]]  # (price_low, price_high, volume)
    hvn_levels: List[float] = field(default_factory=list)
    lvn_levels: List[float] = field(default_factory=list)
    direction: str = "FLAT"           # LONG / SHORT / FLAT inference
    confidence: float = 0.0
    factors: List[str] = field(default_factory=list)
    rationale: str = ""


def _sturges_bins(n: int) -> int:
    """Sturges' rule for histogram bin count: ceil(1 + log2(n))."""
    if n < 2:
        return 1
    return max(8, min(60, int(math.ceil(1 + math.log2(n)))))


def _build_profile(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    n_bins: Optional[int] = None,
) -> List[Tuple[float, float, float]]:
    """Build volume-by-price histogram. Each bar's volume distributed
    uniformly across the bins its high-low range overlaps."""
    n = len(closes)
    if n < 5:
        return []
    if n_bins is None:
        n_bins = _sturges_bins(n)
    p_min = min(lows)
    p_max = max(highs)
    if p_max - p_min <= 0:
        return []
    bin_width = (p_max - p_min) / n_bins
    bins: List[Tuple[float, float, float]] = [
        (p_min + i * bin_width, p_min + (i + 1) * bin_width, 0.0)
        for i in range(n_bins)
    ]
    vol_acc = [0.0] * n_bins
    for i in range(n):
        h, l = highs[i], lows[i]
        v = volumes[i] if i < len(volumes) else 0.0
        if v <= 0:
            continue
        # Distribute v uniformly across bins overlapping [l, h]
        lo_bin = max(0, int((l - p_min) / bin_width))
        hi_bin = min(n_bins - 1, int((h - p_min) / bin_width))
        n_overlap = hi_bin - lo_bin + 1
        if n_overlap <= 0:
            continue
        share = v / n_overlap
        for b in range(lo_bin, hi_bin + 1):
            vol_acc[b] += share
    return [(bins[i][0], bins[i][1], vol_acc[i]) for i in range(n_bins)]


def _compute_value_area(bins: List[Tuple[float, float, float]], pct: float = 0.70) -> Tuple[int, int]:
    """Find indices of bins forming the central pct (default 70%) of total volume.
    Standard algorithm: start from POC, expand outward greedily."""
    if not bins:
        return (0, 0)
    total_vol = sum(b[2] for b in bins)
    if total_vol <= 0:
        return (0, 0)
    target = total_vol * pct
    poc_idx = max(range(len(bins)), key=lambda i: bins[i][2])
    accumulated = bins[poc_idx][2]
    lo = poc_idx
    hi = poc_idx
    while accumulated < target and (lo > 0 or hi < len(bins) - 1):
        # Compare two-bin sums on each side (standard CME-style algorithm)
        left_sum = bins[lo - 1][2] + (bins[lo - 2][2] if lo - 2 >= 0 else 0.0) if lo > 0 else -1
        right_sum = bins[hi + 1][2] + (bins[hi + 2][2] if hi + 2 < len(bins) else 0.0) if hi < len(bins) - 1 else -1
        if right_sum > left_sum and hi < len(bins) - 1:
            hi += 1
            accumulated += bins[hi][2]
        elif lo > 0:
            lo -= 1
            accumulated += bins[lo][2]
        else:
            break
    return (lo, hi)


def _find_hvn_lvn(
    bins: List[Tuple[float, float, float]],
    n_top: int = 3,
) -> Tuple[List[float], List[float]]:
    """Find local volume peaks (HVN) and troughs (LVN)."""
    if len(bins) < 5:
        return [], []
    vols = [b[2] for b in bins]
    centers = [(b[0] + b[1]) / 2 for b in bins]
    hvn: List[Tuple[float, float]] = []
    lvn: List[Tuple[float, float]] = []
    for i in range(1, len(bins) - 1):
        if vols[i] > vols[i - 1] and vols[i] > vols[i + 1]:
            hvn.append((centers[i], vols[i]))
        if vols[i] < vols[i - 1] and vols[i] < vols[i + 1]:
            lvn.append((centers[i], vols[i]))
    hvn.sort(key=lambda t: -t[1])
    lvn.sort(key=lambda t: t[1])
    return [p for p, _ in hvn[:n_top]], [p for p, _ in lvn[:n_top]]


def evaluate(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    value_area_pct: float = 0.70,
    n_bins: Optional[int] = None,
) -> VolumeProfile:
    """Build VPVR + infer directional bias from current price relative
    to POC / VAH / VAL."""
    if len(closes) < 10:
        return VolumeProfile(
            poc_price=float('nan'), vah_price=float('nan'), val_price=float('nan'),
            bins=[], direction="FLAT", confidence=0.0,
            rationale="insufficient bars (<10)",
        )
    if not volumes or sum(volumes) <= 0:
        return VolumeProfile(
            poc_price=float('nan'), vah_price=float('nan'), val_price=float('nan'),
            bins=[], direction="FLAT", confidence=0.0,
            rationale="no volume data",
        )

    bins = _build_profile(highs, lows, closes, volumes, n_bins)
    if not bins:
        return VolumeProfile(
            poc_price=float('nan'), vah_price=float('nan'), val_price=float('nan'),
            bins=[], direction="FLAT", confidence=0.0,
            rationale="bin construction failed",
        )

    val_idx, vah_idx = _compute_value_area(bins, value_area_pct)
    poc_idx = max(range(len(bins)), key=lambda i: bins[i][2])
    poc_price = (bins[poc_idx][0] + bins[poc_idx][1]) / 2.0
    val_price = bins[val_idx][0]
    vah_price = bins[vah_idx][1]

    hvn_levels, lvn_levels = _find_hvn_lvn(bins)

    current = closes[-1]
    factors: List[str] = []
    direction = "FLAT"
    confidence = 0.5

    if current > vah_price:
        direction = "LONG"
        factors.append(f"price_above_VAH ({current:.2f} > {vah_price:.2f})")
        confidence = 0.65
    elif current < val_price:
        direction = "SHORT"
        factors.append(f"price_below_VAL ({current:.2f} < {val_price:.2f})")
        confidence = 0.65
    elif abs(current - poc_price) / max(poc_price, 1e-9) < 0.005:
        direction = "FLAT"
        factors.append(f"price_at_POC ({current:.2f} ≈ {poc_price:.2f}) — consolidation")
        confidence = 0.4
    else:
        # In value area but not at POC — slight mean-reversion bias toward POC
        if current > poc_price:
            direction = "SHORT"
            factors.append(f"in_VA_above_POC — mean-revert SHORT toward {poc_price:.2f}")
            confidence = 0.35
        else:
            direction = "LONG"
            factors.append(f"in_VA_below_POC — mean-revert LONG toward {poc_price:.2f}")
            confidence = 0.35

    # Boost if near an LVN (fast-traversal expected)
    near_lvn = any(abs(current - lvn) / max(current, 1e-9) < 0.01 for lvn in lvn_levels)
    if near_lvn and direction != "FLAT":
        confidence = min(1.0, confidence + 0.15)
        factors.append("near_LVN_fast_move_expected")

    rationale = (
        f"VPVR: POC={poc_price:.2f} VAH={vah_price:.2f} VAL={val_price:.2f} "
        f"current={current:.2f} → {direction} (conf={confidence:.2f}); "
        f"{len(hvn_levels)} HVN, {len(lvn_levels)} LVN"
    )
    return VolumeProfile(
        poc_price=float(poc_price),
        vah_price=float(vah_price),
        val_price=float(val_price),
        bins=bins,
        hvn_levels=hvn_levels,
        lvn_levels=lvn_levels,
        direction=direction,
        confidence=confidence,
        factors=factors,
        rationale=rationale,
    )


def evaluate_dict(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
) -> Dict[str, Any]:
    vp = evaluate(highs, lows, closes, volumes)
    return {
        "strategy": "volume_profile",
        "direction": vp.direction,
        "confidence": round(vp.confidence, 3),
        "poc": round(vp.poc_price, 4) if not math.isnan(vp.poc_price) else None,
        "vah": round(vp.vah_price, 4) if not math.isnan(vp.vah_price) else None,
        "val": round(vp.val_price, 4) if not math.isnan(vp.val_price) else None,
        "hvn_levels": [round(x, 4) for x in vp.hvn_levels],
        "lvn_levels": [round(x, 4) for x in vp.lvn_levels],
        "factors": vp.factors[:4],
        "rationale": vp.rationale[:200],
    }
