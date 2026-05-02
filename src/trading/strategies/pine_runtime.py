"""Pine v5 runtime helpers — Python implementations of Pine's `ta.*` library.

These are the functions that auto-imported Pine strategies call. Each
maps a Pine `ta.X(...)` call to a Python function that takes lists of
floats and returns lists of floats (or scalars for crossover/crossunder).

All functions:
  * Accept lists (current bar is the LAST element)
  * Return either a list of the same length OR a scalar
  * Handle insufficient data gracefully (return NaN-padded list)
  * Have no external dependencies beyond numpy (which is already used)
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def _hist(series: List[Any], offset: int = 0) -> Any:
    """Pine `series[offset]` → return value `offset` bars ago.

    series[0] = current, series[1] = previous, etc.
    """
    if not series or offset >= len(series):
        return float('nan')
    idx = len(series) - 1 - offset
    if idx < 0:
        return float('nan')
    return series[idx]


def _safe_last(series: List[float]) -> float:
    return _hist(series, 0)


# ── Moving averages ──────────────────────────────────────────────

def _pine_sma(series: List[float], length: int) -> List[float]:
    """Simple moving average. Returns same-length list with NaN padding."""
    if length <= 0 or not series:
        return [float('nan')] * len(series)
    out = [float('nan')] * len(series)
    for i in range(length - 1, len(series)):
        window = [x for x in series[i - length + 1: i + 1] if not _is_nan(x)]
        if len(window) == length:
            out[i] = sum(window) / length
    return out


def _pine_ema(series: List[float], length: int) -> List[float]:
    """Exponential moving average."""
    if length <= 0 or not series:
        return [float('nan')] * len(series)
    out = [float('nan')] * len(series)
    alpha = 2.0 / (length + 1)
    # Seed with SMA of first `length` values
    if len(series) < length:
        return out
    seed = sum(series[:length]) / length
    out[length - 1] = seed
    prev = seed
    for i in range(length, len(series)):
        if _is_nan(series[i]):
            out[i] = prev
            continue
        prev = alpha * series[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def _pine_wma(series: List[float], length: int) -> List[float]:
    """Weighted moving average."""
    if length <= 0 or not series:
        return [float('nan')] * len(series)
    out = [float('nan')] * len(series)
    weights = list(range(1, length + 1))
    weight_sum = sum(weights)
    for i in range(length - 1, len(series)):
        window = series[i - length + 1: i + 1]
        if any(_is_nan(x) for x in window):
            continue
        weighted = sum(w * x for w, x in zip(weights, window))
        out[i] = weighted / weight_sum
    return out


# ── RSI ──────────────────────────────────────────────────────────

def _pine_rsi(series: List[float], length: int = 14) -> List[float]:
    """Wilder RSI."""
    if length <= 0 or len(series) < length + 1:
        return [float('nan')] * len(series)
    out = [float('nan')] * len(series)
    gains = [0.0] * len(series)
    losses = [0.0] * len(series)
    for i in range(1, len(series)):
        delta = series[i] - series[i - 1]
        gains[i] = max(delta, 0.0)
        losses[i] = -min(delta, 0.0)
    avg_gain = sum(gains[1:length + 1]) / length
    avg_loss = sum(losses[1:length + 1]) / length
    out[length] = 100 - 100 / (1 + (avg_gain / max(avg_loss, 1e-9)))
    for i in range(length + 1, len(series)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        rs = avg_gain / max(avg_loss, 1e-9)
        out[i] = 100 - 100 / (1 + rs)
    return out


# ── ATR ──────────────────────────────────────────────────────────

def _pine_tr(highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
    """True Range."""
    out = [float('nan')] * len(closes)
    if not closes:
        return out
    out[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        out[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    return out


def _pine_atr(highs: List[float], lows: List[float], closes: List[float], length: int = 14) -> List[float]:
    tr = _pine_tr(highs, lows, closes)
    return _pine_ema(tr, length)


# ── MACD ─────────────────────────────────────────────────────────

def _pine_macd(
    series: List[float],
    fast_length: int = 12,
    slow_length: int = 26,
    signal_length: int = 9,
) -> Tuple[List[float], List[float], List[float]]:
    """Returns (macd_line, signal_line, histogram)."""
    fast = _pine_ema(series, fast_length)
    slow = _pine_ema(series, slow_length)
    macd_line = [
        (f - s) if not (_is_nan(f) or _is_nan(s)) else float('nan')
        for f, s in zip(fast, slow)
    ]
    signal_line = _pine_ema(macd_line, signal_length)
    histogram = [
        (m - s) if not (_is_nan(m) or _is_nan(s)) else float('nan')
        for m, s in zip(macd_line, signal_line)
    ]
    return macd_line, signal_line, histogram


# ── Stochastic ───────────────────────────────────────────────────

def _pine_stoch(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    length: int = 14,
) -> List[float]:
    """Stochastic %K."""
    out = [float('nan')] * len(closes)
    for i in range(length - 1, len(closes)):
        window_h = max(highs[i - length + 1: i + 1])
        window_l = min(lows[i - length + 1: i + 1])
        if window_h - window_l <= 0:
            continue
        out[i] = 100 * (closes[i] - window_l) / (window_h - window_l)
    return out


# ── Bollinger Bands ──────────────────────────────────────────────

def _pine_bb(
    series: List[float],
    length: int = 20,
    mult: float = 2.0,
) -> Tuple[List[float], List[float], List[float]]:
    """Returns (lower, basis_sma, upper)."""
    basis = _pine_sma(series, length)
    upper = [float('nan')] * len(series)
    lower = [float('nan')] * len(series)
    for i in range(length - 1, len(series)):
        window = series[i - length + 1: i + 1]
        if any(_is_nan(x) for x in window):
            continue
        mean = sum(window) / length
        std = math.sqrt(sum((x - mean) ** 2 for x in window) / length)
        upper[i] = basis[i] + mult * std
        lower[i] = basis[i] - mult * std
    return lower, basis, upper


# ── VWAP ─────────────────────────────────────────────────────────

def _pine_vwap(
    closes: List[float],
    volumes: List[float],
) -> List[float]:
    """Cumulative VWAP (Pine session-VWAP equivalent)."""
    out = [float('nan')] * len(closes)
    cum_vol = 0.0
    cum_pv = 0.0
    for i in range(len(closes)):
        v = volumes[i] if i < len(volumes) else 0.0
        cum_vol += v
        cum_pv += closes[i] * v
        if cum_vol > 0:
            out[i] = cum_pv / cum_vol
    return out


# ── OBV ──────────────────────────────────────────────────────────

def _pine_obv(closes: List[float], volumes: List[float]) -> List[float]:
    out = [0.0] * len(closes)
    for i in range(1, len(closes)):
        v = volumes[i] if i < len(volumes) else 0.0
        if closes[i] > closes[i - 1]:
            out[i] = out[i - 1] + v
        elif closes[i] < closes[i - 1]:
            out[i] = out[i - 1] - v
        else:
            out[i] = out[i - 1]
    return out


# ── ADX / DMI ────────────────────────────────────────────────────

def _pine_adx(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    length: int = 14,
) -> List[float]:
    """Wilder ADX."""
    out = [float('nan')] * len(closes)
    if len(closes) < length * 2:
        return out
    tr = _pine_tr(highs, lows, closes)
    plus_dm = [0.0] * len(closes)
    minus_dm = [0.0] * len(closes)
    for i in range(1, len(closes)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    atr = _pine_ema(tr, length)
    plus_di = [
        100 * pdm / atr[i] if atr[i] and atr[i] > 0 else 0.0
        for i, pdm in enumerate(_pine_ema(plus_dm, length))
    ]
    minus_di = [
        100 * mdm / atr[i] if atr[i] and atr[i] > 0 else 0.0
        for i, mdm in enumerate(_pine_ema(minus_dm, length))
    ]
    dx = [
        100 * abs(p - m) / max(p + m, 1e-9)
        for p, m in zip(plus_di, minus_di)
    ]
    return _pine_ema(dx, length)


# ── CCI ──────────────────────────────────────────────────────────

def _pine_cci(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    length: int = 20,
) -> List[float]:
    out = [float('nan')] * len(closes)
    typ = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
    sma = _pine_sma(typ, length)
    for i in range(length - 1, len(closes)):
        window = typ[i - length + 1: i + 1]
        mean = sma[i]
        if _is_nan(mean):
            continue
        mad = sum(abs(x - mean) for x in window) / length
        if mad <= 0:
            continue
        out[i] = (typ[i] - mean) / (0.015 * mad)
    return out


# ── MFI ──────────────────────────────────────────────────────────

def _pine_mfi(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    length: int = 14,
) -> List[float]:
    out = [float('nan')] * len(closes)
    if len(closes) < length + 1:
        return out
    typ = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
    raw_mf = [t * v for t, v in zip(typ, volumes)]
    for i in range(length, len(closes)):
        pos_flow = 0.0
        neg_flow = 0.0
        for j in range(i - length + 1, i + 1):
            if j == 0:
                continue
            if typ[j] > typ[j - 1]:
                pos_flow += raw_mf[j]
            elif typ[j] < typ[j - 1]:
                neg_flow += raw_mf[j]
        if neg_flow <= 0:
            out[i] = 100.0
        else:
            ratio = pos_flow / neg_flow
            out[i] = 100 - 100 / (1 + ratio)
    return out


# ── Crossover / Crossunder (scalar at current bar) ───────────────

def _pine_crossover(a: Any, b: Any) -> bool:
    """Returns True if `a` crossed above `b` on the most recent bar."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return False  # need history
    a_curr = _hist(a, 0) if isinstance(a, list) else a
    a_prev = _hist(a, 1) if isinstance(a, list) else a
    b_curr = _hist(b, 0) if isinstance(b, list) else b
    b_prev = _hist(b, 1) if isinstance(b, list) else b
    if any(_is_nan(x) for x in (a_curr, a_prev, b_curr, b_prev)):
        return False
    return a_prev <= b_prev and a_curr > b_curr


def _pine_crossunder(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return False
    a_curr = _hist(a, 0) if isinstance(a, list) else a
    a_prev = _hist(a, 1) if isinstance(a, list) else a
    b_curr = _hist(b, 0) if isinstance(b, list) else b
    b_prev = _hist(b, 1) if isinstance(b, list) else b
    if any(_is_nan(x) for x in (a_curr, a_prev, b_curr, b_prev)):
        return False
    return a_prev >= b_prev and a_curr < b_curr


# ── highest / lowest / change / roc ──────────────────────────────

def _pine_highest(series: List[float], length: int) -> List[float]:
    out = [float('nan')] * len(series)
    for i in range(length - 1, len(series)):
        out[i] = max(series[i - length + 1: i + 1])
    return out


def _pine_lowest(series: List[float], length: int) -> List[float]:
    out = [float('nan')] * len(series)
    for i in range(length - 1, len(series)):
        out[i] = min(series[i - length + 1: i + 1])
    return out


def _pine_change(series: List[float], length: int = 1) -> List[float]:
    out = [float('nan')] * len(series)
    for i in range(length, len(series)):
        out[i] = series[i] - series[i - length]
    return out


def _pine_roc(series: List[float], length: int) -> List[float]:
    """Rate of change (percentage)."""
    out = [float('nan')] * len(series)
    for i in range(length, len(series)):
        if series[i - length] != 0:
            out[i] = 100 * (series[i] - series[i - length]) / series[i - length]
    return out


# ── Helpers ──────────────────────────────────────────────────────

def _is_nan(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return False
