"""
Strategy Universe Generator -- Hundreds of Strategies from Combinations
========================================================================
Instead of manually coding strategies, this engine GENERATES them
programmatically from combinations of indicators and rules.

Architecture:
  20 indicators x 3 param sets x 5 entry rules x 3 exit rules = 900 combos
  Filter by: backtested positive PnL after 1.69% spread = ~50-100 viable

This is how quantitative hedge funds work -- they test thousands of
strategy hypotheses and deploy only the survivors.

Categories:
  1. SINGLE INDICATOR strategies (RSI oversold, MACD cross, BB touch, etc.)
  2. DUAL INDICATOR combos (RSI + MACD, EMA + Volume, BB + RSI, etc.)
  3. TRIPLE INDICATOR combos (RSI + MACD + Volume, EMA + ADX + ATR, etc.)
  4. MATHEMATICAL strategies (z-score, regression, statistical arb)
  5. PATTERN strategies (candle patterns + confirmation)
  6. VOLATILITY strategies (squeeze, expansion, regime)
  7. MOMENTUM strategies (ROC, Williams %R, MFI combos)
  8. VOLUME strategies (OBV divergence, volume climax, CMF)
"""

import logging
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from src.indicators.indicators import (
    sma, ema, rsi, macd, bollinger_bands, atr, stochastic,
    vwap, obv, adx, bb_width, roc, williams_r,
    chaikin_money_flow, mfi, supertrend, volume_delta, choppiness_index
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers -- safe value extraction
# ---------------------------------------------------------------------------

def _safe_last(series: List[float], offset: int = 0) -> float:
    """Return series[-1-offset], or NaN if out of bounds or NaN."""
    idx = len(series) - 1 - offset
    if idx < 0:
        return float('nan')
    v = series[idx]
    if isinstance(v, float) and math.isnan(v):
        return float('nan')
    return float(v)


def _safe_last_n(series: List[float], n: int) -> List[float]:
    """Return last n values, padding with NaN if needed."""
    if len(series) >= n:
        return [float(x) for x in series[-n:]]
    pad = [float('nan')] * (n - len(series))
    return pad + [float(x) for x in series]


def _is_valid(*values) -> bool:
    """Check none of the values are NaN."""
    for v in values:
        if isinstance(v, float) and math.isnan(v):
            return False
        try:
            if np.isnan(v):
                return False
        except (TypeError, ValueError):
            pass
    return True


# ---------------------------------------------------------------------------
# IndicatorSignal dataclass
# ---------------------------------------------------------------------------

@dataclass
class IndicatorSignal:
    """Wraps any indicator into a buy/sell signal generator.

    Attributes:
        name:     unique strategy identifier
        compute:  function(closes, highs, lows, volumes) -> int signal (-1/0/+1)
        category: one of trend, momentum, volatility, volume, math, combo, pattern
        params:   parameter dict for reproducibility
    """
    name: str
    compute: Callable  # (closes, highs, lows, volumes) -> int
    category: str
    params: dict = field(default_factory=dict)


# ===================================================================
# SECTION 1: Single-indicator signal helpers
# ===================================================================

def _rsi_signal(closes, oversold, overbought, period=14):
    if len(closes) < period + 2:
        return 0
    vals = rsi(closes, period)
    cur = _safe_last(vals)
    prev = _safe_last(vals, 1)
    if not _is_valid(cur, prev):
        return 0
    if prev <= oversold and cur > oversold:
        return 1
    if prev >= overbought and cur < overbought:
        return -1
    if cur < oversold:
        return 1
    if cur > overbought:
        return -1
    return 0


def _ema_cross_signal(closes, fast_period, slow_period):
    if len(closes) < slow_period + 2:
        return 0
    fast = ema(closes, fast_period)
    slow = ema(closes, slow_period)
    cur_f, prev_f = _safe_last(fast), _safe_last(fast, 1)
    cur_s, prev_s = _safe_last(slow), _safe_last(slow, 1)
    if not _is_valid(cur_f, prev_f, cur_s, prev_s):
        return 0
    if prev_f <= prev_s and cur_f > cur_s:
        return 1
    if prev_f >= prev_s and cur_f < cur_s:
        return -1
    return 0


def _sma_cross_signal(closes, fast_period, slow_period):
    if len(closes) < slow_period + 2:
        return 0
    fast = sma(closes, fast_period)
    slow = sma(closes, slow_period)
    cur_f, prev_f = _safe_last(fast), _safe_last(fast, 1)
    cur_s, prev_s = _safe_last(slow), _safe_last(slow, 1)
    if not _is_valid(cur_f, prev_f, cur_s, prev_s):
        return 0
    if prev_f <= prev_s and cur_f > cur_s:
        return 1
    if prev_f >= prev_s and cur_f < cur_s:
        return -1
    return 0


def _bb_signal(closes, period, num_std):
    if len(closes) < period + 2:
        return 0
    upper, mid, lower = bollinger_bands(closes, period, num_std)
    price = closes[-1]
    prev_price = closes[-2]
    lo = _safe_last(lower)
    hi = _safe_last(upper)
    prev_lo = _safe_last(lower, 1)
    prev_hi = _safe_last(upper, 1)
    if not _is_valid(lo, hi, prev_lo, prev_hi):
        return 0
    # Touch lower band and bounce
    if prev_price <= prev_lo and price > lo:
        return 1
    # Touch upper band and drop
    if prev_price >= prev_hi and price < hi:
        return -1
    # Below lower band = oversold
    if price < lo:
        return 1
    # Above upper band = overbought
    if price > hi:
        return -1
    return 0


def _macd_signal(closes, fast, slow, signal_period):
    if len(closes) < slow + signal_period + 2:
        return 0
    macd_line, sig_line, hist = macd(closes, fast, slow, signal_period)
    cur_h, prev_h = _safe_last(hist), _safe_last(hist, 1)
    if not _is_valid(cur_h, prev_h):
        return 0
    # Histogram cross zero
    if prev_h <= 0 and cur_h > 0:
        return 1
    if prev_h >= 0 and cur_h < 0:
        return -1
    return 0


def _macd_divergence_signal(closes, fast, slow, signal_period):
    """MACD line vs signal line crossover."""
    if len(closes) < slow + signal_period + 2:
        return 0
    macd_line, sig_line, _ = macd(closes, fast, slow, signal_period)
    cur_m, prev_m = _safe_last(macd_line), _safe_last(macd_line, 1)
    cur_s, prev_s = _safe_last(sig_line), _safe_last(sig_line, 1)
    if not _is_valid(cur_m, prev_m, cur_s, prev_s):
        return 0
    if prev_m <= prev_s and cur_m > cur_s:
        return 1
    if prev_m >= prev_s and cur_m < cur_s:
        return -1
    return 0


def _stochastic_signal(highs, lows, closes, k_period, d_period, oversold, overbought):
    if len(closes) < k_period + d_period + 2:
        return 0
    k_vals, d_vals = stochastic(highs, lows, closes, k_period, d_period)
    cur_k, prev_k = _safe_last(k_vals), _safe_last(k_vals, 1)
    cur_d, prev_d = _safe_last(d_vals), _safe_last(d_vals, 1)
    if not _is_valid(cur_k, prev_k, cur_d, prev_d):
        return 0
    # %K crosses above %D in oversold
    if cur_k < oversold and prev_k <= prev_d and cur_k > cur_d:
        return 1
    # %K crosses below %D in overbought
    if cur_k > overbought and prev_k >= prev_d and cur_k < cur_d:
        return -1
    # Simple threshold
    if cur_k < oversold:
        return 1
    if cur_k > overbought:
        return -1
    return 0


def _williams_r_signal(highs, lows, closes, period, oversold, overbought):
    if len(closes) < period + 2:
        return 0
    vals = williams_r(highs, lows, closes, period)
    cur = _safe_last(vals)
    prev = _safe_last(vals, 1)
    if not _is_valid(cur, prev):
        return 0
    # Williams %R: -100 = oversold, 0 = overbought
    if prev <= oversold and cur > oversold:
        return 1
    if prev >= overbought and cur < overbought:
        return -1
    return 0


def _roc_signal(closes, period, threshold):
    if len(closes) < period + 2:
        return 0
    vals = roc(closes, period)
    cur = _safe_last(vals)
    prev = _safe_last(vals, 1)
    if not _is_valid(cur, prev):
        return 0
    # ROC cross above threshold
    if prev <= threshold and cur > threshold:
        return 1
    if prev >= -threshold and cur < -threshold:
        return -1
    return 0


def _mfi_signal(highs, lows, closes, volumes, period, oversold, overbought):
    if len(closes) < period + 2:
        return 0
    vals = mfi(highs, lows, closes, volumes, period)
    cur = _safe_last(vals)
    prev = _safe_last(vals, 1)
    if not _is_valid(cur, prev):
        return 0
    if prev <= oversold and cur > oversold:
        return 1
    if prev >= overbought and cur < overbought:
        return -1
    if cur < oversold:
        return 1
    if cur > overbought:
        return -1
    return 0


def _adx_signal(highs, lows, closes, period, threshold):
    """ADX trend strength: trade when ADX > threshold and +DI/-DI cross."""
    if len(closes) < period * 2 + 2:
        return 0
    adx_line, plus_di, minus_di = adx(highs, lows, closes, period)
    cur_adx = _safe_last(adx_line)
    cur_pdi, prev_pdi = _safe_last(plus_di), _safe_last(plus_di, 1)
    cur_mdi, prev_mdi = _safe_last(minus_di), _safe_last(minus_di, 1)
    if not _is_valid(cur_adx, cur_pdi, prev_pdi, cur_mdi, prev_mdi):
        return 0
    if cur_adx < threshold:
        return 0  # No trend
    if prev_pdi <= prev_mdi and cur_pdi > cur_mdi:
        return 1
    if prev_pdi >= prev_mdi and cur_pdi < cur_mdi:
        return -1
    return 0


def _supertrend_signal(highs, lows, closes, period, multiplier):
    if len(closes) < period + 2:
        return 0
    st_line, side = supertrend(highs, lows, closes, period, multiplier)
    cur = side[-1] if side else 0
    prev = side[-2] if len(side) >= 2 else 0
    # Signal on flip
    if prev <= 0 and cur > 0:
        return 1
    if prev >= 0 and cur < 0:
        return -1
    return 0


def _choppiness_signal(highs, lows, closes, period, choppy_thresh, trend_thresh):
    """Choppiness index: <38.2 = trending (trade breakout), >61.8 = choppy (fade)."""
    if len(closes) < period + 2:
        return 0
    vals = choppiness_index(highs, lows, closes, period)
    cur = _safe_last(vals)
    if not _is_valid(cur):
        return 0
    # Trending market -- follow momentum
    if cur < trend_thresh:
        recent_change = closes[-1] - closes[-min(5, len(closes))]
        if recent_change > 0:
            return 1
        elif recent_change < 0:
            return -1
    return 0


def _obv_divergence_signal(closes, volumes, lookback):
    """OBV divergence: price up but OBV down = bearish, vice versa."""
    if len(closes) < lookback + 2:
        return 0
    obv_vals = obv(closes, volumes)
    price_slope = closes[-1] - closes[-lookback]
    obv_slope = obv_vals[-1] - obv_vals[-lookback]
    if price_slope > 0 and obv_slope < 0:
        return -1  # Bearish divergence
    if price_slope < 0 and obv_slope > 0:
        return 1   # Bullish divergence
    return 0


def _cmf_signal(highs, lows, closes, volumes, period, threshold):
    """Chaikin Money Flow: >threshold = buying pressure, <-threshold = selling."""
    if len(closes) < period + 2:
        return 0
    vals = chaikin_money_flow(highs, lows, closes, volumes, period)
    cur = _safe_last(vals)
    prev = _safe_last(vals, 1)
    if not _is_valid(cur, prev):
        return 0
    if prev <= threshold and cur > threshold:
        return 1
    if prev >= -threshold and cur < -threshold:
        return -1
    return 0


def _vwap_signal(closes, volumes):
    """Price vs VWAP: cross above = buy, cross below = sell."""
    if len(closes) < 5:
        return 0
    vwap_vals = vwap(closes, volumes)
    cur_p, prev_p = closes[-1], closes[-2]
    cur_v, prev_v = _safe_last(vwap_vals), _safe_last(vwap_vals, 1)
    if not _is_valid(cur_v, prev_v):
        return 0
    if prev_p <= prev_v and cur_p > cur_v:
        return 1
    if prev_p >= prev_v and cur_p < cur_v:
        return -1
    return 0


def _bb_width_squeeze_signal(closes, period, num_std, squeeze_percentile):
    """BB Width squeeze: when width hits low, prepare for breakout."""
    if len(closes) < max(period + 20, 50):
        return 0
    widths = bb_width(closes, period, num_std)
    valid_widths = [w for w in widths if _is_valid(w)]
    if len(valid_widths) < 20:
        return 0
    cur_width = _safe_last(widths)
    if not _is_valid(cur_width):
        return 0
    threshold = np.percentile(valid_widths[-100:] if len(valid_widths) >= 100
                              else valid_widths, squeeze_percentile)
    if cur_width <= threshold:
        # Squeeze detected -- follow recent direction
        change = closes[-1] - closes[-3] if len(closes) >= 3 else 0
        if change > 0:
            return 1
        elif change < 0:
            return -1
    return 0


def _volume_delta_signal(closes, volumes, lookback):
    """Volume delta: sustained buying/selling pressure."""
    if len(closes) < lookback + 2:
        return 0
    # Approximate opens from previous closes for volume_delta
    opens = [closes[0]] + list(closes[:-1])
    vd = volume_delta(opens, closes, volumes)
    recent = vd[-lookback:]
    if not recent:
        return 0
    avg_delta = sum(recent) / len(recent)
    total_vol = sum(abs(v) for v in recent)
    if total_vol == 0:
        return 0
    ratio = avg_delta / (total_vol / len(recent))
    if ratio > 0.3:
        return 1
    if ratio < -0.3:
        return -1
    return 0


# ===================================================================
# SECTION 2: Generate single-indicator strategies (~60)
# ===================================================================

def generate_single_indicator_strategies() -> List[IndicatorSignal]:
    """Generate ~60 strategies from single indicators with varied params."""
    strategies: List[IndicatorSignal] = []

    # --- RSI strategies (4 threshold pairs x 2 periods = 8) ---
    for period in [14, 21]:
        for oversold, overbought in [(20, 80), (25, 75), (30, 70), (35, 65)]:
            strategies.append(IndicatorSignal(
                name=f"RSI_{period}_{oversold}_{overbought}",
                compute=lambda c, h, l, v, os=oversold, ob=overbought, p=period:
                    _rsi_signal(c, os, ob, p),
                category="momentum",
                params={"period": period, "oversold": oversold, "overbought": overbought}
            ))

    # --- EMA crossover (6 pairs) ---
    for fast, slow in [(5, 13), (8, 21), (10, 30), (12, 26), (20, 50), (50, 200)]:
        strategies.append(IndicatorSignal(
            name=f"EMA_Cross_{fast}_{slow}",
            compute=lambda c, h, l, v, f=fast, s=slow: _ema_cross_signal(c, f, s),
            category="trend",
            params={"fast": fast, "slow": slow}
        ))

    # --- SMA crossover (4 pairs) ---
    for fast, slow in [(10, 30), (20, 50), (50, 100), (50, 200)]:
        strategies.append(IndicatorSignal(
            name=f"SMA_Cross_{fast}_{slow}",
            compute=lambda c, h, l, v, f=fast, s=slow: _sma_cross_signal(c, f, s),
            category="trend",
            params={"fast": fast, "slow": slow}
        ))

    # --- Bollinger Band strategies (4 param sets) ---
    for period, num_std in [(20, 2.0), (20, 2.5), (20, 1.5), (30, 2.0)]:
        strategies.append(IndicatorSignal(
            name=f"BB_Touch_{period}_{num_std}",
            compute=lambda c, h, l, v, p=period, ns=num_std: _bb_signal(c, p, ns),
            category="volatility",
            params={"period": period, "num_std": num_std}
        ))

    # --- MACD histogram cross (3 param sets) ---
    for fast, slow, sig in [(12, 26, 9), (8, 17, 9), (5, 35, 5)]:
        strategies.append(IndicatorSignal(
            name=f"MACD_Hist_{fast}_{slow}_{sig}",
            compute=lambda c, h, l, v, f=fast, s=slow, sp=sig:
                _macd_signal(c, f, s, sp),
            category="momentum",
            params={"fast": fast, "slow": slow, "signal": sig}
        ))

    # --- MACD line/signal crossover (3 param sets) ---
    for fast, slow, sig in [(12, 26, 9), (8, 17, 9), (5, 35, 5)]:
        strategies.append(IndicatorSignal(
            name=f"MACD_Cross_{fast}_{slow}_{sig}",
            compute=lambda c, h, l, v, f=fast, s=slow, sp=sig:
                _macd_divergence_signal(c, f, s, sp),
            category="momentum",
            params={"fast": fast, "slow": slow, "signal": sig}
        ))

    # --- Stochastic strategies (3 param sets x 2 thresholds = 6) ---
    for k_p, d_p in [(14, 3), (9, 3), (21, 7)]:
        for os, ob in [(20, 80), (30, 70)]:
            strategies.append(IndicatorSignal(
                name=f"Stoch_{k_p}_{d_p}_{os}_{ob}",
                compute=lambda c, h, l, v, kp=k_p, dp=d_p, ov=os, ob_=ob:
                    _stochastic_signal(h, l, c, kp, dp, ov, ob_),
                category="momentum",
                params={"k_period": k_p, "d_period": d_p, "oversold": os, "overbought": ob}
            ))

    # --- Williams %R strategies (3 periods x 2 thresholds = 6) ---
    for period in [14, 21, 28]:
        for os, ob in [(-80, -20), (-90, -10)]:
            strategies.append(IndicatorSignal(
                name=f"WillR_{period}_{os}_{ob}",
                compute=lambda c, h, l, v, p=period, ov=os, ob_=ob:
                    _williams_r_signal(h, l, c, p, ov, ob_),
                category="momentum",
                params={"period": period, "oversold": os, "overbought": ob}
            ))

    # --- ROC strategies (4 periods x 2 thresholds = 8) ---
    for period in [5, 10, 14, 21]:
        for threshold in [2.0, 5.0]:
            strategies.append(IndicatorSignal(
                name=f"ROC_{period}_{threshold}",
                compute=lambda c, h, l, v, p=period, t=threshold: _roc_signal(c, p, t),
                category="momentum",
                params={"period": period, "threshold": threshold}
            ))

    # --- MFI strategies (2 periods x 2 thresholds = 4) ---
    for period in [14, 21]:
        for os, ob in [(20, 80), (30, 70)]:
            strategies.append(IndicatorSignal(
                name=f"MFI_{period}_{os}_{ob}",
                compute=lambda c, h, l, v, p=period, ov=os, ob_=ob:
                    _mfi_signal(h, l, c, v, p, ov, ob_),
                category="volume",
                params={"period": period, "oversold": os, "overbought": ob}
            ))

    # --- ADX trend strategies (3 periods x 2 thresholds = 6) ---
    for period in [14, 20, 28]:
        for threshold in [20, 25]:
            strategies.append(IndicatorSignal(
                name=f"ADX_{period}_{threshold}",
                compute=lambda c, h, l, v, p=period, t=threshold:
                    _adx_signal(h, l, c, p, t),
                category="trend",
                params={"period": period, "threshold": threshold}
            ))

    # --- Supertrend strategies (4 param sets) ---
    for period, mult in [(7, 3.0), (10, 2.0), (10, 3.0), (14, 2.5)]:
        strategies.append(IndicatorSignal(
            name=f"Supertrend_{period}_{mult}",
            compute=lambda c, h, l, v, p=period, m=mult:
                _supertrend_signal(h, l, c, p, m),
            category="trend",
            params={"period": period, "multiplier": mult}
        ))

    # --- BB Width squeeze strategies (3 param sets) ---
    for period, ns, pctile in [(20, 2.0, 20), (20, 2.0, 10), (30, 2.0, 15)]:
        strategies.append(IndicatorSignal(
            name=f"BB_Squeeze_{period}_{ns}_{pctile}",
            compute=lambda c, h, l, v, p=period, n=ns, pc=pctile:
                _bb_width_squeeze_signal(c, p, n, pc),
            category="volatility",
            params={"period": period, "num_std": ns, "squeeze_percentile": pctile}
        ))

    # --- Choppiness index strategies (3 periods) ---
    for period in [14, 21, 28]:
        strategies.append(IndicatorSignal(
            name=f"Chop_{period}",
            compute=lambda c, h, l, v, p=period:
                _choppiness_signal(h, l, c, p, 61.8, 38.2),
            category="volatility",
            params={"period": period, "choppy": 61.8, "trend": 38.2}
        ))

    # --- OBV divergence strategies (3 lookbacks) ---
    for lookback in [10, 20, 30]:
        strategies.append(IndicatorSignal(
            name=f"OBV_Div_{lookback}",
            compute=lambda c, h, l, v, lb=lookback: _obv_divergence_signal(c, v, lb),
            category="volume",
            params={"lookback": lookback}
        ))

    # --- CMF strategies (2 periods x 2 thresholds = 4) ---
    for period in [20, 30]:
        for threshold in [0.05, 0.10]:
            strategies.append(IndicatorSignal(
                name=f"CMF_{period}_{threshold}",
                compute=lambda c, h, l, v, p=period, t=threshold:
                    _cmf_signal(h, l, c, v, p, t),
                category="volume",
                params={"period": period, "threshold": threshold}
            ))

    # --- VWAP cross strategy ---
    strategies.append(IndicatorSignal(
        name="VWAP_Cross",
        compute=lambda c, h, l, v: _vwap_signal(c, v),
        category="volume",
        params={}
    ))

    # --- Volume delta strategies (3 lookbacks) ---
    for lookback in [5, 10, 20]:
        strategies.append(IndicatorSignal(
            name=f"VolDelta_{lookback}",
            compute=lambda c, h, l, v, lb=lookback: _volume_delta_signal(c, v, lb),
            category="volume",
            params={"lookback": lookback}
        ))

    logger.info("Generated %d single-indicator strategies", len(strategies))
    return strategies


# ===================================================================
# SECTION 3: Combination strategies (~100)
# ===================================================================

def _combo_and(signals: List[int]) -> int:
    """AND logic: all must agree, else 0."""
    if all(s > 0 for s in signals):
        return 1
    if all(s < 0 for s in signals):
        return -1
    return 0


def _combo_majority(signals: List[int]) -> int:
    """Majority vote: >50% must agree."""
    if not signals:
        return 0
    buy = sum(1 for s in signals if s > 0)
    sell = sum(1 for s in signals if s < 0)
    n = len(signals)
    if buy > n / 2:
        return 1
    if sell > n / 2:
        return -1
    return 0


def _combo_weighted(signals: List[int], weights: List[float]) -> int:
    """Weighted sum: positive = buy, negative = sell, threshold 0.5."""
    if not signals:
        return 0
    total = sum(s * w for s, w in zip(signals, weights))
    if total > 0.5:
        return 1
    if total < -0.5:
        return -1
    return 0


def _combo_primary_confirm(primary: int, confirmer: int) -> int:
    """Primary signal confirmed by secondary (non-opposing)."""
    if primary == 0:
        return 0
    if confirmer == -primary:
        return 0  # Conflict
    return primary


def generate_combo_strategies() -> List[IndicatorSignal]:
    """Generate 2-indicator and 3-indicator combinations."""
    combos: List[IndicatorSignal] = []

    # ---- RSI + MACD (momentum confirmation) ----
    for rsi_os, rsi_ob in [(25, 75), (30, 70)]:
        for macd_f, macd_s, macd_sig in [(12, 26, 9), (8, 17, 9)]:
            combos.append(IndicatorSignal(
                name=f"RSI_MACD_AND_{rsi_os}_{macd_f}_{macd_s}",
                compute=lambda c, h, l, v, os=rsi_os, ob=rsi_ob,
                       mf=macd_f, ms=macd_s, msig=macd_sig:
                    _combo_and([
                        _rsi_signal(c, os, ob),
                        _macd_signal(c, mf, ms, msig)
                    ]),
                category="combo",
                params={"rsi_os": rsi_os, "rsi_ob": rsi_ob,
                        "macd_fast": macd_f, "macd_slow": macd_s}
            ))

    # ---- RSI + BB (mean reversion with bands) ----
    for rsi_os, rsi_ob in [(25, 75), (30, 70)]:
        for bb_p, bb_std in [(20, 2.0), (20, 2.5)]:
            combos.append(IndicatorSignal(
                name=f"RSI_BB_AND_{rsi_os}_{bb_p}_{bb_std}",
                compute=lambda c, h, l, v, os=rsi_os, ob=rsi_ob,
                       bp=bb_p, bs=bb_std:
                    _combo_and([
                        _rsi_signal(c, os, ob),
                        _bb_signal(c, bp, bs)
                    ]),
                category="combo",
                params={"rsi_os": rsi_os, "rsi_ob": rsi_ob,
                        "bb_period": bb_p, "bb_std": bb_std}
            ))

    # ---- EMA + Volume Delta (trend with volume confirmation) ----
    for ema_f, ema_s in [(8, 21), (12, 26), (20, 50)]:
        for vd_lb in [5, 10]:
            combos.append(IndicatorSignal(
                name=f"EMA_VolDelta_{ema_f}_{ema_s}_{vd_lb}",
                compute=lambda c, h, l, v, ef=ema_f, es=ema_s, lb=vd_lb:
                    _combo_primary_confirm(
                        _ema_cross_signal(c, ef, es),
                        _volume_delta_signal(c, v, lb)
                    ),
                category="combo",
                params={"ema_fast": ema_f, "ema_slow": ema_s, "vol_lookback": vd_lb}
            ))

    # ---- MACD + ADX (trend direction + strength) ----
    for macd_f, macd_s in [(12, 26), (8, 17)]:
        for adx_p, adx_t in [(14, 20), (14, 25)]:
            combos.append(IndicatorSignal(
                name=f"MACD_ADX_{macd_f}_{macd_s}_{adx_p}_{adx_t}",
                compute=lambda c, h, l, v, mf=macd_f, ms=macd_s,
                       ap=adx_p, at=adx_t:
                    _combo_and([
                        _macd_signal(c, mf, ms, 9),
                        _adx_signal(h, l, c, ap, at)
                    ]),
                category="combo",
                params={"macd_fast": macd_f, "macd_slow": macd_s,
                        "adx_period": adx_p, "adx_threshold": adx_t}
            ))

    # ---- BB + Stochastic (squeeze + momentum) ----
    for bb_p, bb_std in [(20, 2.0), (20, 2.5)]:
        for k_p in [14, 9]:
            combos.append(IndicatorSignal(
                name=f"BB_Stoch_{bb_p}_{bb_std}_{k_p}",
                compute=lambda c, h, l, v, bp=bb_p, bs=bb_std, kp=k_p:
                    _combo_and([
                        _bb_signal(c, bp, bs),
                        _stochastic_signal(h, l, c, kp, 3, 20, 80)
                    ]),
                category="combo",
                params={"bb_period": bb_p, "bb_std": bb_std, "k_period": k_p}
            ))

    # ---- RSI + Williams %R (double momentum) ----
    for rsi_os, rsi_ob in [(25, 75), (30, 70)]:
        for wr_p in [14, 21]:
            combos.append(IndicatorSignal(
                name=f"RSI_WillR_{rsi_os}_{wr_p}",
                compute=lambda c, h, l, v, os=rsi_os, ob=rsi_ob, wp=wr_p:
                    _combo_and([
                        _rsi_signal(c, os, ob),
                        _williams_r_signal(h, l, c, wp, -80, -20)
                    ]),
                category="combo",
                params={"rsi_os": rsi_os, "rsi_ob": rsi_ob, "wr_period": wr_p}
            ))

    # ---- EMA + RSI (trend + momentum) ----
    for ema_f, ema_s in [(8, 21), (12, 26), (20, 50)]:
        for rsi_os, rsi_ob in [(30, 70), (25, 75)]:
            combos.append(IndicatorSignal(
                name=f"EMA_RSI_{ema_f}_{ema_s}_{rsi_os}",
                compute=lambda c, h, l, v, ef=ema_f, es=ema_s,
                       os=rsi_os, ob=rsi_ob:
                    _combo_primary_confirm(
                        _ema_cross_signal(c, ef, es),
                        _rsi_signal(c, os, ob)
                    ),
                category="combo",
                params={"ema_fast": ema_f, "ema_slow": ema_s,
                        "rsi_os": rsi_os, "rsi_ob": rsi_ob}
            ))

    # ---- Supertrend + RSI (trend + momentum confirmation) ----
    for st_p, st_m in [(7, 3.0), (10, 2.0)]:
        for rsi_os, rsi_ob in [(30, 70), (25, 75)]:
            combos.append(IndicatorSignal(
                name=f"ST_RSI_{st_p}_{st_m}_{rsi_os}",
                compute=lambda c, h, l, v, sp=st_p, sm=st_m,
                       os=rsi_os, ob=rsi_ob:
                    _combo_primary_confirm(
                        _supertrend_signal(h, l, c, sp, sm),
                        _rsi_signal(c, os, ob)
                    ),
                category="combo",
                params={"st_period": st_p, "st_mult": st_m,
                        "rsi_os": rsi_os, "rsi_ob": rsi_ob}
            ))

    # ---- CMF + RSI (money flow + momentum) ----
    for cmf_p in [20, 30]:
        for rsi_os, rsi_ob in [(30, 70), (25, 75)]:
            combos.append(IndicatorSignal(
                name=f"CMF_RSI_{cmf_p}_{rsi_os}",
                compute=lambda c, h, l, v, cp=cmf_p, os=rsi_os, ob=rsi_ob:
                    _combo_and([
                        _cmf_signal(h, l, c, v, cp, 0.05),
                        _rsi_signal(c, os, ob)
                    ]),
                category="combo",
                params={"cmf_period": cmf_p, "rsi_os": rsi_os, "rsi_ob": rsi_ob}
            ))

    # ---- MFI + BB (volume-weighted momentum + bands) ----
    for mfi_p in [14, 21]:
        for bb_p, bb_std in [(20, 2.0), (20, 2.5)]:
            combos.append(IndicatorSignal(
                name=f"MFI_BB_{mfi_p}_{bb_p}_{bb_std}",
                compute=lambda c, h, l, v, mp=mfi_p, bp=bb_p, bs=bb_std:
                    _combo_and([
                        _mfi_signal(h, l, c, v, mp, 20, 80),
                        _bb_signal(c, bp, bs)
                    ]),
                category="combo",
                params={"mfi_period": mfi_p, "bb_period": bb_p, "bb_std": bb_std}
            ))

    # ---- OBV + EMA (volume divergence + trend) ----
    for obv_lb in [10, 20]:
        for ema_f, ema_s in [(8, 21), (12, 26)]:
            combos.append(IndicatorSignal(
                name=f"OBV_EMA_{obv_lb}_{ema_f}_{ema_s}",
                compute=lambda c, h, l, v, lb=obv_lb, ef=ema_f, es=ema_s:
                    _combo_and([
                        _obv_divergence_signal(c, v, lb),
                        _ema_cross_signal(c, ef, es)
                    ]),
                category="combo",
                params={"obv_lookback": obv_lb, "ema_fast": ema_f, "ema_slow": ema_s}
            ))

    # ---- Stochastic + MACD (momentum double confirmation) ----
    for k_p in [9, 14]:
        for macd_f, macd_s in [(12, 26), (8, 17)]:
            combos.append(IndicatorSignal(
                name=f"Stoch_MACD_{k_p}_{macd_f}_{macd_s}",
                compute=lambda c, h, l, v, kp=k_p, mf=macd_f, ms=macd_s:
                    _combo_and([
                        _stochastic_signal(h, l, c, kp, 3, 20, 80),
                        _macd_signal(c, mf, ms, 9)
                    ]),
                category="combo",
                params={"k_period": k_p, "macd_fast": macd_f, "macd_slow": macd_s}
            ))

    # ---- Supertrend + ADX (trend + strength) ----
    for st_p, st_m in [(7, 3.0), (10, 2.0)]:
        for adx_t in [20, 25]:
            combos.append(IndicatorSignal(
                name=f"ST_ADX_{st_p}_{st_m}_{adx_t}",
                compute=lambda c, h, l, v, sp=st_p, sm=st_m, at=adx_t:
                    _combo_primary_confirm(
                        _supertrend_signal(h, l, c, sp, sm),
                        _adx_signal(h, l, c, 14, at)
                    ),
                category="combo",
                params={"st_period": st_p, "st_mult": st_m, "adx_threshold": adx_t}
            ))

    # ---- VWAP + RSI (institutional level + momentum) ----
    for rsi_os, rsi_ob in [(25, 75), (30, 70)]:
        combos.append(IndicatorSignal(
            name=f"VWAP_RSI_{rsi_os}",
            compute=lambda c, h, l, v, os=rsi_os, ob=rsi_ob:
                _combo_and([
                    _vwap_signal(c, v),
                    _rsi_signal(c, os, ob)
                ]),
            category="combo",
            params={"rsi_os": rsi_os, "rsi_ob": rsi_ob}
        ))

    # ---- BB Squeeze + MACD (volatility expansion + direction) ----
    for macd_f, macd_s in [(12, 26), (8, 17)]:
        combos.append(IndicatorSignal(
            name=f"BBSq_MACD_{macd_f}_{macd_s}",
            compute=lambda c, h, l, v, mf=macd_f, ms=macd_s:
                _combo_and([
                    _bb_width_squeeze_signal(c, 20, 2.0, 20),
                    _macd_signal(c, mf, ms, 9)
                ]),
            category="combo",
            params={"macd_fast": macd_f, "macd_slow": macd_s}
        ))

    # ---- ROC + ADX (momentum + trend strength) ----
    for roc_p in [10, 14]:
        for adx_t in [20, 25]:
            combos.append(IndicatorSignal(
                name=f"ROC_ADX_{roc_p}_{adx_t}",
                compute=lambda c, h, l, v, rp=roc_p, at=adx_t:
                    _combo_and([
                        _roc_signal(c, rp, 3.0),
                        _adx_signal(h, l, c, 14, at)
                    ]),
                category="combo",
                params={"roc_period": roc_p, "adx_threshold": adx_t}
            ))

    # ===== TRIPLE INDICATOR COMBOS (majority vote) =====

    # ---- EMA + RSI + Volume Delta (triple confirmation) ----
    for ema_f, ema_s in [(8, 21), (12, 26)]:
        combos.append(IndicatorSignal(
            name=f"EMA_RSI_VD_{ema_f}_{ema_s}",
            compute=lambda c, h, l, v, ef=ema_f, es=ema_s:
                _combo_majority([
                    _ema_cross_signal(c, ef, es),
                    _rsi_signal(c, 30, 70),
                    _volume_delta_signal(c, v, 10)
                ]),
            category="combo",
            params={"ema_fast": ema_f, "ema_slow": ema_s}
        ))

    # ---- MACD + BB + ADX (full analysis) ----
    for macd_f, macd_s in [(12, 26), (8, 17)]:
        combos.append(IndicatorSignal(
            name=f"MACD_BB_ADX_{macd_f}_{macd_s}",
            compute=lambda c, h, l, v, mf=macd_f, ms=macd_s:
                _combo_majority([
                    _macd_signal(c, mf, ms, 9),
                    _bb_signal(c, 20, 2.0),
                    _adx_signal(h, l, c, 14, 20)
                ]),
            category="combo",
            params={"macd_fast": macd_f, "macd_slow": macd_s}
        ))

    # ---- RSI + Stochastic + MFI (triple momentum) ----
    for rsi_os, rsi_ob in [(25, 75), (30, 70)]:
        combos.append(IndicatorSignal(
            name=f"RSI_Stoch_MFI_{rsi_os}",
            compute=lambda c, h, l, v, os=rsi_os, ob=rsi_ob:
                _combo_majority([
                    _rsi_signal(c, os, ob),
                    _stochastic_signal(h, l, c, 14, 3, 20, 80),
                    _mfi_signal(h, l, c, v, 14, 20, 80)
                ]),
            category="combo",
            params={"rsi_os": rsi_os, "rsi_ob": rsi_ob}
        ))

    # ---- Supertrend + MACD + RSI (trend + momentum + confirmation) ----
    for st_p, st_m in [(7, 3.0), (10, 2.0)]:
        combos.append(IndicatorSignal(
            name=f"ST_MACD_RSI_{st_p}_{st_m}",
            compute=lambda c, h, l, v, sp=st_p, sm=st_m:
                _combo_majority([
                    _supertrend_signal(h, l, c, sp, sm),
                    _macd_signal(c, 12, 26, 9),
                    _rsi_signal(c, 30, 70)
                ]),
            category="combo",
            params={"st_period": st_p, "st_mult": st_m}
        ))

    # ---- EMA + ADX + CMF (trend + strength + money flow) ----
    for ema_f, ema_s in [(8, 21), (12, 26)]:
        combos.append(IndicatorSignal(
            name=f"EMA_ADX_CMF_{ema_f}_{ema_s}",
            compute=lambda c, h, l, v, ef=ema_f, es=ema_s:
                _combo_majority([
                    _ema_cross_signal(c, ef, es),
                    _adx_signal(h, l, c, 14, 20),
                    _cmf_signal(h, l, c, v, 20, 0.05)
                ]),
            category="combo",
            params={"ema_fast": ema_f, "ema_slow": ema_s}
        ))

    # ---- BB + RSI + OBV (mean reversion with volume confirmation) ----
    for bb_std in [2.0, 2.5]:
        combos.append(IndicatorSignal(
            name=f"BB_RSI_OBV_{bb_std}",
            compute=lambda c, h, l, v, bs=bb_std:
                _combo_majority([
                    _bb_signal(c, 20, bs),
                    _rsi_signal(c, 30, 70),
                    _obv_divergence_signal(c, v, 20)
                ]),
            category="combo",
            params={"bb_std": bb_std}
        ))

    # ---- MACD + Stochastic + Williams %R (triple oscillator) ----
    combos.append(IndicatorSignal(
        name="MACD_Stoch_WillR",
        compute=lambda c, h, l, v:
            _combo_majority([
                _macd_signal(c, 12, 26, 9),
                _stochastic_signal(h, l, c, 14, 3, 20, 80),
                _williams_r_signal(h, l, c, 14, -80, -20)
            ]),
        category="combo",
        params={}
    ))

    # ---- Supertrend + EMA + OBV (trend + MA + volume) ----
    for st_p, st_m in [(7, 3.0), (10, 2.0)]:
        combos.append(IndicatorSignal(
            name=f"ST_EMA_OBV_{st_p}_{st_m}",
            compute=lambda c, h, l, v, sp=st_p, sm=st_m:
                _combo_majority([
                    _supertrend_signal(h, l, c, sp, sm),
                    _ema_cross_signal(c, 12, 26),
                    _obv_divergence_signal(c, v, 15)
                ]),
            category="combo",
            params={"st_period": st_p, "st_mult": st_m}
        ))

    # ---- Weighted combos ----
    # RSI (0.4) + MACD (0.35) + Volume (0.25) weighted
    combos.append(IndicatorSignal(
        name="Weighted_RSI_MACD_Vol",
        compute=lambda c, h, l, v:
            _combo_weighted(
                [_rsi_signal(c, 30, 70), _macd_signal(c, 12, 26, 9),
                 _volume_delta_signal(c, v, 10)],
                [0.4, 0.35, 0.25]
            ),
        category="combo",
        params={"weights": [0.4, 0.35, 0.25]}
    ))

    # EMA (0.3) + ADX (0.3) + Stochastic (0.4) weighted
    combos.append(IndicatorSignal(
        name="Weighted_EMA_ADX_Stoch",
        compute=lambda c, h, l, v:
            _combo_weighted(
                [_ema_cross_signal(c, 12, 26), _adx_signal(h, l, c, 14, 20),
                 _stochastic_signal(h, l, c, 14, 3, 20, 80)],
                [0.3, 0.3, 0.4]
            ),
        category="combo",
        params={"weights": [0.3, 0.3, 0.4]}
    ))

    # Supertrend (0.5) + RSI (0.25) + MFI (0.25) weighted
    combos.append(IndicatorSignal(
        name="Weighted_ST_RSI_MFI",
        compute=lambda c, h, l, v:
            _combo_weighted(
                [_supertrend_signal(h, l, c, 7, 3.0), _rsi_signal(c, 30, 70),
                 _mfi_signal(h, l, c, v, 14, 20, 80)],
                [0.5, 0.25, 0.25]
            ),
        category="combo",
        params={"weights": [0.5, 0.25, 0.25]}
    ))

    # BB (0.4) + CMF (0.3) + ROC (0.3) weighted
    combos.append(IndicatorSignal(
        name="Weighted_BB_CMF_ROC",
        compute=lambda c, h, l, v:
            _combo_weighted(
                [_bb_signal(c, 20, 2.0), _cmf_signal(h, l, c, v, 20, 0.05),
                 _roc_signal(c, 10, 3.0)],
                [0.4, 0.3, 0.3]
            ),
        category="combo",
        params={"weights": [0.4, 0.3, 0.3]}
    ))

    logger.info("Generated %d combo strategies", len(combos))
    return combos


# ===================================================================
# SECTION 4: Mathematical strategies (~40)
# ===================================================================

def _zscore_mean_reversion(closes, window, entry_z, exit_z):
    """Z-score mean reversion: buy when z < -entry_z, sell when z > entry_z."""
    if len(closes) < window + 2:
        return 0
    recent = closes[-window:]
    mean = sum(recent) / window
    variance = sum((x - mean) ** 2 for x in recent) / window
    std = math.sqrt(variance) if variance > 0 else 0
    if std == 0:
        return 0
    z = (closes[-1] - mean) / std
    z_prev_arr = closes[-(window + 1):-1]
    mean_prev = sum(z_prev_arr) / len(z_prev_arr)
    var_prev = sum((x - mean_prev) ** 2 for x in z_prev_arr) / len(z_prev_arr)
    std_prev = math.sqrt(var_prev) if var_prev > 0 else 0
    z_prev = (closes[-2] - mean_prev) / std_prev if std_prev > 0 else 0

    # Entry signals
    if z < -entry_z:
        return 1   # Oversold
    if z > entry_z:
        return -1  # Overbought
    return 0


def _linear_regression_signal(closes, window, dev_mult):
    """Linear regression channel: trade when price deviates from trend."""
    if len(closes) < window + 2:
        return 0
    y = np.array(closes[-window:], dtype=float)
    x = np.arange(window, dtype=float)
    # Least squares: slope = cov(x,y)/var(x), intercept = mean(y) - slope*mean(x)
    x_mean = x.mean()
    y_mean = y.mean()
    cov_xy = np.sum((x - x_mean) * (y - y_mean))
    var_x = np.sum((x - x_mean) ** 2)
    if var_x == 0:
        return 0
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    predicted = slope * (window - 1) + intercept
    residuals = y - (slope * x + intercept)
    std_err = np.std(residuals)
    if std_err == 0:
        return 0
    deviation = (closes[-1] - predicted) / std_err
    if deviation < -dev_mult:
        return 1   # Below regression channel
    if deviation > dev_mult:
        return -1  # Above regression channel
    return 0


def _donchian_signal(highs, lows, closes, period):
    """Donchian channel breakout: buy at new high, sell at new low."""
    if len(closes) < period + 2:
        return 0
    upper = max(highs[-period:])
    lower = min(lows[-period:])
    prev_upper = max(highs[-(period + 1):-1])
    prev_lower = min(lows[-(period + 1):-1])
    price = closes[-1]
    prev_price = closes[-2]
    if prev_price < prev_upper and price >= upper:
        return 1   # Breakout up
    if prev_price > prev_lower and price <= lower:
        return -1  # Breakout down
    return 0


def _keltner_signal(highs, lows, closes, ema_period, atr_mult):
    """Keltner channel: buy when price breaks above, sell below."""
    if len(closes) < max(ema_period, 14) + 2:
        return 0
    ema_vals = ema(closes, ema_period)
    atr_vals = atr(highs, lows, closes, 14)
    cur_ema = _safe_last(ema_vals)
    cur_atr = _safe_last(atr_vals)
    if not _is_valid(cur_ema, cur_atr) or cur_atr == 0:
        return 0
    upper = cur_ema + atr_mult * cur_atr
    lower = cur_ema - atr_mult * cur_atr
    price = closes[-1]
    prev = closes[-2]
    # Price crossing channel
    prev_ema = _safe_last(ema_vals, 1)
    prev_atr = _safe_last(atr_vals, 1)
    if _is_valid(prev_ema, prev_atr) and prev_atr > 0:
        prev_upper = prev_ema + atr_mult * prev_atr
        prev_lower = prev_ema - atr_mult * prev_atr
        if prev < prev_upper and price >= upper:
            return 1
        if prev > prev_lower and price <= lower:
            return -1
    return 0


def _moving_avg_envelope_signal(closes, period, pct_width):
    """Moving average envelope: buy at lower envelope, sell at upper."""
    if len(closes) < period + 2:
        return 0
    ma = sma(closes, period)
    cur_ma = _safe_last(ma)
    if not _is_valid(cur_ma) or cur_ma == 0:
        return 0
    upper = cur_ma * (1 + pct_width / 100.0)
    lower = cur_ma * (1 - pct_width / 100.0)
    price = closes[-1]
    if price <= lower:
        return 1
    if price >= upper:
        return -1
    return 0


def _momentum_signal(closes, period):
    """Pure price momentum: compare current to N bars ago."""
    if len(closes) < period + 2:
        return 0
    mom = closes[-1] - closes[-period]
    prev_mom = closes[-2] - closes[-(period + 1)]
    # Momentum cross zero
    if prev_mom <= 0 and mom > 0:
        return 1
    if prev_mom >= 0 and mom < 0:
        return -1
    return 0


def _volatility_breakout_signal(highs, lows, closes, atr_period, atr_mult):
    """ATR-based volatility breakout: large move from close = breakout."""
    if len(closes) < atr_period + 2:
        return 0
    atr_vals = atr(highs, lows, closes, atr_period)
    cur_atr = _safe_last(atr_vals)
    if not _is_valid(cur_atr) or cur_atr == 0:
        return 0
    move = closes[-1] - closes[-2]
    if move > atr_mult * cur_atr:
        return 1
    if move < -atr_mult * cur_atr:
        return -1
    return 0


def _range_contraction_signal(highs, lows, closes, lookback, expansion_mult):
    """Range contraction then expansion: trade the breakout direction."""
    if len(closes) < lookback + 5:
        return 0
    ranges = [highs[i] - lows[i] for i in range(-lookback, 0)]
    avg_range = sum(ranges) / len(ranges)
    if avg_range == 0:
        return 0
    cur_range = highs[-1] - lows[-1]
    prev_range = highs[-2] - lows[-2]
    # Contraction then expansion
    if prev_range < avg_range * 0.5 and cur_range > avg_range * expansion_mult:
        if closes[-1] > closes[-2]:
            return 1
        else:
            return -1
    return 0


def _price_vs_sma_signal(closes, period, threshold_pct):
    """Price distance from SMA: extreme deviation = reversion signal."""
    if len(closes) < period + 2:
        return 0
    ma = sma(closes, period)
    cur_ma = _safe_last(ma)
    if not _is_valid(cur_ma) or cur_ma == 0:
        return 0
    pct_away = ((closes[-1] - cur_ma) / cur_ma) * 100
    if pct_away < -threshold_pct:
        return 1
    if pct_away > threshold_pct:
        return -1
    return 0


def _dual_momentum_signal(closes, short_period, long_period):
    """Dual momentum: compare short-term and long-term ROC."""
    if len(closes) < long_period + 2:
        return 0
    short_roc = (closes[-1] - closes[-short_period]) / closes[-short_period] * 100
    long_roc = (closes[-1] - closes[-long_period]) / closes[-long_period] * 100
    # Both positive = strong buy, both negative = strong sell
    if short_roc > 0 and long_roc > 0:
        return 1
    if short_roc < 0 and long_roc < 0:
        return -1
    return 0


def _mean_reversion_bands_signal(closes, window, num_std):
    """Custom mean reversion bands using rolling mean/std (not Bollinger)."""
    if len(closes) < window + 2:
        return 0
    arr = np.array(closes[-window:], dtype=float)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0
    upper = mean + num_std * std
    lower = mean - num_std * std
    price = closes[-1]
    prev = closes[-2]
    if prev > lower and price <= lower:
        return 1
    if prev < upper and price >= upper:
        return -1
    return 0


def generate_math_strategies() -> List[IndicatorSignal]:
    """Generate pure mathematical approach strategies."""
    strategies: List[IndicatorSignal] = []

    # --- Z-score mean reversion (4 windows x 3 z-thresholds = 12) ---
    for window in [10, 20, 50, 100]:
        for entry_z in [1.5, 2.0, 2.5]:
            strategies.append(IndicatorSignal(
                name=f"ZScore_{window}_{entry_z}",
                compute=lambda c, h, l, v, w=window, ez=entry_z:
                    _zscore_mean_reversion(c, w, ez, ez * 0.5),
                category="math",
                params={"window": window, "entry_z": entry_z}
            ))

    # --- Linear regression (3 windows x 2 deviations = 6) ---
    for window in [20, 50, 100]:
        for dev in [1.5, 2.0]:
            strategies.append(IndicatorSignal(
                name=f"LinReg_{window}_{dev}",
                compute=lambda c, h, l, v, w=window, d=dev:
                    _linear_regression_signal(c, w, d),
                category="math",
                params={"window": window, "deviation": dev}
            ))

    # --- Moving average envelope (3 periods x 2 widths = 6) ---
    for period in [20, 50, 100]:
        for width in [2.0, 3.0]:
            strategies.append(IndicatorSignal(
                name=f"MA_Envelope_{period}_{width}",
                compute=lambda c, h, l, v, p=period, w=width:
                    _moving_avg_envelope_signal(c, p, w),
                category="math",
                params={"period": period, "width": width}
            ))

    # --- Donchian channel (4 periods) ---
    for period in [10, 20, 50, 100]:
        strategies.append(IndicatorSignal(
            name=f"Donchian_{period}",
            compute=lambda c, h, l, v, p=period: _donchian_signal(h, l, c, p),
            category="math",
            params={"period": period}
        ))

    # --- Keltner channel (3 periods x 2 multipliers = 6) ---
    for ema_p in [20, 30, 50]:
        for mult in [1.5, 2.0]:
            strategies.append(IndicatorSignal(
                name=f"Keltner_{ema_p}_{mult}",
                compute=lambda c, h, l, v, ep=ema_p, m=mult:
                    _keltner_signal(h, l, c, ep, m),
                category="math",
                params={"ema_period": ema_p, "atr_mult": mult}
            ))

    # --- Price momentum (4 periods) ---
    for period in [5, 10, 20, 50]:
        strategies.append(IndicatorSignal(
            name=f"Momentum_{period}",
            compute=lambda c, h, l, v, p=period: _momentum_signal(c, p),
            category="math",
            params={"period": period}
        ))

    # --- Volatility breakout (3 periods x 2 multipliers = 6) ---
    for atr_p in [10, 14, 20]:
        for mult in [1.5, 2.0]:
            strategies.append(IndicatorSignal(
                name=f"VolBreakout_{atr_p}_{mult}",
                compute=lambda c, h, l, v, ap=atr_p, m=mult:
                    _volatility_breakout_signal(h, l, c, ap, m),
                category="math",
                params={"atr_period": atr_p, "atr_mult": mult}
            ))

    # --- Range contraction/expansion (3 lookbacks x 2 multipliers = 6) ---
    for lookback in [10, 20, 30]:
        for mult in [1.5, 2.0]:
            strategies.append(IndicatorSignal(
                name=f"RangeExpansion_{lookback}_{mult}",
                compute=lambda c, h, l, v, lb=lookback, m=mult:
                    _range_contraction_signal(h, l, c, lb, m),
                category="math",
                params={"lookback": lookback, "expansion_mult": mult}
            ))

    # --- Price vs SMA distance (3 periods x 2 thresholds = 6) ---
    for period in [20, 50, 100]:
        for pct in [3.0, 5.0]:
            strategies.append(IndicatorSignal(
                name=f"SMA_Distance_{period}_{pct}",
                compute=lambda c, h, l, v, p=period, t=pct:
                    _price_vs_sma_signal(c, p, t),
                category="math",
                params={"period": period, "threshold_pct": pct}
            ))

    # --- Dual momentum (3 pairs) ---
    for short_p, long_p in [(5, 20), (10, 50), (20, 100)]:
        strategies.append(IndicatorSignal(
            name=f"DualMom_{short_p}_{long_p}",
            compute=lambda c, h, l, v, sp=short_p, lp=long_p:
                _dual_momentum_signal(c, sp, lp),
            category="math",
            params={"short_period": short_p, "long_period": long_p}
        ))

    # --- Mean reversion bands (3 windows x 2 std = 6) ---
    for window in [20, 50, 100]:
        for ns in [2.0, 2.5]:
            strategies.append(IndicatorSignal(
                name=f"MR_Bands_{window}_{ns}",
                compute=lambda c, h, l, v, w=window, n=ns:
                    _mean_reversion_bands_signal(c, w, n),
                category="math",
                params={"window": window, "num_std": ns}
            ))

    logger.info("Generated %d math strategies", len(strategies))
    return strategies


# ===================================================================
# SECTION 5: Pattern strategies (~20)
# ===================================================================

def _engulfing_signal(closes, highs, lows):
    """Bullish/bearish engulfing candle pattern."""
    if len(closes) < 3:
        return 0
    # Approximate opens from prev close
    o2, c2 = closes[-3], closes[-2]   # Previous candle
    o1, c1 = closes[-2], closes[-1]   # Current candle
    body2 = c2 - o2
    body1 = c1 - o1
    # Bullish engulfing: prev red, current green, current body > prev body
    if body2 < 0 and body1 > 0 and abs(body1) > abs(body2):
        return 1
    # Bearish engulfing
    if body2 > 0 and body1 < 0 and abs(body1) > abs(body2):
        return -1
    return 0


def _hammer_signal(closes, highs, lows):
    """Hammer / shooting star pattern."""
    if len(closes) < 3:
        return 0
    o, c, h, l = closes[-2], closes[-1], highs[-1], lows[-1]
    body = abs(c - o)
    full_range = h - l
    if full_range == 0:
        return 0
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    # Hammer: small body at top, long lower wick (downtrend reversal)
    if lower_wick > 2 * body and upper_wick < body:
        # Confirm downtrend
        if closes[-1] < closes[-3]:
            return 1
    # Shooting star: small body at bottom, long upper wick
    if upper_wick > 2 * body and lower_wick < body:
        if closes[-1] > closes[-3]:
            return -1
    return 0


def _three_bar_pattern_signal(closes):
    """Three consecutive bars in same direction = continuation."""
    if len(closes) < 5:
        return 0
    d1 = closes[-3] - closes[-4]
    d2 = closes[-2] - closes[-3]
    d3 = closes[-1] - closes[-2]
    if d1 > 0 and d2 > 0 and d3 > 0:
        return 1
    if d1 < 0 and d2 < 0 and d3 < 0:
        return -1
    return 0


def _doji_reversal_signal(closes, highs, lows):
    """Doji candle (tiny body) as reversal signal."""
    if len(closes) < 10:
        return 0
    o, c, h, l = closes[-2], closes[-1], highs[-1], lows[-1]
    body = abs(c - o)
    full_range = h - l
    if full_range == 0:
        return 0
    # Doji: body < 10% of range
    if body / full_range < 0.1:
        # Check for prior trend
        trend = closes[-1] - closes[-5]
        if trend > 0:
            return -1  # Reversal from uptrend
        elif trend < 0:
            return 1   # Reversal from downtrend
    return 0


def _inside_bar_breakout_signal(closes, highs, lows):
    """Inside bar: current range inside prev range, trade the breakout."""
    if len(closes) < 4:
        return 0
    # Check if bar -2 is inside bar -3
    if highs[-2] <= highs[-3] and lows[-2] >= lows[-3]:
        # Inside bar detected, check breakout on current bar
        if closes[-1] > highs[-2]:
            return 1
        if closes[-1] < lows[-2]:
            return -1
    return 0


def generate_pattern_strategies() -> List[IndicatorSignal]:
    """Generate candle pattern strategies with indicator confirmation."""
    strategies: List[IndicatorSignal] = []

    # --- Basic patterns ---
    strategies.append(IndicatorSignal(
        name="Engulfing",
        compute=lambda c, h, l, v: _engulfing_signal(c, h, l),
        category="pattern",
        params={}
    ))
    strategies.append(IndicatorSignal(
        name="Hammer",
        compute=lambda c, h, l, v: _hammer_signal(c, h, l),
        category="pattern",
        params={}
    ))
    strategies.append(IndicatorSignal(
        name="ThreeBar",
        compute=lambda c, h, l, v: _three_bar_pattern_signal(c),
        category="pattern",
        params={}
    ))
    strategies.append(IndicatorSignal(
        name="Doji_Reversal",
        compute=lambda c, h, l, v: _doji_reversal_signal(c, h, l),
        category="pattern",
        params={}
    ))
    strategies.append(IndicatorSignal(
        name="InsideBar_Breakout",
        compute=lambda c, h, l, v: _inside_bar_breakout_signal(c, h, l),
        category="pattern",
        params={}
    ))

    # --- Patterns with RSI confirmation ---
    for rsi_os, rsi_ob in [(30, 70), (25, 75)]:
        strategies.append(IndicatorSignal(
            name=f"Engulfing_RSI_{rsi_os}",
            compute=lambda c, h, l, v, os=rsi_os, ob=rsi_ob:
                _combo_and([_engulfing_signal(c, h, l), _rsi_signal(c, os, ob)]),
            category="pattern",
            params={"rsi_os": rsi_os, "rsi_ob": rsi_ob}
        ))
        strategies.append(IndicatorSignal(
            name=f"Hammer_RSI_{rsi_os}",
            compute=lambda c, h, l, v, os=rsi_os, ob=rsi_ob:
                _combo_primary_confirm(_hammer_signal(c, h, l), _rsi_signal(c, os, ob)),
            category="pattern",
            params={"rsi_os": rsi_os, "rsi_ob": rsi_ob}
        ))
        strategies.append(IndicatorSignal(
            name=f"Doji_RSI_{rsi_os}",
            compute=lambda c, h, l, v, os=rsi_os, ob=rsi_ob:
                _combo_and([_doji_reversal_signal(c, h, l), _rsi_signal(c, os, ob)]),
            category="pattern",
            params={"rsi_os": rsi_os, "rsi_ob": rsi_ob}
        ))

    # --- Patterns with volume confirmation ---
    strategies.append(IndicatorSignal(
        name="Engulfing_VolDelta",
        compute=lambda c, h, l, v:
            _combo_primary_confirm(_engulfing_signal(c, h, l), _volume_delta_signal(c, v, 5)),
        category="pattern",
        params={}
    ))
    strategies.append(IndicatorSignal(
        name="InsideBar_VolDelta",
        compute=lambda c, h, l, v:
            _combo_primary_confirm(_inside_bar_breakout_signal(c, h, l), _volume_delta_signal(c, v, 5)),
        category="pattern",
        params={}
    ))

    # --- Pattern + BB (reversal at band boundaries) ---
    strategies.append(IndicatorSignal(
        name="Hammer_BB",
        compute=lambda c, h, l, v:
            _combo_and([_hammer_signal(c, h, l), _bb_signal(c, 20, 2.0)]),
        category="pattern",
        params={}
    ))
    strategies.append(IndicatorSignal(
        name="Doji_BB",
        compute=lambda c, h, l, v:
            _combo_and([_doji_reversal_signal(c, h, l), _bb_signal(c, 20, 2.0)]),
        category="pattern",
        params={}
    ))

    logger.info("Generated %d pattern strategies", len(strategies))
    return strategies


# ===================================================================
# SECTION 6: StrategyUniverse master class
# ===================================================================

class StrategyUniverse:
    """
    Generates and manages hundreds of trading strategies.

    Usage:
        universe = StrategyUniverse()
        print(f"Generated {universe.total_strategies} strategies")

        # Get signal from all strategies
        signals = universe.evaluate_all(closes, highs, lows, volumes)
        # Returns: Dict[str, int] mapping strategy_name -> signal

        # Get consensus
        action, confidence = universe.get_consensus(signals)

        # Get top N by backtest performance
        top = universe.get_top_strategies(n=10, metric='win_rate')
    """

    SPREAD_COST = 0.0169  # 1.69% round-trip spread

    def __init__(self):
        self.strategies: List[IndicatorSignal] = []
        self.strategies.extend(generate_single_indicator_strategies())
        self.strategies.extend(generate_combo_strategies())
        self.strategies.extend(generate_math_strategies())
        self.strategies.extend(generate_pattern_strategies())
        self.performance: Dict[str, Dict] = {}
        self._strategy_map: Dict[str, IndicatorSignal] = {
            s.name: s for s in self.strategies
        }
        logger.info("StrategyUniverse initialized with %d total strategies",
                     self.total_strategies)

    @property
    def total_strategies(self) -> int:
        return len(self.strategies)

    def get_strategy(self, name: str) -> Optional[IndicatorSignal]:
        """Lookup a strategy by name."""
        return self._strategy_map.get(name)

    def get_by_category(self, category: str) -> List[IndicatorSignal]:
        """Get all strategies in a given category."""
        return [s for s in self.strategies if s.category == category]

    def categories(self) -> Dict[str, int]:
        """Return category -> count mapping."""
        cats: Dict[str, int] = {}
        for s in self.strategies:
            cats[s.category] = cats.get(s.category, 0) + 1
        return cats

    def evaluate_all(self, closes: List[float], highs: List[float],
                     lows: List[float], volumes: List[float]) -> Dict[str, int]:
        """Run ALL strategies and return signals.

        Args:
            closes:  list of close prices
            highs:   list of high prices
            lows:    list of low prices
            volumes: list of volumes

        Returns:
            Dict mapping strategy_name -> signal (-1, 0, or +1)
        """
        results: Dict[str, int] = {}
        for strat in self.strategies:
            try:
                sig = strat.compute(closes, highs, lows, volumes)
                results[strat.name] = int(sig) if sig in (-1, 0, 1) else 0
            except Exception:
                results[strat.name] = 0
        return results

    def evaluate_category(self, category: str, closes: List[float],
                          highs: List[float], lows: List[float],
                          volumes: List[float]) -> Dict[str, int]:
        """Run only strategies in a specific category."""
        results: Dict[str, int] = {}
        for strat in self.strategies:
            if strat.category != category:
                continue
            try:
                sig = strat.compute(closes, highs, lows, volumes)
                results[strat.name] = int(sig) if sig in (-1, 0, 1) else 0
            except Exception:
                results[strat.name] = 0
        return results

    def get_consensus(self, signals: Dict[str, int],
                      threshold: float = 0.6) -> Tuple[str, float]:
        """Get weighted consensus from all strategies.

        Args:
            signals:   dict of strategy_name -> signal from evaluate_all
            threshold: fraction of strategies needed for directional signal

        Returns:
            (action, confidence) where action is "BUY", "SELL", or "NEUTRAL"
            and confidence is the fraction of agreeing strategies.
        """
        if not signals:
            return "NEUTRAL", 0.0

        # Weight by performance if available
        buy_weight = 0.0
        sell_weight = 0.0
        total_weight = 0.0

        for name, sig in signals.items():
            w = 1.0
            perf = self.performance.get(name)
            if perf and perf.get('trades', 0) >= 5:
                wr = perf.get('win_rate', 0.5)
                # Weight by win rate, clamped to [0.2, 3.0]
                w = max(0.2, min(3.0, wr / 0.5))
            total_weight += w
            if sig > 0:
                buy_weight += w
            elif sig < 0:
                sell_weight += w

        if total_weight == 0:
            return "NEUTRAL", 0.0

        buy_pct = buy_weight / total_weight
        sell_pct = sell_weight / total_weight

        if buy_pct > threshold:
            return "BUY", buy_pct
        elif sell_pct > threshold:
            return "SELL", sell_pct
        return "NEUTRAL", max(buy_pct, sell_pct)

    def get_category_consensus(self, signals: Dict[str, int]) -> Dict[str, Tuple[str, float]]:
        """Get consensus broken down by category."""
        by_cat: Dict[str, Dict[str, int]] = {}
        for name, sig in signals.items():
            strat = self._strategy_map.get(name)
            if strat:
                cat = strat.category
                if cat not in by_cat:
                    by_cat[cat] = {}
                by_cat[cat][name] = sig

        results: Dict[str, Tuple[str, float]] = {}
        for cat, cat_signals in by_cat.items():
            results[cat] = self.get_consensus(cat_signals)
        return results

    def update_performance(self, name: str, won: bool, pnl: float):
        """Track per-strategy performance for adaptive weighting.

        Args:
            name: strategy name
            won:  whether the trade was profitable after spread
            pnl:  profit/loss percentage
        """
        if name not in self.performance:
            self.performance[name] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'max_win': 0.0,
                'max_loss': 0.0,
                'pnl_after_spread': 0.0,
            }
        p = self.performance[name]
        p['trades'] += 1
        pnl_net = pnl - self.SPREAD_COST
        if won:
            p['wins'] += 1
        else:
            p['losses'] += 1
        p['total_pnl'] += pnl
        p['pnl_after_spread'] += pnl_net
        p['win_rate'] = p['wins'] / p['trades'] if p['trades'] > 0 else 0.0
        p['avg_pnl'] = p['total_pnl'] / p['trades']
        p['max_win'] = max(p['max_win'], pnl)
        p['max_loss'] = min(p['max_loss'], pnl)

    def get_top_strategies(self, n: int = 10,
                           metric: str = 'win_rate',
                           min_trades: int = 5) -> List[Tuple[str, Dict]]:
        """Get top N strategies ranked by a performance metric.

        Args:
            n:          number of strategies to return
            metric:     one of 'win_rate', 'avg_pnl', 'total_pnl', 'pnl_after_spread'
            min_trades: minimum number of trades to qualify

        Returns:
            List of (strategy_name, performance_dict) sorted descending
        """
        qualified = [
            (name, perf) for name, perf in self.performance.items()
            if perf.get('trades', 0) >= min_trades
        ]
        qualified.sort(key=lambda x: x[1].get(metric, 0), reverse=True)
        return qualified[:n]

    def get_worst_strategies(self, n: int = 10,
                             metric: str = 'pnl_after_spread',
                             min_trades: int = 5) -> List[Tuple[str, Dict]]:
        """Get bottom N strategies (candidates for removal)."""
        qualified = [
            (name, perf) for name, perf in self.performance.items()
            if perf.get('trades', 0) >= min_trades
        ]
        qualified.sort(key=lambda x: x[1].get(metric, 0))
        return qualified[:n]

    def summary(self) -> str:
        """Return a human-readable summary of the universe."""
        cats = self.categories()
        lines = [
            f"StrategyUniverse: {self.total_strategies} total strategies",
            "Categories:"
        ]
        for cat, count in sorted(cats.items()):
            lines.append(f"  {cat}: {count}")

        if self.performance:
            tracked = len(self.performance)
            profitable = sum(
                1 for p in self.performance.values()
                if p.get('pnl_after_spread', 0) > 0 and p.get('trades', 0) >= 5
            )
            lines.append(f"Performance tracked: {tracked} strategies")
            lines.append(f"Profitable (after {self.SPREAD_COST*100:.2f}% spread, >=5 trades): {profitable}")

        return "\n".join(lines)
