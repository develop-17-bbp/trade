"""
L1 Quantitative Engine — Technical Indicators Module
=====================================================
Implements all core technical indicators with pure-Python math.
Each function operates on List[float] and returns List[float].

Equations implemented:
  SMA_t(n) = (1/n) * Σ P_{t-i}  for i=0..n-1
  EMA_t    = α·P_t + (1-α)·EMA_{t-1},  α = 2/(n+1)
  Institutions: 50 / 100 / 200 periods
  RSI      = 100 - 100/(1+RS),  RS = AvgGain/AvgLoss
  MACD     = EMA(12) - EMA(26),  Signal = EMA(MACD, 9)
  BB       = SMA ± k·σ  (Bollinger Bands)
  ATR_t    = EMA of True Range
  Stoch %K = (C - L_n) / (H_n - L_n) * 100
  VWAP     = Σ(Price * Volume) / Σ(Volume)
  OBV      = cumulative signed volume
  ADX      = smoothed DX from +DI / -DI
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Simple Moving Average
# ---------------------------------------------------------------------------
def sma(values: List[float], period: int) -> List[float]:
    """SMA_t(n) = (1/n) * Σ_{i=0}^{n-1} P_{t-i}"""
    if period <= 0:
        raise ValueError("period must be > 0")
    out: List[float] = []
    for i in range(len(values)):
        if i + 1 < period:
            out.append(float('nan'))
        else:
            window = values[i + 1 - period: i + 1]
            out.append(sum(window) / period)
    return out


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------
def ema(values: List[float], period: int) -> List[float]:
    """EMA_t = α·P_t + (1-α)·EMA_{t-1},  α = 2/(n+1)"""
    if period <= 0:
        raise ValueError("period must be > 0")
    out: List[float] = []
    k = 2.0 / (period + 1)
    prev: Optional[float] = None
    for v in values:
        if prev is None:
            prev = v
            out.append(prev)
        else:
            prev = (v - prev) * k + prev
            out.append(prev)
    return out


# ---------------------------------------------------------------------------
# Relative Strength Index
# ---------------------------------------------------------------------------
def rsi(values: List[float], period: int = 14) -> List[float]:
    """RSI = 100 - 100/(1+RS) where RS = AvgGain / AvgLoss (Wilder smoothing)"""
    if len(values) < 2:
        return [float('nan')] * len(values)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    out: List[float] = []
    avg_gain: Optional[float] = None
    avg_loss: Optional[float] = None
    for i in range(len(values)):
        if i < period:
            out.append(float('nan'))
            continue
        if avg_gain is None:
            avg_gain = sum(gains[1:period + 1]) / period
            avg_loss = sum(losses[1:period + 1]) / period
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out.append(100.0)
        else:
            rs = avg_gain / avg_loss
            out.append(100.0 - (100.0 / (1.0 + rs)))
    return out


# ---------------------------------------------------------------------------
# MACD  (Moving Average Convergence/Divergence)
# ---------------------------------------------------------------------------
def macd(values: List[float],
         fast: int = 12, slow: int = 26, signal_period: int = 9
         ) -> Tuple[List[float], List[float], List[float]]:
    """
    MACD line   = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD, signal_period)
    Histogram   = MACD - Signal
    Returns (macd_line, signal_line, histogram)
    """
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line: List[float] = []
    for f_val, s_val in zip(ema_fast, ema_slow):
        macd_line.append(f_val - s_val)
    signal_line = ema(macd_line, signal_period)
    histogram = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------
def bollinger_bands(values: List[float], period: int = 20, num_std: float = 2.0
                    ) -> Tuple[List[float], List[float], List[float]]:
    """
    Middle = SMA(period)
    Upper  = SMA + num_std * σ
    Lower  = SMA - num_std * σ
    Returns (upper, middle, lower)
    """
    middle = sma(values, period)
    upper: List[float] = []
    lower: List[float] = []
    for i in range(len(values)):
        if i + 1 < period:
            upper.append(float('nan'))
            lower.append(float('nan'))
        else:
            window = values[i + 1 - period: i + 1]
            mean = middle[i]
            variance = sum((x - mean) ** 2 for x in window) / period
            std = math.sqrt(variance)
            upper.append(mean + num_std * std)
            lower.append(mean - num_std * std)
    return upper, middle, lower


# ---------------------------------------------------------------------------
# Average True Range  (volatility measure)
# ---------------------------------------------------------------------------
def true_range(highs: List[float], lows: List[float], closes: List[float]
               ) -> List[float]:
    """TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)"""
    n = len(closes)
    tr: List[float] = []
    for i in range(n):
        hl = highs[i] - lows[i]
        if i == 0:
            tr.append(hl)
        else:
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr.append(max(hl, hc, lc))
    return tr


def atr(highs: List[float], lows: List[float], closes: List[float],
        period: int = 14) -> List[float]:
    """ATR = EMA(True Range, period)  — or Wilder smoothing"""
    tr_vals = true_range(highs, lows, closes)
    return ema(tr_vals, period)


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------
def stochastic(highs: List[float], lows: List[float], closes: List[float],
               k_period: int = 14, d_period: int = 3
               ) -> Tuple[List[float], List[float]]:
    """
    %K = (C - L_n) / (H_n - L_n) * 100
    %D = SMA(%K, d_period)
    """
    n = len(closes)
    k_vals: List[float] = []
    for i in range(n):
        if i + 1 < k_period:
            k_vals.append(float('nan'))
        else:
            h_max = max(highs[i + 1 - k_period: i + 1])
            l_min = min(lows[i + 1 - k_period: i + 1])
            if h_max == l_min:
                k_vals.append(50.0)
            else:
                k_vals.append(((closes[i] - l_min) / (h_max - l_min)) * 100.0)
    d_vals = sma(k_vals, d_period)
    return k_vals, d_vals


# ---------------------------------------------------------------------------
# Volume-Weighted Average Price
# ---------------------------------------------------------------------------
def vwap(closes: List[float], volumes: List[float]) -> List[float]:
    """VWAP = Σ(P_i * V_i) / Σ(V_i)  (cumulative intraday)"""
    cum_pv = 0.0
    cum_v = 0.0
    out: List[float] = []
    for p, v in zip(closes, volumes):
        cum_pv += p * v
        cum_v += v
        out.append(cum_pv / cum_v if cum_v > 0 else p)
    return out


# ---------------------------------------------------------------------------
# On-Balance Volume
# ---------------------------------------------------------------------------
def obv(closes: List[float], volumes: List[float]) -> List[float]:
    """OBV: cumulative volume with sign determined by price direction"""
    out: List[float] = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            out.append(out[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            out.append(out[-1] - volumes[i])
        else:
            out.append(out[-1])
    return out


# ---------------------------------------------------------------------------
# ADX (Average Directional Index)
# ---------------------------------------------------------------------------
def adx(highs: List[float], lows: List[float], closes: List[float],
        period: int = 14) -> Tuple[List[float], List[float], List[float]]:
    """
    +DM = H_t - H_{t-1}  if positive and > |L_{t-1} - L_t|
    -DM = L_{t-1} - L_t   if positive and > H_t - H_{t-1}
    +DI = EMA(+DM) / ATR * 100
    -DI = EMA(-DM) / ATR * 100
    DX  = |+DI - -DI| / (+DI + -DI) * 100
    ADX = EMA(DX, period)
    Returns (adx_line, plus_di, minus_di)
    """
    n = len(closes)
    plus_dm: List[float] = [0.0]
    minus_dm: List[float] = [0.0]
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
    atr_vals = atr(highs, lows, closes, period)
    smooth_plus = ema(plus_dm, period)
    smooth_minus = ema(minus_dm, period)
    plus_di: List[float] = []
    minus_di: List[float] = []
    dx: List[float] = []
    for i in range(n):
        a = atr_vals[i] if i < len(atr_vals) else 1.0
        if a == 0:
            a = 1e-10
        pdi = (smooth_plus[i] / a) * 100.0
        mdi = (smooth_minus[i] / a) * 100.0
        plus_di.append(pdi)
        minus_di.append(mdi)
        denom = pdi + mdi
        if denom == 0:
            dx.append(0.0)
        else:
            dx.append(abs(pdi - mdi) / denom * 100.0)
    adx_line = ema(dx, period)
    return adx_line, plus_di, minus_di


# ---------------------------------------------------------------------------
# Bollinger Band Width  (volatility expansion/contraction)
# ---------------------------------------------------------------------------
def bb_width(values: List[float], period: int = 20, num_std: float = 2.0
             ) -> List[float]:
    """BB Width = (Upper - Lower) / Middle — measures volatility squeeze"""
    upper, middle, lower = bollinger_bands(values, period, num_std)
    out: List[float] = []
    for u, m, l in zip(upper, middle, lower):
        if math.isnan(u) or m == 0:
            out.append(float('nan'))
        else:
            out.append((u - l) / m)
    return out


# ---------------------------------------------------------------------------
# Rate of Change (momentum)
# ---------------------------------------------------------------------------
def roc(values: List[float], period: int = 12) -> List[float]:
    """ROC = (P_t - P_{t-n}) / P_{t-n} * 100"""
    out: List[float] = []
    for i in range(len(values)):
        if i < period:
            out.append(float('nan'))
        else:
            prev = values[i - period]
            if prev == 0:
                out.append(0.0)
            else:
                out.append(((values[i] - prev) / prev) * 100.0)
    return out


# ---------------------------------------------------------------------------
# Williams %R  (overbought/oversold oscillator)
# ---------------------------------------------------------------------------
def williams_r(highs: List[float], lows: List[float], closes: List[float],
               period: int = 14) -> List[float]:
    """%R = (H_n - C) / (H_n - L_n) * -100"""
    out: List[float] = []
    for i in range(len(closes)):
        if i + 1 < period:
            out.append(float('nan'))
        else:
            h_max = max(highs[i + 1 - period: i + 1])
            l_min = min(lows[i + 1 - period: i + 1])
            if h_max == l_min:
                out.append(-50.0)
            else:
                out.append(((h_max - closes[i]) / (h_max - l_min)) * -100.0)
    return out


# ---------------------------------------------------------------------------
# Bulk Indicator Generator (120+ features)
# ---------------------------------------------------------------------------

def bulk_indicators(closes: List[float],
                    highs: Optional[List[float]] = None,
                    lows: Optional[List[float]] = None,
                    volumes: Optional[List[float]] = None
                    ) -> Dict[str, List[float]]:
    """Compute a large suite of features by varying periods and indicator types.

    Returns a dict mapping feature name -> series list.  By default this routine
    will generate well over 120 distinct indicators (SMA, EMA, RSI, ROC, BB width
    across multiple lookback periods) suitable for machine learning feature
    vectors.
    """
    n = len(closes)
    if highs is None:
        highs = closes
    if lows is None:
        lows = closes
    if volumes is None:
        volumes = [1.0] * n

    features: Dict[str, List[float]] = {}
    periods = list(range(2, 32))  # 2..31 inclusive gives 30 periods

    for p in periods:
        features[f'sma_{p}'] = sma(closes, p)
        features[f'ema_{p}'] = ema(closes, p)
        features[f'rsi_{p}'] = rsi(closes, p)
        features[f'roc_{p}'] = roc(closes, p)
        # BB width uses existing helper
        features[f'bb_width_{p}'] = bb_width(closes, p)
        features[f'williams_r_{p}'] = williams_r(highs, lows, closes, p)
        k_vals, d_vals = stochastic(highs, lows, closes, p, max(3, p // 4))
        features[f'stoch_k_{p}'] = k_vals
        features[f'stoch_d_{p}'] = d_vals
        # adaptive moving average
        features[f'kama_{p}'] = kama(closes, p)
    # additional fixed indicators
    features['obv'] = obv(closes, volumes)
    features['vwap'] = vwap(closes, volumes)
    # OU mean-reversion signal using popular windows
    for w in [10, 20, 30]:
        features[f'ou_{w}'] = ou_signal(closes, window=w)
    # wavelet cycle strength (single series)
    features['wavelet_strength'] = wavelet_cycle_strength(closes)
    # additional fixed indicators
    features['obv'] = obv(closes, volumes)
    features['vwap'] = vwap(closes, volumes)
    # ADX with multiple periods
    for p in [10, 14, 20, 30]:
        adx_line, plus_di, minus_di = adx(highs, lows, closes, p)
        features[f'adx_{p}'] = adx_line
        features[f'plus_di_{p}'] = plus_di
        features[f'minus_di_{p}'] = minus_di
    return features


# ---------------------------------------------------------------------------
# Kaufman Adaptive Moving Average (KAMA)
# ---------------------------------------------------------------------------

def kama(values: List[float], period: int) -> List[float]:
    """Kaufman's Adaptive Moving Average.

    KAMA adapts its smoothing constant based on market efficiency.  It
    reacts quickly during trends and slows during sideways chop.  Implementation
    follows Perry Kaufman's formula.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    n = len(values)
    out: List[float] = [float('nan')] * n
    if n < period:
        return out

    # efficiency ratio: ER = |P_t - P_{t-n}| / sum_{i=1}^n |P_i - P_{i-1}|
    er: List[float] = [0.0] * n
    for i in range(period, n):
        change = abs(values[i] - values[i - period])
        volatility = sum(abs(values[j] - values[j - 1]) for j in range(i - period + 1, i + 1))
        er[i] = 0.0 if volatility == 0 else change / volatility

    # smoothing constants
    sc_fast = 2.0 / (2 + 1)
    sc_slow = 2.0 / (30 + 1)

    prev = values[period - 1]
    out[period - 1] = prev
    for i in range(period, n):
        sc = (er[i] * (sc_fast - sc_slow) + sc_slow) ** 2
        prev = prev + sc * (values[i] - prev)
        out[i] = prev
    return out


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck mean-reversion signal
# ---------------------------------------------------------------------------

def ou_signal(values: List[float], window: int = 20) -> List[float]:
    """Simple OU-style mean-reversion z-score: -(X_t - μ_t)/σ_t over rolling window.

    This serves as a derisking indicator during sideways markets; large
    positive values (price > mean) generate negative signal and vice versa.
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    n = len(values)
    out: List[float] = []
    for i in range(n):
        if i + 1 < window:
            out.append(0.0)
        else:
            window_vals = values[i + 1 - window: i + 1]
            mean = sum(window_vals) / window
            variance = sum((x - mean) ** 2 for x in window_vals) / window
            std = math.sqrt(variance) if variance > 0 else 0.0
            z = 0.0 if std == 0 else -(values[i] - mean) / std
            out.append(z)
    return out


# ---------------------------------------------------------------------------
# Wavelet cycle strength indicator
# ---------------------------------------------------------------------------

try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False


def wavelet_cycle_strength(values: List[float], wavelet: str = 'db4', level: int = 3) -> List[float]:
    """Estimate cycle strength via discrete wavelet transform detail energy.

    Returns a series where higher values indicate stronger cyclical (high-frequency)
    activity.  If `pywt` is not installed the function returns zeros.
    """
    n = len(values)
    if not WAVELET_AVAILABLE or n == 0:
        return [0.0] * n

    # perform a single decomposition
    coeffs = pywt.wavedec(values, wavelet, level=level)
    # detail coefficients at levels 1..level reside in coeffs[1:]
    energies = [sum(c ** 2 for c in detail) for detail in coeffs[1:]]
    total_energy = sum(energies) if energies else 1.0
    # normalize per-level energy by total and broadcast to length n
    strength = total_energy / (len(energies) or 1)
    return [strength] * n


# ---------------------------------------------------------------------------
# Bulk Indicator Generator (120+ features)

# ---------------------------------------------------------------------------
# Chaikin Money Flow (CMF)
# ---------------------------------------------------------------------------
def chaikin_money_flow(highs: List[float], lows: List[float], closes: List[float], 
                       volumes: List[float], period: int = 20) -> List[float]:
    """
    MFM = [(C - L) - (H - C)] / (H - L)
    MFV = MFM * V
    CMF = sum(MFV, n) / sum(V, n)
    """
    n = len(closes)
    arr_h = np.asarray(highs)
    arr_l = np.asarray(lows)
    arr_c = np.asarray(closes)
    arr_v = np.asarray(volumes)
    
    hl = arr_h - arr_l
    mfm = np.where(hl == 0, 0.0, ((arr_c - arr_l) - (arr_h - arr_c)) / hl)
    mfv = mfm * arr_v
    
    out: List[float] = [float('nan')] * n
    for i in range(period - 1, n):
        sum_mfv = np.sum(mfv[i + 1 - period: i + 1])
        sum_vol = np.sum(arr_v[i + 1 - period: i + 1])
        out[i] = float(sum_mfv / sum_vol) if sum_vol != 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Money Flow Index (MFI)
# ---------------------------------------------------------------------------
def mfi(highs: List[float], lows: List[float], closes: List[float],
        volumes: List[float], period: int = 14) -> List[float]:
    """Typical Price = (H+L+C)/3. Like RSI but volume-weighted."""
    arr_h = np.asarray(highs)
    arr_l = np.asarray(lows)
    arr_c = np.asarray(closes)
    arr_v = np.asarray(volumes)
    
    tp = (arr_h + arr_l + arr_c) / 3.0
    n = len(closes)
    pos_mf = np.zeros(n)
    neg_mf = np.zeros(n)
    
    for i in range(1, n):
        mf = tp[i] * arr_v[i]
        if tp[i] > tp[i-1]:
            pos_mf[i] = mf
        elif tp[i] < tp[i-1]:
            neg_mf[i] = mf
            
    out: List[float] = [float('nan')] * n
    for i in range(period - 1, n):
        s_pos = np.sum(pos_mf[i + 1 - period: i + 1])
        s_neg = np.sum(neg_mf[i + 1 - period: i + 1])
        if s_neg == 0:
            out[i] = 100.0
        else:
            mfr = s_pos / s_neg
            out[i] = float(100.0 - (100.0 / (1.0 + mfr)))
    return out


# ---------------------------------------------------------------------------
# SuperTrend
# ---------------------------------------------------------------------------
def supertrend(highs: List[float], lows: List[float], closes: List[float],
               period: int = 7, multiplier: float = 3.0) -> Tuple[List[float], List[int]]:
    """
    Returns (trend_line, side) where side is 1 for long, -1 for short.
    """
    atr_vals = atr(highs, lows, closes, period)
    n = len(closes)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    trend = [1] * n
    st = [0.0] * n
    
    for i in range(n):
        if math.isnan(atr_vals[i]): continue
        
        mid = (highs[i] + lows[i]) / 2.0
        ub = mid + multiplier * atr_vals[i]
        lb = mid - multiplier * atr_vals[i]
        
        if i == 0:
            upper_band[i] = ub
            lower_band[i] = lb
        else:
            upper_band[i] = ub if ub < upper_band[i-1] or closes[i-1] > upper_band[i-1] else upper_band[i-1]
            lower_band[i] = lb if lb > lower_band[i-1] or closes[i-1] < lower_band[i-1] else lower_band[i-1]
            
        if i > 0:
            if trend[i-1] == 1:
                trend[i] = 1 if closes[i] > lower_band[i] else -1
            else:
                trend[i] = -1 if closes[i] < upper_band[i] else 1
                
        st[i] = lower_band[i] if trend[i] == 1 else upper_band[i]
        
    return st, trend


# ---------------------------------------------------------------------------
# Parabolic SAR
# ---------------------------------------------------------------------------
def parabolic_sar(highs: List[float], lows: List[float], closes: List[float],
                  step: float = 0.02, max_step: float = 0.2) -> List[float]:
    """Simplified PSAR implementation."""
    n = len(closes)
    if n < 2: return closes
    sar = [0.0] * n
    trend = 1 # 1 for up, -1 for down
    ep = highs[0] # extreme point
    af = step # acceleration factor
    sar[0] = lows[0]
    
    for i in range(1, n):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if trend == 1:
            if lows[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = lows[i]
                af = step
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + step, max_step)
                if i > 1:
                    sar[i] = min(sar[i], lows[i-1], lows[i-2] if i > 2 else lows[i-1])
        else:
            if highs[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = highs[i]
                af = step
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + step, max_step)
                if i > 1:
                    sar[i] = max(sar[i], highs[i-1], highs[i-2] if i > 2 else highs[i-1])
    return sar


# ---------------------------------------------------------------------------
# Kalman Filter (Trend Smoothing)
# ---------------------------------------------------------------------------
def kalman_filter(values: List[float], process_variance: float = 1e-5, estimated_measurement_variance: float = 0.01) -> List[float]:
    """
    Applies a basic 1D Kalman filter to smooth price data.
    Excellent for removing noise while preserving trend changes.
    """
    n = len(values)
    out: List[float] = [0.0] * n
    if n == 0: return out
    
    post_estimate = values[0]
    post_error_est = 1.0
    
    for i in range(n):
        # Time update
        priori_estimate = post_estimate
        priori_error_est = post_error_est + process_variance
        
        # Measurement update
        blending_factor = priori_error_est / (priori_error_est + estimated_measurement_variance)
        post_estimate = priori_estimate + blending_factor * (values[i] - priori_estimate)
        post_error_est = (1 - blending_factor) * priori_error_est
        out[i] = float(post_estimate)
        
    return out


# ---------------------------------------------------------------------------
# Hilbert Transform (Cycle Phase)
# ---------------------------------------------------------------------------
def hilbert_transform(values: List[float]) -> Tuple[List[float], List[float]]:
    """
    Approximation of the Hilbert Transform for Dominant Cycle Phase detection.
    Returns (In-Phase, Quadrature) components.
    """
    n = len(values)
    arr = np.asarray(values)
    # Simple 7-tap Hilbert filter
    hilbert_filter = np.array([0.1, 0.2, 0.4, 0.0, -0.4, -0.2, -0.1]) # Simplified
    
    q = np.zeros(n)
    for i in range(7, n):
        q[i] = np.sum(arr[i-7:i] * hilbert_filter)
        
    return values, q.tolist()
def choppiness_index(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    """
    CHOP = 100 * LOG10( SUM(ATR(1), n) / (MaxHi(n) - MinLo(n)) ) / LOG10(n)
    Returns values 0-100. >61.8 = choppy, <38.2 = trending.
    """
    atr1 = atr(highs, lows, closes, 1)
    n = len(closes)
    arr_h = np.asarray(highs)
    arr_l = np.asarray(lows)
    arr_atr = np.asarray(atr1)
    
    out: List[float] = [float('nan')] * n
    for i in range(period - 1, n):
        sum_atr = np.sum(arr_atr[i + 1 - period: i + 1])
        max_hi = np.max(arr_h[i + 1 - period: i + 1])
        min_lo = np.min(arr_l[i + 1 - period: i + 1])
        
        range_val = max_hi - min_lo
        if range_val == 0:
            out[i] = 50.0
        else:
            val = float(sum_atr / range_val)
            if val <= 0:
                out[i] = 0.0
            else:
                chop = 100 * math.log10(val) / math.log10(period)
                out[i] = float(max(0.0, min(100.0, chop)))
    return out

# ---------------------------------------------------------------------------
# Volume Delta (Aggressor Volume)
# ---------------------------------------------------------------------------
def volume_delta(opens, closes, volumes):
    """Simplified Volume Delta: Positive = Buy Bias, Negative = Sell Bias."""
    n = len(closes)
    out = []
    for i in range(n):
        try:
            dr = abs(closes[i] - opens[i]) or 1e-10
            bias = (closes[i] - opens[i]) / dr
            out.append(float(bias * volumes[i]))
        except:
            out.append(0.0)
    return out

# ---------------------------------------------------------------------------
# Liquidity Sweep Detection
# ---------------------------------------------------------------------------
def liquidity_sweep(highs, lows, closes, lookback=20):
    """Detects stop-hunting sweeps below Low/above High."""
    n = len(closes)
    out = [0.0] * n
    for i in range(lookback, n):
        rh, rl = max(highs[i-lookback:i]), min(lows[i-lookback:i])
        if lows[i] < rl and closes[i] > rl: out[i] = 1.0 # Bullish
        elif highs[i] > rh and closes[i] < rh: out[i] = -1.0 # Bearish
    return out

# ---------------------------------------------------------------------------
# VWAP Deviation
# ---------------------------------------------------------------------------
def vwap_deviation(closes, vwap_vals, period=20):
    """Z-score of Price relative to VWAP."""
    n = len(closes)
    out = [0.0] * n
    for i in range(period, n):
        try:
            import numpy as np
            devs = [c - v for c, v in zip(closes[i+1-period:i+1], vwap_vals[i+1-period:i+1])]
            std = float(np.std(devs)) or 1e-10
            out[i] = float((closes[i] - vwap_vals[i]) / std)
        except:
            out[i] = 0.0
    return out

def vpin(opens: List[float], closes: List[float], volumes: List[float], bucket_size: int = 50) -> List[float]:
    """
    Volume-synchronous Probability of Informed Trading (VPIN).
    A core HFT metric for detecting Toxic Order Flow (Adverse Selection).
    
    In a professional context: 
    VPIN = Σ|Buy_i - Sell_i| / (V * bucket_count)
    """
    n = len(closes)
    vpin_values = [0.0] * n
    
    # Calculate per-bar imbalance
    imbalances = []
    for i in range(n):
        try:
            dr = abs(closes[i] - opens[i]) or 1e-10
            bias = (closes[i] - opens[i]) / dr # 1 if buy, -1 if sell
            imbalances.append(abs(bias * volumes[i])) # Absolute imbalance per bar
        except:
            imbalances.append(0.0)
            
    # Calculate rolling VPIN
    for i in range(bucket_size, n):
        try:
            bucket_imbalance = sum(imbalances[i-bucket_size:i])
            bucket_volume = sum(volumes[i-bucket_size:i]) or 1e-10
            vpin_values[i] = float(bucket_imbalance / bucket_volume)
        except:
            vpin_values[i] = 0.0
            
    return vpin_values


# Optional: expose in __all__
__all__ = ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands',
           'true_range', 'atr', 'stochastic', 'vwap', 'obv', 'adx',
           'bb_width', 'roc', 'williams_r', 'bulk_indicators',
           'kama', 'ou_signal', 'wavelet_cycle_strength',
           'volume_delta', 'liquidity_sweep', 'vwap_deviation', 'vpin']
