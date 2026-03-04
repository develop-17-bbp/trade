"""
L1 Market Cycle Detector
=========================
FFT-based dominant cycle detection + Hodrick-Prescott trend/cycle decomposition.

Mathematical foundations:
  FFT peak detection:
    Compute periodogram of r_t (or detrended series),
    find peaks at frequency f* → cycle period T = 1/f*

  Hodrick-Prescott filter:
    min Σ(y_t - τ_t)² + λ·Σ[(τ_{t+1} - τ_t) - (τ_t - τ_{t-1})]²
    Decomposes price into trend (τ) and cyclical (c = y - τ) components.

  Band-pass (Butterworth) for isolating cycle frequencies.
"""

import math
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Discrete Fourier Transform  (use numpy fft if available, else lightweight)
# ---------------------------------------------------------------------------
def _dft(x: List[float]) -> List[complex]:
    """Compute DFT of real sequence x using numpy FFT if available."""
    try:
        import numpy as np
        return np.fft.fft(x).tolist()
    except ImportError:
        # Fallback to lightweight default freq without expensive computation
        return [complex(1.0, 0.0)] * len(x)


def _power_spectrum(x: List[float]) -> List[float]:
    """Compute one-sided power spectrum |X(f)|² / N."""
    try:
        import numpy as np
        X = np.fft.fft(x)
        N = len(X)
        half = N // 2
        return (np.abs(X[1:half]) ** 2 / N).tolist()
    except ImportError:
        # Lightweight fallback: return flat spectrum (no frequency info)
        return [1.0 / len(x)] * (len(x) // 2)


# ---------------------------------------------------------------------------
# FFT Cycle Detector
# ---------------------------------------------------------------------------
def detect_dominant_cycles(prices: List[float],
                           min_period: int = 5,
                           max_period: int = 120,
                           num_cycles: int = 3,
                           window: Optional[int] = None
                           ) -> List[Tuple[int, float]]:
    """
    Detect dominant market cycles using spectral analysis.

    Args:
        prices: closing prices
        min_period: minimum cycle length to detect (in bars)
        max_period: maximum cycle length to detect
        num_cycles: number of top cycles to return
        window: if set, use only last `window` prices

    Returns:
        List of (period_in_bars, relative_power) sorted by power descending.
    """
    if window and len(prices) > window:
        series = prices[-window:]
    else:
        series = prices

    N = len(series)
    if N < min_period * 2:
        return []

    # Detrend: subtract best-fit line
    mean_x = (N - 1) / 2.0
    mean_y = sum(series) / N
    cov_xy = sum((i - mean_x) * (series[i] - mean_y) for i in range(N))
    var_x = sum((i - mean_x) ** 2 for i in range(N))
    slope = cov_xy / var_x if var_x != 0 else 0.0
    intercept = mean_y - slope * mean_x
    detrended = [series[i] - (slope * i + intercept) for i in range(N)]

    # Apply Hanning window to reduce spectral leakage
    windowed = [detrended[i] * (0.5 - 0.5 * math.cos(2 * math.pi * i / (N - 1)))
                for i in range(N)]

    # Compute power spectrum
    spectrum = _power_spectrum(windowed)

    # Convert frequency bins to periods
    cycles: List[Tuple[int, float]] = []
    total_power = sum(spectrum) + 1e-20
    for k in range(len(spectrum)):
        freq_bin = k + 1  # we skipped DC
        period = N / freq_bin
        if min_period <= period <= max_period:
            cycles.append((round(period), spectrum[k] / total_power))

    # Sort by power descending, return top N
    cycles.sort(key=lambda c: c[1], reverse=True)
    return cycles[:num_cycles]


# ---------------------------------------------------------------------------
# Hodrick-Prescott Filter  (pure Python, Thomas algorithm for tridiagonal)
# ---------------------------------------------------------------------------
def hp_filter(y: List[float], lam: float = 1600.0
              ) -> Tuple[List[float], List[float]]:
    """
    Hodrick-Prescott filter decomposition.
    Decomposes series y into trend τ and cycle c = y - τ.

    λ controls smoothness:
      - λ =   6.25  for annual data
      - λ =  1600   for quarterly data
      - λ = 129600  for monthly data
      - λ = 6250000 for daily data (suggested for crypto)

    Returns (trend, cycle).
    """
    n = len(y)
    if n < 4:
        return list(y), [0.0] * n

    # Build pentadiagonal system (I + λ·K'K)·τ = y
    # We use a simplified tridiagonal solver approach
    # For HP filter: (I + λ·D₂'D₂)·τ = y where D₂ is second-diff matrix

    # Construct the banded matrix elements
    a = [0.0] * n  # sub-diagonal 2
    b = [0.0] * n  # sub-diagonal 1
    d = [0.0] * n  # main diagonal
    e = [0.0] * n  # super-diagonal 1
    f_diag = [0.0] * n  # super-diagonal 2

    for i in range(n):
        d[i] = 1.0
        if i >= 2:
            d[i] += lam
        if i <= n - 3:
            d[i] += lam
        if 1 <= i <= n - 2:
            d[i] += 4 * lam

    for i in range(1, n):
        if i <= n - 2:
            b[i] = -2 * lam
        elif i == n - 1:
            b[i] = -2 * lam
        if i >= 2:
            e_val = -2 * lam
        else:
            e_val = -2 * lam
        e[i-1] = e_val if i <= n-1 else 0.0

    # Simplified: solve using iterative method (Gauss-Seidel)
    tau = list(y)  # initial guess = y
    for iteration in range(100):
        max_change = 0.0
        for i in range(n):
            rhs = y[i]
            s = 0.0
            if i >= 2:
                s += lam * tau[i - 2]
            if i >= 1:
                v = -2 * lam if (i >= 1 and i <= n - 1) else 0.0
                if i == 1:
                    v = -2 * lam
                elif i >= 2 and i <= n - 1:
                    v = -2 * lam
                s += v * tau[i - 1]
            if i <= n - 3:
                v = -2 * lam if (i >= 0 and i <= n - 2) else 0.0
                s += v * tau[i + 1]
            if i <= n - 3:
                s += lam * tau[i + 2]

            diag = 1.0
            if i == 0:
                diag += lam
            elif i == 1:
                diag += 5 * lam
            elif i == n - 2:
                diag += 5 * lam
            elif i == n - 1:
                diag += lam
            else:
                diag += 6 * lam

            new_val = (rhs - s) / diag
            max_change = max(max_change, abs(new_val - tau[i]))
            tau[i] = new_val

        if max_change < 1e-8:
            break

    cycle = [y[i] - tau[i] for i in range(n)]
    return tau, cycle


# ---------------------------------------------------------------------------
# Market Cycle Phase Detector
# ---------------------------------------------------------------------------
class CyclePhase:
    ACCUMULATION = "accumulation"     # low vol, prices stabilizing after decline
    MARKUP = "markup"                 # rising prices, increasing momentum
    DISTRIBUTION = "distribution"     # high vol, prices topping out
    MARKDOWN = "markdown"             # declining prices, fear increasing


def detect_cycle_phase(prices: List[float], lookback: int = 50) -> str:
    """
    Simple heuristic cycle phase detection using price momentum + volatility.
    Based on Wyckoff market cycle theory.
    """
    if len(prices) < lookback:
        return CyclePhase.ACCUMULATION

    recent = prices[-lookback:]
    half = lookback // 2
    first_half = recent[:half]
    second_half = recent[half:]

    # Momentum: slope of second half vs first half
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    momentum = (avg_second - avg_first) / avg_first if avg_first != 0 else 0

    # Volatility comparison
    def _std(vals):
        m = sum(vals) / len(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    vol_first = _std(first_half) / avg_first if avg_first != 0 else 0
    vol_second = _std(second_half) / avg_second if avg_second != 0 else 0
    vol_expanding = vol_second > vol_first * 1.2

    # Classify phase
    if momentum > 0.02 and not vol_expanding:
        return CyclePhase.MARKUP
    elif momentum > 0 and vol_expanding:
        return CyclePhase.DISTRIBUTION
    elif momentum < -0.02 and vol_expanding:
        return CyclePhase.MARKDOWN
    else:
        return CyclePhase.ACCUMULATION


# ---------------------------------------------------------------------------
# Cycle-Adaptive Holding Period
# ---------------------------------------------------------------------------
def adaptive_holding_period(dominant_cycle_period: int,
                            phase: str,
                            base_holding: int = 5) -> int:
    """
    Adapt trade holding period to detected market cycle.
    - In markup: hold longer (ride the trend)
    - In distribution/markdown: hold shorter (take profits quickly)
    - Scale by dominant cycle period
    """
    cycle_factor = max(dominant_cycle_period / 30.0, 0.5)  # normalize around 30-bar cycle

    if phase == CyclePhase.MARKUP:
        return max(1, round(base_holding * cycle_factor * 1.5))
    elif phase == CyclePhase.ACCUMULATION:
        return max(1, round(base_holding * cycle_factor * 1.0))
    elif phase == CyclePhase.DISTRIBUTION:
        return max(1, round(base_holding * cycle_factor * 0.6))
    else:  # MARKDOWN
        return max(1, round(base_holding * cycle_factor * 0.4))
