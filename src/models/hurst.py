"""
Hurst Exponent Estimator
=========================
Computes the Hurst exponent H via the Rescaled Range (R/S) method.

  H > 0.55  → Trending (persistent)     → favor trend-following strategies
  H < 0.45  → Mean-reverting (anti-persistent) → favor mean-reversion strategies
  H ≈ 0.50  → Random walk               → favor scalping or stay flat

Mathematical basis:
  E[R(n)/S(n)] = C · n^H  as n → ∞
  where R(n) = range of cumulative deviations, S(n) = standard deviation
  Estimate H from log-log regression: log(R/S) = H·log(n) + log(C)

Usage:
    from src.models.hurst import HurstExponent
    h = HurstExponent()
    result = h.compute(prices)
    # result = {'hurst': 0.62, 'regime': 'trending', 'confidence': 0.94}
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class HurstExponent:
    """Rescaled Range (R/S) Hurst exponent estimator."""

    TRENDING_THRESHOLD = 0.55
    MEAN_REVERT_THRESHOLD = 0.45

    def __init__(self, min_window: int = 20, max_window: Optional[int] = None):
        """
        Args:
            min_window: Minimum sub-range size for R/S calculation
            max_window: Maximum sub-range size (default: len(series) // 2)
        """
        self.min_window = min_window
        self.max_window = max_window

    @staticmethod
    def _rs_for_window(series: np.ndarray, window: int) -> float:
        """Compute average R/S statistic for a given window size."""
        n = len(series)
        num_chunks = n // window
        if num_chunks == 0:
            return np.nan

        rs_values = []
        for i in range(num_chunks):
            chunk = series[i * window:(i + 1) * window]
            mean = np.mean(chunk)
            deviations = chunk - mean
            cumulative = np.cumsum(deviations)

            r = np.max(cumulative) - np.min(cumulative)  # Range
            s = np.std(chunk, ddof=1)  # Std dev

            if s > 1e-12:
                rs_values.append(r / s)

        return np.mean(rs_values) if rs_values else np.nan

    def compute(self, prices: np.ndarray, window: int = 200) -> Dict:
        """
        Compute Hurst exponent from price series.

        Args:
            prices: Array of prices (at least 100 values recommended)
            window: Lookback window (uses last `window` prices)

        Returns:
            Dict with keys: hurst, regime, confidence, r_squared
        """
        if len(prices) < self.min_window * 2:
            return {'hurst': 0.5, 'regime': 'random', 'confidence': 0.0, 'r_squared': 0.0}

        series = np.asarray(prices[-window:], dtype=float)
        # Use log returns for stationarity
        log_returns = np.diff(np.log(series + 1e-12))

        if len(log_returns) < self.min_window * 2:
            return {'hurst': 0.5, 'regime': 'random', 'confidence': 0.0, 'r_squared': 0.0}

        max_w = self.max_window or (len(log_returns) // 2)
        max_w = min(max_w, len(log_returns) // 2)

        # Generate window sizes (log-spaced for better regression)
        window_sizes = []
        w = self.min_window
        while w <= max_w:
            window_sizes.append(w)
            w = int(w * 1.4)  # ~40% steps
            if w == window_sizes[-1]:
                w += 1

        if len(window_sizes) < 3:
            return {'hurst': 0.5, 'regime': 'random', 'confidence': 0.0, 'r_squared': 0.0}

        # Compute R/S for each window size
        log_n = []
        log_rs = []
        for w in window_sizes:
            rs = self._rs_for_window(log_returns, w)
            if not np.isnan(rs) and rs > 0:
                log_n.append(np.log(w))
                log_rs.append(np.log(rs))

        if len(log_n) < 3:
            return {'hurst': 0.5, 'regime': 'random', 'confidence': 0.0, 'r_squared': 0.0}

        # OLS regression: log(R/S) = H * log(n) + c
        log_n = np.array(log_n)
        log_rs = np.array(log_rs)

        n_pts = len(log_n)
        x_mean = np.mean(log_n)
        y_mean = np.mean(log_rs)

        ss_xx = np.sum((log_n - x_mean) ** 2)
        ss_xy = np.sum((log_n - x_mean) * (log_rs - y_mean))

        if ss_xx < 1e-12:
            return {'hurst': 0.5, 'regime': 'random', 'confidence': 0.0, 'r_squared': 0.0}

        hurst = ss_xy / ss_xx  # Slope = H

        # R-squared for confidence
        y_pred = hurst * log_n + (y_mean - hurst * x_mean)
        ss_res = np.sum((log_rs - y_pred) ** 2)
        ss_tot = np.sum((log_rs - y_mean) ** 2)
        r_squared = 1.0 - (ss_res / (ss_tot + 1e-12))
        r_squared = max(0.0, min(1.0, r_squared))

        # Clamp hurst to [0, 1]
        hurst = float(np.clip(hurst, 0.0, 1.0))

        # Classify regime
        if hurst > self.TRENDING_THRESHOLD:
            regime = 'trending'
        elif hurst < self.MEAN_REVERT_THRESHOLD:
            regime = 'mean_reverting'
        else:
            regime = 'random'

        return {
            'hurst': hurst,
            'regime': regime,
            'confidence': r_squared,
            'r_squared': r_squared,
        }

    def rolling(self, prices: np.ndarray, window: int = 200,
                step: int = 1) -> List[Dict]:
        """
        Compute rolling Hurst exponent.

        Args:
            prices: Full price series
            window: Lookback for each computation
            step: Step size between computations

        Returns:
            List of result dicts (aligned to end of each window)
        """
        results = []
        for i in range(window, len(prices) + 1, step):
            segment = prices[i - window:i]
            result = self.compute(segment, window)
            results.append(result)
        return results

    def regime_encoded(self, regime: str) -> int:
        """Encode regime as integer for ML features."""
        return {'mean_reverting': 0, 'random': 1, 'trending': 2}.get(regime, 1)
