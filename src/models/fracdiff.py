"""
Fractional Differencing — Memory-Preserving Stationarity Transform
=====================================================================
From Marcos López de Prado's "Advances in Financial Machine Learning" (Ch. 5).

Problem: Standard differencing (d=1) makes series stationary but destroys memory.
Solution: Fractional differencing (0 < d < 1) achieves stationarity while
preserving long-range dependence needed for ML prediction.

Method:
  (1-B)^d = Σ_{k=0}^{∞} C(d,k) · (-B)^k
  where C(d,k) = Π_{i=0}^{k-1} (d-i)/(k!) = (-1)^k · Γ(d+1) / (Γ(k+1)·Γ(d-k+1))

  Truncated at weight threshold τ (default 1e-5) for finite computation.

Usage:
    from src.models.fracdiff import FractionalDiff
    fd = FractionalDiff()
    d_opt = fd.find_min_d(prices)  # Minimum d for stationarity
    stationary = fd.frac_diff(prices, d=d_opt)
"""

import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class FractionalDiff:
    """
    Fractional differencing with automatic d-selection via ADF test.
    """

    def __init__(self, threshold: float = 1e-5, max_lag: Optional[int] = None):
        """
        Args:
            threshold: Weight truncation threshold (smaller = more memory preserved)
            max_lag: Maximum lag for weights (None = auto from threshold)
        """
        self.threshold = threshold
        self.max_lag = max_lag

    @staticmethod
    def _get_weights(d: float, threshold: float = 1e-5, max_lag: int = 500) -> np.ndarray:
        """
        Compute fractional differencing weights.

        w_k = (-1)^k · C(d, k) = w_{k-1} · (k - 1 - d) / k

        Recursive formula avoids factorial overflow.
        """
        weights = [1.0]
        k = 1
        while k < max_lag:
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < threshold:
                break
            weights.append(w)
            k += 1
        return np.array(weights[::-1])  # Reverse for convolution

    def frac_diff(self, series: np.ndarray, d: float) -> np.ndarray:
        """
        Apply fractional differencing of order d to a series.

        Args:
            series: Price or log-price series
            d: Differencing order (0 < d < 1 typically)

        Returns:
            Fractionally differenced series (shorter by len(weights)-1)
        """
        x = np.asarray(series, dtype=float)
        max_lag = self.max_lag or min(len(x), 500)
        weights = self._get_weights(d, self.threshold, max_lag)
        width = len(weights)

        if width > len(x):
            logger.warning(f"FracDiff: series too short ({len(x)}) for d={d}")
            return np.diff(x)  # Fallback to d=1

        # Apply weights via dot product (convolution)
        result = np.zeros(len(x) - width + 1)
        for i in range(width - 1, len(x)):
            result[i - width + 1] = np.dot(weights, x[i - width + 1:i + 1])

        return result

    def frac_diff_log(self, prices: np.ndarray, d: float) -> np.ndarray:
        """Apply fractional differencing to log prices (recommended)."""
        log_prices = np.log(np.asarray(prices, dtype=float))
        return self.frac_diff(log_prices, d)

    def find_min_d(self, series: np.ndarray, d_range: Tuple[float, float] = (0.0, 1.0),
                   step: float = 0.05, adf_pvalue: float = 0.05) -> float:
        """
        Find minimum d that achieves stationarity (ADF p-value < threshold).

        Binary-search-like: tests d from low to high, returns first d where ADF passes.
        This preserves maximum memory while achieving stationarity.

        Args:
            series: Price series (will use log internally)
            d_range: (min_d, max_d) search range
            step: Step size for d search
            adf_pvalue: Target ADF p-value threshold

        Returns:
            Optimal d value (minimum for stationarity)
        """
        log_series = np.log(np.asarray(series, dtype=float))

        d_values = np.arange(d_range[0] + step, d_range[1] + step, step)
        best_d = 1.0  # Fallback to full differencing

        for d in d_values:
            try:
                diffed = self.frac_diff(log_series, d)
                if len(diffed) < 20:
                    continue

                p_value = self._adf_pvalue(diffed)
                if p_value < adf_pvalue:
                    best_d = float(d)
                    logger.info(f"FracDiff: min d={d:.2f} achieves stationarity (p={p_value:.4f})")
                    break
            except Exception as e:
                logger.debug(f"FracDiff test failed at d={d:.2f}: {e}")
                continue

        return best_d

    def compute_features(self, prices: np.ndarray, d: Optional[float] = None) -> dict:
        """
        Compute fractionally differenced features for ML pipeline.

        Returns:
            {fracdiff_d, fracdiff_series, fracdiff_last, fracdiff_momentum}
        """
        prices = np.asarray(prices, dtype=float)

        if d is None:
            d = self.find_min_d(prices)

        diffed = self.frac_diff_log(prices, d)

        if len(diffed) < 5:
            return {
                'fracdiff_d': float(d),
                'fracdiff_last': 0.0,
                'fracdiff_momentum': 0.0,
                'fracdiff_mean': 0.0,
                'fracdiff_std': 0.0,
            }

        return {
            'fracdiff_d': float(d),
            'fracdiff_last': float(diffed[-1]),
            'fracdiff_momentum': float(np.mean(diffed[-5:])),
            'fracdiff_mean': float(np.mean(diffed)),
            'fracdiff_std': float(np.std(diffed)),
        }

    def _adf_pvalue(self, series: np.ndarray) -> float:
        """Get ADF test p-value."""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series, maxlag=int(np.sqrt(len(series))), autolag='AIC')
            return float(result[1])
        except ImportError:
            # Approximate: use AR(1) coefficient
            if len(series) < 10:
                return 1.0
            y = series[1:]
            x = series[:-1]
            x_m = np.mean(x)
            phi = np.sum((x - x_m) * (y - np.mean(y))) / (np.sum((x - x_m)**2) + 1e-12)
            # Rough approximation: if phi < 0.9, likely stationary
            if phi < 0.85:
                return 0.01
            elif phi < 0.95:
                return 0.05
            else:
                return 0.50
