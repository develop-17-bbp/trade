"""
Alpha Decay Model — Signal Expiry & Optimal Holding Period
=============================================================
Models the decay of trading signal alpha over time to determine:
  1. When to exit a trade (signal has expired)
  2. Optimal holding period for each signal type
  3. Signal freshness weighting for the meta-controller

Alpha decay follows exponential decay: α(t) = α₀ · exp(-λt)
where λ is estimated from historical signal-to-PnL correlation at different lags.

Usage:
    from src.models.alpha_decay import AlphaDecayModel
    model = AlphaDecayModel()
    model.fit(signals_history, returns_history)
    holding = model.optimal_holding_period()
    freshness = model.signal_freshness(bars_since_signal=5)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AlphaDecayModel:
    """
    Estimates alpha decay rate from signal-return correlations across horizons.
    """

    def __init__(self, max_horizon: int = 48, min_horizon: int = 1):
        """
        Args:
            max_horizon: Maximum forward horizon to test (in bars)
            min_horizon: Minimum horizon
        """
        self.max_horizon = max_horizon
        self.min_horizon = min_horizon
        self.decay_rate = 0.1  # Default λ
        self.half_life = 7.0  # Default half-life in bars
        self.optimal_hold = 6  # Default holding period
        self.peak_alpha = 0.0
        self._fitted = False

    def fit(self, signals: np.ndarray, returns: np.ndarray) -> Dict:
        """
        Estimate decay rate from signal-return IC (Information Coefficient)
        at multiple forward horizons.

        IC(h) = corr(signal_t, return_{t, t+h}) for h = 1, 2, ..., max_horizon

        Then fit: IC(h) = IC_peak · exp(-λ · (h - h_peak))

        Args:
            signals: (N,) signal values at each bar
            returns: (N,) per-bar returns

        Returns:
            {decay_rate, half_life, optimal_holding, peak_ic, ic_curve}
        """
        signals = np.asarray(signals, dtype=float)
        returns = np.asarray(returns, dtype=float)
        n = min(len(signals), len(returns))

        if n < 50:
            return self._default_result()

        signals = signals[:n]
        returns = returns[:n]

        # Compute IC at each horizon
        horizons = range(self.min_horizon, min(self.max_horizon + 1, n // 5))
        ic_curve = {}

        for h in horizons:
            # Forward returns over h bars
            fwd_returns = np.zeros(n - h)
            for i in range(n - h):
                fwd_returns[i] = np.sum(returns[i + 1:i + h + 1])

            sig_slice = signals[:n - h]

            # Remove NaN/Inf
            valid = np.isfinite(sig_slice) & np.isfinite(fwd_returns)
            if np.sum(valid) < 20:
                continue

            # Rank IC (Spearman correlation)
            ic = self._rank_correlation(sig_slice[valid], fwd_returns[valid])
            ic_curve[h] = float(ic)

        if not ic_curve:
            return self._default_result()

        # Find peak IC
        horizons_arr = np.array(list(ic_curve.keys()))
        ics_arr = np.array(list(ic_curve.values()))

        peak_idx = np.argmax(np.abs(ics_arr))
        peak_horizon = int(horizons_arr[peak_idx])
        peak_ic = float(ics_arr[peak_idx])

        # Fit exponential decay after peak
        post_peak = horizons_arr >= peak_horizon
        if np.sum(post_peak) >= 3:
            h_post = horizons_arr[post_peak] - peak_horizon
            ic_post = np.abs(ics_arr[post_peak])

            # Log-linear fit: log(IC) = log(IC_peak) - λ·h
            # Filter out zeros/negatives
            valid_ic = ic_post > 1e-6
            if np.sum(valid_ic) >= 2:
                y = np.log(ic_post[valid_ic])
                x = h_post[valid_ic].astype(float)

                # OLS
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                slope = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean)**2) + 1e-12)
                self.decay_rate = max(0.01, -slope)  # λ > 0
            else:
                self.decay_rate = 0.1
        else:
            self.decay_rate = 0.1

        self.half_life = np.log(2) / self.decay_rate
        self.peak_alpha = abs(peak_ic)
        self.optimal_hold = peak_horizon
        self._fitted = True

        logger.info(f"Alpha Decay: lambda={self.decay_rate:.4f}, half_life={self.half_life:.1f} bars, "
                     f"optimal_hold={self.optimal_hold} bars, peak_IC={peak_ic:.4f}")

        return {
            'decay_rate': self.decay_rate,
            'half_life': self.half_life,
            'optimal_holding': self.optimal_hold,
            'peak_ic': peak_ic,
            'peak_horizon': peak_horizon,
            'ic_curve': ic_curve,
        }

    def signal_freshness(self, bars_since_signal: int) -> float:
        """
        Compute signal freshness (0-1) based on bars since signal was generated.

        freshness = exp(-λ · max(0, t - t_peak))
        """
        excess = max(0, bars_since_signal - self.optimal_hold)
        return float(np.exp(-self.decay_rate * excess))

    def should_exit(self, bars_held: int, freshness_threshold: float = 0.3) -> bool:
        """
        Recommend exit when signal alpha has decayed below threshold.
        """
        freshness = self.signal_freshness(bars_held)
        return freshness < freshness_threshold

    def adjust_confidence(self, base_confidence: float,
                          bars_since_signal: int) -> float:
        """
        Scale confidence by signal freshness.
        Fresh signal → full confidence. Stale signal → reduced confidence.
        """
        freshness = self.signal_freshness(bars_since_signal)
        return float(base_confidence * freshness)

    def compute_features(self, bars_held: int = 0) -> Dict:
        """Generate features for ML pipeline."""
        freshness = self.signal_freshness(bars_held)
        return {
            'alpha_decay_rate': self.decay_rate,
            'alpha_half_life': self.half_life,
            'alpha_freshness': freshness,
            'alpha_optimal_hold': self.optimal_hold,
            'alpha_should_exit': 1.0 if self.should_exit(bars_held) else 0.0,
        }

    @staticmethod
    def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Spearman rank correlation."""
        n = len(x)
        if n < 5:
            return 0.0
        rank_x = np.argsort(np.argsort(x)).astype(float)
        rank_y = np.argsort(np.argsort(y)).astype(float)
        d = rank_x - rank_y
        return float(1 - 6 * np.sum(d**2) / (n * (n**2 - 1) + 1e-12))

    def _default_result(self) -> Dict:
        return {
            'decay_rate': self.decay_rate,
            'half_life': self.half_life,
            'optimal_holding': self.optimal_hold,
            'peak_ic': 0.0,
            'peak_horizon': 1,
            'ic_curve': {},
        }
