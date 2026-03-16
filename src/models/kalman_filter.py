"""
Kalman Filter Trend Estimator
===============================
Linear Kalman filter with 2D state [level, slope] for optimal adaptive smoothing.

State-space model:
    State:       x_t = [level_t, slope_t]^T
    Transition:  x_t = F · x_{t-1} + w_t    (w ~ N(0, Q))
    Observation: y_t = H · x_t + v_t         (v ~ N(0, R))

    F = [[1, 1],    H = [1, 0]
         [0, 1]]

Advantages over SMA/EMA:
    - Mathematically optimal (minimum variance) estimation
    - Adapts to regime changes via innovation-based noise estimation
    - Provides signal-to-noise ratio as confidence measure
    - Slope estimate = filtered first derivative (no lag from differencing)

Outputs:
    kalman_level     — Smoothed price level (replaces SMA/EMA)
    kalman_slope     — Estimated rate of change (filtered trend)
    kalman_accel     — Change in slope (momentum divergence)
    kalman_snr       — Signal-to-noise ratio (high = clear trend)
    kalman_residual  — Innovation (observation - prediction); spikes = breakouts

Usage:
    from src.models.kalman_filter import KalmanTrendFilter
    kf = KalmanTrendFilter()
    results = kf.filter(prices)
    # results['level'][-1], results['slope'][-1], results['snr'][-1]
"""

import numpy as np
from typing import Dict, List, Optional


class KalmanTrendFilter:
    """
    2D Kalman filter for price trend estimation with adaptive noise.
    """

    def __init__(self, q_level: float = 1e-5, q_slope: float = 1e-7,
                 r_obs: float = 1e-3, adaptive: bool = True,
                 adaptive_window: int = 20):
        """
        Args:
            q_level: Process noise variance for level state
            q_slope: Process noise variance for slope state
            r_obs: Observation noise variance
            adaptive: If True, dynamically estimate R from innovation sequence
            adaptive_window: Window for adaptive noise estimation
        """
        self.q_level = q_level
        self.q_slope = q_slope
        self.r_obs = r_obs
        self.adaptive = adaptive
        self.adaptive_window = adaptive_window

        # State transition matrix
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]])

        # Observation matrix
        self.H = np.array([[1.0, 0.0]])

        # Process noise covariance
        self.Q = np.array([[q_level, 0.0],
                           [0.0, q_slope]])

    def filter(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run Kalman filter over price series.

        Args:
            prices: 1D array of prices

        Returns:
            Dict with arrays: level, slope, accel, snr, residual, signal
        """
        prices = np.asarray(prices, dtype=float)
        n = len(prices)

        if n < 3:
            empty = np.zeros(n)
            return {
                'level': prices.copy(),
                'slope': empty,
                'accel': empty,
                'snr': empty,
                'residual': empty,
                'signal': np.zeros(n, dtype=int),
            }

        # Normalize prices to log-scale for numerical stability
        log_prices = np.log(prices + 1e-12)

        # Initialize state
        x = np.array([log_prices[0], 0.0])  # [level, slope]
        P = np.array([[1.0, 0.0],
                       [0.0, 1.0]])  # State covariance

        R = np.array([[self.r_obs]])  # Observation noise

        # Output arrays
        levels = np.zeros(n)
        slopes = np.zeros(n)
        residuals = np.zeros(n)
        kalman_gains = np.zeros(n)
        innovation_sq = np.zeros(n)

        levels[0] = prices[0]

        for t in range(1, n):
            # === PREDICT ===
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q

            # === UPDATE ===
            y_obs = log_prices[t]
            y_pred = (self.H @ x_pred)[0]
            innovation = y_obs - y_pred  # Residual (surprise)

            S = (self.H @ P_pred @ self.H.T + R)[0, 0]  # Innovation covariance
            K = (P_pred @ self.H.T) / (S + 1e-15)  # Kalman gain (2x1)

            x = x_pred + K.flatten() * innovation
            P = (np.eye(2) - K @ self.H) @ P_pred

            # Store results
            levels[t] = np.exp(x[0]) if x[0] < 20 else prices[t]  # Back to price space
            slopes[t] = x[1]  # In log-space (≈ percentage rate of change)
            residuals[t] = innovation
            kalman_gains[t] = K[0, 0]
            innovation_sq[t] = innovation ** 2

            # === ADAPTIVE NOISE ===
            if self.adaptive and t >= self.adaptive_window:
                # Estimate observation noise from recent innovations
                recent_innov = innovation_sq[t - self.adaptive_window + 1:t + 1]
                r_est = np.mean(recent_innov)
                R[0, 0] = max(r_est, 1e-8)

        # Compute derived signals
        accel = np.diff(slopes, prepend=slopes[0])

        # Signal-to-noise ratio: |slope| / std(residuals)
        snr = np.zeros(n)
        win = min(20, n)
        for t in range(win, n):
            res_std = np.std(residuals[t - win + 1:t + 1])
            if res_std > 1e-10:
                snr[t] = abs(slopes[t]) / res_std
            else:
                snr[t] = 0.0

        # Direction signal
        signal = np.zeros(n, dtype=int)
        for t in range(1, n):
            if slopes[t] > 0 and accel[t] >= 0 and snr[t] > 0.5:
                signal[t] = 1  # Uptrend with positive acceleration
            elif slopes[t] < 0 and accel[t] <= 0 and snr[t] > 0.5:
                signal[t] = -1  # Downtrend with negative acceleration
            else:
                signal[t] = 0

        return {
            'level': levels,
            'slope': slopes,
            'accel': accel,
            'snr': snr,
            'residual': residuals,
            'signal': signal,
            'kalman_gain': kalman_gains,
        }

    def latest(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Get only the latest Kalman filter output (convenience method).

        Args:
            prices: Price series

        Returns:
            Dict with scalar values for the last bar
        """
        result = self.filter(prices)
        return {
            'kalman_level': float(result['level'][-1]),
            'kalman_slope': float(result['slope'][-1]),
            'kalman_accel': float(result['accel'][-1]),
            'kalman_snr': float(result['snr'][-1]),
            'kalman_residual': float(result['residual'][-1]),
            'kalman_signal': int(result['signal'][-1]),
        }
