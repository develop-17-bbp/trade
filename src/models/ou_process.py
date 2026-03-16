"""
Ornstein-Uhlenbeck Mean Reversion Model
==========================================
Fits the OU process to log-prices for statistically-grounded mean reversion signals.

Model:
    dX_t = κ(θ - X_t)dt + σ dW_t

Parameters:
    κ (kappa)  — Speed of mean reversion (higher = faster reversion)
    θ (theta)  — Long-run equilibrium level
    σ (sigma)  — Diffusion (noise amplitude)
    half_life  = ln(2) / κ  — Time to revert halfway to equilibrium

Signal logic:
    1. Fit OU parameters via OLS on discretized process
    2. Run ADF test — only trade if unit root rejected (p < 0.05)
    3. Compute z-score relative to OU-calibrated equilibrium
    4. Entry when |z| > entry_threshold, exit at θ

Usage:
    from src.models.ou_process import OUProcess
    ou = OUProcess()
    result = ou.fit_and_signal(prices)
    # result = {'half_life': 12.5, 'theta': 42100.0, 'z_score': -2.1, 'signal': 1, ...}
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("statsmodels not installed — ADF test disabled")


class OUParams:
    """Container for fitted OU process parameters."""
    __slots__ = ('kappa', 'theta', 'sigma', 'half_life', 'adf_pvalue', 'is_stationary')

    def __init__(self, kappa: float, theta: float, sigma: float,
                 adf_pvalue: float = 1.0):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.half_life = np.log(2) / kappa if kappa > 1e-8 else float('inf')
        self.adf_pvalue = adf_pvalue
        self.is_stationary = adf_pvalue < 0.05

    def __repr__(self):
        return (f"OUParams(κ={self.kappa:.4f}, θ={self.theta:.2f}, "
                f"σ={self.sigma:.4f}, half_life={self.half_life:.1f}, "
                f"stationary={self.is_stationary})")


class OUProcess:
    """
    Ornstein-Uhlenbeck mean reversion estimator and signal generator.
    """

    def __init__(self, entry_threshold: float = 1.5, exit_threshold: float = 0.5,
                 max_half_life: float = 100, adf_pvalue: float = 0.05):
        """
        Args:
            entry_threshold: Z-score threshold for entry signals
            exit_threshold: Z-score threshold for exit (back near equilibrium)
            max_half_life: Maximum acceptable half-life in bars (skip slow reversion)
            adf_pvalue: Maximum ADF p-value to consider series stationary
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_half_life = max_half_life
        self.adf_pvalue_threshold = adf_pvalue

    def fit(self, prices: np.ndarray, dt: float = 1.0) -> OUParams:
        """
        Fit OU parameters via OLS on the discretized Euler scheme:
            X_{t+1} - X_t = κ(θ - X_t)Δt + ε_t

        Rearranged as OLS:
            ΔX = a + b·X_t + ε
            where b = -κΔt, a = κθΔt

        Args:
            prices: Price series (raw prices, not log)
            dt: Time step (1.0 for unit bars)

        Returns:
            OUParams with fitted parameters
        """
        prices = np.asarray(prices, dtype=float)
        if len(prices) < 30:
            return OUParams(0.0, np.mean(prices), 0.0, 1.0)

        # Work in log space for better stationarity
        x = np.log(prices + 1e-12)
        dx = np.diff(x)
        x_lag = x[:-1]

        # OLS: dx = a + b * x_lag + noise
        n = len(dx)
        x_mean = np.mean(x_lag)
        dx_mean = np.mean(dx)

        ss_xx = np.sum((x_lag - x_mean) ** 2)
        ss_xy = np.sum((x_lag - x_mean) * (dx - dx_mean))

        if ss_xx < 1e-15:
            return OUParams(0.0, np.mean(prices), 0.0, 1.0)

        b = ss_xy / ss_xx
        a = dx_mean - b * x_mean

        # Extract OU parameters
        kappa = max(-b / dt, 0.0)  # -b/dt = κ (b should be negative for mean reversion)
        theta_log = a / (kappa * dt) if kappa > 1e-8 else np.mean(x)
        theta = np.exp(theta_log)  # Convert back from log space

        # Residual volatility
        residuals = dx - (a + b * x_lag)
        sigma = np.std(residuals, ddof=2) / np.sqrt(dt)

        # ADF test for stationarity
        adf_p = 1.0
        if HAS_STATSMODELS:
            try:
                adf_result = adfuller(x, maxlag=int(np.sqrt(len(x))), regression='c', autolag=None)
                adf_p = adf_result[1]
            except Exception:
                adf_p = 1.0

        return OUParams(kappa, theta, sigma, adf_p)

    def compute_z_score(self, current_price: float, params: OUParams) -> float:
        """
        Compute OU-calibrated z-score.

        z = (log(P) - log(θ)) / (σ / sqrt(2κ))

        The denominator is the OU stationary standard deviation.
        """
        if params.kappa < 1e-8 or params.sigma < 1e-12:
            return 0.0

        log_p = np.log(current_price + 1e-12)
        log_theta = np.log(params.theta + 1e-12)
        ou_std = params.sigma / np.sqrt(2 * params.kappa)

        if ou_std < 1e-12:
            return 0.0

        return (log_p - log_theta) / ou_std

    def signal(self, z_score: float, params: OUParams) -> int:
        """
        Generate trading signal from OU z-score.

        Returns:
            +1: LONG (price below equilibrium, expect reversion up)
            -1: SHORT (price above equilibrium, expect reversion down)
             0: FLAT (no signal or conditions not met)
        """
        # No signal if series isn't mean-reverting
        if not params.is_stationary:
            return 0

        # No signal if reversion too slow
        if params.half_life > self.max_half_life:
            return 0

        if z_score < -self.entry_threshold:
            return 1   # Price below θ → go LONG, expect reversion up
        elif z_score > self.entry_threshold:
            return -1  # Price above θ → go SHORT, expect reversion down
        else:
            return 0

    def fit_and_signal(self, prices: np.ndarray, window: int = 252) -> Dict:
        """
        Complete pipeline: fit OU params and generate signal.

        Args:
            prices: Price series
            window: Lookback window for fitting

        Returns:
            Dict with all OU metrics and trading signal
        """
        prices = np.asarray(prices, dtype=float)
        segment = prices[-window:] if len(prices) > window else prices

        params = self.fit(segment)
        current_price = float(prices[-1])
        z = self.compute_z_score(current_price, params)
        sig = self.signal(z, params)

        return {
            'ou_kappa': params.kappa,
            'ou_theta': params.theta,
            'ou_sigma': params.sigma,
            'ou_half_life': params.half_life,
            'ou_z_score': z,
            'ou_signal': sig,
            'ou_adf_pvalue': params.adf_pvalue,
            'ou_is_stationary': params.is_stationary,
            'ou_entry_level': params.theta * np.exp(-self.entry_threshold * params.sigma / np.sqrt(2 * params.kappa + 1e-12)),
            'ou_exit_level': params.theta,
        }

    def rolling(self, prices: np.ndarray, window: int = 252,
                step: int = 1) -> list:
        """Compute rolling OU signals."""
        results = []
        for i in range(window, len(prices) + 1, step):
            segment = prices[i - window:i]
            result = self.fit_and_signal(segment, window)
            results.append(result)
        return results
