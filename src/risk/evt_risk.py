"""
Extreme Value Theory (EVT) Tail Risk Engine
===============================================
Uses the Peaks-Over-Threshold (POT) method with Generalized Pareto Distribution
(GPD) to model extreme tail losses beyond what normal distributions capture.

Why: Normal VaR underestimates tail risk by 3-5x in crypto.
     EVT captures the actual fat-tail behavior of crypto returns.

Method (POT-GPD):
  1. Select threshold u (e.g., 95th percentile of losses)
  2. Extract exceedances: Y_i = X_i - u for X_i > u
  3. Fit GPD: F(y) = 1 - (1 + ξy/σ)^{-1/ξ}
     ξ (shape): >0 = heavy tail (Pareto), 0 = exponential, <0 = bounded
     σ (scale): spread of exceedances
  4. Compute EVT-VaR and EVT-ES at extreme quantiles (99%, 99.5%)

Usage:
    from src.risk.evt_risk import EVTRisk
    evt = EVTRisk()
    result = evt.fit_and_assess(returns)
    # result = {'evt_var_99': -0.082, 'evt_es_99': -0.115, 'tail_index': 3.2, ...}
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class EVTRisk:
    """
    Peaks-Over-Threshold (POT) with Generalized Pareto Distribution.
    """

    def __init__(self, threshold_quantile: float = 0.95,
                 var_level: float = 0.99, min_exceedances: int = 20):
        """
        Args:
            threshold_quantile: Quantile for threshold selection (0.90 - 0.95 typical)
            var_level: Confidence level for VaR/ES computation
            min_exceedances: Minimum exceedances needed for reliable fit
        """
        self.threshold_quantile = threshold_quantile
        self.var_level = var_level
        self.min_exceedances = min_exceedances
        self.xi = 0.0   # Shape parameter (tail index)
        self.sigma = 1.0  # Scale parameter
        self.threshold = 0.0
        self._fitted = False

    def fit(self, losses: np.ndarray) -> Dict:
        """
        Fit GPD to tail exceedances via Method of Moments (robust).

        Args:
            losses: Array of losses (positive values = losses)

        Returns:
            {xi, sigma, threshold, n_exceedances, tail_index}
        """
        losses = np.asarray(losses, dtype=float)
        losses = losses[np.isfinite(losses)]

        if len(losses) < 50:
            return self._default_params()

        # Threshold: quantile of losses
        self.threshold = float(np.quantile(losses, self.threshold_quantile))

        # Exceedances
        exceedances = losses[losses > self.threshold] - self.threshold
        n_exc = len(exceedances)

        if n_exc < self.min_exceedances:
            logger.warning(f"EVT: Only {n_exc} exceedances (need {self.min_exceedances})")
            return self._default_params()

        # Method of Moments for GPD
        # E[Y] = σ / (1 - ξ)
        # Var[Y] = σ² / ((1-ξ)²(1-2ξ))
        mean_y = np.mean(exceedances)
        var_y = np.var(exceedances)

        if mean_y <= 0:
            return self._default_params()

        # Solve for ξ and σ:
        # From moments: ξ = 0.5 * (1 - mean²/var)
        # σ = mean * (1 - ξ)
        ratio = mean_y**2 / (var_y + 1e-12)
        self.xi = 0.5 * (1 - ratio)

        # Clamp ξ to reasonable range for financial data
        self.xi = float(np.clip(self.xi, -0.5, 1.0))
        self.sigma = float(mean_y * (1 - self.xi))

        if self.sigma <= 0:
            self.sigma = float(mean_y)

        self._fitted = True

        # Tail index: 1/ξ (higher = thinner tail)
        tail_index = 1.0 / self.xi if abs(self.xi) > 0.01 else float('inf')

        logger.info(f"EVT fit: ξ={self.xi:.4f}, σ={self.sigma:.4f}, "
                     f"threshold={self.threshold:.4f}, exceedances={n_exc}")

        return {
            'xi': self.xi,
            'sigma': self.sigma,
            'threshold': self.threshold,
            'n_exceedances': n_exc,
            'tail_index': tail_index,
            'mean_exceedance': float(mean_y),
        }

    def var(self, n_total: int, quantile: Optional[float] = None) -> float:
        """
        Compute EVT-VaR at given quantile.

        VaR_p = u + (σ/ξ) · [(n/N_u · (1-p))^{-ξ} - 1]

        Args:
            n_total: Total number of observations
            quantile: Confidence level (default: self.var_level)

        Returns:
            VaR value (as a positive loss)
        """
        q = quantile or self.var_level
        n_u = int(n_total * (1 - self.threshold_quantile))

        if n_u <= 0 or not self._fitted:
            return self.threshold * 1.5

        if abs(self.xi) < 1e-6:
            # Exponential case (ξ → 0)
            return self.threshold + self.sigma * np.log(n_u / (n_total * (1 - q) + 1e-12))

        ratio = (n_u / (n_total * (1 - q) + 1e-12))
        var_val = self.threshold + (self.sigma / self.xi) * (ratio**self.xi - 1)

        return float(max(var_val, self.threshold))

    def expected_shortfall(self, n_total: int,
                           quantile: Optional[float] = None) -> float:
        """
        Compute EVT Expected Shortfall (CVaR).

        ES_p = VaR_p / (1 - ξ) + (σ - ξ·u) / (1 - ξ)
        """
        q = quantile or self.var_level
        var_val = self.var(n_total, q)

        if abs(1 - self.xi) < 1e-6:
            return var_val * 1.5  # Approximate

        es = var_val / (1 - self.xi) + (self.sigma - self.xi * self.threshold) / (1 - self.xi)
        return float(max(es, var_val))

    def fit_and_assess(self, returns: np.ndarray) -> Dict:
        """
        Full EVT risk assessment from return series.

        Args:
            returns: Array of returns (negative = losses)

        Returns:
            Complete risk assessment with EVT metrics
        """
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]

        if len(returns) < 50:
            return self._default_result()

        # Convert to losses (positive values)
        losses = -returns  # Flip sign so losses are positive

        # Fit GPD
        fit_result = self.fit(losses)

        n = len(returns)

        # EVT risk metrics
        evt_var_95 = -self.var(n, 0.95)   # Negative for loss
        evt_var_99 = -self.var(n, 0.99)
        evt_var_995 = -self.var(n, 0.995)
        evt_es_95 = -self.expected_shortfall(n, 0.95)
        evt_es_99 = -self.expected_shortfall(n, 0.99)

        # Normal VaR for comparison (shows how much EVT adds)
        normal_var_99 = float(np.mean(returns) - 2.326 * np.std(returns))

        # Tail risk ratio: EVT_VaR / Normal_VaR (>1 means fatter tails)
        tail_ratio = abs(evt_var_99) / (abs(normal_var_99) + 1e-12)

        # Risk score (0-1): based on tail heaviness
        if self.xi > 0:
            # Heavy tail: score increases with ξ
            risk_score = float(np.clip(self.xi * 3, 0, 1))
        else:
            risk_score = 0.2  # Light/bounded tail

        # Position scale recommendation
        if risk_score > 0.7:
            position_scale = 0.3
        elif risk_score > 0.5:
            position_scale = 0.5
        elif risk_score > 0.3:
            position_scale = 0.75
        else:
            position_scale = 1.0

        return {
            'evt_var_95': evt_var_95,
            'evt_var_99': evt_var_99,
            'evt_var_995': evt_var_995,
            'evt_es_95': evt_es_95,
            'evt_es_99': evt_es_99,
            'evt_xi': self.xi,
            'evt_sigma': self.sigma,
            'evt_threshold': self.threshold,
            'evt_tail_ratio': tail_ratio,
            'evt_risk_score': risk_score,
            'evt_position_scale': position_scale,
            'normal_var_99': normal_var_99,
            'tail_index': fit_result.get('tail_index', float('inf')),
            'n_exceedances': fit_result.get('n_exceedances', 0),
        }

    def _default_params(self) -> Dict:
        self._fitted = False
        return {
            'xi': 0.0,
            'sigma': 1.0,
            'threshold': 0.0,
            'n_exceedances': 0,
            'tail_index': float('inf'),
            'mean_exceedance': 0.0,
        }

    def _default_result(self) -> Dict:
        return {
            'evt_var_95': -0.03,
            'evt_var_99': -0.06,
            'evt_var_995': -0.08,
            'evt_es_95': -0.04,
            'evt_es_99': -0.08,
            'evt_xi': 0.0,
            'evt_sigma': 0.02,
            'evt_threshold': 0.02,
            'evt_tail_ratio': 1.0,
            'evt_risk_score': 0.3,
            'evt_position_scale': 0.75,
            'normal_var_99': -0.05,
            'tail_index': float('inf'),
            'n_exceedances': 0,
        }
