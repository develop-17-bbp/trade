"""
Black-Litterman Portfolio Allocation Model
=============================================
Combines market equilibrium returns (CAPM prior) with investor views
to produce optimal portfolio weights that are more stable and intuitive
than raw mean-variance optimization.

Formula:
  μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} · [(τΣ)^{-1}·π + P'Ω^{-1}·Q]

  where:
    π = equilibrium excess returns (δ·Σ·w_mkt)
    P = view matrix (which assets the views are about)
    Q = view vector (expected returns from views)
    Ω = uncertainty of views
    τ = scaling factor (~0.05)
    δ = risk aversion coefficient
    Σ = covariance matrix

Usage:
    from src.portfolio.black_litterman import BlackLitterman
    bl = BlackLitterman()
    weights = bl.optimize(returns_matrix, market_caps, views, view_confidences)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BlackLitterman:
    """
    Black-Litterman portfolio optimization with sentiment-derived views.
    """

    def __init__(self, tau: float = 0.05, risk_aversion: float = 2.5,
                 risk_free_rate: float = 0.0):
        """
        Args:
            tau: Scaling factor for prior uncertainty (typically 0.025 - 0.05)
            risk_aversion: Risk aversion coefficient δ (higher = more conservative)
            risk_free_rate: Annual risk-free rate
        """
        self.tau = tau
        self.delta = risk_aversion
        self.rf = risk_free_rate

    def equilibrium_returns(self, cov_matrix: np.ndarray,
                            market_weights: np.ndarray) -> np.ndarray:
        """
        Compute implied equilibrium excess returns (CAPM prior).
        π = δ · Σ · w_mkt
        """
        return self.delta * cov_matrix @ market_weights

    def optimize(self,
                 returns_matrix: np.ndarray,
                 market_weights: np.ndarray,
                 views: Optional[List[Dict]] = None,
                 min_weight: float = 0.0,
                 max_weight: float = 0.5) -> Dict:
        """
        Full Black-Litterman optimization.

        Args:
            returns_matrix: (T, N) matrix of historical returns for N assets
            market_weights: (N,) market cap weights or equal weights
            views: List of view dicts:
                [{'assets': [0, 2], 'weights': [1, -1], 'return': 0.02, 'confidence': 0.8}]
                Each view: "asset 0 will outperform asset 2 by 2% with 80% confidence"
            min_weight: Minimum allocation per asset
            max_weight: Maximum allocation per asset

        Returns:
            {weights, expected_returns, risk, sharpe, details}
        """
        R = np.asarray(returns_matrix, dtype=float)
        w_mkt = np.asarray(market_weights, dtype=float)
        n_assets = R.shape[1] if R.ndim == 2 else len(w_mkt)

        # Covariance matrix
        if R.ndim == 2 and R.shape[0] >= n_assets:
            Sigma = np.cov(R, rowvar=False)
        else:
            # Fallback: identity scaled by average variance
            Sigma = np.eye(n_assets) * 0.04

        # Ensure positive definite
        Sigma = self._make_positive_definite(Sigma)

        # Equilibrium returns
        pi = self.equilibrium_returns(Sigma, w_mkt)

        if views and len(views) > 0:
            # Build view matrices P, Q, Omega
            P, Q, Omega = self._build_view_matrices(views, n_assets, Sigma)
            # Black-Litterman posterior
            mu_bl = self._bl_posterior(Sigma, pi, P, Q, Omega)
        else:
            mu_bl = pi

        # Optimal weights via analytical solution (unconstrained)
        # w* = (δΣ)^{-1} · μ_BL
        try:
            Sigma_inv = np.linalg.inv(self.delta * Sigma)
            w_opt = Sigma_inv @ mu_bl
        except np.linalg.LinAlgError:
            w_opt = w_mkt.copy()

        # Apply constraints
        w_opt = np.clip(w_opt, min_weight, max_weight)
        # Normalize to sum to 1
        w_sum = np.sum(w_opt)
        if w_sum > 0:
            w_opt = w_opt / w_sum
        else:
            w_opt = w_mkt.copy()

        # Portfolio metrics
        port_return = float(w_opt @ mu_bl)
        port_risk = float(np.sqrt(w_opt @ Sigma @ w_opt))
        sharpe = float(port_return / (port_risk + 1e-10))

        return {
            'weights': w_opt.tolist(),
            'expected_returns': mu_bl.tolist(),
            'portfolio_return': port_return,
            'portfolio_risk': port_risk,
            'sharpe_ratio': sharpe,
            'equilibrium_returns': pi.tolist(),
            'market_weights': w_mkt.tolist(),
        }

    def optimize_from_signals(self,
                              returns_matrix: np.ndarray,
                              symbols: List[str],
                              signal_dict: Dict[str, Dict],
                              max_weight: float = 0.4) -> Dict:
        """
        Convenience: Build views from trading signals (sentiment + model predictions).

        Args:
            returns_matrix: (T, N) returns
            symbols: Asset names matching columns
            signal_dict: {symbol: {'signal': int, 'confidence': float, 'expected_return': float}}
            max_weight: Max per-asset allocation

        Returns:
            Black-Litterman optimized weights
        """
        n = len(symbols)
        market_weights = np.ones(n) / n  # Equal weight prior

        views = []
        for i, sym in enumerate(symbols):
            if sym in signal_dict:
                sig = signal_dict[sym]
                direction = sig.get('signal', 0)
                conf = sig.get('confidence', 0.5)
                exp_ret = sig.get('expected_return', direction * 0.01)

                if direction != 0:
                    views.append({
                        'assets': [i],
                        'weights': [1.0],
                        'return': exp_ret,
                        'confidence': conf,
                    })

        result = self.optimize(returns_matrix, market_weights, views,
                               max_weight=max_weight)
        result['symbols'] = symbols
        return result

    def _build_view_matrices(self, views: List[Dict], n_assets: int,
                             Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build P (view matrix), Q (view returns), Omega (view uncertainty)."""
        k = len(views)
        P = np.zeros((k, n_assets))
        Q = np.zeros(k)

        for i, view in enumerate(views):
            assets = view['assets']
            weights = view.get('weights', [1.0 / len(assets)] * len(assets))
            for j, (a, w) in enumerate(zip(assets, weights)):
                if 0 <= a < n_assets:
                    P[i, a] = w
            Q[i] = view['return']

        # Omega: uncertainty diagonal = τ · diag(P·Σ·P') / confidence
        P_Sigma_Pt = P @ (self.tau * Sigma) @ P.T
        omega_diag = np.diag(P_Sigma_Pt).copy()

        for i, view in enumerate(views):
            conf = view.get('confidence', 0.5)
            # Lower confidence = higher uncertainty
            omega_diag[i] = omega_diag[i] / (conf + 1e-6)

        Omega = np.diag(omega_diag)
        return P, Q, Omega

    def _bl_posterior(self, Sigma: np.ndarray, pi: np.ndarray,
                      P: np.ndarray, Q: np.ndarray,
                      Omega: np.ndarray) -> np.ndarray:
        """
        Compute BL posterior mean:
        μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} · [(τΣ)^{-1}·π + P'Ω^{-1}·Q]
        """
        tau_Sigma = self.tau * Sigma

        try:
            tau_Sigma_inv = np.linalg.inv(tau_Sigma)
            Omega_inv = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            return pi  # Fallback

        # Posterior precision
        precision = tau_Sigma_inv + P.T @ Omega_inv @ P

        # Posterior mean
        try:
            precision_inv = np.linalg.inv(precision)
            mu_bl = precision_inv @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)
        except np.linalg.LinAlgError:
            mu_bl = pi

        return mu_bl

    @staticmethod
    def _make_positive_definite(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Ensure matrix is positive definite via eigenvalue floor."""
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, eps)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T


class RiskParity:
    """
    Risk Parity allocation — equal risk contribution from each asset.

    Objective: w_i · (Σw)_i = w_j · (Σw)_j for all i, j
    Solved via iterative inverse-volatility with covariance adjustment.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def optimize(self, cov_matrix: np.ndarray,
                 min_weight: float = 0.02,
                 max_weight: float = 0.5) -> Dict:
        """
        Risk parity allocation.

        Args:
            cov_matrix: (N, N) covariance matrix
            min_weight: Minimum per-asset weight
            max_weight: Maximum per-asset weight

        Returns:
            {weights, risk_contributions, portfolio_risk}
        """
        Sigma = np.asarray(cov_matrix, dtype=float)
        n = Sigma.shape[0]

        # Initial guess: inverse volatility
        vols = np.sqrt(np.diag(Sigma))
        w = (1.0 / vols) / np.sum(1.0 / vols)

        for _ in range(self.max_iter):
            # Marginal risk contribution: MRC_i = (Σw)_i / σ_p
            Sigma_w = Sigma @ w
            port_var = w @ Sigma_w
            port_vol = np.sqrt(port_var + 1e-12)

            # Risk contribution: RC_i = w_i · MRC_i
            mrc = Sigma_w / port_vol
            rc = w * mrc

            # Target: equal risk contribution = port_vol / n
            target_rc = port_vol / n

            # Update: w_new ∝ w · (target_rc / rc)
            w_new = w * (target_rc / (rc + 1e-12))
            w_new = np.clip(w_new, min_weight, max_weight)
            w_new = w_new / np.sum(w_new)

            if np.max(np.abs(w_new - w)) < self.tol:
                w = w_new
                break
            w = w_new

        # Final metrics
        Sigma_w = Sigma @ w
        port_var = w @ Sigma_w
        port_vol = np.sqrt(port_var)
        rc = w * (Sigma_w / port_vol)

        return {
            'weights': w.tolist(),
            'risk_contributions': rc.tolist(),
            'portfolio_risk': float(port_vol),
            'risk_parity_score': float(1.0 - np.std(rc) / (np.mean(rc) + 1e-12)),
        }
