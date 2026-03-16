"""
Monte Carlo Portfolio Risk Engine
====================================
Forward-looking risk assessment via Monte Carlo simulation.

Simulates N price paths using Geometric Brownian Motion (GBM) with
regime-conditional drift and volatility from GARCH/EWMA + HMM.

Outputs:
    VaR (95%, 99%)   — Maximum expected loss at confidence level
    CVaR (95%)       — Expected Shortfall (average loss in worst 5%)
    Max Drawdown     — 95th percentile of simulated drawdowns
    Risk Score (0-1) — Normalized composite risk metric
    Position Scale   — Suggested position multiplier based on risk budget

Model:
    S_{t+1} = S_t · exp((μ - σ²/2)Δt + σ·√Δt·Z)
    where Z ~ N(0,1), μ = regime-conditional drift, σ = current vol estimate

Usage:
    from src.risk.monte_carlo_risk import MonteCarloRisk
    mc = MonteCarloRisk()
    result = mc.simulate(current_price=42000, volatility=0.03, drift=0.0001)
    # result = {'var_95': -0.032, 'cvar_95': -0.048, 'risk_score': 0.45, ...}
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MonteCarloRisk:
    """
    Monte Carlo VaR/CVaR risk engine with regime-conditional parameters.
    """

    def __init__(self, n_simulations: int = 10000, horizon: int = 24,
                 var_confidence: float = 0.95, risk_budget: float = 0.02):
        """
        Args:
            n_simulations: Number of MC paths to simulate
            horizon: Forward horizon in bars (24 = 1 day for 1h bars)
            var_confidence: Confidence level for VaR (0.95 or 0.99)
            risk_budget: Maximum acceptable VaR as fraction of portfolio
        """
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.var_confidence = var_confidence
        self.risk_budget = risk_budget

    def simulate(self, current_price: float, volatility: float,
                 drift: float = 0.0, regime: str = 'normal',
                 regime_vol_multiplier: Optional[float] = None) -> Dict:
        """
        Run Monte Carlo simulation and compute risk metrics.

        Args:
            current_price: Current asset price
            volatility: Annualized or per-bar volatility estimate (from GARCH/EWMA)
            drift: Expected per-bar return (typically near 0 for crypto)
            regime: Current market regime ('bull', 'bear', 'sideways', 'crisis')
            regime_vol_multiplier: Override vol multiplier for regime (auto if None)

        Returns:
            Dict with VaR, CVaR, risk score, position scale, and simulation stats
        """
        if current_price <= 0 or volatility <= 0:
            return self._default_result()

        # Regime-conditional adjustments
        if regime_vol_multiplier is not None:
            vol = volatility * regime_vol_multiplier
        else:
            vol = volatility * self._regime_vol_factor(regime)

        mu = drift + self._regime_drift_adjustment(regime)

        # Simulate GBM paths: S_{t+1} = S_t * exp((mu - vol^2/2)*dt + vol*sqrt(dt)*Z)
        dt = 1.0  # Per-bar
        n = self.n_simulations
        h = self.horizon

        # Random innovations (n_sims × horizon)
        z = np.random.randn(n, h)

        # Cumulative log returns
        log_returns = (mu - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * z
        cumulative_log_returns = np.cumsum(log_returns, axis=1)

        # Price paths
        price_paths = current_price * np.exp(cumulative_log_returns)

        # Terminal returns (fraction of initial price)
        terminal_returns = (price_paths[:, -1] - current_price) / current_price

        # === VaR / CVaR ===
        sorted_returns = np.sort(terminal_returns)

        # VaR at 95%
        var_95_idx = int((1 - 0.95) * n)
        var_95 = float(sorted_returns[var_95_idx])

        # CVaR at 95% (Expected Shortfall)
        cvar_95 = float(np.mean(sorted_returns[:var_95_idx + 1]))

        # VaR at 99%
        var_99_idx = int((1 - 0.99) * n)
        var_99 = float(sorted_returns[max(var_99_idx, 0)])

        # CVaR at 99%
        cvar_99 = float(np.mean(sorted_returns[:max(var_99_idx + 1, 1)]))

        # === Max Drawdown ===
        # For each path, compute peak-to-trough drawdown
        cummax = np.maximum.accumulate(price_paths, axis=1)
        drawdowns = (price_paths - cummax) / (cummax + 1e-12)
        max_drawdowns = np.min(drawdowns, axis=1)  # Most negative per path

        dd_95 = float(np.percentile(max_drawdowns, 5))  # 95th percentile worst DD

        # === Risk Score (0-1 normalized) ===
        # Based on how extreme current VaR is relative to risk budget
        risk_ratio = abs(var_95) / (self.risk_budget + 1e-10)
        risk_score = float(np.clip(risk_ratio, 0.0, 1.0))

        # === Position Scale ===
        # Scale position so that VaR stays within risk budget
        if abs(var_95) > 1e-8:
            position_scale = float(np.clip(self.risk_budget / abs(var_95), 0.05, 1.0))
        else:
            position_scale = 1.0

        # === Simulation stats ===
        mean_return = float(np.mean(terminal_returns))
        std_return = float(np.std(terminal_returns))
        prob_profit = float(np.mean(terminal_returns > 0))
        prob_loss_5pct = float(np.mean(terminal_returns < -0.05))

        return {
            'mc_var_95': var_95,
            'mc_cvar_95': cvar_95,
            'mc_var_99': var_99,
            'mc_cvar_99': cvar_99,
            'mc_max_drawdown_95': dd_95,
            'mc_risk_score': risk_score,
            'mc_position_scale': position_scale,
            'mc_mean_return': mean_return,
            'mc_std_return': std_return,
            'mc_prob_profit': prob_profit,
            'mc_prob_loss_5pct': prob_loss_5pct,
            'mc_regime': regime,
            'mc_vol_used': vol,
            'mc_horizon': self.horizon,
            'mc_n_sims': self.n_simulations,
        }

    def simulate_portfolio(self, positions: Dict[str, Dict],
                           correlation_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        Multi-asset portfolio Monte Carlo with optional correlation.

        Args:
            positions: {symbol: {'price': float, 'weight': float, 'vol': float, 'drift': float}}
            correlation_matrix: NxN correlation matrix (Cholesky decomposition for correlated draws)

        Returns:
            Portfolio-level risk metrics
        """
        symbols = list(positions.keys())
        n_assets = len(symbols)

        if n_assets == 0:
            return self._default_result()

        weights = np.array([positions[s].get('weight', 1.0 / n_assets) for s in symbols])
        vols = np.array([positions[s].get('vol', 0.02) for s in symbols])
        drifts = np.array([positions[s].get('drift', 0.0) for s in symbols])

        n = self.n_simulations
        h = self.horizon

        # Generate correlated random draws
        if correlation_matrix is not None and correlation_matrix.shape == (n_assets, n_assets):
            try:
                L = np.linalg.cholesky(correlation_matrix)
                z_raw = np.random.randn(n, h, n_assets)
                z = np.einsum('ij,...j->...i', L, z_raw)
            except np.linalg.LinAlgError:
                z = np.random.randn(n, h, n_assets)
        else:
            z = np.random.randn(n, h, n_assets)

        # Simulate each asset
        dt = 1.0
        log_returns = np.zeros((n, h, n_assets))
        for a in range(n_assets):
            log_returns[:, :, a] = (drifts[a] - 0.5 * vols[a]**2) * dt + vols[a] * np.sqrt(dt) * z[:, :, a]

        # Portfolio returns (weighted sum of asset returns)
        asset_cum_returns = np.exp(np.cumsum(log_returns, axis=1)) - 1  # (n, h, n_assets)
        portfolio_returns = np.sum(asset_cum_returns * weights[np.newaxis, np.newaxis, :], axis=2)

        terminal = portfolio_returns[:, -1]
        sorted_term = np.sort(terminal)

        var_95_idx = int(0.05 * n)
        var_95 = float(sorted_term[var_95_idx])
        cvar_95 = float(np.mean(sorted_term[:var_95_idx + 1]))
        var_99_idx = int(0.01 * n)
        var_99 = float(sorted_term[max(var_99_idx, 0)])

        risk_ratio = abs(var_95) / (self.risk_budget + 1e-10)
        risk_score = float(np.clip(risk_ratio, 0.0, 1.0))
        position_scale = float(np.clip(self.risk_budget / (abs(var_95) + 1e-10), 0.05, 1.0))

        return {
            'mc_var_95': var_95,
            'mc_cvar_95': cvar_95,
            'mc_var_99': var_99,
            'mc_risk_score': risk_score,
            'mc_position_scale': position_scale,
            'mc_mean_return': float(np.mean(terminal)),
            'mc_prob_profit': float(np.mean(terminal > 0)),
            'mc_n_assets': n_assets,
            'mc_horizon': self.horizon,
            'mc_n_sims': self.n_simulations,
        }

    @staticmethod
    def _regime_vol_factor(regime: str) -> float:
        """Volatility multiplier by regime."""
        return {
            'bull': 0.9,
            'bear': 1.3,
            'sideways': 0.7,
            'crisis': 2.0,
            'normal': 1.0,
        }.get(regime, 1.0)

    @staticmethod
    def _regime_drift_adjustment(regime: str) -> float:
        """Drift adjustment by regime (per bar)."""
        return {
            'bull': 0.0002,
            'bear': -0.0003,
            'sideways': 0.0,
            'crisis': -0.001,
            'normal': 0.0,
        }.get(regime, 0.0)

    def _default_result(self) -> Dict:
        """Default result when simulation can't run."""
        return {
            'mc_var_95': -0.02,
            'mc_cvar_95': -0.03,
            'mc_var_99': -0.04,
            'mc_cvar_99': -0.05,
            'mc_max_drawdown_95': -0.05,
            'mc_risk_score': 0.5,
            'mc_position_scale': 0.5,
            'mc_mean_return': 0.0,
            'mc_std_return': 0.02,
            'mc_prob_profit': 0.5,
            'mc_prob_loss_5pct': 0.1,
            'mc_regime': 'normal',
            'mc_vol_used': 0.02,
            'mc_horizon': self.horizon,
            'mc_n_sims': self.n_simulations,
        }
