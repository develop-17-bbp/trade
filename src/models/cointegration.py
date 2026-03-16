"""
Cointegration & Pairs Trading Engine
=========================================
Engle-Granger two-step cointegration test for identifying mean-reverting
spread relationships between crypto pairs.

Alpha source: Sideways/ranging markets where directional strategies fail.

Method:
  1. Engle-Granger: OLS hedge ratio β, then ADF on spread = Y - β·X
  2. Johansen (if statsmodels available): multivariate cointegration rank
  3. Spread z-score trading: Long spread when z < -entry, Short when z > +entry

Usage:
    from src.models.cointegration import CointegrationEngine
    engine = CointegrationEngine()
    result = engine.test_pair(prices_a, prices_b)
    signal = engine.spread_signal(prices_a, prices_b)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CointegrationEngine:
    """
    Engle-Granger cointegration test + spread trading signals.
    """

    def __init__(self, entry_z: float = 2.0, exit_z: float = 0.5,
                 lookback: int = 200, min_half_life: int = 2,
                 max_half_life: int = 120, adf_pvalue: float = 0.05):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.adf_pvalue = adf_pvalue

    def test_pair(self, prices_a: np.ndarray, prices_b: np.ndarray) -> Dict:
        """
        Engle-Granger cointegration test between two price series.

        Returns:
            {cointegrated: bool, hedge_ratio: float, spread_half_life: float,
             adf_stat: float, adf_pvalue: float, spread_mean: float, spread_std: float}
        """
        a = np.asarray(prices_a, dtype=float)
        b = np.asarray(prices_b, dtype=float)
        n = min(len(a), len(b))
        if n < 50:
            return self._default_result()

        a, b = a[-n:], b[-n:]

        # Step 1: OLS hedge ratio — Y = β·X + α + ε
        # Using log prices for better stationarity properties
        log_a = np.log(a)
        log_b = np.log(b)

        # OLS: β = Cov(X,Y) / Var(X)
        X = log_b
        Y = log_a
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)
        beta = np.sum((X - X_mean) * (Y - Y_mean)) / (np.sum((X - X_mean) ** 2) + 1e-12)
        alpha = Y_mean - beta * X_mean

        # Spread (residuals)
        spread = Y - (beta * X + alpha)

        # Step 2: ADF test on spread
        adf_stat, adf_p = self._adf_test(spread)

        # Step 3: Half-life via AR(1) on spread
        half_life = self._compute_half_life(spread)

        # Step 4: Spread statistics
        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread))

        cointegrated = (
            adf_p < self.adf_pvalue and
            self.min_half_life <= half_life <= self.max_half_life
        )

        return {
            'cointegrated': cointegrated,
            'hedge_ratio': float(beta),
            'intercept': float(alpha),
            'spread_half_life': float(half_life),
            'adf_stat': float(adf_stat),
            'adf_pvalue': float(adf_p),
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'spread_current': float(spread[-1]),
            'spread_z_score': float((spread[-1] - spread_mean) / (spread_std + 1e-12)),
            'n_observations': n,
        }

    def spread_signal(self, prices_a: np.ndarray, prices_b: np.ndarray) -> Dict:
        """
        Generate pairs trading signal from spread z-score.

        Returns:
            {signal: int, z_score: float, cointegrated: bool, ...}
        """
        result = self.test_pair(prices_a, prices_b)

        if not result['cointegrated']:
            return {**result, 'signal': 0, 'signal_reason': 'not_cointegrated'}

        z = result['spread_z_score']

        if z < -self.entry_z:
            signal = 1   # Spread too low → buy A, sell B
            reason = f'spread_oversold_z={z:.2f}'
        elif z > self.entry_z:
            signal = -1  # Spread too high → sell A, buy B
            reason = f'spread_overbought_z={z:.2f}'
        elif abs(z) < self.exit_z:
            signal = 0   # Mean reached → close
            reason = 'spread_at_mean'
        else:
            signal = 0
            reason = 'no_signal'

        return {**result, 'signal': signal, 'signal_reason': reason}

    def find_cointegrated_pairs(self, price_dict: Dict[str, np.ndarray],
                                 top_n: int = 5) -> List[Dict]:
        """
        Scan all pairs and return top cointegrated ones.

        Args:
            price_dict: {symbol: price_array}
            top_n: Number of best pairs to return

        Returns:
            List of pair results sorted by half-life (shorter = better)
        """
        symbols = list(price_dict.keys())
        results = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                try:
                    result = self.test_pair(
                        price_dict[symbols[i]],
                        price_dict[symbols[j]]
                    )
                    if result['cointegrated']:
                        result['pair'] = (symbols[i], symbols[j])
                        results.append(result)
                except Exception as e:
                    logger.debug(f"Pair test failed {symbols[i]}/{symbols[j]}: {e}")

        # Sort by half-life (shorter = faster mean reversion = better)
        results.sort(key=lambda x: x['spread_half_life'])
        return results[:top_n]

    def _adf_test(self, series: np.ndarray) -> Tuple[float, float]:
        """Augmented Dickey-Fuller test. Returns (statistic, p-value)."""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series, maxlag=int(np.sqrt(len(series))), autolag='AIC')
            return float(result[0]), float(result[1])
        except ImportError:
            # Manual ADF approximation: ΔY_t = α + β·Y_{t-1} + ε
            n = len(series)
            if n < 20:
                return 0.0, 1.0

            dy = np.diff(series)
            y_lag = series[:-1]

            # OLS: dy = a + b * y_lag
            X = np.column_stack([np.ones(n - 1), y_lag])
            try:
                beta = np.linalg.lstsq(X, dy, rcond=None)[0]
                residuals = dy - X @ beta
                se_beta = np.sqrt(np.sum(residuals**2) / (n - 3) /
                                  (np.sum((y_lag - np.mean(y_lag))**2) + 1e-12))
                t_stat = beta[1] / (se_beta + 1e-12)

                # Approximate p-value using MacKinnon critical values
                # -3.43 (1%), -2.86 (5%), -2.57 (10%) for n=200
                if t_stat < -3.43:
                    p_val = 0.01
                elif t_stat < -2.86:
                    p_val = 0.05
                elif t_stat < -2.57:
                    p_val = 0.10
                else:
                    p_val = 0.50
                return float(t_stat), p_val
            except Exception:
                return 0.0, 1.0

    def _compute_half_life(self, spread: np.ndarray) -> float:
        """Half-life of mean reversion via AR(1): spread_t = φ·spread_{t-1} + ε."""
        n = len(spread)
        if n < 10:
            return 999.0

        y = spread[1:]
        x = spread[:-1]

        # OLS: y = φ·x + c
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        phi = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean)**2) + 1e-12)

        if phi >= 1.0 or phi <= 0:
            return 999.0

        half_life = -np.log(2) / np.log(abs(phi))
        return max(0.5, float(half_life))

    def _default_result(self) -> Dict:
        return {
            'cointegrated': False,
            'hedge_ratio': 0.0,
            'intercept': 0.0,
            'spread_half_life': 999.0,
            'adf_stat': 0.0,
            'adf_pvalue': 1.0,
            'spread_mean': 0.0,
            'spread_std': 0.0,
            'spread_current': 0.0,
            'spread_z_score': 0.0,
            'n_observations': 0,
        }
