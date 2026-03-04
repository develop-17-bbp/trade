"""
L1 / L3 Volatility Models
==========================
EWMA, GARCH(1,1), and Volatility Regime Detection.

Mathematical foundations:
  Log return:  r_t = ln(P_t / P_{t-1})
  EWMA vol:   σ²_t = (1−λ)·r²_t + λ·σ²_{t-1}
  GARCH(1,1): σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
              Constraints:  ω>0, α≥0, β≥0, α+β<1
  Regime:     LOW if σ_t < median(σ), HIGH if σ_t > k·median(σ), else MEDIUM
"""

import math
from typing import List, Tuple, Optional
from enum import Enum


class VolRegime(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


# ---------------------------------------------------------------------------
# Log returns
# ---------------------------------------------------------------------------
def log_returns(prices: List[float]) -> List[float]:
    """r_t = ln(P_t / P_{t-1})"""
    out: List[float] = [0.0]  # first return = 0
    for i in range(1, len(prices)):
        if prices[i - 1] <= 0:
            out.append(0.0)
        else:
            out.append(math.log(prices[i] / prices[i - 1]))
    return out


def simple_returns(prices: List[float]) -> List[float]:
    """R_t = (P_t - P_{t-1}) / P_{t-1}"""
    out: List[float] = [0.0]
    for i in range(1, len(prices)):
        if prices[i - 1] == 0:
            out.append(0.0)
        else:
            out.append((prices[i] - prices[i - 1]) / prices[i - 1])
    return out


# ---------------------------------------------------------------------------
# EWMA Volatility  (RiskMetrics style)
# ---------------------------------------------------------------------------
def ewma_volatility(prices: List[float], lam: float = 0.94) -> List[float]:
    """
    Exponentially Weighted Moving Average volatility.
    σ²_t = (1−λ)·r²_t + λ·σ²_{t-1}
    Returns annualized volatility (assuming daily data, 365 days for crypto).
    """
    rets = log_returns(prices)
    n = len(rets)
    if n < 2:
        return [0.0] * n

    # Initialize with first squared return
    var: float = rets[1] ** 2 if n > 1 else 0.0
    out: List[float] = [0.0, math.sqrt(var)]

    for i in range(2, n):
        var = (1 - lam) * rets[i] ** 2 + lam * var
        out.append(math.sqrt(max(var, 0.0)))

    return out


def ewma_volatility_annualized(prices: List[float], lam: float = 0.94,
                                trading_days: int = 365) -> List[float]:
    """Annualized EWMA vol: σ_annual = σ_daily * √(trading_days)"""
    daily = ewma_volatility(prices, lam)
    factor = math.sqrt(trading_days)
    return [v * factor for v in daily]


# ---------------------------------------------------------------------------
# GARCH(1,1) estimation (simple MLE-free calibration)
# ---------------------------------------------------------------------------
class GARCH11:
    """
    GARCH(1,1): σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
    Constraints: ω>0, α≥0, β≥0, α+β<1 (stationarity)

    Uses a practical quasi-MLE approach with grid search for simplicity
    (no external optimizer required).
    """

    def __init__(self, omega: float = 1e-6, alpha: float = 0.1, beta: float = 0.85):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self._validate()

    def _validate(self):
        assert self.omega > 0, "ω must be > 0"
        assert self.alpha >= 0, "α must be ≥ 0"
        assert self.beta >= 0, "β must be ≥ 0"
        persistence = self.alpha + self.beta
        if persistence >= 1.0:
            # Enforce stationarity by scaling down
            scale = 0.99 / persistence
            self.alpha *= scale
            self.beta *= scale

    def fit(self, prices: List[float]) -> 'GARCH11':
        """
        Calibrate GARCH(1,1) parameters using grid search on log-likelihood.
        This avoids scipy dependency while still finding reasonable parameters.
        """
        rets = log_returns(prices)
        if len(rets) < 10:
            return self

        # Variance of returns for initialization
        mean_r = sum(rets[1:]) / max(len(rets) - 1, 1)
        var_r = sum((r - mean_r) ** 2 for r in rets[1:]) / max(len(rets) - 2, 1)

        best_ll = float('-inf')
        best_params = (self.omega, self.alpha, self.beta)

        # Grid search over reasonable parameter ranges
        alphas = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
        betas = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94]

        for a in alphas:
            for b in betas:
                if a + b >= 0.99:
                    continue
                w = var_r * (1 - a - b)
                if w <= 0:
                    continue
                ll = self._log_likelihood(rets, w, a, b)
                if ll > best_ll:
                    best_ll = ll
                    best_params = (w, a, b)

        self.omega, self.alpha, self.beta = best_params
        return self

    @staticmethod
    def _log_likelihood(rets: List[float], omega: float, alpha: float,
                        beta: float) -> float:
        """Negative of Gaussian log-likelihood for GARCH(1,1)."""
        n = len(rets)
        if n < 3:
            return float('-inf')
        # Initialize variance
        var = sum(r ** 2 for r in rets[1:]) / (n - 1)
        ll = 0.0
        for i in range(2, n):
            var = omega + alpha * rets[i - 1] ** 2 + beta * var
            var = max(var, 1e-20)
            ll += -0.5 * (math.log(2 * math.pi) + math.log(var) + rets[i] ** 2 / var)
        return ll

    def forecast(self, prices: List[float]) -> List[float]:
        """
        Compute GARCH(1,1) conditional volatility for each timestamp.
        Returns daily volatility σ_t.
        """
        rets = log_returns(prices)
        n = len(rets)
        if n < 2:
            return [0.0] * n

        # Initialize with unconditional variance
        var = self.omega / max(1 - self.alpha - self.beta, 0.01)
        out: List[float] = [math.sqrt(max(var, 0.0))]

        for i in range(1, n):
            var = self.omega + self.alpha * rets[i] ** 2 + self.beta * var
            var = max(var, 1e-20)
            out.append(math.sqrt(var))

        return out

    def forecast_n_step(self, current_var: float, steps: int = 5) -> List[float]:
        """
        N-step ahead variance forecast.
        σ²_{t+h} = ω·Σ(α+β)^i + (α+β)^h · σ²_t
        """
        ab = self.alpha + self.beta
        uncond_var = self.omega / max(1 - ab, 0.01)
        forecasts = []
        var = current_var
        for _ in range(steps):
            var = self.omega + ab * var
            forecasts.append(math.sqrt(max(var, 0.0)))
        return forecasts


# ---------------------------------------------------------------------------
# Volatility Regime Classifier
# ---------------------------------------------------------------------------
def classify_volatility_regime(vol_series: List[float],
                                low_mult: float = 0.75,
                                high_mult: float = 1.5,
                                extreme_mult: float = 2.5
                                ) -> List[VolRegime]:
    """
    Classify each point into a volatility regime based on rolling median.
    LOW:     σ < low_mult * median(σ)
    MEDIUM:  low_mult * median ≤ σ ≤ high_mult * median
    HIGH:    high_mult * median < σ ≤ extreme_mult * median
    EXTREME: σ > extreme_mult * median
    """
    valid = [v for v in vol_series if v > 0 and not math.isnan(v)]
    if not valid:
        return [VolRegime.MEDIUM] * len(vol_series)

    sorted_vals = sorted(valid)
    median_vol = sorted_vals[len(sorted_vals) // 2]
    if median_vol == 0:
        median_vol = 1e-10

    out: List[VolRegime] = []
    for v in vol_series:
        if v <= 0 or math.isnan(v):
            out.append(VolRegime.MEDIUM)
        elif v < low_mult * median_vol:
            out.append(VolRegime.LOW)
        elif v > extreme_mult * median_vol:
            out.append(VolRegime.EXTREME)
        elif v > high_mult * median_vol:
            out.append(VolRegime.HIGH)
        else:
            out.append(VolRegime.MEDIUM)
    return out


# ---------------------------------------------------------------------------
# Volatility-Adjusted Returns
# ---------------------------------------------------------------------------
def volatility_adjusted_returns(prices: List[float], lam: float = 0.94
                                 ) -> List[float]:
    """R^adj_t = R_t / σ_t  — normalize returns by current volatility"""
    rets = simple_returns(prices)
    vols = ewma_volatility(prices, lam)
    out: List[float] = []
    for r, v in zip(rets, vols):
        if v == 0 or math.isnan(v):
            out.append(0.0)
        else:
            out.append(r / v)
    return out


# ---------------------------------------------------------------------------
# Realized Volatility (rolling window)
# ---------------------------------------------------------------------------
def realized_volatility(prices: List[float], window: int = 20) -> List[float]:
    """Rolling realized vol = std(log_returns) over window, annualized"""
    rets = log_returns(prices)
    n = len(rets)
    out: List[float] = []
    for i in range(n):
        if i + 1 < window:
            out.append(float('nan'))
        else:
            w = rets[i + 1 - window: i + 1]
            mean_w = sum(w) / window
            var_w = sum((x - mean_w) ** 2 for x in w) / (window - 1)
            out.append(math.sqrt(max(var_w, 0.0)) * math.sqrt(365))
    return out
