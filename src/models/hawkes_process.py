"""
Hawkes Process — Self-Exciting Point Process for Trade Event Clustering
=========================================================================
Models the arrival intensity of significant market events (large moves,
liquidations, volume spikes) as a self-exciting process where past events
increase the probability of future events.

Intensity:  λ(t) = μ + Σ_{t_i < t} α · exp(-β · (t - t_i))

  μ = baseline intensity (events per unit time)
  α = excitation factor (how much each event boosts intensity)
  β = decay rate (how fast excitation fades)
  α/β = branching ratio (< 1 for stability)

Applications:
  - Predict clustering of volatile moves (flash crashes, cascading liquidations)
  - Optimal trade timing: enter when intensity is LOW (calm after storm)
  - Risk: increase caution when intensity is HIGH (event clustering)

Usage:
    from src.models.hawkes_process import HawkesProcess
    hp = HawkesProcess()
    hp.fit(event_times)
    intensity = hp.current_intensity(event_times)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HawkesProcess:
    """
    Univariate Hawkes process with exponential kernel.
    """

    def __init__(self, mu: float = 0.1, alpha: float = 0.5,
                 beta: float = 1.0):
        """
        Args:
            mu: Baseline intensity
            alpha: Excitation amplitude
            beta: Decay rate (higher = faster decay)
        """
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self._fitted = False

    def fit(self, event_times: np.ndarray, max_iter: int = 100,
            lr: float = 0.01) -> Dict:
        """
        Fit Hawkes parameters via MLE using gradient ascent.

        Log-likelihood:
          L = Σ log(λ(t_i)) - ∫_0^T λ(t) dt

        Args:
            event_times: Array of event timestamps (sorted, non-negative)
            max_iter: Maximum optimization iterations
            lr: Learning rate

        Returns:
            {mu, alpha, beta, branching_ratio, log_likelihood}
        """
        times = np.sort(np.asarray(event_times, dtype=float))
        n = len(times)
        if n < 5:
            return self._default_params()

        T = times[-1] - times[0]
        if T <= 0:
            return self._default_params()

        # Normalize times to start at 0
        times = times - times[0]

        # Initialize
        mu = max(0.01, n / (2 * T))  # Start with half the empirical rate
        alpha = 0.3
        beta = 1.0

        best_ll = -np.inf

        for iteration in range(max_iter):
            # Compute intensity at each event
            intensities = np.zeros(n)
            A = np.zeros(n)  # Recursive auxiliary variable

            for i in range(n):
                if i == 0:
                    A[i] = 0
                else:
                    dt = times[i] - times[i - 1]
                    A[i] = np.exp(-beta * dt) * (1 + A[i - 1])
                intensities[i] = mu + alpha * A[i]

            # Ensure positive intensities
            intensities = np.maximum(intensities, 1e-10)

            # Log-likelihood
            # L = Σ log(λ(t_i)) - μ·T - (α/β) · Σ(1 - exp(-β(T - t_i)))
            ll_events = np.sum(np.log(intensities))
            ll_integral = mu * T + (alpha / beta) * np.sum(
                1 - np.exp(-beta * (T - times))
            )
            ll = ll_events - ll_integral

            if ll > best_ll:
                best_ll = ll
                self.mu = mu
                self.alpha = alpha
                self.beta = beta

            # Gradients (simplified)
            # dL/dmu = Σ(1/λ_i) - T
            grad_mu = np.sum(1.0 / intensities) - T

            # dL/dalpha = Σ(A_i/λ_i) - (1/β)·Σ(1 - exp(-β(T-t_i)))
            grad_alpha = np.sum(A / intensities) - \
                         (1 / beta) * np.sum(1 - np.exp(-beta * (T - times)))

            # Update with constraints
            mu = max(1e-4, mu + lr * grad_mu)
            alpha = max(1e-4, alpha + lr * grad_alpha)
            # Keep branching ratio < 1 for stability
            if alpha / beta >= 0.99:
                beta = alpha / 0.95

        self._fitted = True
        branching = self.alpha / self.beta

        logger.info(f"Hawkes fit: μ={self.mu:.4f}, α={self.alpha:.4f}, "
                     f"β={self.beta:.4f}, branching={branching:.3f}")

        return {
            'mu': self.mu,
            'alpha': self.alpha,
            'beta': self.beta,
            'branching_ratio': branching,
            'log_likelihood': best_ll,
            'n_events': n,
            'time_span': T,
        }

    def current_intensity(self, event_times: np.ndarray,
                          current_time: Optional[float] = None) -> float:
        """
        Compute current Hawkes intensity.

        λ(t) = μ + α · Σ exp(-β·(t - t_i))
        """
        times = np.asarray(event_times, dtype=float)
        t = current_time if current_time is not None else times[-1]

        past = times[times < t]
        if len(past) == 0:
            return self.mu

        excitation = self.alpha * np.sum(np.exp(-self.beta * (t - past)))
        return self.mu + excitation

    def intensity_series(self, event_times: np.ndarray,
                         eval_times: np.ndarray) -> np.ndarray:
        """Compute intensity at each evaluation time."""
        result = np.zeros(len(eval_times))
        times = np.asarray(event_times, dtype=float)

        for i, t in enumerate(eval_times):
            past = times[times < t]
            if len(past) == 0:
                result[i] = self.mu
            else:
                excitation = self.alpha * np.sum(np.exp(-self.beta * (t - past)))
                result[i] = self.mu + excitation

        return result

    def detect_events(self, prices: np.ndarray, threshold_pct: float = 0.02,
                      volume: Optional[np.ndarray] = None,
                      vol_threshold: float = 2.0) -> np.ndarray:
        """
        Extract event times from price/volume data.

        Events = bars where |return| > threshold OR volume > vol_threshold × avg.
        """
        prices = np.asarray(prices, dtype=float)
        returns = np.abs(np.diff(prices) / (prices[:-1] + 1e-12))

        events = returns > threshold_pct

        if volume is not None:
            vol = np.asarray(volume, dtype=float)
            avg_vol = np.mean(vol[-50:]) if len(vol) >= 50 else np.mean(vol)
            vol_spikes = vol[1:] > vol_threshold * avg_vol
            events = events | vol_spikes

        # Return indices (as float "times")
        return np.where(events)[0].astype(float)

    def trade_timing_signal(self, prices: np.ndarray,
                            volume: Optional[np.ndarray] = None) -> Dict:
        """
        Generate trade timing signal based on Hawkes intensity.

        HIGH intensity → Wait (event clustering, dangerous)
        LOW intensity → Trade (calm period, favorable entry)
        MEDIUM → Normal

        Returns:
            {intensity, regime, trade_allowed, intensity_percentile}
        """
        event_times = self.detect_events(prices, volume=volume)

        if len(event_times) < 3:
            return {
                'intensity': self.mu,
                'regime': 'calm',
                'trade_allowed': True,
                'intensity_percentile': 0.5,
                'event_count': 0,
            }

        # Fit if not already
        if not self._fitted:
            self.fit(event_times)

        # Current intensity
        current = self.current_intensity(event_times)

        # Historical intensity distribution
        eval_times = np.arange(max(0, event_times[0]), event_times[-1] + 1)
        hist_intensity = self.intensity_series(event_times, eval_times)
        percentile = float(np.mean(hist_intensity <= current))

        # Regime classification
        if percentile > 0.85:
            regime = 'clustering'
            trade_allowed = False
        elif percentile > 0.65:
            regime = 'elevated'
            trade_allowed = True
        elif percentile < 0.25:
            regime = 'calm'
            trade_allowed = True
        else:
            regime = 'normal'
            trade_allowed = True

        return {
            'intensity': float(current),
            'regime': regime,
            'trade_allowed': trade_allowed,
            'intensity_percentile': percentile,
            'event_count': len(event_times),
            'branching_ratio': self.alpha / self.beta,
        }

    def _default_params(self) -> Dict:
        return {
            'mu': self.mu,
            'alpha': self.alpha,
            'beta': self.beta,
            'branching_ratio': self.alpha / self.beta,
            'log_likelihood': 0.0,
            'n_events': 0,
            'time_span': 0.0,
        }
