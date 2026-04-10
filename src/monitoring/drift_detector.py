"""
Drift Detector — Feature & Model Output Distribution Monitoring
================================================================
Detects when live feature distributions shift vs training baselines.
Uses simple statistical tests (no heavy deps) suitable for real-time trading.

Methods:
  - PSI (Population Stability Index) for feature drift
  - Z-score monitoring for prediction output drift
  - Rolling window comparison (recent vs baseline)

Thresholds:
  - PSI < 0.10 = no drift
  - 0.10 <= PSI < 0.25 = moderate drift (warning)
  - PSI >= 0.25 = significant drift (alert)
"""

import math
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Monitors feature and prediction distributions for drift.

    Usage:
        detector = DriftDetector()
        detector.set_baseline("rsi", baseline_values)
        detector.update("rsi", new_value)
        result = detector.check_all()
        # result = {"drifted": True, "alerts": ["rsi: PSI=0.31 (SIGNIFICANT)"]}
    """

    # PSI thresholds
    PSI_OK = 0.10
    PSI_WARN = 0.25

    def __init__(self, window_size: int = 500, n_bins: int = 10):
        self.window_size = window_size
        self.n_bins = n_bins

        # Baseline distributions (from training data)
        self._baselines: Dict[str, List[float]] = {}
        # Live rolling windows
        self._live: Dict[str, deque] = {}
        # Cached PSI scores
        self._psi_cache: Dict[str, float] = {}
        # Last check time
        self._last_check: float = 0
        self._check_interval: float = 300  # Re-check every 5 min

    def set_baseline(self, feature_name: str, values: List[float]):
        """Set baseline distribution for a feature (from training data)."""
        self._baselines[feature_name] = list(values)
        if feature_name not in self._live:
            self._live[feature_name] = deque(maxlen=self.window_size)

    def update(self, feature_name: str, value: float):
        """Push a new observation for a feature."""
        if feature_name not in self._live:
            self._live[feature_name] = deque(maxlen=self.window_size)
        if math.isfinite(value):
            self._live[feature_name].append(value)

    def update_batch(self, features: Dict[str, float]):
        """Push multiple feature observations at once."""
        for name, value in features.items():
            self.update(name, value)

    @staticmethod
    def _compute_psi(baseline: List[float], live: List[float], n_bins: int = 10) -> float:
        """
        Compute Population Stability Index between two distributions.

        PSI = SUM( (live_pct - base_pct) * ln(live_pct / base_pct) )

        Lower is better:
          < 0.10 = stable
          0.10-0.25 = moderate shift
          >= 0.25 = significant shift
        """
        if len(baseline) < 10 or len(live) < 10:
            return 0.0

        # Determine bin edges from baseline
        sorted_base = sorted(baseline)
        bin_edges = []
        for i in range(1, n_bins):
            idx = int(len(sorted_base) * i / n_bins)
            bin_edges.append(sorted_base[min(idx, len(sorted_base) - 1)])

        def _bin_counts(data: List[float]) -> List[int]:
            counts = [0] * (len(bin_edges) + 1)
            for v in data:
                placed = False
                for j, edge in enumerate(bin_edges):
                    if v <= edge:
                        counts[j] += 1
                        placed = True
                        break
                if not placed:
                    counts[-1] += 1
            return counts

        base_counts = _bin_counts(baseline)
        live_counts = _bin_counts(live)

        total_base = len(baseline)
        total_live = len(live)

        psi = 0.0
        eps = 1e-6  # Avoid log(0)
        for bc, lc in zip(base_counts, live_counts):
            base_pct = (bc / total_base) + eps
            live_pct = (lc / total_live) + eps
            psi += (live_pct - base_pct) * math.log(live_pct / base_pct)

        return psi

    @staticmethod
    def _zscore_drift(baseline: List[float], live: List[float]) -> float:
        """Check if live mean has shifted significantly from baseline mean."""
        if len(baseline) < 10 or len(live) < 10:
            return 0.0

        base_mean = sum(baseline) / len(baseline)
        base_var = sum((x - base_mean) ** 2 for x in baseline) / len(baseline)
        base_std = math.sqrt(base_var) if base_var > 0 else 1e-6

        live_mean = sum(live) / len(live)
        return abs(live_mean - base_mean) / base_std

    def check_feature(self, feature_name: str) -> Tuple[float, str]:
        """
        Check drift for a single feature.

        Returns:
            (psi_score, severity) where severity is "OK", "WARNING", or "ALERT"
        """
        if feature_name not in self._baselines or feature_name not in self._live:
            return 0.0, "OK"

        baseline = self._baselines[feature_name]
        live = list(self._live[feature_name])

        if len(live) < 30:
            return 0.0, "OK"  # Not enough data yet

        psi = self._compute_psi(baseline, live, self.n_bins)
        self._psi_cache[feature_name] = psi

        if psi >= self.PSI_WARN:
            return psi, "ALERT"
        elif psi >= self.PSI_OK:
            return psi, "WARNING"
        return psi, "OK"

    def check_all(self) -> Dict[str, Any]:
        """
        Check all tracked features for drift.

        Returns:
            {
                "drifted": bool,
                "alerts": List[str],
                "warnings": List[str],
                "scores": Dict[str, float],
                "timestamp": float
            }
        """
        alerts = []
        warnings = []
        scores = {}

        for feature_name in self._baselines:
            psi, severity = self.check_feature(feature_name)
            scores[feature_name] = round(psi, 4)
            if severity == "ALERT":
                alerts.append(f"{feature_name}: PSI={psi:.3f} (SIGNIFICANT DRIFT)")
                logger.warning(f"[DRIFT] ALERT: {feature_name} PSI={psi:.3f} — distribution shifted significantly")
            elif severity == "WARNING":
                warnings.append(f"{feature_name}: PSI={psi:.3f} (moderate drift)")
                logger.info(f"[DRIFT] WARNING: {feature_name} PSI={psi:.3f} — moderate distribution shift")

        self._last_check = time.time()

        return {
            "drifted": len(alerts) > 0,
            "alerts": alerts,
            "warnings": warnings,
            "scores": scores,
            "timestamp": self._last_check,
        }


# ── Backward-compatible function API ──

_global_detector: Optional[DriftDetector] = None


def get_detector() -> DriftDetector:
    """Get or create the global drift detector singleton."""
    global _global_detector
    if _global_detector is None:
        _global_detector = DriftDetector()
    return _global_detector


def check_drift(stats: Dict[str, Any]) -> bool:
    """
    Check if any tracked feature has drifted significantly.

    Args:
        stats: Dict of feature_name -> current_value. Values are pushed
               to the rolling window and checked against baselines.

    Returns:
        True if significant drift detected, False otherwise.
    """
    detector = get_detector()

    # Push new observations
    for key, value in stats.items():
        if isinstance(value, (int, float)) and math.isfinite(value):
            detector.update(key, value)

    # Only run full check periodically (avoid CPU overhead per bar)
    if time.time() - detector._last_check < detector._check_interval:
        # Quick check: return cached result
        return any(psi >= DriftDetector.PSI_WARN for psi in detector._psi_cache.values())

    result = detector.check_all()
    return result["drifted"]
