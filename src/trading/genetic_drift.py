"""
Drift-Triggered Immigrants (P3 of genetic-loop audit)
======================================================
DRED — Detection of Regime/Environmental Drift — using a lightweight
rolling-window statistical detector. When drift is detected, the
genetic engine injects fresh-random "immigrant" DNAs into the
population to regain exploration after a long, converged run.

Why this matters: ACT's existing genetic loop is run on a 2h cadence.
After 5-10 cycles the population converges around the dominant regime
of the last few months. When the regime *changes* (HMM transition,
vol regime shift, macro pivot), that converged population is unfit
for the new world. Without drift-triggered immigration, the bot keeps
recycling the previous regime's winners while live PnL bleeds.

Detector (lightweight, no dependency on HMM):
  * Rolling 60-bar return mean and stdev
  * Compare to baseline (older 240-bar window)
  * Z-score of mean shift > 2.0 OR variance ratio > 2.0 → drift
  * Optional: feed in HMM regime labels if available; immigrate when
    regime label changes.

Action on drift:
  * Replace bottom 20% of population with random immigrants
  * Reset adaptive mutation rate to 0.3 (force exploration)
  * Log to logs/genetic_drift.jsonl with the trigger reason
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRIFT_LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "genetic_drift.jsonl")


@dataclass
class DriftSignal:
    drift_detected: bool
    z_score_mean: float
    variance_ratio: float
    regime_label_changed: bool
    last_regime: Optional[str]
    current_regime: Optional[str]
    triggers: List[str] = field(default_factory=list)
    detector_window_bars: int = 60
    baseline_window_bars: int = 240

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "z_score_mean": round(float(self.z_score_mean), 4),
            "variance_ratio": round(float(self.variance_ratio), 4),
            "regime_label_changed": self.regime_label_changed,
            "last_regime": self.last_regime,
            "current_regime": self.current_regime,
            "triggers": self.triggers,
            "detector_window_bars": self.detector_window_bars,
            "baseline_window_bars": self.baseline_window_bars,
        }


def detect_drift(
    closes: List[float],
    detector_window: int = 60,
    baseline_window: int = 240,
    z_threshold: float = 2.0,
    variance_ratio_threshold: float = 2.0,
    last_regime: Optional[str] = None,
    current_regime: Optional[str] = None,
) -> DriftSignal:
    """Detect regime drift using rolling-window statistics.

    Compares the most-recent `detector_window` returns to the older
    `baseline_window` returns immediately preceding the detector
    window.

    Returns DriftSignal with `.drift_detected = True` if any of:
      * |z_score of return mean shift| >= z_threshold
      * variance ratio (recent / baseline) >= variance_ratio_threshold
      * regime label changed (if both last/current provided)
    """
    triggers: List[str] = []
    z_score = 0.0
    var_ratio = 1.0
    label_changed = False

    if len(closes) >= detector_window + baseline_window + 1:
        prices = np.array(closes, dtype=float)
        rets = np.diff(prices) / prices[:-1]
        recent = rets[-detector_window:]
        base = rets[-(detector_window + baseline_window):-detector_window]
        if len(recent) > 1 and len(base) > 1:
            mu_r = float(np.mean(recent))
            mu_b = float(np.mean(base))
            sd_b = float(np.std(base)) or 1e-9
            sd_r = float(np.std(recent)) or 1e-9
            z_score = (mu_r - mu_b) / (sd_b / math.sqrt(len(recent)))
            var_ratio = (sd_r ** 2) / (sd_b ** 2 + 1e-12)
            if abs(z_score) >= z_threshold:
                triggers.append(f"z_score_mean_shift={z_score:.2f}")
            if var_ratio >= variance_ratio_threshold:
                triggers.append(f"variance_ratio={var_ratio:.2f}")

    if last_regime and current_regime and last_regime != current_regime:
        label_changed = True
        triggers.append(f"regime_label:{last_regime}->{current_regime}")

    return DriftSignal(
        drift_detected=bool(triggers),
        z_score_mean=z_score,
        variance_ratio=var_ratio,
        regime_label_changed=label_changed,
        last_regime=last_regime,
        current_regime=current_regime,
        triggers=triggers,
        detector_window_bars=detector_window,
        baseline_window_bars=baseline_window,
    )


def inject_immigrants(
    engine: Any,
    fraction: float = 0.20,
    reset_mutation_rate: bool = True,
    logger_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Replace the bottom `fraction` of the population with random immigrants.

    Returns a dict with the count replaced and the previous mutation rate.
    """
    from src.trading.genetic_strategy_engine import StrategyDNA

    if not engine.population:
        return {"replaced": 0, "reason": "empty_population"}

    pop = engine.population
    pop.sort(key=lambda d: getattr(d, "fitness", 0.0))  # ascending
    n_replace = max(1, int(len(pop) * fraction))

    prev_mut_rate = float(getattr(engine, "_current_mutation_rate", 0.15))
    new_mut_rate = max(0.30, prev_mut_rate)

    for i in range(n_replace):
        immigrant = StrategyDNA()
        immigrant.mutate(mutation_rate=0.8)
        immigrant.name = f"IMMIG_{getattr(engine, '_total_generations', 0)}_{i:02d}"
        pop[i] = immigrant

    if reset_mutation_rate:
        engine._current_mutation_rate = new_mut_rate
        engine._stagnation_counter = 0  # reset stagnation

    stats = {
        "replaced": n_replace,
        "population_size": len(pop),
        "prev_mutation_rate": round(prev_mut_rate, 4),
        "new_mutation_rate": round(new_mut_rate, 4),
        "extra": logger_extra or {},
    }

    # Persist to drift log
    try:
        os.makedirs(os.path.dirname(DRIFT_LOG_PATH), exist_ok=True)
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **stats,
        }
        with open(DRIFT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass

    return stats


def maybe_inject_immigrants_on_drift(
    engine: Any,
    closes: List[float],
    detector_window: int = 60,
    baseline_window: int = 240,
    immigrant_fraction: float = 0.20,
    last_regime: Optional[str] = None,
    current_regime: Optional[str] = None,
) -> Dict[str, Any]:
    """High-level convenience: detect drift, inject immigrants if drifted."""
    signal = detect_drift(
        closes,
        detector_window=detector_window,
        baseline_window=baseline_window,
        last_regime=last_regime,
        current_regime=current_regime,
    )
    out = {"drift": signal.to_dict()}
    if signal.drift_detected:
        out["immigration"] = inject_immigrants(
            engine,
            fraction=immigrant_fraction,
            logger_extra={"drift_signal": signal.to_dict()},
        )
        logger.info(
            "[DRIFT] triggers=%s replaced=%d", signal.triggers,
            out["immigration"].get("replaced", 0),
        )
    else:
        out["immigration"] = {"replaced": 0, "reason": "no_drift"}
    return out


__all__ = [
    "DriftSignal",
    "detect_drift",
    "inject_immigrants",
    "maybe_inject_immigrants_on_drift",
]
