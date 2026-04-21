"""ML support utilities: probability calibration, score-delta mapping, champion gating."""
from .calibration import (
    fit_calibration,
    load_calibration,
    apply_calibration,
    score_delta_for,
    CalibrationBundle,
    DEFAULT_FALLBACK_DELTAS,
)
from .champion_gate import evaluate_and_gate, save_challenger
from .gpu import lgbm_device_params, describe_device

__all__ = [
    "fit_calibration",
    "load_calibration",
    "apply_calibration",
    "score_delta_for",
    "CalibrationBundle",
    "DEFAULT_FALLBACK_DELTAS",
    "evaluate_and_gate",
    "save_challenger",
    "lgbm_device_params",
    "describe_device",
]
