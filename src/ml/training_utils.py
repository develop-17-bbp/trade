"""Shared training-loop helpers used by meta-label + shadow retrainers.

Consolidates duplicates that were present in src/scripts/train_meta_label.py and
src/scripts/shadow_retrain.py so future callers (a third retrainer, a notebook)
have one place to import from.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def inverse_freq_weights(y: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """Return per-sample inverse-frequency weights for `y`.

    For balanced classes this returns all-1.0. For a 90/10 split on binary y,
    the minority class gets weight ~5.0 and the majority ~0.555. When any class
    has zero samples, returns uniform 1.0 (caller usually guards against this
    earlier).
    """
    y_arr = np.asarray(y, dtype=int)
    n = len(y_arr)
    counts = np.bincount(y_arr, minlength=n_classes).astype(float)
    if np.any(counts == 0):
        return np.ones(n, dtype=float)
    per_class = n / (n_classes * counts)
    return per_class[y_arr].astype(float)


def best_f1_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    positive_cls: int = 1,
    lo: float = 0.20,
    hi: float = 0.85,
    step: float = 0.01,
) -> Tuple[float, float]:
    """Sweep TAKE thresholds over [lo, hi) and return (best_threshold, best_f1).

    Produces (0.5, 0.0) if no threshold yields a TP.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_true, dtype=int)
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(lo, hi, step):
        preds = (p >= t).astype(int)
        tp = int(np.sum((preds == positive_cls) & (y == positive_cls)))
        fp = int(np.sum((preds == positive_cls) & (y != positive_cls)))
        fn = int(np.sum((preds != positive_cls) & (y == positive_cls)))
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1
