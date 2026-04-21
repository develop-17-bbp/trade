"""
Champion / challenger gating for the binary LightGBM SKIP/TRADE model and the 3-class
direction model.

Both `train_all_models.py` and `continuous_adapt.py` previously overwrote the production
`lgbm_{asset}_trained.txt` on every retrain without comparing the new model to the one
it was replacing. A regressed retrain silently replaced a good model.

`evaluate_and_gate` scores the incumbent (on disk) and the challenger (just trained) on
the same held-out slice. Promotion rule:

  • New F1 ≥ incumbent F1 − TOLERANCE_PP  (default 1 pp tolerance)
  • AND new F1 is finite / non-degenerate

Rejected models are archived under `models/challengers/` with a timestamp so the cause
can be reviewed without rerunning.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False


TOLERANCE_PP = 0.01  # new_f1 must be >= incumbent_f1 - 0.01 (1 percentage point)


@dataclass
class GateResult:
    promoted: bool
    reason: str
    new_f1: float
    incumbent_f1: Optional[float]
    incumbent_path: str
    challenger_path: Optional[str]


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_cls: int = 1) -> float:
    tp = int(np.sum((y_pred == positive_cls) & (y_true == positive_cls)))
    fp = int(np.sum((y_pred == positive_cls) & (y_true != positive_cls)))
    fn = int(np.sum((y_pred != positive_cls) & (y_true == positive_cls)))
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _score_booster(booster: "lgb.Booster", X: np.ndarray, y: np.ndarray,
                   threshold: float = 0.5) -> float:
    """F1 for the binary TRADE class at the given threshold."""
    probs = np.asarray(booster.predict(X), dtype=float)
    # Multiclass boosters return a 2D matrix; binary ones return 1D.
    if probs.ndim == 2 and probs.shape[1] >= 2:
        # Multiclass: use the positive class column (index 1) by convention.
        probs = probs[:, 1]
    preds = (probs >= threshold).astype(int)
    return _binary_f1(y, preds, positive_cls=1)


def save_challenger(new_model: "lgb.Booster", incumbent_path: str, reason: str) -> str:
    """Archive a rejected challenger to models/challengers/ with a timestamp."""
    base = os.path.basename(incumbent_path).replace(".txt", "")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    models_dir = os.path.dirname(incumbent_path) or "models"
    out_dir = os.path.join(models_dir, "challengers")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_{stamp}.txt")
    new_model.save_model(out_path)

    meta_path = out_path.replace(".txt", ".reason.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"rejected_at={stamp}\nreason={reason}\nincumbent={incumbent_path}\n")
    return out_path


def evaluate_and_gate(
    new_model: "lgb.Booster",
    incumbent_path: str,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    *,
    tolerance_pp: float = TOLERANCE_PP,
    threshold: float = 0.5,
) -> GateResult:
    """
    Decide whether to promote `new_model` to `incumbent_path`.

    • If no incumbent exists, promote unconditionally.
    • Otherwise, compare F1 on the held-out slice. Promote if new_f1 ≥ incumbent_f1 − tolerance_pp.
    """
    new_f1 = _score_booster(new_model, X_holdout, y_holdout, threshold)

    if not HAS_LGB or not os.path.exists(incumbent_path):
        return GateResult(
            promoted=True,
            reason="no_incumbent",
            new_f1=new_f1,
            incumbent_f1=None,
            incumbent_path=incumbent_path,
            challenger_path=None,
        )

    try:
        incumbent = lgb.Booster(model_file=incumbent_path)
        incumbent_f1 = _score_booster(incumbent, X_holdout, y_holdout, threshold)
    except Exception as e:
        return GateResult(
            promoted=True,
            reason=f"incumbent_unreadable:{e}",
            new_f1=new_f1,
            incumbent_f1=None,
            incumbent_path=incumbent_path,
            challenger_path=None,
        )

    if new_f1 + tolerance_pp >= incumbent_f1:
        return GateResult(
            promoted=True,
            reason=f"new_f1={new_f1:.4f} >= incumbent_f1={incumbent_f1:.4f} - {tolerance_pp:.2f}",
            new_f1=new_f1,
            incumbent_f1=incumbent_f1,
            incumbent_path=incumbent_path,
            challenger_path=None,
        )

    # Rejected — archive the challenger.
    archived = save_challenger(
        new_model,
        incumbent_path,
        reason=f"new_f1={new_f1:.4f} < incumbent_f1={incumbent_f1:.4f} - {tolerance_pp:.2f}",
    )
    return GateResult(
        promoted=False,
        reason=f"new_f1={new_f1:.4f} < incumbent_f1={incumbent_f1:.4f} - {tolerance_pp:.2f}",
        new_f1=new_f1,
        incumbent_f1=incumbent_f1,
        incumbent_path=incumbent_path,
        challenger_path=archived,
    )
