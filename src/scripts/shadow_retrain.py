"""
Shadow-retrain — rebuild the meta model from LIVE trade outcomes.

The forward-simulated labels used by train_meta_label.py are an approximation.
Once the bot has run with `ACT_META_SHADOW_MODE=1` for enough trades, we have
ground-truth data: features at entry + actual win/loss. This script uses that
data to replace the simulated-label meta model.

Workflow:
  1. Read logs/meta_shadow.jsonl
  2. Join shadow_predict and shadow_outcome records by trade_id
  3. Report: n trades, actual WR, shadow-model veto precision
  4. If n >= MIN_JOINED, train LightGBM on (features -> actual win)
  5. Calibrate + champion-gate + persist to lgbm_{asset}_meta.txt

Run after >= 100 joined trades. Reports stats first and bails with a clear
message if the dataset is too small.

Usage:
    python -m src.scripts.shadow_retrain --asset BTC
    python -m src.scripts.shadow_retrain --asset ETH --min-joined 150 --force
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("shadow_retrain")


MIN_JOINED = 100            # Refuse to retrain below this many completed trades
TRAIN_VAL_SPLIT = 0.80
BOOST_ROUNDS = 300
EARLY_STOPPING = 30


def _filter_joined(joined: List[Dict], asset: str) -> List[Dict]:
    """Keep only records for the requested asset with a real features vector."""
    out = []
    for r in joined:
        if r.get("asset") != asset:
            continue
        feats = r.get("features") or []
        if not isinstance(feats, list) or len(feats) < 10:
            continue
        out.append(r)
    return out


def _class_weights(y: np.ndarray) -> np.ndarray:
    n = len(y)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return np.ones(n, dtype=float)
    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    return np.where(y == 1, w_pos, w_neg).astype(float)


def _optimal_threshold(probs: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Sweep TAKE thresholds; return the one that maximizes F1."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.20, 0.85, 0.01):
        p = (probs >= t).astype(int)
        tp = int(np.sum((p == 1) & (y == 1)))
        fp = int(np.sum((p == 1) & (y == 0)))
        fn = int(np.sum((p == 0) & (y == 1)))
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def retrain(asset: str, min_joined: int, models_dir: str, force: bool) -> Dict:
    from src.ml import shadow_log
    from src.ml.calibration import fit_calibration, save_calibration, calibration_path_for
    from src.ml.champion_gate import evaluate_and_gate
    from src.ml.gpu import lgbm_device_params
    import lightgbm as lgb

    logger.info(f"Loading shadow log (default path: {shadow_log.DEFAULT_LOG_PATH})...")
    records = shadow_log.read_all()
    logger.info(f"  {len(records)} total records")

    joined = shadow_log.join_predict_outcome(records)
    logger.info(f"  {len(joined)} joined trades (predict + outcome present)")

    asset_rows = _filter_joined(joined, asset)
    logger.info(f"  {len(asset_rows)} for asset={asset} with valid features")

    stats = shadow_log.shadow_stats(asset_rows)
    logger.info(f"  stats: {stats}")

    if len(asset_rows) < min_joined:
        logger.warning(
            f"Need at least {min_joined} joined trades for {asset}; have {len(asset_rows)}. "
            f"Retrain blocked — keep shadow-mode running and try again later."
        )
        return {"asset": asset, "n": len(asset_rows), "promoted": False, "reason": "insufficient_data"}

    # Build X, y time-ordered (trade_id suffix usually monotonic)
    asset_rows_sorted = sorted(asset_rows, key=lambda r: r.get("trade_id", ""))
    X = np.asarray([r["features"] for r in asset_rows_sorted], dtype=np.float32)
    y = np.asarray([int(r["win"]) for r in asset_rows_sorted], dtype=int)
    logger.info(f"Training matrix: X={X.shape} y_win_rate={y.mean():.3f}")

    split = int(len(X) * TRAIN_VAL_SPLIT)
    if split < 20 or len(X) - split < 20:
        logger.warning(f"Split too small (train={split}, test={len(X)-split}); need 20+ per side")
        return {"asset": asset, "n": len(X), "promoted": False, "reason": "split_too_small"}

    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    w_tr = _class_weights(y_tr)

    # Force-reset incumbent
    incumbent_path = os.path.join(models_dir, f"lgbm_{asset.lower()}_meta.txt")
    if force:
        for suffix in ("meta.txt", "meta_calibration.json", "meta_thresholds.json"):
            stale = os.path.join(models_dir, f"lgbm_{asset.lower()}_{suffix}")
            if os.path.exists(stale):
                os.remove(stale)
                logger.info(f"--force: removed {stale}")

    device_params = lgbm_device_params()
    logger.info(f"LightGBM device: {device_params.get('device', 'cpu')}")

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_data_in_leaf": 10,
        "lambda_l1": 0.1,
        "lambda_l2": 0.2,
        "verbose": -1,
        **device_params,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    dval = lgb.Dataset(X_te, label=y_te, reference=dtrain)
    model = lgb.train(
        params, dtrain, num_boost_round=BOOST_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(50)],
    )

    # Metrics on held-out
    probs = np.asarray(model.predict(X_te), dtype=float)
    preds = (probs > 0.5).astype(int)
    acc = float(np.mean(preds == y_te))
    tp = int(np.sum((preds == 1) & (y_te == 1)))
    fp = int(np.sum((preds == 1) & (y_te == 0)))
    fn = int(np.sum((preds == 0) & (y_te == 1)))
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    logger.info(f"Held-out: acc={acc:.3f} | TAKE P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    # Champion gate
    os.makedirs(models_dir, exist_ok=True)
    gate = evaluate_and_gate(model, incumbent_path, X_te, y_te, threshold=0.5)
    logger.info(f"Gate: {'PROMOTED' if gate.promoted else 'REJECTED'} — {gate.reason}")
    if not gate.promoted:
        return {"asset": asset, "n": len(X), "promoted": False, "reason": gate.reason,
                "test_accuracy": acc, "test_f1": f1}

    model.save_model(incumbent_path)

    # Calibration
    bundle = fit_calibration(raw_probs=list(probs), y_true=list(y_te), asset=asset)
    if bundle is not None:
        save_calibration(bundle, calibration_path_for(models_dir, asset))
        # Also write the meta-specific calibration filename used by the executor
        save_calibration(bundle, os.path.join(models_dir, f"lgbm_{asset.lower()}_meta_calibration.json"))
        logger.info(f"Calibration: deltas={bundle.deltas} base_wr={bundle.baseline_win_rate:.3f} n={bundle.fit_n_samples}")

    # Optimal threshold for veto
    best_t, best_f1 = _optimal_threshold(probs, y_te)
    thresh_path = os.path.join(models_dir, f"lgbm_{asset.lower()}_meta_thresholds.json")
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump({
            "take_threshold": best_t, "take_f1": best_f1,
            "n_train": len(y_tr), "n_test": len(y_te),
            "base_win_rate": float(y.mean()),
            "source": "shadow_retrain",
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)
    logger.info(f"Thresholds: take>={best_t:.2f} (F1={best_f1:.3f})")

    return {
        "asset": asset, "n": len(X), "promoted": True,
        "test_accuracy": acc, "test_f1": f1,
        "optimal_take_threshold": best_t, "optimal_f1": best_f1,
        "shadow_stats": stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--min-joined", type=int, default=MIN_JOINED,
                    help="Minimum joined predict+outcome records required to retrain")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--force", action="store_true",
                    help="Delete existing meta model files before training")
    ap.add_argument("--stats-only", action="store_true",
                    help="Print shadow stats without retraining")
    args = ap.parse_args()

    if args.stats_only:
        from src.ml import shadow_log
        recs = shadow_log.read_all()
        joined = shadow_log.join_predict_outcome(recs)
        for asset in ("BTC", "ETH"):
            rows = _filter_joined(joined, asset)
            stats = shadow_log.shadow_stats(rows)
            print(f"\n=== {asset} shadow stats ===")
            print(json.dumps(stats, indent=2))
        return 0

    t0 = time.time()
    try:
        result = retrain(
            asset=args.asset, min_joined=args.min_joined,
            models_dir=args.models_dir, force=args.force,
        )
    except Exception as e:
        logger.error(f"Retrain failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 2
    logger.info(f"Done in {time.time() - t0:.1f}s. Result: {json.dumps(result, default=str)}")
    return 0 if result.get("promoted") else 1


if __name__ == "__main__":
    sys.exit(main())
