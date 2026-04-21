"""
Meta-label trainer (López de Prado style) — ACT v1.

The existing binary trainer (train_all_models.py:1313-1354) labels EVERY bar by
forward-simulating a trailing-SL trade. That trains the model to answer "would
a random trade at this bar have worked?" — a question that has nothing to do
with whether the rule strategy would actually enter here. Result: 2026-04-22
ablation showed ML-on losing $330 more than ML-off on BTC 5m / 8 days.

This trainer fixes the training question:

  1. Walk historical bars and harvest every bar where the rule-based
     strategy's signal_generator would fire an entry (signal != NEUTRAL,
     score >= min_entry_score).
  2. For each harvested (rule-signaled) bar, simulate the trade forward
     with the safe-entries stop floor + 3:1 R:R + 60-bar time cap; label
     `win = final_pnl_pct > 0`.
  3. Train LightGBM on (features_at_signal_bar -> win_label) with
     inverse-frequency sample weights + time-ordered 80/20 split.
  4. Calibrate via src.ml.calibration.fit_calibration so the executor's
     delta-mapping reflects actual per-bucket rule-signal win rate.
  5. Champion-gate against any existing meta model; persist to
     models/lgbm_{asset}_meta.txt + lgbm_{asset}_meta_calibration.json.

The executor loads `_meta.txt` if present and uses it as a VETO-ONLY gate
AFTER rule signals pass (see executor wiring). The model can only subtract
from entry score, never add — so meta-model false positives can only reduce
trade count, never take trades the rules didn't already approve.

Usage:
    python -m src.scripts.train_meta_label --asset BTC --days 180 --primary-tf 5m
    python -m src.scripts.train_meta_label --asset ETH --days 180
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("meta_label")


MIN_ENTRY_SCORE = 4            # Default rule-signal threshold (matches adaptive.min_entry_score)
WARMUP_BARS = 100              # Skip the first N bars so indicators have data
FORWARD_SIM_MAX_BARS = 60      # Longest a simulated trade can run before time-exit
TRAIN_VAL_SPLIT = 0.80         # Time-ordered split
MIN_SAMPLES = 200              # Refuse to train below this many rule signals
BOOST_ROUNDS = 400
EARLY_STOPPING = 40


def _simulate_forward(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    entry_idx: int,
    direction: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    max_bars: int = FORWARD_SIM_MAX_BARS,
) -> Tuple[int, float, str]:
    """Walk forward and return (exit_idx, pnl_pct, reason).

    v2 (2026-04-22): the original 3:1 R:R exits produced a 6.7% label WR on BTC
    because prices rarely move 5% in 60 5m bars. Now SL is still honored (the
    safe-entries floor protects against tail loss), but TP is NOT — trades exit
    only on SL hit or the time cap. Label = final pnl sign after spread, which
    matches how the live executor actually closes most positions (trailing SL
    ratchet or time-based exit near entry).

    Exits on first of:
      * SL hit (low<=sl for LONG, high>=sl for SHORT) -> reason="sl"
      * Bar `entry_idx + max_bars` -> reason="time"; uses close price
    """
    # tp_price kept in signature for backwards compat; intentionally ignored here.
    _ = tp_price
    end = min(entry_idx + 1 + max_bars, len(closes) - 1)
    for j in range(entry_idx + 1, end + 1):
        hi = float(highs[j])
        lo = float(lows[j])
        if direction == "LONG":
            if lo <= sl_price:
                pnl_pct = (sl_price - entry_price) / entry_price * 100.0
                return j, pnl_pct, "sl"
        else:
            if hi >= sl_price:
                pnl_pct = (entry_price - sl_price) / entry_price * 100.0
                return j, pnl_pct, "sl"
    # Timed out — use close price of the last bar in the window
    cl = float(closes[end])
    if direction == "LONG":
        pnl_pct = (cl - entry_price) / entry_price * 100.0
    else:
        pnl_pct = (entry_price - cl) / entry_price * 100.0
    return end, pnl_pct, "time"


def _harvest_rule_signals(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    volumes: np.ndarray,
    ema_period: int,
    min_score: int,
    longs_only: bool,
    spread_pct: float,
) -> List[Dict]:
    """Walk bars and emit a record per bar where the rule strategy would enter."""
    from src.backtesting.signal_generator import compute_tf_signal, compute_entry_score, compute_indicator_context
    from src.indicators.indicators import atr as _atr_fn, ema as _ema_fn

    n = len(closes)
    out: List[Dict] = []

    # Precompute EMA + ATR arrays once — using same API as executor
    ema_full = np.asarray(_ema_fn(closes.tolist(), ema_period), dtype=float)
    atr_full = np.asarray(_atr_fn(highs.tolist(), lows.tolist(), closes.tolist(), 14), dtype=float)

    for i in range(WARMUP_BARS, n - FORWARD_SIM_MAX_BARS - 1):
        # Build a rolling OHLCV window ending at bar i (so signal_generator sees
        # the same slice the executor would have at that bar).
        window_start = max(0, i - 200)
        ohlcv = {
            "closes": closes[window_start:i + 1].tolist(),
            "highs": highs[window_start:i + 1].tolist(),
            "lows": lows[window_start:i + 1].tolist(),
            "opens": opens[window_start:i + 1].tolist(),
            "volumes": volumes[window_start:i + 1].tolist(),
        }
        sig = compute_tf_signal(ohlcv, ema_period=ema_period)
        signal = sig.get("signal", "NEUTRAL")
        if signal == "NEUTRAL":
            continue
        if longs_only and signal != "BUY":
            continue

        ema_vals = sig.get("ema_vals") or _ema_fn(ohlcv["closes"], ema_period)
        ema_direction = sig.get("ema_direction", "")
        price = float(sig.get("price", ohlcv["closes"][-2]))

        try:
            ctx = compute_indicator_context(ohlcv, ema_vals, spread_pct=spread_pct)
        except Exception:
            ctx = {}

        try:
            score, reasons = compute_entry_score(
                signal=signal, ohlcv=ohlcv, ema_vals=ema_vals,
                ema_direction=ema_direction, price=price, indicator_context=ctx,
            )
        except Exception:
            continue
        if score < min_score:
            continue

        out.append({
            "bar_idx": i,
            "direction": "LONG" if signal == "BUY" else "SHORT",
            "entry_price": float(closes[i]),
            "entry_score": int(score),
            "atr": float(atr_full[i] if i < len(atr_full) else 0.0),
            "ema_at_signal": float(ema_full[i] if i < len(ema_full) else price),
        })
    logger.info(f"Harvested {len(out)} rule-signaled bars from {n} total")
    return out


def _label_signal(record: Dict, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                  rt_spread_pct: float) -> Optional[Dict]:
    """Compute the win/loss label for one rule signal by forward-simulating its trade."""
    from src.trading.safe_entries import apply_stop_floor, synthesize_tp, DEFAULT_CONFIG

    entry = record["entry_price"]
    direction = record["direction"]
    atr_val = record["atr"]
    if atr_val <= 0 or entry <= 0:
        return None

    # Use the safe-entries stop floor so the training labels match the gate the
    # executor applies at runtime. Without this, training trades have tight
    # stops that the live bot never uses.
    naive_sl = entry - 1.5 * atr_val if direction == "LONG" else entry + 1.5 * atr_val
    sl, _reason = apply_stop_floor(
        entry=entry, sl_price=naive_sl, direction=direction,
        atr=atr_val, rt_spread_pct=rt_spread_pct, config=DEFAULT_CONFIG,
    )
    tp = synthesize_tp(entry=entry, sl=sl, direction=direction, min_rr=float(DEFAULT_CONFIG["min_rr"]))

    exit_idx, pnl_pct, reason = _simulate_forward(
        closes=closes, highs=highs, lows=lows,
        entry_idx=record["bar_idx"], direction=direction,
        entry_price=entry, sl_price=sl, tp_price=tp,
    )
    # Deduct round-trip spread so label = "profit net of transaction cost"
    pnl_net = pnl_pct - rt_spread_pct
    return {
        **record,
        "sl": sl, "tp": tp,
        "exit_idx": exit_idx, "pnl_pct": pnl_pct, "pnl_net_pct": pnl_net,
        "exit_reason": reason,
        "label": 1 if pnl_net > 0 else 0,
    }


def train(
    asset: str,
    days: int,
    primary_tf: str,
    min_score: int,
    longs_only: bool,
    spread_pct: float,
    models_dir: str,
    force: bool = False,
) -> Dict:
    from src.backtesting.data_loader import fetch_backtest_data
    from src.scripts.train_all_models import compute_strategy_features

    # --force wipes the incumbent + its calibration so champion gate promotes freely.
    # Rationale: prior bad-label runs leave a worse-than-nothing F1 that blocks future
    # attempts with different labeling schemes. We only force-delete when explicitly asked.
    if force:
        for suffix in ('meta.txt', 'meta_calibration.json', 'meta_thresholds.json'):
            stale = os.path.join(models_dir, f"lgbm_{asset.lower()}_{suffix}")
            if os.path.exists(stale):
                os.remove(stale)
                logger.info(f"--force: removed stale {stale}")

    logger.info(f"Fetching {days}d of {asset} @ {primary_tf}...")
    data = fetch_backtest_data(asset=asset, days=days, primary_tf=primary_tf, local_only=False)
    primary = data.primary
    closes = np.asarray(primary["closes"], dtype=float)
    highs = np.asarray(primary["highs"], dtype=float)
    lows = np.asarray(primary["lows"], dtype=float)
    opens = np.asarray(primary.get("opens", primary["closes"]), dtype=float)
    volumes = np.asarray(primary.get("volumes", [0.0] * len(closes)), dtype=float)
    logger.info(f"Loaded {len(closes)} bars")

    logger.info("Extracting 50-feature matrix over all bars...")
    X_all, _ = compute_strategy_features(
        closes.tolist(), highs.tolist(), lows.tolist(),
        opens.tolist(), volumes.tolist(),
        seq_len=1, n_features=50, spread_cost_pct=spread_pct,
    )
    if X_all is None or len(X_all) == 0:
        raise RuntimeError("compute_strategy_features returned empty matrix")
    feat_mat = np.asarray(X_all).reshape(-1, X_all.shape[-1])
    logger.info(f"Feature matrix: {feat_mat.shape}")

    logger.info("Harvesting rule signals...")
    signals = _harvest_rule_signals(
        closes=closes, highs=highs, lows=lows, opens=opens, volumes=volumes,
        ema_period=8, min_score=min_score, longs_only=longs_only, spread_pct=spread_pct,
    )
    if len(signals) < MIN_SAMPLES:
        raise RuntimeError(
            f"Only {len(signals)} rule signals harvested (need {MIN_SAMPLES}). "
            f"Try --days larger or --primary-tf longer."
        )

    logger.info("Labeling each signal by forward simulation...")
    labeled: List[Dict] = []
    for rec in signals:
        lab = _label_signal(rec, closes, highs, lows, rt_spread_pct=spread_pct)
        if lab is not None:
            labeled.append(lab)
    logger.info(
        f"Labeled {len(labeled)} signals; win rate = "
        f"{sum(1 for r in labeled if r['label'] == 1) / max(1, len(labeled)):.3f}"
    )

    if len(labeled) < MIN_SAMPLES:
        raise RuntimeError(f"Only {len(labeled)} labeled signals after sim; need {MIN_SAMPLES}")

    # Build (X, y) — features at signal bar, label = win/loss
    idxs = [r["bar_idx"] for r in labeled]
    # feat_mat row i corresponds to bar i (seq_len=1). Guard shape.
    if feat_mat.shape[0] != len(closes):
        # Re-align by trimming — train_all_models uses 55-bar warmup internally
        # so feat_mat may be shorter than closes. Map via warmup offset.
        offset = len(closes) - feat_mat.shape[0]
        logger.info(f"Feature matrix offset from closes: {offset}")
        idxs_adj = [i - offset for i in idxs if i - offset >= 0 and i - offset < feat_mat.shape[0]]
        labels = [labeled[k]["label"] for k, i in enumerate(idxs)
                  if i - offset >= 0 and i - offset < feat_mat.shape[0]]
        X = feat_mat[idxs_adj]
        y = np.asarray(labels, dtype=int)
    else:
        X = feat_mat[idxs]
        y = np.asarray([r["label"] for r in labeled], dtype=int)
    logger.info(f"Training matrix: X={X.shape}, y win_rate={y.mean():.3f}")

    # Time-ordered split
    split = int(len(X) * TRAIN_VAL_SPLIT)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    if len(y_tr) < 50 or len(y_te) < 20:
        raise RuntimeError(f"Split too small after filtering: train={len(y_tr)}, test={len(y_te)}")

    # Inverse-frequency weights
    w_tr = _class_weights(y_tr)

    import lightgbm as lgb
    from src.ml.gpu import lgbm_device_params
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
        "min_data_in_leaf": 20,
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

    # Evaluate
    probs = np.asarray(model.predict(X_te), dtype=float)
    preds = (probs > 0.5).astype(int)
    acc = float(np.mean(preds == y_te))
    tp = int(np.sum((preds == 1) & (y_te == 1)))
    fp = int(np.sum((preds == 1) & (y_te == 0)))
    fn = int(np.sum((preds == 0) & (y_te == 1)))
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    logger.info(f"Test: acc={acc:.3f} | TAKE P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    # Save — champion gate against existing meta model
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"lgbm_{asset.lower()}_meta.txt")
    from src.ml.champion_gate import evaluate_and_gate
    gate = evaluate_and_gate(model, model_path, X_te, y_te, threshold=0.5)
    logger.info(f"Gate: {'PROMOTED' if gate.promoted else 'REJECTED'} — {gate.reason}")
    if not gate.promoted:
        logger.info(f"Kept incumbent; challenger archived: {gate.challenger_path}")
        return {
            "asset": asset, "n_signals": len(labeled),
            "promoted": False, "reason": gate.reason,
            "test_accuracy": acc, "test_f1": f1,
        }

    model.save_model(model_path)

    # Calibration
    from src.ml import calibration as _calib
    bundle = _calib.fit_calibration(
        raw_probs=list(probs), y_true=list(y_te), asset=asset,
    )
    if bundle is not None:
        cal_path = os.path.join(models_dir, f"lgbm_{asset.lower()}_meta_calibration.json")
        _calib.save_calibration(bundle, cal_path)
        logger.info(f"Calibration: buckets={bundle.buckets} deltas={bundle.deltas} n={bundle.fit_n_samples}")
    else:
        logger.info("Calibration: skipped (degenerate holdout)")

    # Thresholds (optimal TAKE cutoff)
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.30, 0.75, 0.01):
        p = (probs >= t).astype(int)
        _tp = int(np.sum((p == 1) & (y_te == 1)))
        _fp = int(np.sum((p == 1) & (y_te == 0)))
        _fn = int(np.sum((p == 0) & (y_te == 1)))
        if _tp == 0:
            continue
        _pr = _tp / (_tp + _fp)
        _re = _tp / (_tp + _fn)
        _f1 = 2 * _pr * _re / (_pr + _re + 1e-10)
        if _f1 > best_f1:
            best_f1 = _f1
            best_thresh = float(t)
    thresh_path = os.path.join(models_dir, f"lgbm_{asset.lower()}_meta_thresholds.json")
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump({"take_threshold": best_thresh, "take_f1": best_f1,
                   "n_train": len(y_tr), "n_test": len(y_te),
                   "base_win_rate": float(y.mean())}, f, indent=2)
    logger.info(f"Thresholds: take>={best_thresh:.2f} (F1={best_f1:.3f})")

    return {
        "asset": asset, "n_signals": len(labeled),
        "promoted": True, "test_accuracy": acc, "test_f1": f1,
        "model_path": model_path,
    }


def _class_weights(y: np.ndarray) -> np.ndarray:
    n = len(y)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return np.ones(n, dtype=float)
    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    return np.where(y == 1, w_pos, w_neg).astype(float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--days", type=int, default=180,
                    help="Days of history (default 180 — recent regime only)")
    ap.add_argument("--primary-tf", default="5m")
    ap.add_argument("--min-score", type=int, default=MIN_ENTRY_SCORE)
    ap.add_argument("--longs-only", action="store_true", default=True)
    ap.add_argument("--allow-shorts", dest="longs_only", action="store_false")
    ap.add_argument("--spread-pct", type=float, default=0.0,
                    help="Round-trip spread deducted from label (default 0 — model predicts "
                         "signal direction-correctness; spread is an EXIT-strategy concern, "
                         "not a signal-quality concern). Pass 1.69 for Robinhood-realistic net-profit labeling.")
    ap.add_argument("--force", action="store_true",
                    help="Delete the incumbent lgbm_{asset}_meta.txt before training, so the "
                         "champion gate promotes the new model unconditionally. Use when a prior "
                         "bad-label training left a stale model blocking fresh attempts.")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    t0 = time.time()
    try:
        result = train(
            asset=args.asset, days=args.days, primary_tf=args.primary_tf,
            min_score=args.min_score, longs_only=args.longs_only,
            spread_pct=args.spread_pct, models_dir=args.models_dir,
            force=args.force,
        )
    except Exception as e:
        logger.error(f"Training failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 2
    dt = time.time() - t0
    logger.info(f"Done in {dt:.1f}s. Result: {result}")
    return 0 if result.get("promoted") else 1


if __name__ == "__main__":
    sys.exit(main())
