"""
Shadow-mode logging for the meta model.

Lets the meta model run in "observe-only" mode: it predicts on every rule-
signaled entry and logs the prediction, but doesn't veto anything. When the
trade closes, the outcome gets appended to the same log keyed by trade_id.
A later retrain (`src/scripts/shadow_retrain.py`) joins predictions with
actual outcomes to train the next meta model on ground-truth live data
instead of simulated forward-labels.

Log format: append-only JSONL at `logs/meta_shadow.jsonl`. Two record types:

    # At entry, when rule signals would take the trade:
    {"event": "shadow_predict",
     "trade_id": "BTC_42",
     "ts": "2026-04-22T16:30:00Z",
     "asset": "BTC", "direction": "LONG",
     "entry_price": 89000.0, "entry_score": 5,
     "meta_prob_raw": 0.381, "meta_prob_cal": 0.412,
     "meta_decision": "SKIP", "take_threshold": 0.32,
     "would_veto": true,
     "features": [f0, f1, ..., f49]}

    # At exit, when the position closes:
    {"event": "shadow_outcome",
     "trade_id": "BTC_42",
     "ts": "2026-04-22T17:45:00Z",
     "pnl_pct": -0.56, "pnl_usd": -11.20,
     "exit_price": 88500.0, "bars_held": 24,
     "win": 0, "exit_reason": "SL L2 hit"}

Enabled by `ACT_META_SHADOW_MODE=1`. Also respects `ACT_DISABLE_ML=1` —
if ML is fully disabled, shadow logs nothing. Orthogonal to the ordinary
meta-gate veto: when shadow mode is on, the veto is suppressed (predict-only).
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_LOG_PATH = os.path.join("logs", "meta_shadow.jsonl")

_write_lock = threading.Lock()  # single-process, multi-thread safety


def is_enabled() -> bool:
    """Shadow mode is on when ACT_META_SHADOW_MODE=1 AND ACT_DISABLE_ML != 1."""
    if (os.environ.get("ACT_DISABLE_ML") or "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return (os.environ.get("ACT_META_SHADOW_MODE") or "").strip().lower() in ("1", "true", "yes", "on")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_predict(
    trade_id: str,
    asset: str,
    direction: str,
    entry_price: float,
    entry_score: int,
    meta_prob_raw: float,
    meta_prob_cal: float,
    take_threshold: float,
    features: List[float],
    path: Optional[str] = None,
) -> None:
    """Append a shadow_predict record. Never raises — shadow logging cannot
    break the decision path."""
    if not is_enabled():
        return
    rec = {
        "event": "shadow_predict",
        "trade_id": str(trade_id),
        "ts": _now_iso(),
        "asset": asset,
        "direction": direction,
        "entry_price": float(entry_price),
        "entry_score": int(entry_score),
        "meta_prob_raw": round(float(meta_prob_raw), 6),
        "meta_prob_cal": round(float(meta_prob_cal), 6),
        "meta_decision": "SKIP" if meta_prob_cal < take_threshold else "TAKE",
        "take_threshold": round(float(take_threshold), 4),
        "would_veto": bool(meta_prob_cal < take_threshold),
        "features": [round(float(f), 6) for f in features],
    }
    _append(rec, path)


def log_outcome(
    trade_id: str,
    pnl_pct: float,
    pnl_usd: float,
    exit_price: float,
    bars_held: int,
    exit_reason: str = "",
    path: Optional[str] = None,
) -> None:
    """Append a shadow_outcome record. Matched against shadow_predict by trade_id
    at retrain time. Never raises."""
    if not is_enabled():
        return
    rec = {
        "event": "shadow_outcome",
        "trade_id": str(trade_id),
        "ts": _now_iso(),
        "pnl_pct": round(float(pnl_pct), 6),
        "pnl_usd": round(float(pnl_usd), 4),
        "exit_price": float(exit_price),
        "bars_held": int(bars_held),
        "win": 1 if pnl_pct > 0 else 0,
        "exit_reason": str(exit_reason)[:100],
    }
    _append(rec, path)


def _append(rec: Dict[str, Any], path: Optional[str]) -> None:
    try:
        p = path or DEFAULT_LOG_PATH
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception:
        # Swallow — shadow logging must never impact the trading decision path.
        pass


def read_all(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read every record. Used by shadow_retrain. Missing file returns []."""
    p = path or DEFAULT_LOG_PATH
    if not os.path.exists(p):
        return []
    out: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def join_predict_outcome(
    records: Optional[List[Dict[str, Any]]] = None,
    path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Join shadow_predict and shadow_outcome by trade_id; returns completed trades
    with features+label. Predictions without outcomes (position still open) are skipped."""
    recs = records if records is not None else read_all(path)
    preds: Dict[str, Dict[str, Any]] = {}
    outs: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        tid = r.get("trade_id")
        if not tid:
            continue
        if r.get("event") == "shadow_predict":
            preds[tid] = r
        elif r.get("event") == "shadow_outcome":
            outs[tid] = r

    joined: List[Dict[str, Any]] = []
    for tid, p in preds.items():
        o = outs.get(tid)
        if o is None:
            continue
        joined.append({
            "trade_id": tid,
            "asset": p.get("asset"),
            "direction": p.get("direction"),
            "entry_score": p.get("entry_score"),
            "meta_prob_raw": p.get("meta_prob_raw"),
            "meta_prob_cal": p.get("meta_prob_cal"),
            "would_veto": p.get("would_veto", False),
            "features": p.get("features") or [],
            "pnl_pct": o.get("pnl_pct", 0.0),
            "pnl_usd": o.get("pnl_usd", 0.0),
            "bars_held": o.get("bars_held", 0),
            "win": int(o.get("win", 0)),
            "exit_reason": o.get("exit_reason", ""),
        })
    return joined


def shadow_stats(joined: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summary stats for operator reporting. Measures how well the meta model
    would have filtered live trades if its veto had been applied."""
    if not joined:
        return {"n": 0}
    n = len(joined)
    wins = sum(1 for r in joined if r["win"] == 1)
    losses = n - wins
    # How the meta model's veto suggestion compared to actual outcome
    meta_would_veto = [r for r in joined if r.get("would_veto")]
    meta_would_take = [r for r in joined if not r.get("would_veto")]
    veto_correct = sum(1 for r in meta_would_veto if r["win"] == 0)
    take_correct = sum(1 for r in meta_would_take if r["win"] == 1)
    return {
        "n": n,
        "actual_wr": round(wins / n, 4),
        "meta_veto_count": len(meta_would_veto),
        "meta_take_count": len(meta_would_take),
        # Precision of the veto suggestion: P(loss | meta would veto)
        "veto_precision_loss": round(veto_correct / max(1, len(meta_would_veto)), 4),
        # Precision of the take suggestion: P(win | meta would take)
        "take_precision_win": round(take_correct / max(1, len(meta_would_take)), 4),
        # Total pnl distribution
        "total_pnl_pct": round(sum(r["pnl_pct"] for r in joined), 4),
        "if_vetoed_pnl_pct": round(sum(r["pnl_pct"] for r in joined if not r.get("would_veto")), 4),
    }
