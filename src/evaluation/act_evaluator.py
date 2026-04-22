"""
ACT evaluator — reads everything the bot has written and answers:

    1. What's ON right now? (env flags + config)
    2. Which components correlate with wins vs losses? (attribution)
    3. What should I consider turning off? (recommendations)

Inputs (all optional — missing files degrade gracefully):
    logs/robinhood_paper.jsonl      — ENTRY/EXIT events
    logs/meta_shadow.jsonl          — shadow_predict + shadow_outcome
    logs/safe_entries_state.json    — rolling per-asset Sharpe + consec losses
    models/retrain_history.json     — retrain timestamps per asset/tf
    config.yaml                     — current config
    env vars                        — ACT_*, DASHBOARD_API_KEY, etc

Output: a single dict with the shape documented in `build_report()`.
Stable contract — consumers (CLI + Streamlit) can rely on these keys.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Default paths — overridable per-call for tests
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
PAPER_JOURNAL = "logs/robinhood_paper.jsonl"
SHADOW_LOG = "logs/meta_shadow.jsonl"
SAFE_STATE = "logs/safe_entries_state.json"
RETRAIN_HISTORY = "models/retrain_history.json"
CONFIG_YAML = "config.yaml"

# Attribution bucket edges
SCORE_BUCKETS = [(-10, 3), (3, 5), (5, 7), (7, 10), (10, 99)]
CONF_BUCKETS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]

# Recommendation thresholds
RECO_BUCKET_MIN_N = 10          # need at least N trades in a bucket before recommending
RECO_LOSING_WR_UPPER = 0.40     # bucket with WR < this = flagged
RECO_FLAT_SHARPE_LOW = 0.3      # rolling sharpe below this = flagged


# ─────────────────────────────────────────────────────────────────────
# COMPONENT STATE
# ─────────────────────────────────────────────────────────────────────

# Every toggleable subsystem. `invert=True` means the env var is a KILL switch
# (var=1 -> feature OFF), so is_on is inverted after parsing. Default False
# means regular enable-switch (var=1 -> feature ON).
COMPONENTS: List[Dict[str, Any]] = [
    # ACT_DISABLE_ML is a kill switch — 1 disables ML. Feature "ML Gate" is ON
    # when the kill switch is OFF (var unset or var=0).
    {"name": "ML Gate (base)",        "env": "ACT_DISABLE_ML",       "invert": True,  "on_when_unset": True,  "toggle_on": "0", "toggle_off": "1"},
    {"name": "Safe-entries gate",     "env": "ACT_SAFE_ENTRIES",     "invert": False, "on_when_unset": False, "toggle_on": "1", "toggle_off": "0"},
    {"name": "Shadow-mode meta",      "env": "ACT_META_SHADOW_MODE", "invert": False, "on_when_unset": False, "toggle_on": "1", "toggle_off": "0"},
    {"name": "LightGBM GPU",          "env": "ACT_LGBM_DEVICE",      "invert": False, "on_when_unset": False, "toggle_on": "gpu", "toggle_off": "cpu"},
    {"name": "Prometheus metrics",    "env": "ACT_METRICS_ENABLED",  "invert": False, "on_when_unset": False, "toggle_on": "1", "toggle_off": "0"},
    {"name": "OTel tracing",          "env": "ACT_TRACING_ENABLED",  "invert": False, "on_when_unset": False, "toggle_on": "1", "toggle_off": "0"},
    {"name": "Real-capital gate flag", "env": "ACT_REAL_CAPITAL_ENABLED", "invert": False, "on_when_unset": False, "toggle_on": "1", "toggle_off": "0"},
    {"name": "Rolling-Sharpe gate min", "env": "ACT_GATE_MIN_SHARPE", "invert": False, "on_when_unset": False, "toggle_on": "1.0", "toggle_off": "0"},
]


def load_component_state(root: Optional[Path] = None) -> Dict[str, Any]:
    """Return current state of every toggleable component.

    Each entry:
        {name, env, value, is_on, toggle_cmd_on, toggle_cmd_off, description?}

    `is_on` reflects the USER-VISIBLE FEATURE state, which for kill-switch vars
    (invert=True) is the logical NOT of the raw env value.
    """
    out: List[Dict[str, Any]] = []
    for c in COMPONENTS:
        raw = os.environ.get(c["env"])
        invert = bool(c.get("invert", False))
        # First: parse the raw env value into a "var is truthy" bool
        var_truthy: bool
        if raw is None or raw == "":
            # Unset — for kill switches, unset=0 (feature ON). For enable
            # switches, unset=0 (feature OFF). on_when_unset captures both.
            is_on = bool(c.get("on_when_unset", False))
            var_value = "(unset)"
        else:
            if c["env"] == "ACT_LGBM_DEVICE":
                var_truthy = raw.strip().lower() == "gpu"
            elif c["env"] == "ACT_GATE_MIN_SHARPE":
                try:
                    var_truthy = float(raw) > 0.0
                except Exception:
                    var_truthy = False
            else:
                var_truthy = raw.strip().lower() in ("1", "true", "yes", "on")
            # Kill switches: feature is ON when the var is NOT truthy
            is_on = (not var_truthy) if invert else var_truthy
            var_value = raw
        out.append({
            "name": c["name"],
            "env": c["env"],
            "value": var_value,
            "is_on": is_on,
            "toggle_cmd_on": f'setx {c["env"]} {c["toggle_on"]}',
            "toggle_cmd_off": f'setx {c["env"]} {c["toggle_off"]}',
        })
    return {"components": out}


# ─────────────────────────────────────────────────────────────────────
# PAPER JOURNAL → TRADE RECORDS
# ─────────────────────────────────────────────────────────────────────

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Delegate to src.ml.shadow_log.read_all — same contract, one place to maintain."""
    from src.ml.shadow_log import read_all
    return read_all(path)


def load_paper_trades(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Join ENTRY + EXIT events from the paper journal into complete trade records.

    Matching rule: EXIT matches the most recent ENTRY for the same
    (asset, direction, entry_price ~=). Imperfect when assets recycle prices,
    but the journal rarely produces collisions under normal cadence.

    Returns list of:
        {asset, direction, entry_price, exit_price, entry_time, exit_time,
         pnl_pct, pnl_usd, bars_held, score, ml_conf, llm_conf, spread_pct,
         size_pct, reason, reasoning, win}
    """
    p = path or os.path.join(DEFAULT_ROOT, PAPER_JOURNAL)
    records = _read_jsonl(p)

    # Build running FIFO queue of open entries per (asset, direction)
    open_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    trades: List[Dict[str, Any]] = []

    for row in records:
        event = row.get("event")
        asset = row.get("asset")
        direction = row.get("direction")
        if not asset or not direction:
            continue
        key = (asset, direction)
        if event == "ENTRY":
            open_by_key.setdefault(key, []).append(row)
        elif event == "EXIT":
            queue = open_by_key.get(key, [])
            if not queue:
                continue
            entry = queue.pop(0)  # FIFO
            pnl_pct = float(row.get("pnl_pct") or 0.0)
            trades.append({
                "asset": asset,
                "direction": direction,
                "entry_price": float(entry.get("fill_price") or 0.0),
                "exit_price": float(row.get("exit_price") or 0.0),
                "entry_time": entry.get("timestamp"),
                "exit_time": row.get("timestamp"),
                "pnl_pct": pnl_pct,
                "pnl_usd": float(row.get("pnl_usd") or 0.0),
                "bars_held": int(row.get("bars_held") or 0),
                "score": int(entry.get("score") or 0),
                "ml_conf": float(entry.get("ml_confidence") or 0.0),
                "llm_conf": float(entry.get("llm_confidence") or 0.0),
                "spread_pct": float(entry.get("spread_pct") or 0.0),
                "size_pct": float(entry.get("size_pct") or 0.0),
                "reason": str(row.get("reason") or "")[:120],
                "reasoning": str(entry.get("reasoning") or "")[:120],
                "win": 1 if pnl_pct > 0 else 0,
            })
    return trades


# ─────────────────────────────────────────────────────────────────────
# BUCKET ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────

def _wr_and_pnl(rows: List[Dict[str, Any]]) -> Tuple[int, float, float]:
    if not rows:
        return 0, 0.0, 0.0
    n = len(rows)
    wins = sum(1 for r in rows if r["win"] == 1)
    wr = wins / n
    mean_pnl = sum(r["pnl_pct"] for r in rows) / n
    return n, wr, mean_pnl


def bucket_attribution(
    trades: List[Dict[str, Any]],
    field: str,
    buckets: List[Tuple[float, float]],
    label_fmt: str = "[{lo:g}, {hi:g})",
) -> List[Dict[str, Any]]:
    """Generic bucketer — returns list of {bucket, n, wr, mean_pnl_pct, total_pnl_pct}."""
    out: List[Dict[str, Any]] = []
    for lo, hi in buckets:
        rows = [r for r in trades if lo <= r.get(field, 0) < hi]
        n, wr, mean_pnl = _wr_and_pnl(rows)
        out.append({
            "bucket": label_fmt.format(lo=lo, hi=hi),
            "field": field,
            "lo": lo, "hi": hi,
            "n": n,
            "wr": round(wr, 4) if n else None,
            "mean_pnl_pct": round(mean_pnl, 4) if n else None,
            "total_pnl_pct": round(sum(r["pnl_pct"] for r in rows), 4),
        })
    return out


def categorical_attribution(
    trades: List[Dict[str, Any]],
    field: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Group by a discrete field (asset, direction, exit_reason_family)."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in trades:
        key = str(r.get(field) or "UNKNOWN")
        groups.setdefault(key, []).append(r)
    out = []
    for k, rows in groups.items():
        n, wr, mean_pnl = _wr_and_pnl(rows)
        out.append({
            "bucket": k, "field": field, "n": n,
            "wr": round(wr, 4), "mean_pnl_pct": round(mean_pnl, 4),
            "total_pnl_pct": round(sum(r["pnl_pct"] for r in rows), 4),
        })
    out.sort(key=lambda r: r["n"], reverse=True)
    return out[:limit]


def exit_reason_family(reason: str) -> str:
    """Normalize the free-form exit reason into a small set of families."""
    s = (reason or "").lower()
    if "sl l" in s:
        return "SL ratchet"
    if "sl hit" in s or "hard stop" in s:
        return "SL hard stop"
    if "roi" in s:
        return "ROI target"
    if "time" in s:
        return "Time exit"
    if "ema" in s:
        return "EMA flip"
    if "bear" in s:
        return "Bear override"
    return "Other"


# ─────────────────────────────────────────────────────────────────────
# ROLLING METRICS
# ─────────────────────────────────────────────────────────────────────

def rolling_sharpe_series(
    trades: List[Dict[str, Any]],
    window: int = 30,
) -> List[Dict[str, Any]]:
    """Rolling per-trade Sharpe over the last `window` trades at each step.

    Returns list of {idx, n, mean, std, sharpe} covering each step after the
    first `window` trades. Sharpe is per-trade (not annualized).
    """
    out: List[Dict[str, Any]] = []
    pnls = [r["pnl_pct"] for r in trades]
    for i in range(window, len(pnls) + 1):
        slice_ = pnls[i - window:i]
        mean = sum(slice_) / window
        var = sum((x - mean) ** 2 for x in slice_) / (window - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        sharpe = (mean / std) if std > 0 else 0.0
        out.append({
            "idx": i, "n": window,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "sharpe": round(sharpe, 4),
        })
    return out


# ─────────────────────────────────────────────────────────────────────
# SHADOW LOG INTEGRATION
# ─────────────────────────────────────────────────────────────────────

def shadow_summary(path: Optional[str] = None) -> Dict[str, Any]:
    """Pull shadow-log stats via the canonical helper."""
    try:
        from src.ml.shadow_log import read_all, join_predict_outcome, shadow_stats
    except Exception as e:
        return {"available": False, "error": str(e)}
    p = path or os.path.join(DEFAULT_ROOT, SHADOW_LOG)
    recs = read_all(p)
    joined = join_predict_outcome(recs)
    per_asset = {}
    for asset in set(r.get("asset") for r in joined):
        if asset:
            rows = [r for r in joined if r.get("asset") == asset]
            per_asset[asset] = shadow_stats(rows)
    return {
        "available": True,
        "total_records": len(recs),
        "joined_trades": len(joined),
        "per_asset": per_asset,
        "combined": shadow_stats(joined),
    }


# ─────────────────────────────────────────────────────────────────────
# SAFE-ENTRIES STATE
# ─────────────────────────────────────────────────────────────────────

def safe_entries_summary(path: Optional[str] = None) -> Dict[str, Any]:
    p = path or os.path.join(DEFAULT_ROOT, SAFE_STATE)
    if not os.path.exists(p):
        return {"available": False}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"available": False, "error": str(e)}
    try:
        from src.trading.safe_entries import SafeEntryState
        state = SafeEntryState.from_dict(data)
        per_asset = {}
        for a, st in state.assets.items():
            per_asset[a] = {
                "consecutive_losses": st.consecutive_losses,
                "paused_until": st.paused_until,
                "n_trades": len(st.trade_pnl_pcts),
                "rolling_sharpe_30": round(state.rolling_sharpe(a, n=30), 4),
            }
        return {
            "available": True,
            "per_asset": per_asset,
            "combined_rolling_sharpe_30": round(state.combined_rolling_sharpe(n=30), 4),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────
# RETRAIN HISTORY
# ─────────────────────────────────────────────────────────────────────

def retrain_summary(path: Optional[str] = None, tail: int = 10) -> Dict[str, Any]:
    p = path or os.path.join(DEFAULT_ROOT, RETRAIN_HISTORY)
    if not os.path.exists(p):
        return {"available": False}
    try:
        with open(p, "r", encoding="utf-8") as f:
            hist = json.load(f)
    except Exception as e:
        return {"available": False, "error": str(e)}
    if not isinstance(hist, list):
        return {"available": False, "error": "unexpected format"}
    recent = hist[-tail:] if len(hist) > tail else hist
    # Per-asset latest
    per_asset: Dict[str, Dict[str, Any]] = {}
    for row in hist:
        a = row.get("asset")
        if not a:
            continue
        ts = row.get("timestamp", "")
        if a not in per_asset or ts > per_asset[a].get("timestamp", ""):
            per_asset[a] = {
                "timestamp": ts,
                "accuracy": row.get("new_accuracy"),
                "high_conf_accuracy": row.get("new_high_conf_accuracy"),
                "timeframe": row.get("timeframe"),
            }
    return {
        "available": True,
        "total_retrains": len(hist),
        "per_asset_latest": per_asset,
        "recent": recent,
    }


# ─────────────────────────────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────

def recommendations(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate actionable "consider turning off / investigating" items based on
    the rest of the report. Each recommendation has severity + reason + action."""
    recs: List[Dict[str, Any]] = []

    n_trades = len(report.get("trades") or [])
    if n_trades < 10:
        recs.append({
            "severity": "info",
            "area": "paper soak",
            "reason": f"Only {n_trades} trades — not enough data for reliable attribution",
            "action": "Let the bot run for at least 30 more trades before retuning",
        })
        return recs

    # Overall performance
    overall_wr = sum(1 for t in (report.get("trades") or []) if t["win"] == 1) / max(1, n_trades)
    overall_pnl = sum(t["pnl_pct"] for t in (report.get("trades") or []))
    if overall_wr < 0.50:
        recs.append({
            "severity": "high",
            "area": "overall strategy",
            "reason": f"WR = {overall_wr:.1%} over {n_trades} trades — negative hit rate",
            "action": "Consider exchange migration (lower spread) or timeframe change (1h/4h)",
        })

    # Score bucket analysis
    score_attrs = report.get("attribution", {}).get("by_score", [])
    for b in score_attrs:
        if b.get("n") and b["n"] >= RECO_BUCKET_MIN_N and b.get("wr") is not None:
            if b["wr"] < RECO_LOSING_WR_UPPER:
                recs.append({
                    "severity": "medium",
                    "area": f"score {b['bucket']}",
                    "reason": f"bucket n={b['n']} WR={b['wr']:.1%} — losing disproportionately",
                    "action": f"Raise min_entry_score above {b['lo']} in config.yaml (adaptive.min_entry_score)",
                })

    # LLM confidence bucket
    llm_attrs = report.get("attribution", {}).get("by_llm_conf", [])
    for b in llm_attrs:
        if b.get("n") and b["n"] >= RECO_BUCKET_MIN_N and b.get("wr") is not None:
            if b["wr"] < RECO_LOSING_WR_UPPER and b["hi"] >= 0.85:
                recs.append({
                    "severity": "medium",
                    "area": "LLM confidence bucket",
                    "reason": f"LLM conf {b['bucket']} n={b['n']} WR={b['wr']:.1%} — high-conf losing",
                    "action": "Investigate LLM prompt / calibration — confident predictions are wrong",
                })

    # Direction skew
    dir_attrs = report.get("attribution", {}).get("by_direction", [])
    for d in dir_attrs:
        if d.get("n") and d["n"] >= RECO_BUCKET_MIN_N and d.get("wr") is not None:
            if d["wr"] < RECO_LOSING_WR_UPPER:
                recs.append({
                    "severity": "medium",
                    "area": f"direction {d['bucket']}",
                    "reason": f"{d['bucket']} trades n={d['n']} WR={d['wr']:.1%} — net losing",
                    "action": f"Disable {d['bucket']} entries (set longs_only=true if {d['bucket']}==SHORT)",
                })

    # Exit-reason pattern
    exit_attrs = report.get("attribution", {}).get("by_exit_reason", [])
    for e in exit_attrs:
        if e.get("n") and e["n"] >= RECO_BUCKET_MIN_N:
            if e["bucket"] == "SL ratchet" and e.get("wr", 0) < 0.30:
                recs.append({
                    "severity": "medium",
                    "area": "SL ratchet exits",
                    "reason": f"n={e['n']} WR={e['wr']:.1%} — trailing SL exits mostly losing",
                    "action": "Widen SL floor — safe_entries.stop_atr_mult 2.5 -> 3.0",
                })

    # Rolling Sharpe trend
    rolling = report.get("rolling_sharpe_30") or []
    if rolling:
        last = rolling[-1]["sharpe"]
        if last < RECO_FLAT_SHARPE_LOW:
            recs.append({
                "severity": "high",
                "area": "rolling Sharpe",
                "reason": f"last 30-trade Sharpe = {last:.3f} — below soak-promotion floor of 1.0",
                "action": "Readiness gate will refuse to open. Either fix strategy or accept paper-only mode",
            })

    # Shadow shouldn't-vetoed analysis
    shadow = report.get("shadow", {})
    if shadow.get("available") and shadow.get("joined_trades", 0) >= 50:
        combined = shadow.get("combined", {})
        tot = combined.get("total_pnl_pct")
        if_vetoed = combined.get("if_vetoed_pnl_pct")
        if tot is not None and if_vetoed is not None and if_vetoed > tot + 2.0:
            recs.append({
                "severity": "medium",
                "area": "meta shadow model",
                "reason": f"if_vetoed_pnl {if_vetoed:.2f}% > total_pnl {tot:.2f}% — meta model "
                          f"would have improved PnL by ~{if_vetoed-tot:.2f}pp",
                "action": "Consider promoting shadow to live veto: unset ACT_META_SHADOW_MODE",
            })

    # ML kill switch on while meta model trained
    components = report.get("components", {}).get("components", [])
    for c in components:
        if c["env"] == "ACT_DISABLE_ML" and c["is_on"] is False and shadow.get("available") and shadow.get("joined_trades", 0) < 20:
            recs.append({
                "severity": "info",
                "area": "shadow mode off too early",
                "reason": f"ML kill switch on; shadow has only {shadow.get('joined_trades', 0)} joined trades",
                "action": "Set ACT_META_SHADOW_MODE=1 + ACT_DISABLE_ML=0 to accumulate shadow data",
            })

    if not recs:
        recs.append({
            "severity": "info",
            "area": "system",
            "reason": "No immediate red flags in current sample. Keep soaking.",
            "action": "Re-check after 50 more trades.",
        })

    return recs


# ─────────────────────────────────────────────────────────────────────
# TOP-LEVEL REPORT
# ─────────────────────────────────────────────────────────────────────

def build_report(
    paper_journal_path: Optional[str] = None,
    shadow_log_path: Optional[str] = None,
    safe_state_path: Optional[str] = None,
    retrain_history_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Single entry point. Loads everything, returns the full structured report.

    Shape (stable contract):
        {
          components: {components: [ ... ]},
          trades: [ ... ],
          totals: {n, wins, losses, wr, total_pnl_pct, total_pnl_usd, mean_pnl_pct, mean_bars_held},
          attribution: {
            by_score, by_llm_conf, by_ml_conf, by_spread, by_size,
            by_direction, by_asset, by_exit_reason,
          },
          rolling_sharpe_30: [...],
          shadow: {...},
          safe_entries: {...},
          retrain: {...},
          recommendations: [...],
        }
    """
    components = load_component_state()
    trades = load_paper_trades(paper_journal_path)

    # Decorate trades with exit_reason family for categorical grouping
    for t in trades:
        t["exit_reason_family"] = exit_reason_family(t.get("reason", ""))

    totals = {"n": len(trades)}
    if trades:
        wins = sum(1 for t in trades if t["win"] == 1)
        totals.update({
            "wins": wins,
            "losses": len(trades) - wins,
            "wr": round(wins / len(trades), 4),
            "total_pnl_pct": round(sum(t["pnl_pct"] for t in trades), 4),
            "total_pnl_usd": round(sum(t["pnl_usd"] for t in trades), 2),
            "mean_pnl_pct": round(sum(t["pnl_pct"] for t in trades) / len(trades), 4),
            "mean_bars_held": round(sum(t["bars_held"] for t in trades) / len(trades), 1),
        })

    attribution = {
        "by_score": bucket_attribution(trades, "score", SCORE_BUCKETS, "[{lo:g}, {hi:g})"),
        "by_llm_conf": bucket_attribution(trades, "llm_conf", CONF_BUCKETS),
        "by_ml_conf": bucket_attribution(trades, "ml_conf", CONF_BUCKETS),
        "by_spread": bucket_attribution(trades, "spread_pct",
                                        [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 99)]),
        "by_size": bucket_attribution(trades, "size_pct",
                                      [(0, 1), (1, 2), (2, 5), (5, 100)]),
        "by_direction": categorical_attribution(trades, "direction"),
        "by_asset": categorical_attribution(trades, "asset"),
        "by_exit_reason": categorical_attribution(trades, "exit_reason_family"),
    }

    report: Dict[str, Any] = {
        "components": components,
        "trades": trades,
        "totals": totals,
        "attribution": attribution,
        "rolling_sharpe_30": rolling_sharpe_series(trades, 30) if len(trades) >= 30 else [],
        "shadow": shadow_summary(shadow_log_path),
        "safe_entries": safe_entries_summary(safe_state_path),
        "retrain": retrain_summary(retrain_history_path),
    }
    report["recommendations"] = recommendations(report)
    return report
