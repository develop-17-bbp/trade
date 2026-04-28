"""Prediction-accuracy feedback for the macro learn loop.

Operator: "the self-critique / learn which is in macro loop, the LLM
should know how accurate and wrong their predictions are."

Pre-state: trade_verifier writes SelfCritique post-close; brain reads
recent_critiques. Brain has narrative context about prior trades but
NOT a structured calibration of HOW ACCURATE its directional reads
have been. It can't tell whether its bias-score predictions are
working as a class.

This module computes per-prediction accuracy stats and exposes them
to the brain:

  * Win-rate per direction (LONG vs SHORT vs SKIP)
  * Win-rate per conviction tier (sniper / normal / advisory)
  * Win-rate per bias-score bucket (when bias_score embedded in
    thesis text — extracted with regex)
  * Calibration error — does the brain over-estimate its own
    confidence?
  * Most-frequent miss reasons (which factor mismatches predicted
    outcomes most often)

Output is rendered as a compact stats block the brain can read
on every tick. New tool query_prediction_accuracy returns the
structured form for deeper inspection.

Anti-overfit / anti-noise:
  * Aggregates only — no per-trade exposure
  * Sample-size warnings (<10 closed trades = "low_confidence")
  * Time-decayed: only last 60 days of closed trades counted
  * Pure read-only over warm_store + brain_memory (no new schema)
  * Bucket accuracy < 50% sample → flagged "insufficient_sample"
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 60
DEFAULT_MIN_BUCKET_SAMPLES = 5
BIAS_SCORE_PATTERN = re.compile(
    r"(?:bias|long_bias|long_bias_score|score)\s*[=:]\s*([+-]?\d*\.?\d+)",
    re.IGNORECASE,
)


@dataclass
class AccuracyBucket:
    label: str
    n: int = 0
    wins: int = 0
    avg_pnl_pct: float = 0.0
    win_rate: float = 0.0
    sample_warning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "n": int(self.n),
            "wins": int(self.wins),
            "win_rate": round(float(self.win_rate), 3),
            "avg_pnl_pct": round(float(self.avg_pnl_pct), 3),
            "sample_warning": self.sample_warning,
        }


@dataclass
class AccuracySnapshot:
    asset_filter: Optional[str]
    n_closed_trades: int
    overall_win_rate: float
    overall_avg_pnl_pct: float
    by_direction: List[AccuracyBucket] = field(default_factory=list)
    by_tier: List[AccuracyBucket] = field(default_factory=list)
    by_bias_bucket: List[AccuracyBucket] = field(default_factory=list)
    by_regime: List[AccuracyBucket] = field(default_factory=list)
    most_common_miss_reasons: List[Dict[str, Any]] = field(default_factory=list)
    calibration_label: str = "neutral"   # "well_calibrated" / "over_confident" / "under_confident" / "neutral"
    sample_warning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_filter": self.asset_filter or "ALL",
            "n_closed_trades": int(self.n_closed_trades),
            "overall_win_rate": round(float(self.overall_win_rate), 3),
            "overall_avg_pnl_pct": round(float(self.overall_avg_pnl_pct), 3),
            "by_direction": [b.to_dict() for b in self.by_direction],
            "by_tier": [b.to_dict() for b in self.by_tier],
            "by_bias_bucket": [b.to_dict() for b in self.by_bias_bucket],
            "by_regime": [b.to_dict() for b in self.by_regime],
            "most_common_miss_reasons": self.most_common_miss_reasons[:5],
            "calibration_label": self.calibration_label,
            "sample_warning": self.sample_warning,
            "advisory": (
                "Use this to calibrate trust in your own predictions: "
                "if by_bias_bucket shows >+0.5 plans winning <50% of the "
                "time, your strong-LONG calls are over-confident — "
                "weight macro/cycle factors higher next tick. If "
                "by_regime shows risk_off plans losing more than risk_on, "
                "you're trading against macro headwind — bias toward "
                "SKIP in risk_off."
            ),
        }


def _extract_bias_score(text: str) -> Optional[float]:
    """Extract embedded bias_score from thesis or critique text."""
    if not text:
        return None
    m = BIAS_SCORE_PATTERN.search(text)
    if not m:
        return None
    try:
        v = float(m.group(1))
        if -2.0 <= v <= 2.0:  # sanity bound
            return max(-1.0, min(1.0, v))
    except Exception:
        pass
    return None


def _bucket_bias(score: Optional[float]) -> str:
    if score is None:
        return "no_bias_score"
    if score > 0.5:
        return "strong_long"
    if score > 0.2:
        return "mild_long"
    if score >= -0.2:
        return "neutral"
    if score >= -0.5:
        return "mild_short"
    return "strong_short"


def _read_closed_decisions(asset: Optional[str] = None,
                           lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                           limit: int = 500) -> List[Dict[str, Any]]:
    """Pull closed decisions (with critique containing realized_pnl_pct)
    from warm_store within lookback window."""
    try:
        from src.orchestration.warm_store import get_store
        store = get_store()
        cutoff_ns = int((time.time() - lookback_days * 86400) * 1e9)
        conn = sqlite3.connect(store.db_path, timeout=2.0)
        try:
            sql = (
                "SELECT ts_ns, symbol, plan_json, self_critique, component_signals "
                "FROM decisions "
                "WHERE ts_ns >= ? AND self_critique != '{}' "
                "AND self_critique IS NOT NULL "
                + ("AND symbol = ? " if asset else "")
                + "ORDER BY ts_ns DESC LIMIT ?"
            )
            params = ((cutoff_ns, asset, limit) if asset
                      else (cutoff_ns, limit))
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        out: List[Dict[str, Any]] = []
        for ts_ns, sym, plan_raw, crit_raw, comp_raw in rows:
            try:
                plan = json.loads(plan_raw or "{}")
                crit = json.loads(crit_raw or "{}")
                comp = json.loads(comp_raw or "{}")
            except Exception:
                continue
            pnl = crit.get("realized_pnl_pct")
            if pnl is None:
                continue
            thesis = str(plan.get("thesis", ""))
            lessons = str(crit.get("lessons", ""))
            verdict = str(crit.get("verdict", ""))
            bias = _extract_bias_score(thesis) or _extract_bias_score(lessons)
            out.append({
                "ts_ns": int(ts_ns),
                "asset": sym,
                "direction": plan.get("direction", "?"),
                "tier": plan.get("entry_tier", ""),
                "regime": comp.get("regime", "unknown"),
                "realized_pnl_pct": float(pnl),
                "bias_score": bias,
                "miss_reason": str(crit.get("miss_reasons", ""))[:80],
                "verdict": verdict[:40],
                "thesis": thesis[:160],
            })
        return out
    except Exception as e:
        logger.debug("read_closed_decisions failed: %s", e)
        return []


def _aggregate_buckets(rows: List[Dict[str, Any]],
                       key_fn) -> List[AccuracyBucket]:
    """Group rows by key_fn(row), compute WR + avg pnl per bucket."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        k = str(key_fn(r))
        groups.setdefault(k, []).append(r)
    out: List[AccuracyBucket] = []
    for label, items in sorted(groups.items()):
        n = len(items)
        if n == 0:
            continue
        wins = sum(1 for it in items if it["realized_pnl_pct"] > 0)
        avg_pnl = sum(it["realized_pnl_pct"] for it in items) / n
        wr = wins / n
        warning = ""
        if n < DEFAULT_MIN_BUCKET_SAMPLES:
            warning = "insufficient_sample"
        out.append(AccuracyBucket(
            label=label, n=n, wins=wins,
            win_rate=wr, avg_pnl_pct=avg_pnl,
            sample_warning=warning,
        ))
    return out


def _calibration_label(by_bias: List[AccuracyBucket]) -> str:
    """Classify whether the brain is over- or under-confident.

    Look at strong_long bucket: if WR < 0.5 → over_confident.
    Look at neutral bucket: if WR > 0.55 → under_confident (skipping
    when it should have entered).
    """
    strong_long = next((b for b in by_bias if b.label == "strong_long"), None)
    neutral = next((b for b in by_bias if b.label == "neutral"), None)
    if strong_long and strong_long.n >= DEFAULT_MIN_BUCKET_SAMPLES:
        if strong_long.win_rate < 0.45:
            return "over_confident"
        if strong_long.win_rate > 0.65:
            return "well_calibrated"
    if neutral and neutral.n >= DEFAULT_MIN_BUCKET_SAMPLES:
        if neutral.win_rate > 0.55:
            return "under_confident"
    return "neutral"


def compute_accuracy(asset: Optional[str] = None,
                     lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> AccuracySnapshot:
    """Pull closed decisions, bucket, return structured snapshot."""
    rows = _read_closed_decisions(asset=asset, lookback_days=lookback_days)
    n = len(rows)
    if n == 0:
        return AccuracySnapshot(
            asset_filter=asset, n_closed_trades=0,
            overall_win_rate=0.0, overall_avg_pnl_pct=0.0,
            sample_warning="no_closed_trades_in_lookback",
        )

    overall_wins = sum(1 for r in rows if r["realized_pnl_pct"] > 0)
    overall_wr = overall_wins / n
    overall_avg_pnl = sum(r["realized_pnl_pct"] for r in rows) / n

    by_direction = _aggregate_buckets(rows, lambda r: f"direction:{r['direction']}")
    by_tier = _aggregate_buckets(rows, lambda r: f"tier:{r['tier']}")
    by_bias = _aggregate_buckets(rows, lambda r: _bucket_bias(r["bias_score"]))
    by_regime = _aggregate_buckets(rows, lambda r: f"regime:{r['regime']}")

    miss_counts: Dict[str, int] = {}
    for r in rows:
        m = r.get("miss_reason", "")
        if m:
            miss_counts[m] = miss_counts.get(m, 0) + 1
    most_common_misses = sorted(
        miss_counts.items(), key=lambda kv: kv[1], reverse=True,
    )[:5]
    miss_list = [{"reason": k[:80], "count": v} for k, v in most_common_misses]

    calibration = _calibration_label(by_bias)

    sample_warning = ""
    if n < 10:
        sample_warning = "low_sample_under_10_trades"

    return AccuracySnapshot(
        asset_filter=asset,
        n_closed_trades=n,
        overall_win_rate=overall_wr,
        overall_avg_pnl_pct=overall_avg_pnl,
        by_direction=by_direction,
        by_tier=by_tier,
        by_bias_bucket=by_bias,
        by_regime=by_regime,
        most_common_miss_reasons=miss_list,
        calibration_label=calibration,
        sample_warning=sample_warning,
    )


def render_summary_for_tick(asset: Optional[str] = None) -> str:
    """Compact one-line summary for tick_state injection.

    Brain reads this every tick to know its own track record without
    having to call query_prediction_accuracy. Empty string when no
    closed trades exist."""
    snap = compute_accuracy(asset=asset, lookback_days=30)
    if snap.n_closed_trades == 0:
        return ""
    parts = [
        f"PREDICTION_ACCURACY: n={snap.n_closed_trades}",
        f"WR={snap.overall_win_rate:.0%}",
        f"avg_pnl={snap.overall_avg_pnl_pct:+.2f}%",
        f"calibration={snap.calibration_label}",
    ]
    # Add the most-imbalanced bucket so brain sees where it's wrong
    weak_bucket = None
    for bucket in (snap.by_direction + snap.by_tier + snap.by_bias_bucket
                    + snap.by_regime):
        if bucket.n >= DEFAULT_MIN_BUCKET_SAMPLES and bucket.win_rate < 0.4:
            if weak_bucket is None or bucket.n > weak_bucket.n:
                weak_bucket = bucket
    if weak_bucket:
        parts.append(
            f"weakest={weak_bucket.label}({weak_bucket.win_rate:.0%},n={weak_bucket.n})"
        )
    if snap.most_common_miss_reasons:
        top_miss = snap.most_common_miss_reasons[0]
        parts.append(f"top_miss={top_miss['reason'][:30]}")
    return " | ".join(parts)[:300]
