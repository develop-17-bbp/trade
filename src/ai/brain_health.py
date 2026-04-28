"""Brain-health tracker — measures discipline, tool coverage, goal focus.

Operator: "the brain should only think about achieving the goal and
use all integrations."

Three signals the brain (and operator) needs visibility into:

  1. TOOL VARIETY — distinct tools called over a rolling window. Low
     variety means brain is defaulting to a small subset of the 97
     available. High variety means it's exercising the integrations.

  2. DECISION QUALITY — per-tick metrics:
       skip_rate (last N ticks)
       avg_steps_per_tick (ReAct steps used)
       tool_calls_per_tick
       avg_tools_per_decision
       thesis_quality_score (cites factor_synthesis? mentions
                              DSR/PBO? references similar setups?)

  3. GOAL ADHERENCE — how aligned recent activity is with closing
     gap_to_1pct:
       under_trading flag (gap > 0.5% AND fewer than 2 trades today)
       over_trading flag (>10 trades today AND today_pct < 0)
       focus_drift flag (last 10 SKIPs all cite same reason — brain
                          stuck on one factor)

Module is read-only over warm_store + tick_state. Brain reads via
query_brain_health tool. Auto-injected one-line summary lands in
tick_state every tick.

Anti-overfit / anti-noise:
  * Pure aggregates, no learned weights
  * Sample-size warnings (<20 ticks tracked = "low_sample")
  * Bounded variety score [0, 1] = unique_tools / max(1, all_tools_seen)
  * Operator-visible flags help calibration without forcing behavior
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_TICKS = 100
DEFAULT_LOOKBACK_DAYS = 7
MIN_TICKS_FOR_SUMMARY = 5
LOW_VARIETY_THRESHOLD = 0.20
WEAK_THESIS_THRESHOLD = 0.40
CACHE_TTL_S = 60.0  # snapshot drift over many ticks; per-tick recompute wasted


@dataclass
class BrainHealthSnapshot:
    n_ticks_observed: int = 0
    n_distinct_tools_used: int = 0
    tool_variety_score: float = 0.0       # 0-1
    top_tools_by_usage: List[Dict[str, Any]] = field(default_factory=list)
    avg_steps_per_tick: float = 0.0
    avg_tools_per_decision: float = 0.0
    skip_rate: float = 0.0
    submit_rate: float = 0.0
    avg_thesis_quality_score: float = 0.0  # 0-1
    under_trading: bool = False
    over_trading: bool = False
    focus_drift: bool = False
    sample_warning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_ticks_observed": int(self.n_ticks_observed),
            "n_distinct_tools_used": int(self.n_distinct_tools_used),
            "tool_variety_score": round(float(self.tool_variety_score), 3),
            "top_tools_by_usage": self.top_tools_by_usage[:10],
            "avg_steps_per_tick": round(float(self.avg_steps_per_tick), 2),
            "avg_tools_per_decision": round(float(self.avg_tools_per_decision), 2),
            "skip_rate": round(float(self.skip_rate), 3),
            "submit_rate": round(float(self.submit_rate), 3),
            "avg_thesis_quality_score": round(float(self.avg_thesis_quality_score), 3),
            "under_trading": bool(self.under_trading),
            "over_trading": bool(self.over_trading),
            "focus_drift": bool(self.focus_drift),
            "sample_warning": self.sample_warning,
            "advisory": (
                "tool_variety_score < 0.2 = brain is defaulting to a small "
                "subset; consider broader exploration. avg_thesis_quality_"
                "score < 0.5 = theses are missing factor_synthesis / DSR / "
                "similar setups citations — TradePlans should reference "
                "the synthesis explicitly. under_trading/over_trading "
                "flags surface goal-misalignment."
            ),
        }


def _score_thesis_quality(thesis: str) -> float:
    """Heuristic 0-1 score for whether a thesis cites the rich
    integrations: factor_synthesis bias, DSR/PBO, similar setups,
    multi-strategy, agents, etc.
    """
    if not thesis or len(thesis) < 30:
        return 0.0
    t = thesis.lower()
    score = 0.0
    keywords = (
        ("bias", 0.20),                # factor_synthesis bias_score
        ("regime", 0.10),
        ("similar", 0.10),              # decision_graph_similar
        ("dsr", 0.10),                  # deflated sharpe
        ("pbo", 0.05),
        ("conviction", 0.05),           # conviction tier
        ("sniper", 0.05),
        ("pattern_score", 0.05),
        ("ml_meta", 0.05),
        ("macro", 0.05),                # macro_overlay
        ("dominance", 0.05),            # btc_dominance
        ("cvd", 0.05),                  # cumulative volume delta
        ("whale", 0.05),
        ("halving", 0.05),
    )
    for kw, weight in keywords:
        if kw in t:
            score += weight
    return min(1.0, score)


def _read_recent_decisions(lookback_ticks: int = DEFAULT_LOOKBACK_TICKS) -> List[Dict[str, Any]]:
    """Pull last N decisions with their plan_json + tool_calls (when present)."""
    try:
        from src.orchestration.warm_store import get_store
        store = get_store()
        conn = sqlite3.connect(store.db_path, timeout=2.0)
        try:
            rows = conn.execute(
                "SELECT ts_ns, plan_json, payload_json, final_action "
                "FROM decisions ORDER BY ts_ns DESC LIMIT ?",
                (int(lookback_ticks),),
            ).fetchall()
        finally:
            conn.close()
        out: List[Dict[str, Any]] = []
        for ts_ns, plan_raw, payload_raw, action in rows:
            try:
                plan = json.loads(plan_raw or "{}")
                payload = json.loads(payload_raw or "{}")
            except Exception:
                continue
            tool_calls = payload.get("tool_calls", []) if isinstance(payload, dict) else []
            steps = int(payload.get("steps_taken", 0)) if isinstance(payload, dict) else 0
            out.append({
                "ts_ns": int(ts_ns),
                "direction": plan.get("direction", "?"),
                "thesis": str(plan.get("thesis", ""))[:500],
                "tool_calls": tool_calls,
                "steps_taken": steps,
                "final_action": action,
            })
        return out
    except Exception as e:
        logger.debug("brain_health read failed: %s", e)
        return []


_CACHE: Dict[int, "tuple[float, BrainHealthSnapshot]"] = {}


def compute_brain_health(lookback_ticks: int = DEFAULT_LOOKBACK_TICKS) -> BrainHealthSnapshot:
    """Aggregate brain-discipline metrics from recent decisions.

    Cached for CACHE_TTL_S — every executor tick (multi-asset loop)
    would otherwise re-scan 100 rows + parse JSON identically.
    """
    now = time.time()
    cached = _CACHE.get(lookback_ticks)
    if cached and (now - cached[0]) < CACHE_TTL_S:
        return cached[1]
    decisions = _read_recent_decisions(lookback_ticks=lookback_ticks)
    n = len(decisions)
    if n == 0:
        snap = BrainHealthSnapshot(
            n_ticks_observed=0,
            sample_warning="no_decisions_in_warm_store",
        )
        _CACHE[lookback_ticks] = (now, snap)
        return snap

    # Tool usage stats
    all_tools_used: List[str] = []
    n_tool_calls_per_tick: List[int] = []
    n_steps_per_tick: List[int] = []
    skip_count = 0
    submit_count = 0
    thesis_scores: List[float] = []

    for d in decisions:
        tcs = d.get("tool_calls") or []
        n_tool_calls_per_tick.append(len(tcs))
        for tc in tcs:
            if isinstance(tc, dict):
                name = tc.get("name", "")
                if name:
                    all_tools_used.append(name)
        n_steps_per_tick.append(int(d.get("steps_taken", 0) or 0))
        if d.get("direction", "").upper() in ("SKIP", "FLAT", ""):
            skip_count += 1
        else:
            submit_count += 1
        thesis_scores.append(_score_thesis_quality(d.get("thesis", "")))

    distinct_tools = set(all_tools_used)
    # Variety score: distinct used / total ever-callable
    try:
        from src.ai.trade_tools import build_default_registry
        total_tools = len(build_default_registry().list_names())
    except Exception:
        total_tools = 100
    variety_score = len(distinct_tools) / max(1, total_tools)
    variety_score = min(1.0, variety_score)

    tool_counter = Counter(all_tools_used)
    top_tools = [
        {"name": name, "uses": cnt}
        for name, cnt in tool_counter.most_common(10)
    ]

    avg_steps = sum(n_steps_per_tick) / max(1, len(n_steps_per_tick))
    avg_tools = sum(n_tool_calls_per_tick) / max(1, len(n_tool_calls_per_tick))
    skip_rate = skip_count / n
    submit_rate = submit_count / n
    avg_thesis = sum(thesis_scores) / max(1, len(thesis_scores))

    # Goal-adherence flags
    try:
        from src.ai import tick_state as _ts
        snap_btc = _ts.get("BTC") or {}
        gap = float(snap_btc.get("gap_to_1pct", 0.0))
        today_trades = int(snap_btc.get("today_trades", 0))
        today_pct = float(snap_btc.get("today_pct_total", 0.0))
    except Exception:
        gap = 0.0
        today_trades = 0
        today_pct = 0.0

    under_trading = gap > 0.5 and today_trades < 2
    over_trading = today_trades > 10 and today_pct < 0
    # Focus drift: same skip reason 5+ times in last 10 SKIPs
    last_skips = [d for d in decisions if d["direction"].upper() in ("SKIP", "FLAT")][:10]
    focus_drift = False
    if len(last_skips) >= 5:
        # Check if all theses contain the same first-30-chars phrase
        first_phrases = [d["thesis"][:30].lower() for d in last_skips]
        if first_phrases:
            most_common = Counter(first_phrases).most_common(1)[0]
            if most_common[1] >= 5:
                focus_drift = True

    sample_warning = ""
    if n < 20:
        sample_warning = "low_sample_under_20_ticks"

    snap = BrainHealthSnapshot(
        n_ticks_observed=n,
        n_distinct_tools_used=len(distinct_tools),
        tool_variety_score=variety_score,
        top_tools_by_usage=top_tools,
        avg_steps_per_tick=avg_steps,
        avg_tools_per_decision=avg_tools,
        skip_rate=skip_rate,
        submit_rate=submit_rate,
        avg_thesis_quality_score=avg_thesis,
        under_trading=under_trading,
        over_trading=over_trading,
        focus_drift=focus_drift,
        sample_warning=sample_warning,
    )
    _CACHE[lookback_ticks] = (now, snap)
    return snap


def render_summary_for_tick() -> str:
    """Compact one-line tick_state summary the brain reads every tick.

    Empty string when no decisions yet (cold start) so brain doesn't
    see a confusing zeroed snapshot."""
    snap = compute_brain_health(lookback_ticks=DEFAULT_LOOKBACK_TICKS)
    if snap.n_ticks_observed < MIN_TICKS_FOR_SUMMARY:
        return ""
    flags = []
    if snap.under_trading:
        flags.append("UNDER_TRADING")
    if snap.over_trading:
        flags.append("OVER_TRADING")
    if snap.focus_drift:
        flags.append("FOCUS_DRIFT")
    if snap.tool_variety_score < LOW_VARIETY_THRESHOLD:
        flags.append("LOW_TOOL_VARIETY")
    if snap.avg_thesis_quality_score < WEAK_THESIS_THRESHOLD:
        flags.append("WEAK_THESIS")
    flags_str = (" flags=[" + ",".join(flags) + "]") if flags else ""
    return (
        f"BRAIN_HEALTH: skip_rate={snap.skip_rate:.0%} "
        f"thesis_quality={snap.avg_thesis_quality_score:.0%} "
        f"tool_variety={snap.tool_variety_score:.0%} "
        f"({snap.n_distinct_tools_used} tools, "
        f"avg {snap.avg_tools_per_decision:.1f}/tick){flags_str}"
    )[:300]
