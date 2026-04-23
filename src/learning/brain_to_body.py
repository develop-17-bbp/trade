"""
Brain-to-body controller — C9.

Turns the dual-brain's output (scanner opportunity trajectory +
analyst verdict history + self-critique accuracy) into concrete
pressure signals that steer downstream subsystems:

  * exploration_bias  → Thompson bandit (explore vs exploit)
  * genetic_cadence_s → next genetic evolution interval
  * emergency_level   → composite caution flag for multiple subsystems
  * priority_agents   → which of the 10 agent tools the Analyst
                        should query first this tick

Design:
  * Pure-logic module — no new ML, no new storage. Reads existing
    brain_memory (scans, traces) + warm_store (critiques) +
    readiness_gate (emergency flag).
  * Singleton BodyController with `refresh()` + `current()` — consumers
    read the cached BodyControls, controller recomputes periodically.
  * Never raises — missing inputs collapse to neutral defaults.
  * Kill switch: ACT_DISABLE_BODY_CONTROLLER=1 — consumers fall back
    to their static defaults.

Output is SUGGESTIVE, not mandatory. Subsystems CAN override
(e.g. autonomous_loop's emergency_mode rules already exist; the
controller only amplifies signal direction, never flips gates).
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

EmergencyLevel = Literal["normal", "caution", "stress"]

logger = logging.getLogger(__name__)


DISABLE_ENV = "ACT_DISABLE_BODY_CONTROLLER"
DEFAULT_GENETIC_CADENCE_S = 21600.0      # 6h baseline


# Mapping from scanner top_signals keywords → which agent tool to
# prioritize. Lowercase substring match; first hit wins per signal.
_SIGNAL_AGENT_MAP = [
    ("breakout",       "ask_trend_momentum"),
    ("momentum",       "ask_trend_momentum"),
    ("trend",          "ask_trend_momentum"),
    ("reversal",       "ask_mean_reversion"),
    ("oversold",       "ask_mean_reversion"),
    ("overbought",     "ask_mean_reversion"),
    ("mean",           "ask_mean_reversion"),
    ("whale",          "ask_sentiment_decoder"),
    ("sentiment",      "ask_sentiment_decoder"),
    ("news",           "ask_sentiment_decoder"),
    ("fomc",           "ask_regime_intelligence"),
    ("macro",          "ask_regime_intelligence"),
    ("regime",         "ask_regime_intelligence"),
    ("volatility",     "ask_market_structure"),
    ("liquidity",      "ask_market_structure"),
    ("sweep",          "ask_market_structure"),
    ("pattern",        "ask_pattern_matcher"),
    ("etf",            "ask_pattern_matcher"),
    ("correlation",    "ask_portfolio_optimizer"),
    ("drawdown",       "ask_risk_guardian"),
    ("risk",           "ask_risk_guardian"),
    ("stop",           "ask_loss_prevention"),
    ("polymarket",     "ask_polymarket_arb"),
]


EMERGENCY_LEVEL_NORMAL = "normal"
EMERGENCY_LEVEL_CAUTION = "caution"
EMERGENCY_LEVEL_STRESS = "stress"


@dataclass
class BodyControls:
    """One snapshot of brain→body pressure signals."""
    exploration_bias: float = 1.0
    genetic_cadence_s: float = DEFAULT_GENETIC_CADENCE_S
    emergency_level: EmergencyLevel = EMERGENCY_LEVEL_NORMAL
    priority_agents: List[str] = field(default_factory=list)
    reason: str = "neutral defaults (no inputs)"
    computed_at: float = field(default_factory=time.time)
    # Raw aggregates for dashboards / audit:
    avg_opportunity_score: float = 0.0
    analyst_match_rate: float = 0.0
    analyst_skip_rate: float = 0.0
    parse_failure_rate: float = 0.0
    scans_considered: int = 0
    traces_considered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exploration_bias": round(self.exploration_bias, 3),
            "genetic_cadence_s": round(self.genetic_cadence_s, 0),
            "emergency_level": self.emergency_level,
            "priority_agents": list(self.priority_agents),
            "reason": self.reason,
            "computed_at": self.computed_at,
            "avg_opportunity_score": round(self.avg_opportunity_score, 1),
            "analyst_match_rate": round(self.analyst_match_rate, 3),
            "analyst_skip_rate": round(self.analyst_skip_rate, 3),
            "parse_failure_rate": round(self.parse_failure_rate, 3),
            "scans_considered": self.scans_considered,
            "traces_considered": self.traces_considered,
        }


# ── Compute core ───────────────────────────────────────────────────────


def compute_controls(
    *,
    scan_scores: Sequence[float],
    analyst_verdicts: Sequence[str],
    critique_matches: Sequence[bool],
    top_signals: Sequence[str],
    emergency_mode: bool = False,
) -> BodyControls:
    """Pure function — given aggregates, return a BodyControls.

    `scan_scores`: recent scanner opportunity_score values (most-recent last).
    `analyst_verdicts`: recent terminated_reason values ("plan","skip",
        "max_steps","parse_failures","disabled").
    `critique_matches`: recent matched_thesis bool outcomes (live trades).
    `top_signals`: current scanner top_signals (lowercase).
    `emergency_mode`: readiness_gate emergency flag.

    Never raises; empty-inputs collapse to neutral defaults.
    """
    n_scans = len(scan_scores)
    n_traces = len(analyst_verdicts)
    n_crit = len(critique_matches)

    avg_opp = (sum(scan_scores) / n_scans) if n_scans else 50.0
    skip_reasons = {"skip", "max_steps", "parse_failures", "disabled"}
    skip_rate = (sum(1 for v in analyst_verdicts if v in skip_reasons)
                 / n_traces) if n_traces else 0.0
    parse_rate = (sum(1 for v in analyst_verdicts if v == "parse_failures")
                  / n_traces) if n_traces else 0.0
    match_rate = (sum(1 for m in critique_matches if m) / n_crit) if n_crit else 0.5

    # ── Emergency level ────────────────────────────────────────────────
    if emergency_mode or (n_crit >= 10 and match_rate < 0.30) or parse_rate > 0.50:
        emergency_level = EMERGENCY_LEVEL_STRESS
    elif (n_crit >= 10 and match_rate < 0.50) or skip_rate > 0.90 or parse_rate > 0.20:
        emergency_level = EMERGENCY_LEVEL_CAUTION
    else:
        emergency_level = EMERGENCY_LEVEL_NORMAL

    # ── Exploration bias (Thompson sampling multiplier) ────────────────
    # 1.0 = neutral. >1 = exploit more (sharpens ranking). <1 = explore.
    if emergency_level == EMERGENCY_LEVEL_STRESS:
        # When stressed, EXPLORE harder — the current champions aren't working.
        exploration_bias = 0.6
    elif emergency_level == EMERGENCY_LEVEL_CAUTION:
        exploration_bias = 0.85
    elif avg_opp > 70 and match_rate > 0.55 and n_crit >= 10:
        # Hot scanner + analyst accuracy → exploit the champion.
        exploration_bias = 1.8
    elif avg_opp > 60 and match_rate > 0.50:
        exploration_bias = 1.3
    else:
        exploration_bias = 1.0

    # ── Genetic cadence ────────────────────────────────────────────────
    if emergency_level == EMERGENCY_LEVEL_STRESS:
        genetic_cadence_s = DEFAULT_GENETIC_CADENCE_S / 3   # 2h
    elif skip_rate > 0.85 and n_traces >= 20:
        # Scanner sees setups but analyst keeps skipping → strategy pool
        # doesn't match reality; evolve sooner.
        genetic_cadence_s = DEFAULT_GENETIC_CADENCE_S / 2   # 3h
    elif emergency_level == EMERGENCY_LEVEL_CAUTION:
        genetic_cadence_s = DEFAULT_GENETIC_CADENCE_S / 1.5   # 4h
    else:
        genetic_cadence_s = DEFAULT_GENETIC_CADENCE_S

    # ── Priority agents (from scanner top_signals) ─────────────────────
    priority: List[str] = []
    seen: set = set()
    for sig in top_signals:
        sig_lower = str(sig).lower()
        for keyword, agent_tool in _SIGNAL_AGENT_MAP:
            if keyword in sig_lower and agent_tool not in seen:
                priority.append(agent_tool)
                seen.add(agent_tool)
                break
        if len(priority) >= 5:
            break

    # Default priority when scanner has no signals — safety-first chain.
    if not priority:
        priority = ["ask_regime_intelligence", "ask_risk_guardian",
                    "ask_trend_momentum"]

    reason = (
        f"opp={avg_opp:.0f} match={match_rate:.2f} skip={skip_rate:.2f} "
        f"parse={parse_rate:.2f} → {emergency_level} "
        f"exp_bias={exploration_bias:.2f} gen_cadence={int(genetic_cadence_s)}s"
    )
    return BodyControls(
        exploration_bias=exploration_bias,
        genetic_cadence_s=genetic_cadence_s,
        emergency_level=emergency_level,
        priority_agents=priority,
        reason=reason,
        avg_opportunity_score=avg_opp,
        analyst_match_rate=match_rate,
        analyst_skip_rate=skip_rate,
        parse_failure_rate=parse_rate,
        scans_considered=n_scans,
        traces_considered=n_traces,
    )


# ── Controller singleton (caches latest BodyControls) ──────────────────


class BodyController:
    """Caches the latest BodyControls. Refreshed by shadow_tick; read
    by thompson_bandit / autonomous_loop / orchestrator / the LLM."""

    def __init__(self):
        self._lock = threading.Lock()
        self._current: BodyControls = BodyControls()
        self._last_refresh: float = 0.0

    def current(self) -> BodyControls:
        with self._lock:
            return self._current

    def last_refresh_age_s(self) -> float:
        with self._lock:
            return (time.time() - self._last_refresh) if self._last_refresh else float("inf")

    def refresh(
        self, asset: str = "BTC",
        *,
        window_scans: int = 20,
        window_traces: int = 30,
        window_critiques: int = 30,
    ) -> BodyControls:
        """Read live data from the existing stores, recompute controls.

        Never raises — any subsystem failure yields neutral defaults.
        """
        if os.environ.get(DISABLE_ENV, "0") == "1":
            controls = BodyControls(reason="disabled by env")
            with self._lock:
                self._current = controls
                self._last_refresh = time.time()
            return controls

        scan_scores: List[float] = []
        verdicts: List[str] = []
        matches: List[bool] = []
        top_signals: List[str] = []
        emergency_mode = False

        # Scanner history + current top_signals.
        try:
            from src.ai.brain_memory import get_brain_memory
            mem = get_brain_memory()
            latest = mem.read_latest_scan(asset, max_age_s=86400.0)
            if latest is not None:
                scan_scores.append(float(latest.opportunity_score or 0.0))
                top_signals = list(latest.top_signals or [])
        except Exception as e:
            logger.debug("body_controller: scanner read failed: %s", e)

        # Recent analyst verdicts.
        try:
            from src.ai.brain_memory import get_brain_memory
            mem = get_brain_memory()
            traces = mem.read_recent_traces(asset, limit=window_traces, max_age_s=86400.0)
            verdicts = [str(t.verdict or "").lower() for t in traces if t]
        except Exception as e:
            logger.debug("body_controller: traces read failed: %s", e)

        # Recent critiques from warm_store.
        try:
            import sqlite3
            from pathlib import Path
            db = os.getenv(
                "ACT_WARM_DB_PATH",
                str(Path(__file__).resolve().parents[2] / "data" / "warm_store.sqlite"),
            )
            if os.path.exists(db):
                conn = sqlite3.connect(db, timeout=2.0)
                rows = conn.execute(
                    "SELECT json_extract(self_critique, '$.matched_thesis') "
                    "FROM decisions WHERE self_critique != '{}' "
                    "AND self_critique IS NOT NULL "
                    "ORDER BY ts_ns DESC LIMIT ?",
                    (int(window_critiques),),
                ).fetchall()
                conn.close()
                matches = [bool(r[0]) for r in rows if r[0] is not None]
        except Exception as e:
            logger.debug("body_controller: critique read failed: %s", e)

        # Emergency flag.
        try:
            from src.orchestration.readiness_gate import is_emergency_mode
            emergency_mode = bool(is_emergency_mode())
        except Exception:
            pass

        controls = compute_controls(
            scan_scores=scan_scores,
            analyst_verdicts=verdicts,
            critique_matches=matches,
            top_signals=top_signals,
            emergency_mode=emergency_mode,
        )
        with self._lock:
            self._current = controls
            self._last_refresh = time.time()
        return controls


_singleton: Optional[BodyController] = None
_singleton_lock = threading.Lock()


def get_controller() -> BodyController:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = BodyController()
        return _singleton


# ── Convenience functions for consumers ───────────────────────────────


def current_exploration_bias(default: float = 1.0) -> float:
    """thompson_bandit reads this to override EMERGENCY_EXPLOIT_BIAS."""
    try:
        return float(get_controller().current().exploration_bias)
    except Exception:
        return default


def current_genetic_cadence_s(default: float = DEFAULT_GENETIC_CADENCE_S) -> float:
    """autonomous_loop / scheduler reads this to adjust genetic PeriodicJob."""
    try:
        return float(get_controller().current().genetic_cadence_s)
    except Exception:
        return default


def current_priority_agents(default: Optional[List[str]] = None) -> List[str]:
    """orchestrator / agentic_bridge can read to bias which tools to
    ask first this tick."""
    try:
        return list(get_controller().current().priority_agents)
    except Exception:
        return list(default or [])


def current_emergency_level(default: str = EMERGENCY_LEVEL_NORMAL) -> str:
    try:
        return str(get_controller().current().emergency_level)
    except Exception:
        return default
