"""Hierarchical multi-agent orchestrator (TradingAgents-style).

Restructures the agent layer into a four-stage pipeline:

  ANALYST TEAM        →  TRADER         →  RISK TEAM        →  ORCHESTRATOR
  (3 specialist       (synthesizes       (gates plan         (final scoring +
  votes)              into a plan)       against risk)       confidence)

Each stage produces a confidence-scored output that the next stage
reads. Research (TradingAgents, Multi-Agents LLM Financial Trading
Framework arxiv 2412.20138) reports more stable results than single
flat-vote agent layers — explicit role specialization reduces
contradictions and improves auditability.

ACT keeps its existing flat 13-agent layer (src/agents/orchestrator.py)
as the default authority. This hierarchical layer runs **alongside**
when the env flag is set, never replacing the flat layer unless
explicitly promoted.

Activation:
  ACT_AGENT_HIERARCHY unset / "0"  → module dormant
  ACT_AGENT_HIERARCHY = "shadow"   → runs in parallel, logs comparison
                                     to logs/agent_hierarchy_shadow.jsonl,
                                     never affects the authoritative
                                     decision
  ACT_AGENT_HIERARCHY = "1"        → output is authoritative; flat
                                     layer demoted to advisory only

Anti-overfit / anti-noise design:
  * Each stage emits ONE bounded verdict (direction + confidence +
    one-line rationale), not raw debate transcripts
  * Confidence is calibrated against sample-size: <20 prior decisions
    flagged "low_confidence_calibration"
  * The hierarchy uses the SAME 13 underlying agents — it just
    re-organizes how their outputs roll up. No new LLM calls per tick
    beyond the existing flat layer.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SHADOW_LOG_PATH = "logs/agent_hierarchy_shadow.jsonl"
LOW_CONFIDENCE_THRESHOLD_SAMPLES = 20


@dataclass
class StageVerdict:
    """One stage's bounded output."""
    stage: str                    # "analyst" | "trader" | "risk" | "orchestrator"
    direction: int                # +1 / -1 / 0
    confidence: float             # 0.0 - 1.0
    rationale: str = ""           # capped at 200 chars
    contributing_agents: List[str] = field(default_factory=list)
    veto: bool = False            # only the risk team can set this

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "direction": int(self.direction),
            "confidence": round(float(self.confidence), 3),
            "rationale": self.rationale[:200],
            "contributing_agents": self.contributing_agents[:8],
            "veto": bool(self.veto),
        }


@dataclass
class HierarchicalDecision:
    """End-to-end decision trace through the four stages."""
    asset: str
    analyst_team: StageVerdict
    trader: StageVerdict
    risk_team: StageVerdict
    orchestrator: StageVerdict
    final_direction: int
    final_confidence: float
    sample_size: int              # decisions used for confidence calibration
    confidence_calibration: str   # "ok" | "low_sample"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "analyst_team": self.analyst_team.to_dict(),
            "trader": self.trader.to_dict(),
            "risk_team": self.risk_team.to_dict(),
            "orchestrator": self.orchestrator.to_dict(),
            "final_direction": int(self.final_direction),
            "final_confidence": round(float(self.final_confidence), 3),
            "sample_size": int(self.sample_size),
            "confidence_calibration": self.confidence_calibration,
        }


def is_enabled() -> bool:
    val = (os.environ.get("ACT_AGENT_HIERARCHY") or "").strip().lower()
    return val in ("shadow", "1", "true", "on")


def is_authoritative() -> bool:
    val = (os.environ.get("ACT_AGENT_HIERARCHY") or "").strip().lower()
    return val in ("1", "true", "on")


# ── Stage groupings ─────────────────────────────────────────────────────
# These group the existing 13 agents into specialist teams. The
# hierarchy doesn't add agents — it re-organizes the existing votes.

ANALYST_TEAM = (
    "market_structure", "regime_intelligence", "trend_momentum",
    "mean_reversion", "pattern_matcher",
)
TRADER_AGENT = ("trade_timing",)
RISK_TEAM = (
    "risk_guardian", "loss_prevention_guardian",
    "authority_compliance_guardian",
)
ORCHESTRATOR_AGENTS = (
    "portfolio_optimizer", "decision_auditor",
)
INDEPENDENT = ("sentiment_decoder", "polymarket_agent")


def _aggregate(votes: Dict[str, Dict[str, Any]],
               group: tuple) -> tuple:
    """Combine N agent votes into a single direction + confidence +
    contributing-agents list. Returns (direction, confidence, list).
    Empty group → (0, 0.0, [])."""
    if not votes:
        return 0, 0.0, []
    total_dir = 0.0
    total_conf = 0.0
    n = 0
    contributing = []
    for name in group:
        v = votes.get(name)
        if not isinstance(v, dict):
            continue
        d = float(v.get("direction", 0))
        c = float(v.get("confidence", 0))
        total_dir += d * c
        total_conf += c
        n += 1
        if abs(d) > 0:
            contributing.append(name)
    if n == 0:
        return 0, 0.0, []
    blended = total_dir / max(0.001, total_conf) if total_conf > 0 else 0.0
    avg_conf = total_conf / n
    direction = 1 if blended > 0.15 else (-1 if blended < -0.15 else 0)
    return direction, max(0.0, min(1.0, avg_conf)), contributing


def _count_recent_decisions() -> int:
    """How many decisions exist in warm_store — used to calibrate
    confidence (<20 = low-sample warning)."""
    try:
        import sqlite3
        from src.orchestration.warm_store import get_store
        store = get_store()
        conn = sqlite3.connect(store.db_path, timeout=2.0)
        try:
            n = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        finally:
            conn.close()
        return int(n)
    except Exception:
        return 0


def hierarchical_decide(asset: str,
                        agent_votes: Dict[str, Dict[str, Any]]) -> HierarchicalDecision:
    """Roll up agent votes through the 4-stage hierarchy.

    `agent_votes` matches the structure already produced by
    `agents/orchestrator.py::run_orchestrator` — a dict keyed by agent
    name with subdict containing 'direction' and 'confidence'.
    """
    # Stage 1 — analyst team aggregates 5 specialist votes
    a_dir, a_conf, a_contrib = _aggregate(agent_votes, ANALYST_TEAM)
    analyst = StageVerdict(
        stage="analyst", direction=a_dir, confidence=a_conf,
        rationale=f"{len(a_contrib)}/{len(ANALYST_TEAM)} agents voted with direction",
        contributing_agents=a_contrib,
    )

    # Stage 2 — trader synthesizes (uses analyst output + timing agent)
    t_dir, t_conf, t_contrib = _aggregate(agent_votes, TRADER_AGENT)
    # Trader can override analyst direction if timing has high confidence
    if t_conf > 0.7 and t_dir != 0 and t_dir != a_dir:
        synth_dir = t_dir
        synth_rationale = f"timing override of analyst ({a_dir}->{t_dir})"
    else:
        synth_dir = a_dir
        synth_rationale = "trader confirms analyst direction"
    synth_conf = (a_conf + t_conf) / 2 if t_conf > 0 else a_conf
    trader = StageVerdict(
        stage="trader", direction=synth_dir, confidence=synth_conf,
        rationale=synth_rationale,
        contributing_agents=a_contrib + t_contrib,
    )

    # Stage 3 — risk team gates (can veto)
    r_dir, r_conf, r_contrib = _aggregate(agent_votes, RISK_TEAM)
    risk_veto = (r_dir != 0 and r_dir != synth_dir and r_conf >= 0.6)
    risk = StageVerdict(
        stage="risk", direction=r_dir, confidence=r_conf,
        rationale=("VETO: risk team disagrees" if risk_veto
                   else "risk team confirms"),
        contributing_agents=r_contrib, veto=risk_veto,
    )

    # Stage 4 — orchestrator final scoring
    o_dir, o_conf, o_contrib = _aggregate(agent_votes, ORCHESTRATOR_AGENTS)
    if risk_veto:
        final_dir = 0
        final_conf = 0.0
        o_rationale = "risk veto applied"
    else:
        # Final = trader direction, confidence adjusted by orchestrator agreement
        final_dir = synth_dir
        # Orchestrator agreement bonus, disagreement penalty
        if o_dir == synth_dir:
            final_conf = min(1.0, synth_conf + 0.05)
            o_rationale = "orchestrator confirms"
        elif o_dir == 0:
            final_conf = synth_conf
            o_rationale = "orchestrator neutral"
        else:
            final_conf = max(0.0, synth_conf - 0.10)
            o_rationale = "orchestrator dissent — confidence reduced"
    orchestrator = StageVerdict(
        stage="orchestrator", direction=final_dir, confidence=final_conf,
        rationale=o_rationale, contributing_agents=o_contrib,
    )

    sample_size = _count_recent_decisions()
    calibration = ("low_sample" if sample_size < LOW_CONFIDENCE_THRESHOLD_SAMPLES
                   else "ok")

    return HierarchicalDecision(
        asset=asset,
        analyst_team=analyst, trader=trader,
        risk_team=risk, orchestrator=orchestrator,
        final_direction=final_dir, final_confidence=final_conf,
        sample_size=sample_size, confidence_calibration=calibration,
    )


def log_shadow(decision: HierarchicalDecision,
               flat_direction: int, flat_confidence: float) -> None:
    """Append shadow comparison: flat vs hierarchy. Never raises."""
    try:
        path = Path(SHADOW_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.time(),
            "asset": decision.asset,
            "hierarchy": decision.to_dict(),
            "flat_direction": int(flat_direction),
            "flat_confidence": round(float(flat_confidence), 3),
            "agree": decision.final_direction == flat_direction,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception as e:
        logger.debug("hierarchy shadow log failed: %s", e)
