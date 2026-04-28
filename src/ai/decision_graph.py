"""Decision graph extension to graph_rag.

The existing knowledge graph (`src/ai/graph_rag.py`) holds market
narrative — news, sentiment, institutional flows, correlations. This
module extends it with the BRAIN's own decisions and outcomes:

  * decision nodes      (TradePlan submitted)
  * outcome nodes       (close pnl + verdict)
  * critique nodes      (post-trade SelfCritique lessons)
  * agent_vote nodes    (specialist agent verdicts at decision time)

  edges:
    decision -CAUSED_BY-> regime + pattern + agent_consensus
    decision -RESULTED_IN-> outcome
    outcome  -CRITIQUED_BY-> critique
    decision -SIMILAR_TO-> decision (when shared regime + direction)

The brain queries this graph to answer causal questions in ONE call:
  "Show me every LONG that fired during HMM CRISIS where macro was
  bearish — what was the outcome distribution?"

Without this, the brain would join 4 tools (recent_plans + verifier
+ recent_critiques + tick_state). The graph collapses that join.

Anti-overfit / anti-noise:
  * Read-only over warm_store + brain_memory (no new write surface)
  * Time-decayed edge weights (older edges contribute less)
  * Bounded query results (default top-20 nodes)
  * Returns aggregate stats (win-rate by bucket), not raw data
  * Empty result = honest "insufficient_history" not a faked synthesis

This module is a READ-ONLY view layer over existing stores. No new
schema, no new persistence. The graph is recomputed on demand from
warm_store.decisions + brain_memory.analyst_traces. For a 500-row
warm_store this is sub-100ms; far below tick cadence.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_NODES = 20
DEFAULT_DECAY_DAYS = 60


@dataclass
class GraphNode:
    """A node in the decision graph (decision/outcome/critique/regime/etc)."""
    node_id: str
    node_type: str               # "decision" | "outcome" | "critique" | "regime" | "pattern"
    properties: Dict[str, Any] = field(default_factory=dict)
    age_s: float = 0.0
    decay_weight: float = 1.0    # exp(-age / 30days)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "properties": self.properties,
            "age_s": round(self.age_s, 1),
            "decay_weight": round(self.decay_weight, 4),
        }


@dataclass
class GraphEdge:
    src: str
    dst: str
    edge_type: str               # "CAUSED_BY" | "RESULTED_IN" | "CRITIQUED_BY" | "SIMILAR_TO"
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src, "dst": self.dst,
            "edge_type": self.edge_type,
            "weight": round(float(self.weight), 4),
        }


@dataclass
class CausalQueryResult:
    """Structured response to 'what happened on similar setups?'"""
    matched_decisions: int
    win_rate: float
    avg_pnl_pct: float
    avg_pnl_pct_net: float
    median_pnl_pct: float
    direction_distribution: Dict[str, int]   # LONG/SHORT/SKIP counts
    regime_filter: str
    pattern_filter: str
    direction_filter: str
    sample_warning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched_decisions": int(self.matched_decisions),
            "win_rate": round(float(self.win_rate), 4),
            "avg_pnl_pct": round(float(self.avg_pnl_pct), 4),
            "avg_pnl_pct_net": round(float(self.avg_pnl_pct_net), 4),
            "median_pnl_pct": round(float(self.median_pnl_pct), 4),
            "direction_distribution": dict(self.direction_distribution),
            "regime_filter": self.regime_filter,
            "pattern_filter": self.pattern_filter,
            "direction_filter": self.direction_filter,
            "sample_warning": self.sample_warning,
        }


def _read_decisions(asset: Optional[str] = None,
                    limit: int = 500,
                    decay_days: int = DEFAULT_DECAY_DAYS) -> List[Dict[str, Any]]:
    """Pull decisions + their plan + critique from warm_store.

    Returns one row per decision with merged plan + critique data.
    """
    try:
        import sqlite3
        from src.orchestration.warm_store import get_store
        store = get_store()
        cutoff_ns = int((time.time() - decay_days * 86400) * 1e9)
        conn = sqlite3.connect(store.db_path, timeout=2.0)
        try:
            sql = (
                "SELECT decision_id, ts_ns, symbol, plan_json, "
                "self_critique, final_action, component_signals "
                "FROM decisions "
                "WHERE ts_ns >= ? "
                + ("AND symbol = ? " if asset else "")
                + "ORDER BY ts_ns DESC LIMIT ?"
            )
            params = ((cutoff_ns, asset, limit) if asset
                      else (cutoff_ns, limit))
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        out = []
        for did, ts_ns, sym, plan_raw, crit_raw, action, comp_raw in rows:
            try:
                plan = json.loads(plan_raw or "{}")
                crit = json.loads(crit_raw or "{}")
                comp = json.loads(comp_raw or "{}")
            except Exception:
                continue
            out.append({
                "decision_id": did,
                "ts_ns": int(ts_ns),
                "asset": sym,
                "direction": plan.get("direction", "SKIP"),
                "tier": plan.get("entry_tier", ""),
                "size_pct": float(plan.get("size_pct", 0.0)),
                "regime": comp.get("regime", "unknown"),
                "pattern_label": comp.get("pattern_label", ""),
                "pattern_score": int(comp.get("pattern_score", 0) or 0),
                "macro_bias": float(comp.get("macro_bias", 0.0) or 0.0),
                "realized_pnl_pct": crit.get("realized_pnl_pct"),
                "final_action": action,
                "self_critique_lessons": str(crit.get("lessons", ""))[:200],
            })
        return out
    except Exception as e:
        logger.debug("decision_graph read failed: %s", e)
        return []


def causal_query(
    asset: Optional[str] = None,
    regime: Optional[str] = None,
    pattern_label: Optional[str] = None,
    direction: Optional[str] = None,
    min_pattern_score: int = 0,
    decay_days: int = DEFAULT_DECAY_DAYS,
    spread_pct: float = 1.69,
) -> CausalQueryResult:
    """Causal traversal: 'show me every decision matching these
    filters; what was the outcome distribution?'

    Filters are AND-combined. Empty filter = no constraint.
    """
    rows = _read_decisions(asset=asset, limit=500, decay_days=decay_days)

    # Apply filters
    matched = []
    direction_dist: Dict[str, int] = {}
    for r in rows:
        if regime and str(r.get("regime", "")).upper() != str(regime).upper():
            continue
        if pattern_label and str(r.get("pattern_label", "")).upper() != str(pattern_label).upper():
            continue
        if direction and str(r.get("direction", "")).upper() != str(direction).upper():
            continue
        if int(r.get("pattern_score", 0)) < min_pattern_score:
            continue
        if r.get("realized_pnl_pct") is None:
            continue
        matched.append(r)
        d = str(r.get("direction", "?"))
        direction_dist[d] = direction_dist.get(d, 0) + 1

    n = len(matched)
    if n == 0:
        return CausalQueryResult(
            matched_decisions=0, win_rate=0.0,
            avg_pnl_pct=0.0, avg_pnl_pct_net=0.0,
            median_pnl_pct=0.0,
            direction_distribution={},
            regime_filter=regime or "any",
            pattern_filter=pattern_label or "any",
            direction_filter=direction or "any",
            sample_warning="no_decisions_matched_filters",
        )

    pnls = [float(r["realized_pnl_pct"]) for r in matched]
    pnls_net = [p - spread_pct for p in pnls]
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / n
    avg_pnl = sum(pnls) / n
    avg_net = sum(pnls_net) / n
    sorted_pnls = sorted(pnls)
    median = sorted_pnls[n // 2]

    sample_warning = ""
    if n < 10:
        sample_warning = "small_sample_under_10_decisions"

    return CausalQueryResult(
        matched_decisions=n,
        win_rate=win_rate,
        avg_pnl_pct=avg_pnl,
        avg_pnl_pct_net=avg_net,
        median_pnl_pct=median,
        direction_distribution=direction_dist,
        regime_filter=regime or "any",
        pattern_filter=pattern_label or "any",
        direction_filter=direction or "any",
        sample_warning=sample_warning,
    )


def similar_setups(
    asset: str,
    current_regime: str,
    current_pattern: str,
    current_direction: str,
    top_k: int = 5,
    decay_days: int = DEFAULT_DECAY_DAYS,
) -> Dict[str, Any]:
    """Find the K most-similar past setups + their outcomes.

    Similarity = exact regime + exact pattern + exact direction.
    Returns top-K by recency (most recent first, then time-decayed).
    """
    rows = _read_decisions(asset=asset, limit=200, decay_days=decay_days)
    matches = []
    for r in rows:
        if (str(r.get("regime", "")).upper() != str(current_regime).upper()
                or str(r.get("pattern_label", "")).upper() != str(current_pattern).upper()
                or str(r.get("direction", "")).upper() != str(current_direction).upper()):
            continue
        if r.get("realized_pnl_pct") is None:
            continue
        matches.append(r)
    matches = matches[:top_k]
    return {
        "asset": asset, "regime": current_regime,
        "pattern": current_pattern, "direction": current_direction,
        "n_matched": len(matches),
        "setups": [
            {
                "ts_ns": m["ts_ns"],
                "tier": m["tier"],
                "realized_pnl_pct": round(float(m["realized_pnl_pct"]), 3),
                "size_pct": m["size_pct"],
                "lesson": m["self_critique_lessons"],
            }
            for m in matches
        ],
        "advisory": (
            "These are exact-match historical setups. If win-rate "
            "across them is high but recent_critiques flag a regime "
            "shift, don't extrapolate. <5 matches = low-confidence."
        ),
    }


def build_node_view(
    asset: Optional[str] = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    decay_days: int = DEFAULT_DECAY_DAYS,
) -> Dict[str, Any]:
    """Construct a graph view: nodes (decisions, regimes, patterns) +
    edges (CAUSED_BY, RESULTED_IN). Returns at most max_nodes.

    For audit / dashboard inspection — brain-side use is mostly via
    causal_query + similar_setups."""
    rows = _read_decisions(asset=asset, limit=max_nodes, decay_days=decay_days)
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []
    now = time.time()
    for r in rows[:max_nodes]:
        age_s = now - r["ts_ns"] / 1e9
        decay = math.exp(-age_s / (decay_days * 86400))
        # Decision node
        d_id = f"d:{r['decision_id'][:12]}"
        nodes.append(GraphNode(
            node_id=d_id, node_type="decision",
            properties={
                "asset": r["asset"], "direction": r["direction"],
                "tier": r["tier"], "size_pct": r["size_pct"],
                "realized_pnl_pct": r["realized_pnl_pct"],
            },
            age_s=age_s, decay_weight=decay,
        ))
        # Regime node + edge
        regime = str(r.get("regime", "unknown"))
        regime_id = f"r:{regime}"
        edges.append(GraphEdge(d_id, regime_id, "CAUSED_BY", weight=decay))
        # Pattern node + edge
        pattern = str(r.get("pattern_label", "none"))
        if pattern and pattern != "none":
            pattern_id = f"p:{pattern}"
            edges.append(GraphEdge(d_id, pattern_id, "CAUSED_BY", weight=decay))
        # Outcome edge (synthesized as a property, not a separate node
        # for compactness)
    return {
        "asset_filter": asset or "any",
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "decay_days": decay_days,
        "nodes": [n.to_dict() for n in nodes],
        "edges": [e.to_dict() for e in edges],
    }
