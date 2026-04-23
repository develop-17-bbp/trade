"""
Transient persona agents — C14 (MiroFish-inspired, real-time adapted).

Dynamically spawn LLM-backed persona agents from HOT clusters in the
real-time knowledge graph (C12). Personas are TRANSIENT — they exist
only while the cluster they represent remains hot; when the cluster
cools they dissolve. This is the real-time adaptation of MiroFish's
persona-generation: we don't build personas once from static seed docs,
we build them on the fly from what the market is actually doing right
now.

Rules:
  * Fixed 13 specialist agents (risk_guardian, trend_momentum, ...) are
    NEVER replaced — transient personas JOIN the debate alongside them.
  * Each persona inherits `BaseAgent` so it drops into the existing
    orchestrator + debate_engine pipeline with zero architectural change.
  * Personas start with neutral weight (1.0). If they live long enough
    to accumulate trade outcomes, the Bayesian accuracy tracker adjusts
    their weight like any other agent.
  * Cap: `max_dynamic_concurrent` personas at once (default 6) — keeps
    the debate token budget bounded.
  * Min cluster heat required to spawn: `min_cluster_heat` (default 0.6).
    Prevents random-noise clusters from spawning junk personas.

Implementation is thin — 90% of the machinery (AgentVote shape,
BaseAgent accuracy tracking, episodic memory from C12b, orchestrator
registration) already exists. This module:

  1. Scans the graph for hot entity clusters every scheduler tick.
  2. Spawns a TransientPersonaAgent for each new hot cluster above
     the heat threshold, up to the concurrent cap.
  3. Dissolves personas whose cluster heat has dropped below a
     decay threshold.
  4. Exposes `get_active_personas()` so orchestrator can include them
     in the current decision's debate.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base_agent import AgentVote, BaseAgent

logger = logging.getLogger(__name__)


DISABLE_ENV = "ACT_DISABLE_DYNAMIC_PERSONAS"

DEFAULT_MAX_CONCURRENT = int(os.getenv("ACT_PERSONA_MAX_CONCURRENT", "6"))
DEFAULT_MIN_CLUSTER_HEAT = float(os.getenv("ACT_PERSONA_MIN_HEAT", "0.6"))
DEFAULT_DISSOLVE_HEAT = float(os.getenv("ACT_PERSONA_DISSOLVE_HEAT", "0.3"))
DEFAULT_PERSONA_MAX_AGE_S = float(os.getenv("ACT_PERSONA_MAX_AGE_S", "14400.0"))  # 4h hard cap


# ── Transient agent class ─────────────────────────────────────────────


@dataclass
class PersonaDescriptor:
    """What this persona speaks for — derived from one graph cluster."""
    persona_id: str
    name: str                         # e.g. "FOMC-dove", "whale-accumulation-BTC"
    theme: str                        # short description for the LLM prompt
    anchor_entity_id: str             # the central entity in the cluster
    cluster_heat: float               # summed decayed weight at spawn
    neighbor_entity_ids: List[str] = field(default_factory=list)
    spawned_at: float = 0.0
    last_seen_heat: float = 0.0


class TransientPersonaAgent(BaseAgent):
    """LLM-backed agent whose system-prompt is derived from a graph cluster."""

    def __init__(self, descriptor: PersonaDescriptor, config: Optional[Dict] = None):
        super().__init__(name=descriptor.persona_id, config=config or {})
        self.descriptor = descriptor

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        """Ask the Analyst brain to speak as this persona. Never raises."""
        # Build a compact prompt — persona theme + current quant snapshot.
        asset = context.get("asset") or quant_state.get("asset") or "BTC"
        theme = self.descriptor.theme[:400]
        name = self.descriptor.name

        # Pluck a handful of salient metrics so the prompt is grounded.
        salient = {
            k: quant_state.get(k) for k in
            ("ema_slope_pct", "rsi", "atr_pct", "hurst",
             "regime", "zscore", "funding_rate", "fear_greed_index")
            if quant_state.get(k) is not None
        }
        prompt = (
            f"You are a TRANSIENT persona agent named '{name}'. "
            f"You were spawned because ACT's real-time knowledge graph "
            f"detected this cluster:\n  {theme}\n\n"
            f"Current asset: {asset}\n"
            f"Salient metrics: {salient}\n\n"
            f"Return ONE JSON object: "
            f'{{"direction": -1|0|1, "confidence": 0..1, "rationale": "<=200 chars"}}. '
            f"Your job is to vote how your persona would — if you are "
            f"the 'FOMC-dove persona', vote from that viewpoint; if "
            f"'whale-accumulation persona', from that. Keep it short."
        )

        try:
            from src.ai.dual_brain import analyze as _analyze
            resp = _analyze(prompt)
            if resp and resp.ok and resp.text:
                import json as _json
                # Extract first JSON object from the text.
                start = resp.text.find("{")
                end = resp.text.rfind("}")
                if start >= 0 and end > start:
                    obj = _json.loads(resp.text[start : end + 1])
                    return AgentVote(
                        direction=int(obj.get("direction", 0)),
                        confidence=float(obj.get("confidence", 0.5)),
                        reasoning=f"[persona:{name}] {str(obj.get('rationale', ''))[:200]}",
                        metadata={"persona": self.descriptor.persona_id,
                                  "theme": theme},
                    )
        except Exception as e:
            logger.debug("persona %s analyze failed: %s", name, e)

        # Graceful fallback — neutral vote with descriptor in reasoning
        # so the debate combiner sees SOMETHING from this persona.
        return AgentVote(
            direction=0, confidence=0.3,
            reasoning=f"[persona:{name}] LLM unavailable; returning neutral",
            metadata={"persona": self.descriptor.persona_id},
        )


# ── Manager ────────────────────────────────────────────────────────────


class PersonaManager:
    """Owns the lifecycle of transient personas. Singleton per process."""

    def __init__(self,
                 max_concurrent: int = DEFAULT_MAX_CONCURRENT,
                 min_spawn_heat: float = DEFAULT_MIN_CLUSTER_HEAT,
                 dissolve_heat: float = DEFAULT_DISSOLVE_HEAT,
                 max_age_s: float = DEFAULT_PERSONA_MAX_AGE_S):
        self.max_concurrent = max(1, int(max_concurrent))
        self.min_spawn_heat = float(min_spawn_heat)
        self.dissolve_heat = float(dissolve_heat)
        self.max_age_s = float(max_age_s)
        self._lock = threading.Lock()
        self._active: Dict[str, TransientPersonaAgent] = {}

    def active(self) -> List[TransientPersonaAgent]:
        with self._lock:
            return list(self._active.values())

    def refresh(self, asset: str = "BTC") -> Dict[str, Any]:
        """Scan the graph, spawn/dissolve personas. Returns a report dict.

        Called on scheduler cadence (e.g., every tick or every 5 min).
        Never raises — graph/LLM unavailability just yields empty spawn.
        """
        if os.environ.get(DISABLE_ENV, "0") == "1":
            return {"disabled": True}

        report: Dict[str, Any] = {"spawned": [], "dissolved": [], "kept": []}

        # 1. Dissolve old / cold personas.
        now = time.time()
        with self._lock:
            to_drop: List[str] = []
            for pid, agent in self._active.items():
                d = agent.descriptor
                age = now - d.spawned_at
                if age > self.max_age_s:
                    to_drop.append(pid)
                    report["dissolved"].append({"persona_id": pid, "reason": "age"})
            for pid in to_drop:
                self._active.pop(pid, None)

        # 2. Read graph for current hot clusters.
        try:
            from src.ai.graph_rag import get_graph
            g = get_graph()
            asset_eid = f"asset:{asset.lower()}"
            hot = g.top_connected(asset_eid, limit=self.max_concurrent * 2,
                                   since_s=3600.0)
        except Exception as e:
            logger.debug("PersonaManager: graph read failed: %s", e)
            return report

        # 3. Update heat on existing personas; mark cold ones for dissolve.
        with self._lock:
            current_ids = set(self._active.keys())
            hot_by_anchor = {h["neighbor_id"]: h for h in hot}
            for pid, agent in list(self._active.items()):
                d = agent.descriptor
                h = hot_by_anchor.get(d.anchor_entity_id)
                heat = float(h["total_decayed_weight"]) if h else 0.0
                d.last_seen_heat = heat
                if heat < self.dissolve_heat:
                    self._active.pop(pid, None)
                    report["dissolved"].append({
                        "persona_id": pid, "reason": "cold", "heat": heat,
                    })
                else:
                    report["kept"].append({
                        "persona_id": pid, "heat": heat,
                    })

            # 4. Spawn new personas for uncovered hot clusters.
            covered_anchors = {a.descriptor.anchor_entity_id
                               for a in self._active.values()}
            for h in hot:
                if len(self._active) >= self.max_concurrent:
                    break
                anchor = h["neighbor_id"]
                heat = float(h["total_decayed_weight"])
                if heat < self.min_spawn_heat:
                    continue
                if anchor in covered_anchors:
                    continue
                theme = self._build_theme(h)
                desc = PersonaDescriptor(
                    persona_id=f"persona:{anchor}:{int(now)}",
                    name=h["neighbor_name"][:60],
                    theme=theme, anchor_entity_id=anchor,
                    cluster_heat=heat, last_seen_heat=heat,
                    neighbor_entity_ids=[anchor],
                    spawned_at=now,
                )
                agent = TransientPersonaAgent(desc)
                self._active[desc.persona_id] = agent
                covered_anchors.add(anchor)
                report["spawned"].append({
                    "persona_id": desc.persona_id,
                    "name": desc.name, "heat": heat,
                })

        return report

    def _build_theme(self, hot_neighbor: Dict[str, Any]) -> str:
        """Short theme string injected into the persona's LLM prompt.

        Derived purely from the graph's edge/entity labels — no magic.
        A richer version could ask the LLM to summarize a subgraph.
        """
        name = hot_neighbor.get("neighbor_name", "?")
        kind = hot_neighbor.get("neighbor_kind", "?")
        rels = ",".join(hot_neighbor.get("relations", [])[:3])
        heat = hot_neighbor.get("total_decayed_weight", 0.0)
        return (
            f"Hot cluster anchor: '{name}' (kind={kind}, heat={heat:.2f}). "
            f"Related via: {rels}. You represent this cluster's viewpoint "
            f"in the current trade debate."
        )


# ── Process-wide singleton ─────────────────────────────────────────────

_manager_singleton: Optional[PersonaManager] = None
_manager_lock = threading.Lock()


def get_manager() -> PersonaManager:
    global _manager_singleton
    with _manager_lock:
        if _manager_singleton is None:
            _manager_singleton = PersonaManager()
        return _manager_singleton


def get_active_personas() -> List[TransientPersonaAgent]:
    """Orchestrator-facing shortcut."""
    try:
        return get_manager().active()
    except Exception:
        return []
