"""Canonical Decision envelope — Phase 0.

The single Pydantic object that flows L1 → L9 for one decision cycle.
Every field added here becomes a field every Experience carries once
Phase 4.5a extends Decision into Experience, so keep this minimal.

Phase 0 scope:
    - decision_id (ULID, time-sortable)
    - trace_id    (Phase 1 will plumb OTel trace context; for Phase 0
                   trace_id == decision_id so JSONL consumers can start
                   indexing on a stable key today)
    - per-layer timestamps + per-layer outputs
    - agent votes
    - final action + client_order_id
    - provenance block (fields present, empty strings — Phase 1 fills)

Phase 1 will add data_snapshot_hash, prompt_hash, model_versions,
authority_rules_version, config_hash. This file should NOT grow
to the kitchen-sink §3.2 schema yet — ship minimal, iterate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from ulid import ULID


def new_decision_id() -> str:
    """Generate a fresh time-sortable ULID for one decision cycle."""
    return str(ULID())


FinalAction = Literal["LONG", "FLAT", "EXIT", "SHORT"]


class AgentVoteRecord(BaseModel):
    """Lightweight snapshot of a single agent's vote for the audit log.

    Mirrors the runtime AgentVote dataclass (src/agents/base_agent.py)
    but serializable + validated. We do NOT replace the runtime class —
    this is the wire/audit format only.
    """

    model_config = ConfigDict(extra="ignore")

    direction: int = 0
    confidence: float = 0.0
    position_scale: float = 1.0
    veto: bool = False
    reasoning: str = ""

    @field_validator("direction")
    @classmethod
    def _dir_range(cls, v: int) -> int:
        if v not in (-1, 0, 1):
            raise ValueError(f"direction must be -1/0/+1, got {v}")
        return v

    @field_validator("confidence", "position_scale")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"value must be in [0,1], got {v}")
        return v


class Provenance(BaseModel):
    """Reproducibility block. Phase 1 fills these; Phase 0 ships empty."""

    model_config = ConfigDict(extra="ignore")

    data_snapshot_hash: str = ""
    prompt_hash: str = ""
    model_versions: Dict[str, str] = Field(default_factory=dict)
    authority_rules_version: str = ""
    config_hash: str = ""


class Decision(BaseModel):
    """Canonical decision envelope flowing L1 → L9 for one cycle."""

    model_config = ConfigDict(extra="ignore", validate_assignment=False)

    # ── Identity ──
    decision_id: str = Field(default_factory=new_decision_id)
    trace_id: str = ""  # Phase 1 plumbs real OTel trace context; Phase 0: trace_id == decision_id
    symbol: str = "BTC"
    ts_started_ns: int = 0
    ts_per_layer_ns: Dict[str, int] = Field(default_factory=dict)

    # ── L1–L4 scalar outputs ──
    l1_signal: int = 0
    l1_confidence: float = 0.0
    l2_sentiment_score: float = 0.0
    l3_risk_score: float = 0.0
    l4_fused_direction: int = 0
    l4_fused_confidence: float = 0.0

    # ── L6 agent consensus ──
    agent_votes: Dict[str, AgentVoteRecord] = Field(default_factory=dict)
    authority_violations: List[str] = Field(default_factory=list)
    consensus_level: str = "PENDING"

    # ── L7 LLM (Phase 1 will hash; Phase 0 keeps refs) ──
    llm_pass1_model: str = ""
    llm_pass2_model: str = ""

    # ── Final action ──
    final_action: FinalAction = "FLAT"
    final_confidence: float = 0.0
    final_position_scale: float = 0.0
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    client_order_id: Optional[str] = None
    veto: bool = False

    # ── Provenance (Phase 1 fills) ──
    provenance: Provenance = Field(default_factory=Provenance)

    # ── Validators ──

    @field_validator("l1_signal", "l4_fused_direction")
    @classmethod
    def _dir_range(cls, v: int) -> int:
        if v not in (-1, 0, 1):
            raise ValueError(f"direction must be -1/0/+1, got {v}")
        return v

    @field_validator(
        "l1_confidence",
        "l2_sentiment_score",
        "l3_risk_score",
        "l4_fused_confidence",
        "final_confidence",
        "final_position_scale",
    )
    @classmethod
    def _bounded(cls, v: float) -> float:
        # sentiment_score can be [-1,1]; others [0,1]. Widen here; tighten later.
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"value must be in [-1,1], got {v}")
        return v

    @model_validator(mode="after")
    def _trace_id_defaults_to_decision_id(self) -> "Decision":
        # Phase 0: if caller didn't set trace_id, mirror decision_id.
        # Phase 1 will stop doing this once OTel is the producer of trace_id.
        if not self.trace_id:
            object.__setattr__(self, "trace_id", self.decision_id)
        return self

    @model_validator(mode="after")
    def _sl_sane_for_longs(self) -> "Decision":
        if self.final_action == "LONG" and self.sl_price is not None and self.sl_price <= 0:
            raise ValueError("sl_price must be positive for LONG")
        return self

    # ── Helpers ──

    def to_audit_dict(self) -> Dict[str, Any]:
        """Serialize to the dict shape the executor's JSONL audit log uses.

        Phase 0 reshapes just enough that the existing _log_decision_audit
        caller can merge this into its payload without touching downstream
        readers. Backward compatible — old keys stay, new keys add.
        """
        d = self.model_dump(mode="json")
        return {
            "decision_id": d["decision_id"],
            "trace_id": d["trace_id"],
            "ts_started_ns": d["ts_started_ns"],
            "ts_per_layer_ns": d["ts_per_layer_ns"],
            "consensus_level": d["consensus_level"],
            "final_action": d["final_action"],
            "final_confidence": d["final_confidence"],
            "final_position_scale": d["final_position_scale"],
            "client_order_id": d["client_order_id"],
            "veto": d["veto"],
            "provenance": d["provenance"],
        }


def synthetic_decision_id(reason: str) -> str:
    """Label for code paths that emit an audit row outside a real decision.

    Phase 0 inspection confirmed the audit function has only ONE call
    site (executor.py:1508) inside a normal decision cycle, so this
    helper is defensive — if a future error path or shutdown handler
    logs a row, it can pass `synthetic_decision_id("shutdown_flatten")`
    instead of None. The uniqueness test explicitly excludes IDs
    starting with 'synth-'.
    """
    safe = "".join(c if c.isalnum() else "_" for c in reason)[:40]
    return f"synth-{safe}-{new_decision_id()}"
