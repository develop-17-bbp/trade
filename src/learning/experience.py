"""Experience envelope — Phase 4.5a (Learning Mesh §2.1).

Extends the Phase 0 Decision envelope with everything needed to train on
a closed trade. One Experience per closed position. Consumed by:

    - credit_assigner.py         (weights per component)
    - meta_coordinator.py        (fan-out to learners)
    - warm_store.py              (durable log)
    - cold_archive.py            (monthly parquet)

Keep this schema stable — anything that reads from the warm store joins
on these field names. If you must add a field, append it (Pydantic will
default-fill for older rows).

Validators enforced:
  - credit_allocation sums to 1.0 ± 1e-6 (warn at 1e-3, reject at 1e-2)
  - pnl_pct within [-0.5, 1.0] (hard bound — larger values mean a
    data-entry bug, not a real trade)
  - exit_reason in {TP, SL, TIMEOUT, MANUAL}
  - regime_tag in {TRENDING, RANGING, VOLATILE, CHOPPY}
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ExitReason = Literal["TP", "SL", "TIMEOUT", "MANUAL"]
RegimeTag = Literal["TRENDING", "RANGING", "VOLATILE", "CHOPPY"]


class Outcome(BaseModel):
    """Realized trade outcome. One per closed position."""

    model_config = ConfigDict(extra="ignore")

    pnl_pct: float = 0.0
    pnl_usd: float = 0.0
    hold_duration_s: float = 0.0
    exit_reason: ExitReason = "MANUAL"
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_ts: float = 0.0
    exit_ts: float = 0.0

    @field_validator("pnl_pct")
    @classmethod
    def _pnl_sanity(cls, v: float) -> float:
        if not -0.5 <= v <= 1.0:
            raise ValueError(f"pnl_pct outside [-0.5, 1.0] — likely data error: {v}")
        return v


class UncertaintySnapshot(BaseModel):
    """Model-level uncertainty captured at decision time. Phase 4.5b fills."""

    model_config = ConfigDict(extra="ignore")

    entropy: float = 0.0
    variance: float = 0.0
    disagreement_score: float = 0.0


class Counterfactual(BaseModel):
    """What-if estimate for a component held FLAT during the cycle."""

    model_config = ConfigDict(extra="ignore")

    action_taken: str = ""
    alt_action: str = "FLAT"
    estimated_alt_pnl: float = 0.0


class AuthorityContext(BaseModel):
    """Which authority rules were on the edge / fired during this cycle."""

    model_config = ConfigDict(extra="ignore")

    strategies_applied: List[str] = Field(default_factory=list)   # ["S1","S2"] subset of S1/S2/S3
    rules_on_edge: List[str] = Field(default_factory=list)        # which of the 7 universal rules
    violations: List[str] = Field(default_factory=list)           # what vetoed the trade (if any)


class Experience(BaseModel):
    """Enriched decision + outcome for training pipelines.

    The Plan §2.1 requires Experience to *extend* Decision. We model it as
    composition rather than inheritance so pydantic v2 doesn't need to
    revalidate the whole Decision schema on every write — the decision
    block is carried as-is from the executor.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=False)

    # Identity (mirrors Decision)
    decision_id: str
    trace_id: str = ""
    symbol: str = "BTC"

    outcome: Outcome = Field(default_factory=Outcome)
    credit_allocation: Dict[str, float] = Field(default_factory=dict)
    uncertainty_snapshot: Dict[str, UncertaintySnapshot] = Field(default_factory=dict)
    counterfactuals: Dict[str, Counterfactual] = Field(default_factory=dict)
    regime_tag_entry: Optional[RegimeTag] = None
    regime_tag_exit: Optional[RegimeTag] = None
    authority_context: AuthorityContext = Field(default_factory=AuthorityContext)

    # Raw decision payload carried forward (kept loose so old rows deserialize).
    decision: Dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _credit_sums_to_one(self) -> "Experience":
        if not self.credit_allocation:
            return self
        total = sum(self.credit_allocation.values())
        if abs(total - 1.0) > 1e-2:
            raise ValueError(
                f"credit_allocation must sum to 1.0 (±1e-2); got {total:.4f}: {self.credit_allocation}"
            )
        return self

    def to_warm_row(self) -> Dict:
        """Flatten for warm_store.write_outcome()."""
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "direction": self.decision.get("direction"),
            "entry_price": self.outcome.entry_price,
            "exit_price": self.outcome.exit_price,
            "pnl_pct": self.outcome.pnl_pct,
            "pnl_usd": self.outcome.pnl_usd,
            "duration_s": self.outcome.hold_duration_s,
            "exit_reason": self.outcome.exit_reason,
            "regime": self.regime_tag_entry or "unknown",
            "entry_ts": self.outcome.entry_ts,
            "exit_ts": self.outcome.exit_ts,
            "credit": self.credit_allocation,
            "authority_violations": self.authority_context.violations,
        }
