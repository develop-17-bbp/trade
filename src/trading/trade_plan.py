"""
TradePlan — the compiled output of ACT's "plan mode" (Claude-Code analogue).

The agentic LLM trade loop (`src/ai/agentic_trade_loop.py`) iterates over
context-gathering tool calls and ultimately emits ONE of these objects.
It is the complete, validated, executable description of a proposed trade:
entry price, tier, size, stop, take-profit ladder, expected hold, exit
conditions, reasoning trace, and the tool-call evidence that supports it.

Nothing reaches the conviction gate or executor without first being
compiled into a TradePlan. That single-artifact discipline is what makes
post-hoc audit and credit assignment tractable — every realized outcome
has a plan on file describing exactly what the LLM thought would happen.

Design constraints that match existing infra:

  * Pydantic BaseModel + validators, same style as
    src/ai/agentic_strategist.py::TradeActionDecision.
  * Size/tier semantics match src/trading/conviction_gate.py::ConvictionResult
    — sniper=3.0x, normal=1.0x, skip=0.0x — so a plan's tier maps 1:1 onto
    the gate's existing size multiplier math.
  * All fields JSON-serialisable (warm_store logs `plan_json`).
  * `valid_until` prevents a stale plan from firing — if the LLM took 30
    seconds compiling and price has moved, the plan is cheap to reject
    rather than entering mid-move.

Not in scope here (handled elsewhere):
  * Authority-rule enforcement — `conviction_gate.evaluate()` + executor
    authority_rules.py run on the plan; this module only shapes it.
  * Actual submission — executor's existing place-order path.
  * Self-critique — `trade_verifier.py` reads the plan off the decision
    row after close and compares predicted vs actual.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


# Default staleness window. A plan older than this when submit_trade_plan
# runs gets rejected as "stale". 120s is long enough for a 6-8 step LLM
# tool-use loop on Ollama/local models; short enough that the price the
# LLM saw is still reasonable.
DEFAULT_VALIDITY_SECONDS = 120


Direction = Literal["LONG", "SHORT", "FLAT", "SKIP"]
Tier = Literal["sniper", "normal", "skip"]


class TPLevel(BaseModel):
    """One rung of a scaled take-profit ladder.

    The plan can specify 1..N rungs; executor's existing partial-take logic
    walks them in price order and closes `fraction` of the remaining position
    at each. Sum of fractions across rungs should be <= 1.0; the remainder
    trails with `exit_conditions`.
    """
    price: float = Field(..., gt=0.0)
    fraction: float = Field(..., gt=0.0, le=1.0)
    reason: str = Field(default="tp")


class ExitCondition(BaseModel):
    """Named non-price exit trigger the executor can evaluate each tick.

    `kind` is matched by name in the executor's existing exit-switch. Unknown
    kinds are ignored (forward-compat for new condition types).
    """
    kind: str  # 'regime_change' | 'news_flatten' | 'time_decay' | 'ema_flip' | ...
    params: Dict[str, Any] = Field(default_factory=dict)


class ToolCallEvidence(BaseModel):
    """Record of one tool call the LLM made while compiling this plan.

    Stored with the plan so the post-close verifier can answer "which
    evidence actually mattered?" — if the LLM consistently cites
    `get_fear_greed_index` on winning trades and ignores it on losers,
    credit assignment re-weights accordingly.
    """
    tool: str
    args_json: str = "{}"
    summary: str = ""  # one-line digest the LLM wrote itself


class TradePlan(BaseModel):
    """Fully-compiled trade proposal. Executable iff `passed_gates=True`."""

    # ── Identity ─────────────────────────────────────────────────────────
    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    decision_id: Optional[str] = None  # linked to warm_store.decisions row
    compiled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: Optional[datetime] = None

    # ── What to trade ────────────────────────────────────────────────────
    asset: str                  # 'BTC' | 'ETH' (upper-cased by validator)
    direction: Direction
    entry_tier: Tier
    entry_price: float = Field(..., gt=0.0)
    size_pct: float = Field(..., ge=0.0, le=30.0)  # hard upper; gate tightens

    # ── Risk shape ───────────────────────────────────────────────────────
    sl_price: float = Field(..., gt=0.0)            # absolute stop, not %
    tp_levels: List[TPLevel] = Field(default_factory=list)
    expected_hold_bars: int = Field(default=24, ge=1)
    exit_conditions: List[ExitCondition] = Field(default_factory=list)

    # ── Thesis + evidence ────────────────────────────────────────────────
    thesis: str = Field(default="", max_length=2000)
    supporting_evidence: List[ToolCallEvidence] = Field(default_factory=list)
    expected_pnl_pct_range: Tuple[float, float] = (0.0, 0.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # ── Gate status (filled by validate_against_gates) ───────────────────
    passed_gates: bool = False
    gate_reasons: List[str] = Field(default_factory=list)

    # ─────────────────────── Validators ──────────────────────────────────

    @field_validator("asset")
    @classmethod
    def _upcase_asset(cls, v: str) -> str:
        return str(v or "").upper()

    @model_validator(mode="after")
    def _fill_defaults_and_check(self) -> "TradePlan":
        # Default valid_until if caller didn't set one.
        if self.valid_until is None:
            self.valid_until = self.compiled_at + timedelta(seconds=DEFAULT_VALIDITY_SECONDS)

        # If SKIP/FLAT, relax risk-shape checks — a skip-plan only needs
        # the thesis text for the log.
        if self.direction in ("SKIP", "FLAT"):
            self.entry_tier = "skip"
            self.size_pct = 0.0
            return self

        # LONG/SHORT must have a sane stop relative to entry.
        if self.direction == "LONG" and self.sl_price >= self.entry_price:
            raise ValueError("LONG plan: sl_price must be below entry_price")
        if self.direction == "SHORT" and self.sl_price <= self.entry_price:
            raise ValueError("SHORT plan: sl_price must be above entry_price")

        # TP rung prices must make sense in direction.
        for rung in self.tp_levels:
            if self.direction == "LONG" and rung.price <= self.entry_price:
                raise ValueError("LONG plan: all TP rungs must be above entry")
            if self.direction == "SHORT" and rung.price >= self.entry_price:
                raise ValueError("SHORT plan: all TP rungs must be below entry")

        # Fractions across rungs must not exceed 1.0 (remainder trails).
        total = sum(r.fraction for r in self.tp_levels)
        if total > 1.0 + 1e-6:
            raise ValueError(f"TP fractions sum to {total:.3f} > 1.0")

        # Tier ↔ size sanity. The conviction gate will re-check against
        # real-time context; this is a cheap upfront guard against the
        # LLM returning wildly mis-sized tiers.
        if self.entry_tier == "sniper" and self.size_pct > 20.0:
            raise ValueError("sniper tier size_pct capped at 20%")
        if self.entry_tier == "normal" and self.size_pct > 8.0:
            raise ValueError("normal tier size_pct capped at 8%")
        return self

    # ─────────────────────── Helpers ─────────────────────────────────────

    def is_stale(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return self.valid_until is not None and now > self.valid_until

    def size_multiplier(self) -> float:
        """Match conviction_gate.ConvictionResult semantics: 3x/1x/0x."""
        return {"sniper": 3.0, "normal": 1.0, "skip": 0.0}[self.entry_tier]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dict for warm_store plan_json column."""
        d = self.model_dump(mode="json")
        # datetime -> ISO-8601 string for sqlite TEXT storage.
        for k in ("compiled_at", "valid_until"):
            v = d.get(k)
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        return d

    @classmethod
    def skip(cls, asset: str, thesis: str = "no-setup") -> "TradePlan":
        """Convenience constructor for a skip-this-tick plan."""
        return cls(
            asset=asset,
            direction="SKIP",
            entry_tier="skip",
            entry_price=1.0,       # placeholder — skip plans don't trade
            size_pct=0.0,
            sl_price=1.0,
            thesis=thesis,
        )
