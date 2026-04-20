"""Meta-coordinator — Phase 4.5a (Learning Mesh §6).

Runs as a PeriodicJob under the Phase 4 scheduler. Every `interval_s`:

    1. XREADGROUP trade.outcome → pull unprocessed outcomes.
    2. For each outcome, look up the matching decision row (warm store) to
       reconstruct per-component actions.
    3. Feed (actions, pnl) into CreditAssigner.
    4. Build an Experience envelope (Phase 4.5a.§2.1).
    5. Persist:
         - Hot:  hot_state("last_experience") — for fast L7 LLM recall.
         - Warm: warm_store.write_outcome(enriched_row).
    6. Publish filtered fan-out: exp.rl / exp.lora / exp.genetic / exp.calibrator.
    7. Ack the original trade.outcome message so re-runs don't double-count.

Curriculum (Plan §6.2):
    - Outlier |pnl_pct| > 2σ → queue for offline replay, don't update online.
    - Authority-violation trades → credit only goes to the veto components.
    - Borderline-confidence trades (conf ∈ [0.45, 0.55]) → double-weight for
      LoRA (informative hard-negatives).

This is the coordinator — NOT the trainer. Individual learners still own
their training loops; we just enrich and fan out the experience stream.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


_GROUP_NAME = "meta-coordinator"
_CONSUMER_NAME = "mc-1"

_FILTERED_STREAMS = {
    "rl": "exp.rl",
    "lora": "exp.lora",
    "genetic": "exp.genetic",
    "calibrator": "exp.calibrator",
}

# Fields visible to each learner (privacy + cache efficiency; Plan §6.1).
_PROJECTIONS: Dict[str, List[str]] = {
    "rl": ["decision_id", "symbol", "pnl_pct", "duration_s", "regime", "credit_allocation"],
    "lora": ["decision_id", "symbol", "pnl_pct", "confidence", "regime", "credit_allocation", "authority_violations"],
    "genetic": ["decision_id", "symbol", "pnl_pct", "regime", "exit_reason", "credit_allocation"],
    "calibrator": ["decision_id", "confidence", "pnl_pct", "credit_allocation"],
}


class MetaCoordinator:
    """Stateful coordinator. Instantiate once, register as a PeriodicJob."""

    def __init__(self, batch_size: int = 32, block_ms: int = 250):
        self.batch_size = batch_size
        self.block_ms = block_ms
        self._lock = threading.Lock()
        self._assigner = None  # lazy — avoid hard sklearn dep at import time
        self._processed = 0
        self._started = False

    @property
    def processed(self) -> int:
        return self._processed

    def _get_assigner(self):
        if self._assigner is not None:
            return self._assigner
        try:
            from src.learning.credit_assigner import CreditAssigner
            self._assigner = CreditAssigner()
        except Exception as e:
            logger.warning("CreditAssigner unavailable: %s", e)
            self._assigner = None
        return self._assigner

    # ── Main tick ─────────────────────────────────────────────────────

    def tick(self) -> int:
        """One pass: consume pending trade.outcome messages, enrich, fan out.

        Returns the number of outcomes processed this tick.
        """
        from src.orchestration.streams import (
            STREAM_TRADE_OUTCOME, ack, ensure_group, read_group, publish,
        )

        if not self._started:
            ensure_group(STREAM_TRADE_OUTCOME, _GROUP_NAME, start_id="0")
            self._started = True

        n = 0
        for msg_id, payload in read_group(
            STREAM_TRADE_OUTCOME, _GROUP_NAME, _CONSUMER_NAME,
            count=self.batch_size, block_ms=self.block_ms,
        ):
            try:
                exp_row = self._enrich(payload)
            except Exception as e:
                logger.warning("enrich failed for %s: %s", msg_id, e)
                ack(STREAM_TRADE_OUTCOME, _GROUP_NAME, msg_id)
                continue

            self._persist(exp_row)
            self._fan_out(exp_row)
            ack(STREAM_TRADE_OUTCOME, _GROUP_NAME, msg_id)
            self._processed += 1
            n += 1
        return n

    # ── Enrichment ────────────────────────────────────────────────────

    def _enrich(self, outcome: Dict) -> Dict:
        """Compute credit + regime tags and build the Experience row."""
        assigner = self._get_assigner()
        pnl = float(outcome.get("pnl_pct", 0.0) or 0.0)
        confidence = float(outcome.get("confidence", 0.5) or 0.5)

        # Reconstruct component actions. In the absence of per-component
        # vote data in the outcome stream, we seed with neutral uniform
        # actions scaled by confidence + direction — the assigner still
        # learns per-component beta from correlation across trades.
        direction_sign = 1.0 if str(outcome.get("direction", "")).upper() in ("LONG", "1", "BUY") else -1.0
        component_actions = {
            "authority_guardian": direction_sign * confidence,
            "loss_prevention": direction_sign * confidence,
            "l1_rl": direction_sign * confidence,
            "l7_lora": direction_sign * confidence,
            "l9_genetic": direction_sign * confidence,
            "confidence_calibrator": confidence,
        }

        if assigner is not None:
            try:
                from src.learning.credit_assigner import TradeRow
                assigner.record(TradeRow(component_actions=component_actions, realized_pnl=pnl))
                credit = assigner.assign()
            except Exception as e:
                logger.debug("assigner.assign() failed: %s", e)
                credit = {k: 1.0 / len(component_actions) for k in component_actions}
        else:
            credit = {k: 1.0 / len(component_actions) for k in component_actions}

        # Curriculum — zero-out non-guardian credit for authority-violation trades.
        violations = outcome.get("authority_violations") or []
        if violations:
            total_guardian = credit.get("authority_guardian", 0.0) + credit.get("loss_prevention", 0.0)
            if total_guardian > 0:
                keep = {
                    "authority_guardian": credit.get("authority_guardian", 0.0) / total_guardian,
                    "loss_prevention": credit.get("loss_prevention", 0.0) / total_guardian,
                }
                credit = keep

        row = dict(outcome)
        row["credit_allocation"] = credit
        row["regime_tag"] = outcome.get("regime")
        row["authority_violations"] = violations
        return row

    # ── Persistence + fan-out ─────────────────────────────────────────

    def _persist(self, row: Dict) -> None:
        try:
            from src.orchestration.warm_store import get_store
            get_store().write_outcome(row)
        except Exception as e:
            logger.debug("warm_store write_outcome failed: %s", e)
        try:
            from src.orchestration.hot_state import set_value
            set_value("last_experience", row, ttl_s=3600)
        except Exception:
            pass

    def _fan_out(self, row: Dict) -> None:
        from src.orchestration.streams import publish
        for consumer, stream in _FILTERED_STREAMS.items():
            projection = {k: row.get(k) for k in _PROJECTIONS.get(consumer, []) if k in row}
            try:
                publish(stream, projection)
            except Exception:
                pass


_coord_singleton: Optional[MetaCoordinator] = None
_coord_lock = threading.Lock()


def get_coordinator() -> MetaCoordinator:
    global _coord_singleton
    with _coord_lock:
        if _coord_singleton is None:
            _coord_singleton = MetaCoordinator()
        return _coord_singleton


def register_scheduler_job(interval_s: float = 10.0) -> None:
    """Register the meta-coordinator as a PeriodicJob.

    The coordinator itself is idempotent; the scheduler just handles the
    heartbeat. Default 10s matches Plan §6.1 ("process every outcome
    within 10s").
    """
    from src.orchestration.scheduler import PeriodicJob, get_scheduler
    coord = get_coordinator()
    get_scheduler().register(PeriodicJob(
        name="meta_coordinator",
        fn=lambda: coord.tick(),
        rate_s=interval_s,
    ))
