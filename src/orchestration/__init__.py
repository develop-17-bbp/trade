"""Orchestration layer — decision envelope, structured state, scheduler.

This package holds the cross-process plumbing for ACT's decision pipeline:
  - envelope.py:   canonical Decision model that flows L1→L9 (Phase 0)
  - metrics.py:    Prometheus registrars (Phase 1)
  - tracing.py:    OpenTelemetry spans (Phase 1)
  - circuit_breakers.py / retries.py / gpu_scheduler.py / task.py  (Phase 2)
  - scheduler.py / protocols.py / data_contract.py  (Phase 4)
"""

from src.orchestration.envelope import Decision, new_decision_id
from src.orchestration.metrics import (
    record_agent_vote,
    record_authority_violation,
    record_decision,
    record_llm_tokens,
    set_equity,
    start_exporter,
    time_decision,
)
from src.orchestration.tracing import decision_span, init_tracer

__all__ = [
    "Decision",
    "new_decision_id",
    # metrics
    "start_exporter",
    "record_decision",
    "record_agent_vote",
    "record_authority_violation",
    "record_llm_tokens",
    "set_equity",
    "time_decision",
    # tracing
    "init_tracer",
    "decision_span",
]
