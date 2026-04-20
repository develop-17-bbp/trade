"""Orchestration layer — decision envelope, structured state, scheduler.

This package holds the cross-process plumbing for ACT's decision pipeline:
  - envelope.py:          canonical Decision model L1→L9   (Phase 0)
  - metrics.py:           Prometheus registrars             (Phase 1)
  - tracing.py:           OpenTelemetry spans               (Phase 1)
  - streams.py:           Redis Streams pub/sub             (Phase 2)
  - hot_state.py:         Redis TTL'd snapshots             (Phase 2)
  - circuit_breakers.py:  pybreaker registry                (Phase 2)
  - retries.py:           tenacity wrappers                 (Phase 2)
  - warm_store.py:        SQLite WAL durable tier           (Phase 3)
  - cold_archive.py:      monthly parquet rollups           (Phase 3)
  - checkpoint.py:        crash-resume snapshots            (Phase 3)
  - scheduler.py:         PeriodicJob supervisor            (Phase 4)
  - gpu_scheduler.py:     priority-aware GPU lease          (Phase 4)
"""

from src.orchestration.envelope import Decision, new_decision_id
from src.orchestration.metrics import (
    record_agent_vote,
    record_authority_violation,
    record_circuit_breaker_state,
    record_circuit_breaker_trip,
    record_decision,
    record_llm_tokens,
    record_stream_publish,
    set_equity,
    start_exporter,
    time_decision,
)
from src.orchestration.streams import (
    STREAM_DECISION_CYCLE,
    STREAM_TRADE_OUTCOME,
    publish as stream_publish,
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
    "record_circuit_breaker_state",
    "record_circuit_breaker_trip",
    "record_stream_publish",
    "time_decision",
    # tracing
    "init_tracer",
    "decision_span",
    # streams
    "stream_publish",
    "STREAM_DECISION_CYCLE",
    "STREAM_TRADE_OUTCOME",
]
