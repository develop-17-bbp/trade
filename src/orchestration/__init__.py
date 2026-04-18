"""Orchestration layer — decision envelope, structured state, scheduler.

This package holds the cross-process plumbing for ACT's decision pipeline:
  - envelope.py:   canonical Decision model that flows L1→L9 (Phase 0)
  - logging.py:    structlog wrapper (Phase 1)
  - tracing.py:    OpenTelemetry spans (Phase 1)
  - metrics.py:    Prometheus registrars (Phase 1)
  - circuit_breakers.py / retries.py / gpu_scheduler.py / task.py  (Phase 2)
  - scheduler.py / protocols.py / data_contract.py  (Phase 4)

Phase 0 ships envelope.py only.
"""

from src.orchestration.envelope import Decision, new_decision_id

__all__ = ["Decision", "new_decision_id"]
