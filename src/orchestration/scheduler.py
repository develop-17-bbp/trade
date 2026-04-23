"""PeriodicJob scheduler — Phase 4 (Plan §3.3).

Small, in-process supervisor for the four parallel learning loops the bot
already runs (continuous_adapt, autonomous_loop, genetic_loop, retrain)
plus new ones Phase 4.5a adds (MetaLearnerCoordinator). This is NOT a
distributed task queue — anything that needs to survive a process restart
belongs in Phase 3's warm store.

Design:
  - Each PeriodicJob has a name, a callable, and a rate control.
  - Rate can be static (fixed interval) or dynamic (compute_score
    callable returning next interval in seconds). The plan §3.3 calls
    for dynamic intervals so the scheduler can prioritize hot signals.
  - One thread per job; daemon threads so ctrl-C kills cleanly.
  - Per-job Prometheus metrics: last-run ts, duration, failures.
  - Exceptions in a job DO NOT kill the thread; the supervisor logs +
    bumps a counter, then re-sleeps. This matches the resilience model
    for the legacy loops the plan consolidates.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PeriodicJob:
    """One job the scheduler owns.

    rate_s: fixed interval in seconds, OR
    compute_score: callable returning the next-interval seconds (dynamic).
    Exactly one of the two must be set.
    """

    name: str
    fn: Callable[[], None]
    rate_s: Optional[float] = None
    compute_score: Optional[Callable[[], float]] = None
    min_interval_s: float = 0.5
    max_interval_s: float = 3600.0

    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _runs: int = field(default=0, init=False, repr=False)
    _failures: int = field(default=0, init=False, repr=False)
    _last_run_ts: float = field(default=0.0, init=False, repr=False)
    _last_duration_s: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if (self.rate_s is None) == (self.compute_score is None):
            raise ValueError(
                f"PeriodicJob {self.name!r}: set exactly one of rate_s / compute_score"
            )

    def _next_interval(self) -> float:
        try:
            v = self.compute_score() if self.compute_score else self.rate_s
            v = float(v or self.min_interval_s)
        except Exception as e:
            logger.warning("[JOB:%s] compute_score raised %s — using min_interval", self.name, e)
            v = self.min_interval_s
        return max(self.min_interval_s, min(v, self.max_interval_s))

    def _emit_metric(self, success: bool, duration_s: float) -> None:
        try:
            from src.orchestration.metrics import record_job_run
            record_job_run(self.name, ok=success, duration_s=duration_s)
        except Exception:
            pass

    def run(self) -> None:
        """Run once synchronously. Used by tests + the loop body."""
        self._runs += 1
        t0 = time.perf_counter()
        ok = False
        try:
            self.fn()
            ok = True
        except Exception as e:
            self._failures += 1
            logger.warning("[JOB:%s] run #%d raised %s", self.name, self._runs, e)
        finally:
            dt = time.perf_counter() - t0
            self._last_run_ts = time.time()
            self._last_duration_s = dt
            self._emit_metric(ok, dt)

    def _loop(self) -> None:
        logger.info("[JOB:%s] started", self.name)
        while not self._stop.is_set():
            self.run()
            self._stop.wait(self._next_interval())
        logger.info("[JOB:%s] stopped (runs=%d failures=%d)", self.name, self._runs, self._failures)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"job-{self.name}", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)


class Scheduler:
    """Registry of PeriodicJobs. Start/stop together or individually."""

    def __init__(self) -> None:
        self._jobs: List[PeriodicJob] = []
        self._lock = threading.Lock()

    def register(self, job: PeriodicJob) -> PeriodicJob:
        with self._lock:
            # Replace if same name already registered — let tests rewire cleanly.
            self._jobs = [j for j in self._jobs if j.name != job.name]
            self._jobs.append(job)
        return job

    def start_all(self) -> None:
        with self._lock:
            for j in self._jobs:
                j.start()

    def stop_all(self, timeout_s: float = 2.0) -> None:
        with self._lock:
            for j in self._jobs:
                j.stop(timeout_s=timeout_s)

    def jobs(self) -> List[PeriodicJob]:
        with self._lock:
            return list(self._jobs)


_scheduler_singleton: Optional[Scheduler] = None
_sched_lock = threading.Lock()


def get_scheduler() -> Scheduler:
    global _scheduler_singleton
    with _sched_lock:
        if _scheduler_singleton is None:
            _scheduler_singleton = Scheduler()
        return _scheduler_singleton


# ── Emergency-mode aware rate helper (C4a) ──────────────────────────────
#
# The readiness gate publishes `ACT_EMERGENCY_MODE=1` when the rolling
# Sharpe lags target (see src/orchestration/readiness_gate.py::is_emergency_mode).
# Jobs that should "spin harder" under pressure — hyperopt, freqai retrain,
# genetic evolution — wrap their base rate through this helper instead of
# passing a static rate_s.

import os  # noqa: E402  (kept local to module; no other os use above)


def emergency_aware_rate(base_rate_s: float, factor: float = 0.5) -> Callable[[], float]:
    """Return a compute_score callable: base rate normally, base * factor
    in emergency mode. Env is re-read on every call so a flap takes effect
    within one scheduler tick.

    factor=0.5 → halved interval in emergency, doubled cadence.
    factor=1.0 → no-op (pass-through).
    """
    base = float(base_rate_s)
    fac = max(0.1, min(1.0, float(factor)))

    def _compute() -> float:
        if os.environ.get("ACT_EMERGENCY_MODE", "0") == "1":
            return base * fac
        return base

    return _compute
