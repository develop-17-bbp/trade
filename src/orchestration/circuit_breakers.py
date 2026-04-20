"""Named circuit breakers — Phase 2 (§2.4 resilience).

Replaces the ad-hoc host-based breaker in src/data/base_fetcher.py with a
pybreaker-backed registry. Each breaker is keyed by a stable string name
(e.g. "polymarket", "cryptopanic_rss", "bybit_rest") so Prometheus can
expose state per-dependency.

State mapping emitted to Prometheus (act_circuit_breaker_state gauge):
    0  CLOSED     — normal
    1  HALF_OPEN  — recovery probe in flight
    2  OPEN       — tripped, calls blocked

A trip also increments act_circuit_breaker_trips_total (counter).

Soft dep: if pybreaker isn't installed, `get_breaker` returns a no-op that
just runs the callable directly. The bot stays functional — it just loses
the trip/cooldown benefit and the state metric.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import pybreaker
    _HAS_PYBREAKER = True
except ImportError:
    _HAS_PYBREAKER = False
    logger.info("pybreaker not installed — circuit breakers run in pass-through mode.")


_registry: Dict[str, Any] = {}
_registry_lock = threading.Lock()


if _HAS_PYBREAKER:

    class _MetricsListener(pybreaker.CircuitBreakerListener):
        """Pushes state transitions to Prometheus. Defensive — never raises."""

        def __init__(self, name: str):
            self._name = name

        def state_change(self, cb, old_state, new_state):  # noqa: N802 (pybreaker API)
            try:
                from src.orchestration.metrics import (
                    record_circuit_breaker_state,
                    record_circuit_breaker_trip,
                )
                state_str = getattr(new_state, "name", str(new_state)).lower()
                record_circuit_breaker_state(name=self._name, state=state_str)
                if state_str == "open":
                    record_circuit_breaker_trip(name=self._name)
            except Exception:
                pass

        def failure(self, cb, exc):  # noqa: N802
            logger.debug("[CB:%s] failure: %s", self._name, exc)

        def success(self, cb):  # noqa: N802
            pass


def get_breaker(name: str, fail_max: int = 3, reset_timeout: int = 60) -> Any:
    """Return (and cache) a named CircuitBreaker.

    Thread-safe; repeated calls with the same name always return the same
    object so per-caller threshold drift is impossible.
    """
    with _registry_lock:
        b = _registry.get(name)
        if b is not None:
            return b
        if _HAS_PYBREAKER:
            b = pybreaker.CircuitBreaker(
                fail_max=fail_max,
                reset_timeout=reset_timeout,
                name=name,
                listeners=[_MetricsListener(name)],
            )
        else:
            b = _PassthroughBreaker(name)
        _registry[name] = b
        # Seed the metric at 'closed' so dashboards don't show a blank series
        # until the first failure.
        try:
            from src.orchestration.metrics import record_circuit_breaker_state
            record_circuit_breaker_state(name=name, state="closed")
        except Exception:
            pass
        return b


class _PassthroughBreaker:
    """Fallback when pybreaker isn't installed. Just calls the function."""

    def __init__(self, name: str):
        self.name = name

    def call(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        return fn(*args, **kwargs)

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn


def call(name: str, fn: Callable[..., Any], *args, **kwargs) -> Any:
    """Convenience: look up / create breaker `name` and run `fn` through it.

    Raises whatever the underlying breaker raises — typically
    pybreaker.CircuitBreakerError when tripped. Callers that prefer a
    soft-fail should catch broadly and treat as a feed miss.
    """
    return get_breaker(name).call(fn, *args, **kwargs)


def is_open(name: str) -> bool:
    """True if the named breaker is currently OPEN. Missing breaker → False."""
    b = _registry.get(name)
    if b is None or not _HAS_PYBREAKER:
        return False
    try:
        return str(getattr(b, "current_state", "")).lower() == "open"
    except Exception:
        return False
