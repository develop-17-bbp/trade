"""
API Circuit Breaker — Prevent Cascade Failures on Robinhood
=============================================================
Robinhood has undocumented rate limits and can ban accounts that
send too many requests. This circuit breaker:

1. Tracks API failures in a sliding window
2. TRIPS (opens) after N failures in M seconds
3. BLOCKS all API calls for a recovery period
4. PROBES with a single request to test recovery
5. ALERTS via the existing AlertManager

States:
  CLOSED → normal, requests flow through
  OPEN   → tripped, all requests blocked (returns error immediately)
  HALF_OPEN → recovery probe, one request allowed to test

Usage:
    breaker = CircuitBreaker(failure_threshold=3, window_seconds=300)

    # Wrap any API call:
    try:
        result = breaker.call(robinhood_client.get_best_price, "BTC-USD")
    except CircuitBreakerOpen:
        print("API paused — waiting for recovery")
"""

import time
import threading
import logging
from collections import deque
from enum import Enum
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


class BreakerState(Enum):
    CLOSED = "CLOSED"       # Normal — requests flow through
    OPEN = "OPEN"           # Tripped — requests blocked
    HALF_OPEN = "HALF_OPEN"  # Recovery probe — one request allowed


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is OPEN and blocking requests."""
    def __init__(self, remaining_seconds: int):
        self.remaining = remaining_seconds
        super().__init__(f"Circuit breaker OPEN. Retry in {remaining_seconds}s.")


class CircuitBreaker:
    """
    Production circuit breaker for API rate limiting protection.

    Args:
        failure_threshold: Number of failures to trip the breaker (default: 3)
        window_seconds: Sliding window for counting failures (default: 300 = 5min)
        recovery_timeout: How long to stay OPEN before probing (default: 1800 = 30min)
        alert_fn: Optional callback when breaker trips (receives error as arg)
        name: Identifier for logging
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        window_seconds: int = 300,
        recovery_timeout: int = 1800,
        alert_fn: Optional[Callable] = None,
        name: str = "API",
    ):
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.recovery_timeout = recovery_timeout
        self.alert_fn = alert_fn
        self.name = name

        self._state = BreakerState.CLOSED
        self._failures: deque = deque()
        self._opened_at: float = 0
        self._total_trips: int = 0
        self._total_failures: int = 0
        self._total_calls: int = 0
        self._lock = threading.Lock()

        logger.info(f"[CIRCUIT-BREAKER:{name}] Initialized: {failure_threshold} failures in {window_seconds}s → pause {recovery_timeout}s")

    @property
    def state(self) -> BreakerState:
        """Get current state, auto-transitioning OPEN → HALF_OPEN after timeout."""
        with self._lock:
            if self._state == BreakerState.OPEN:
                elapsed = time.time() - self._opened_at
                if elapsed >= self.recovery_timeout:
                    self._state = BreakerState.HALF_OPEN
                    logger.info(f"[CIRCUIT-BREAKER:{self.name}] OPEN → HALF_OPEN (probing recovery)")
            return self._state

    @property
    def is_available(self) -> bool:
        """Check if requests can go through."""
        return self.state != BreakerState.OPEN

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Wrap an API call with circuit breaker protection.

        Args:
            fn: The function to call (e.g., robinhood_client.get_best_price)
            *args, **kwargs: Arguments to pass to fn

        Returns:
            Result of fn(*args, **kwargs)

        Raises:
            CircuitBreakerOpen: If breaker is OPEN
            Original exception: If fn raises and breaker hasn't tripped yet
        """
        self._total_calls += 1
        current_state = self.state

        if current_state == BreakerState.OPEN:
            remaining = int(self.recovery_timeout - (time.time() - self._opened_at))
            raise CircuitBreakerOpen(max(0, remaining))

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def call_safe(self, fn: Callable, *args, default=None, **kwargs) -> Any:
        """
        Like call() but returns default instead of raising on OPEN.
        Use for non-critical calls (price checks, status queries).
        """
        try:
            return self.call(fn, *args, **kwargs)
        except CircuitBreakerOpen:
            return default
        except Exception:
            return default

    def _on_success(self):
        """Record successful API call."""
        with self._lock:
            if self._state == BreakerState.HALF_OPEN:
                self._state = BreakerState.CLOSED
                self._failures.clear()
                logger.info(f"[CIRCUIT-BREAKER:{self.name}] HALF_OPEN → CLOSED (recovered)")
                print(f"  [CIRCUIT-BREAKER] {self.name} recovered — API calls resumed")

    def _on_failure(self, error: Exception):
        """Record failed API call, potentially trip the breaker."""
        with self._lock:
            now = time.time()
            self._failures.append(now)
            self._total_failures += 1

            # Evict failures outside the sliding window
            while self._failures and now - self._failures[0] > self.window_seconds:
                self._failures.popleft()

            count = len(self._failures)
            logger.warning(
                f"[CIRCUIT-BREAKER:{self.name}] Failure {count}/{self.failure_threshold}: "
                f"{type(error).__name__}: {str(error)[:100]}"
            )

            if count >= self.failure_threshold:
                self._state = BreakerState.OPEN
                self._opened_at = now
                self._total_trips += 1

                msg = (
                    f"[CIRCUIT-BREAKER:{self.name}] TRIPPED! "
                    f"{count} failures in {self.window_seconds}s. "
                    f"Pausing all API calls for {self.recovery_timeout}s. "
                    f"Total trips: {self._total_trips}"
                )
                logger.error(msg)
                print(f"  {msg}")

                # Alert
                if self.alert_fn:
                    try:
                        self.alert_fn(
                            'CRITICAL',
                            f'Circuit Breaker Tripped: {self.name}',
                            msg,
                            {'error': str(error), 'failures': count, 'trips': self._total_trips}
                        )
                    except Exception:
                        pass

            # In HALF_OPEN, a single failure re-trips immediately
            if self._state == BreakerState.HALF_OPEN:
                self._state = BreakerState.OPEN
                self._opened_at = now
                logger.warning(f"[CIRCUIT-BREAKER:{self.name}] HALF_OPEN → OPEN (probe failed)")

    def get_status(self) -> dict:
        """Return current breaker status for dashboard/API."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failures_in_window': len(self._failures),
            'threshold': self.failure_threshold,
            'total_trips': self._total_trips,
            'total_failures': self._total_failures,
            'total_calls': self._total_calls,
            'is_available': self.is_available,
        }

    def force_close(self):
        """Manually reset breaker to CLOSED (admin override)."""
        with self._lock:
            self._state = BreakerState.CLOSED
            self._failures.clear()
            logger.info(f"[CIRCUIT-BREAKER:{self.name}] Manually reset to CLOSED")
