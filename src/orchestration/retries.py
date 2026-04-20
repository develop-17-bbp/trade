"""Retry helpers — Phase 2 (§2.4).

tenacity wrappers for flaky external feeds (HTTP, websocket, exchange REST).
Retry policy: up to 3 attempts, exponential backoff 1→10s, full jitter.

Only retry on clearly-transient errors. A 4xx response is NOT retried — it
means the request itself is wrong and will fail identically. Retrying those
wastes rate limit and burns the circuit breaker.

Soft dep: if tenacity isn't installed, `retry_api` degrades to a simple
loop with constant 1s sleeps.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Tuple, Type

logger = logging.getLogger(__name__)

# Default transient error set. Concrete callers can pass their own tuple.
try:
    import requests as _requests
    _REQUESTS_ERRORS: Tuple[Type[BaseException], ...] = (
        _requests.ConnectionError,
        _requests.Timeout,
    )
except ImportError:
    _REQUESTS_ERRORS = ()

DEFAULT_TRANSIENT: Tuple[Type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
) + _REQUESTS_ERRORS


try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential_jitter,
    )
    _HAS_TENACITY = True
except ImportError:
    _HAS_TENACITY = False
    logger.info("tenacity not installed — retry helpers use a simple loop fallback.")


def retry_api(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    transient: Tuple[Type[BaseException], ...] = DEFAULT_TRANSIENT,
) -> Callable:
    """Decorator factory for flaky external calls.

    Usage:
        @retry_api()
        def fetch_prices():
            return requests.get(url, timeout=5).json()
    """
    if _HAS_TENACITY:
        return retry(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=initial_wait, max=max_wait, jitter=initial_wait),
            retry=retry_if_exception_type(transient),
        )

    def _decorator(fn: Callable) -> Callable:
        def _wrapped(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except transient as e:
                    last_exc = e
                    if attempt < max_attempts - 1:
                        time.sleep(min(initial_wait * (2 ** attempt), max_wait))
            if last_exc is not None:
                raise last_exc
            return None
        _wrapped.__name__ = fn.__name__
        _wrapped.__doc__ = fn.__doc__
        return _wrapped

    return _decorator
