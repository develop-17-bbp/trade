"""
Shared Base Fetcher
====================
Common infrastructure for all data fetcher classes:
  - Thread-safe TTL cache with eviction
  - Shared requests.Session with User-Agent
  - Safe HTTP GET helper with logging
  - Host-level circuit breaker (N failures → cooldown window)
    Temporary until Phase 2 replaces this with pybreaker.
"""

import time
import logging
import threading
from collections import defaultdict
from urllib.parse import urlparse

import requests
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Shared User-Agent across all fetchers
USER_AGENT = 'CryptoTradeBot/2.0'

# Default TTLs (seconds)
CACHE_TTL_SHORT = 60       # 1 min — fast-moving data
CACHE_TTL_MEDIUM = 300     # 5 min — moderate data
CACHE_TTL_LONG = 900       # 15 min — slow-moving data

# Max cache entries before eviction sweep
_MAX_CACHE_SIZE = 200

# Circuit breaker: N consecutive failures per host → cool down for D seconds.
# Shared across all fetcher instances so one dead endpoint hit from multiple
# fetchers trips once, not per-fetcher.
_CB_FAIL_THRESHOLD = 3          # trip after 3 consecutive failures
_CB_COOLDOWN_SEC = 300          # 5 min cooldown once tripped
_cb_state: Dict[str, Dict[str, Any]] = defaultdict(
    lambda: {"fails": 0, "open_until": 0.0, "logged_open": False}
)
_cb_lock = threading.Lock()


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _cb_is_open(host: str) -> bool:
    """True if the host is currently in cooldown."""
    if not host:
        return False
    with _cb_lock:
        st = _cb_state[host]
        return time.time() < st["open_until"]


def _cb_record_success(host: str) -> None:
    if not host:
        return
    with _cb_lock:
        st = _cb_state[host]
        if st["fails"] > 0 or st["logged_open"]:
            logger.info(f"[CB] host {host} recovered — resetting")
        st["fails"] = 0
        st["open_until"] = 0.0
        st["logged_open"] = False


def _cb_record_failure(host: str) -> None:
    if not host:
        return
    with _cb_lock:
        st = _cb_state[host]
        st["fails"] += 1
        if st["fails"] >= _CB_FAIL_THRESHOLD and st["open_until"] < time.time():
            st["open_until"] = time.time() + _CB_COOLDOWN_SEC
            st["logged_open"] = True
            logger.warning(
                f"[CB] host {host} tripped after {st['fails']} failures — "
                f"cooling down {_CB_COOLDOWN_SEC}s"
            )


class CachedFetcher:
    """
    Base class providing thread-safe TTL cache, shared HTTP session,
    and a safe GET helper. Subclass this instead of duplicating
    cache/session boilerplate.
    """

    def __init__(self, timeout: int = 8):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.timeout = timeout

        # Thread-safe TTL cache with eviction
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

    # ── Cache ──

    def _get_cached(self, key: str) -> Optional[Any]:
        """Return cached data if present and not expired, else None."""
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry and time.time() - entry['ts'] < entry['ttl']:
                return entry['data']
        return None

    def _set_cached(self, key: str, data: Any, ttl: int):
        """Store data in cache with TTL. Evicts stale entries when cache grows large."""
        with self._cache_lock:
            self._cache[key] = {'data': data, 'ts': time.time(), 'ttl': ttl}
            # Evict expired entries if cache exceeds threshold
            if len(self._cache) > _MAX_CACHE_SIZE:
                now = time.time()
                self._cache = {
                    k: v for k, v in self._cache.items()
                    if now - v['ts'] < v['ttl']
                }

    # ── HTTP ──

    def _safe_get(self, url: str, params: dict = None,
                  headers: dict = None, timeout: int = None) -> Optional[dict]:
        """
        Safe HTTP GET with error handling, logging, and host-level
        circuit breaking (3 consecutive failures → 5min cooldown).
        Returns parsed JSON dict on success, None on failure/skip.
        """
        host = _host_of(url)

        # Short-circuit if host is in cooldown — no request, no log spam
        if _cb_is_open(host):
            return None

        try:
            resp = self.session.get(
                url, params=params, headers=headers,
                timeout=timeout or self.timeout
            )
            if resp.status_code == 200:
                _cb_record_success(host)
                return resp.json()
            logger.warning(f"API {url} returned {resp.status_code}")
            _cb_record_failure(host)
        except requests.exceptions.Timeout:
            logger.warning(f"API timeout: {url}")
            _cb_record_failure(host)
        except Exception as e:
            logger.warning(f"API call failed {url}: {e}")
            _cb_record_failure(host)
        return None
