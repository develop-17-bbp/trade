"""
Shared Base Fetcher
====================
Common infrastructure for all data fetcher classes:
  - Thread-safe TTL cache with eviction
  - Shared requests.Session with User-Agent
  - Safe HTTP GET helper with logging
"""

import time
import logging
import threading
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
        Safe HTTP GET with error handling and logging.
        Returns parsed JSON dict on success, None on failure.
        """
        try:
            resp = self.session.get(
                url, params=params, headers=headers,
                timeout=timeout or self.timeout
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"API {url} returned {resp.status_code}")
        except requests.exceptions.Timeout:
            logger.warning(f"API timeout: {url}")
        except Exception as e:
            logger.warning(f"API call failed {url}: {e}")
        return None
