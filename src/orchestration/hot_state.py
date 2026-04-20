"""Redis-backed hot state — Phase 2 (Plan §1.3 Hot tier).

Three-tier state model from the hardening plan:
  - Hot  (this module): Redis, TTL-bounded. Last-known snapshots that readers
                        need in < 10ms. E.g. last regime, last equity, last
                        position mark. Wiped on Redis restart — that's fine,
                        the warm tier (SQLite, Phase 3) is the source of truth.
  - Warm (Phase 3):     SQLite WAL. Persisted, queryable history.
  - Cold (Phase 3):     parquet archives. Monthly rollups for offline RL.

All ops soft-fail: dead Redis returns None/False rather than raising. The
decision path must never block on this layer.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_KEY_PREFIX = os.getenv("ACT_HOT_STATE_PREFIX", "act:hot:")
_DEFAULT_TTL_S = int(os.getenv("ACT_HOT_STATE_TTL_S", "3600"))  # 1h


def _k(key: str) -> str:
    return f"{_KEY_PREFIX}{key}"


def set_value(key: str, value: Any, ttl_s: Optional[int] = None) -> bool:
    """JSON-encode `value` and SET it with TTL. Returns True on success."""
    from src.orchestration.streams import get_client  # reuse the cached client
    c = get_client()
    if c is None:
        return False
    try:
        payload = json.dumps(value, default=str).encode("utf-8")
        c.set(_k(key), payload, ex=ttl_s if ttl_s is not None else _DEFAULT_TTL_S)
        return True
    except Exception as e:
        logger.debug("hot_state.set(%s) failed: %s", key, e)
        return False


def get_value(key: str) -> Any:
    """JSON-decode and return the value, or None if absent/unavailable."""
    from src.orchestration.streams import get_client
    c = get_client()
    if c is None:
        return None
    try:
        raw = c.get(_k(key))
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.debug("hot_state.get(%s) failed: %s", key, e)
        return None


def delete(key: str) -> bool:
    from src.orchestration.streams import get_client
    c = get_client()
    if c is None:
        return False
    try:
        c.delete(_k(key))
        return True
    except Exception:
        return False
