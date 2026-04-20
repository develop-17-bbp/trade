"""Redis Streams — Phase 2 (§6.1).

Thin wrapper around redis-py's stream API. Every public function is a
soft-fail: if Redis is unreachable / the `redis` package isn't installed /
`ACT_REDIS_URL` is empty, the call returns None instead of raising. The
decision path MUST NOT block on stream availability.

Canonical streams (producer side):
  - decision.cycle  — one message per L1→L9 cycle (carries decision_id,
                      trace_id, symbol, final_action, consensus). Phase 4.5a
                      will enrich this into Experience.
  - trade.outcome   — one message per closed position (pnl_pct, duration,
                      exit_reason, regime). Phase 4.5a's meta-coordinator
                      consumes this.
  - model.signal.*  — Phase 4.5b learner cross-talk (reserved).

Naming follows the Orchestration Hardening Plan §6.1 schema exactly so the
meta-coordinator in Phase 4.5a can XREADGROUP without remapping.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

_REDIS_URL = os.getenv("ACT_REDIS_URL", "redis://localhost:6379/0")
_MAX_STREAM_LEN = int(os.getenv("ACT_STREAM_MAX_LEN", "100000"))

try:
    import redis as _redis
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False
    logger.info("redis-py not installed — streams disabled (no-op).")

_client: Optional[Any] = None
_client_lock = threading.Lock()
_last_connect_warn: float = 0.0


def get_client() -> Optional[Any]:
    """Return a cached Redis client, or None if unavailable.

    Uses a short-circuit warning throttle so a dead Redis doesn't flood
    logs. Reconnects transparently on next call if the cache was cleared.
    """
    global _client, _last_connect_warn
    if not _HAS_REDIS or not _REDIS_URL:
        return None
    with _client_lock:
        if _client is not None:
            return _client
        try:
            c = _redis.Redis.from_url(
                _REDIS_URL,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
                decode_responses=False,  # stream values are bytes → we JSON-decode ourselves
            )
            # Cheap liveness check — avoid silently returning a dead client.
            c.ping()
            _client = c
            logger.info("Redis client connected at %s", _REDIS_URL)
            return _client
        except Exception as e:
            now = time.time()
            if now - _last_connect_warn > 60:
                logger.warning("Redis connect failed at %s: %s", _REDIS_URL, e)
                _last_connect_warn = now
            return None


def _record_publish_metric(stream: str, ok: bool) -> None:
    try:
        from src.orchestration.metrics import record_stream_publish
        record_stream_publish(stream=stream, ok=ok)
    except Exception:
        pass


def publish(stream: str, payload: Dict[str, Any], maxlen: Optional[int] = None) -> Optional[str]:
    """Append one JSON-encoded message to `stream`. Returns the message id or None.

    Uses XADD MAXLEN ~ approximate trim — matches the plan's bounded-stream
    invariant (§6.1). Never raises: a dead Redis returns None, the caller
    treats this as a cache miss and moves on.
    """
    c = get_client()
    if c is None:
        _record_publish_metric(stream, ok=False)
        return None
    try:
        body = json.dumps(payload, default=str).encode("utf-8")
        mid = c.xadd(
            name=stream,
            fields={b"json": body},
            maxlen=maxlen if maxlen is not None else _MAX_STREAM_LEN,
            approximate=True,
        )
        _record_publish_metric(stream, ok=True)
        return mid.decode("utf-8") if isinstance(mid, (bytes, bytearray)) else str(mid)
    except Exception as e:
        _record_publish_metric(stream, ok=False)
        logger.debug("stream publish failed on %s: %s", stream, e)
        return None


def ensure_group(stream: str, group: str, start_id: str = "$") -> bool:
    """Idempotently create a consumer group. Returns True if the group exists now."""
    c = get_client()
    if c is None:
        return False
    try:
        c.xgroup_create(name=stream, groupname=group, id=start_id, mkstream=True)
        return True
    except Exception as e:
        # BUSYGROUP just means it's already there — that's success for us.
        msg = str(e).upper()
        if "BUSYGROUP" in msg:
            return True
        logger.debug("xgroup_create %s/%s failed: %s", stream, group, e)
        return False


def read_group(
    stream: str,
    group: str,
    consumer: str,
    count: int = 32,
    block_ms: int = 1000,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Consume messages as (message_id, decoded_json_payload). Generator.

    Yields nothing on timeout; caller should loop with a kill switch.
    Ack is the CALLER's responsibility — call `ack(stream, group, id)`
    after successful processing, otherwise the message stays pending
    and XPENDING surfaces it for another consumer.
    """
    c = get_client()
    if c is None:
        return
    try:
        resp = c.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: ">"},
            count=count,
            block=block_ms,
        )
    except Exception as e:
        logger.debug("xreadgroup %s/%s failed: %s", stream, group, e)
        return
    for _stream_name, messages in resp or []:
        for msg_id, fields in messages:
            mid = msg_id.decode("utf-8") if isinstance(msg_id, (bytes, bytearray)) else str(msg_id)
            try:
                raw = fields.get(b"json") or fields.get("json")
                payload = json.loads(raw) if raw else {}
            except Exception:
                payload = {"_raw": str(fields)}
            yield mid, payload


def ack(stream: str, group: str, message_id: str) -> bool:
    c = get_client()
    if c is None:
        return False
    try:
        c.xack(stream, group, message_id)
        return True
    except Exception:
        return False


def stream_len(stream: str) -> Optional[int]:
    c = get_client()
    if c is None:
        return None
    try:
        return int(c.xlen(stream))
    except Exception:
        return None


# Canonical stream names — import these instead of stringly-typed calls so
# Phase 4.5a's meta-coordinator and the executor agree on the schema.
STREAM_DECISION_CYCLE = "decision.cycle"
STREAM_TRADE_OUTCOME = "trade.outcome"
