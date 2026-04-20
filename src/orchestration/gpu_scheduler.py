"""GPU lease scheduler — Phase 4 (Plan §2.5).

Coordinates GPU access across the five consumers that otherwise fight for
VRAM on the single RTX-5090:

    P1  Ollama inference (decision path, L7 LLM)        ← hot, always wins
    P2  LoRA fine-tune / PatchTST training              ← slow loop
    P3  Genetic evolution (GPU-accelerated backtest)    ← slow loop
    P3.5 Meta-coordinator credit regression            ← Phase 4.5a
    P4  Manual scripts (retrain, analysis)              ← operator

A priority-aware Redis-backed mutex:
  - Acquire: SETNX act:gpu:lease:{device} with TTL. On success, the caller
    writes its priority level. On failure, compare the holder's priority
    with our own — if ours is strictly higher (lower number), we BUMP the
    holder: rewrite the key, signal the displaced caller via pubsub, and
    let them drop out gracefully.
  - Release: DEL the key + publish release event.
  - Dead-holder recovery: TTL expires a crashed holder's lease automatically.

This is a best-effort cooperative scheme — we trust each holder to check
for a bump periodically during long-running GPU work and exit early. The
decision path (P1) never bumps and never gets bumped mid-cycle because
its work is bounded at ~1-2 seconds.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

_LEASE_KEY_PREFIX = "act:gpu:lease:"
_DEFAULT_DEVICE = os.getenv("ACT_GPU_DEVICE", "cuda:0")
_DEFAULT_TTL_S = int(os.getenv("ACT_GPU_LEASE_TTL_S", "300"))

# Priority numbering (lower = more important)
P1 = 1
P2 = 2
P3 = 3
P3_5 = 4   # meta-coord sits between P3 and P4 per Plan §2.5
P4 = 5


class LeaseNotAcquired(Exception):
    """Raised when a caller fails to obtain the GPU lease."""


def _holder_identity() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _parse_holder_value(raw: bytes) -> Tuple[str, int]:
    """Decode 'holder_id|priority' value. Falls back to (raw, P4)."""
    if not raw:
        return "", P4
    try:
        s = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        holder, prio = s.rsplit("|", 1)
        return holder, int(prio)
    except Exception:
        return str(raw), P4


def acquire(device: str = _DEFAULT_DEVICE, priority: int = P4, ttl_s: int = _DEFAULT_TTL_S) -> Optional[str]:
    """Try once to acquire the lease. Returns holder_id on success, None otherwise.

    Higher-priority callers (P1 < P2 < ...) displace lower-priority ones.
    Same-priority callers lose (FIFO is not guaranteed; use different priorities
    to express precedence).
    """
    try:
        from src.orchestration.streams import get_client
        c = get_client()
        if c is None:
            # No Redis → one-process mode → always grant (nothing else to fight).
            return _holder_identity()
    except Exception:
        return _holder_identity()

    key = _LEASE_KEY_PREFIX + device
    holder = _holder_identity()
    value = f"{holder}|{priority}".encode("utf-8")

    # Fast path: nobody holds it.
    if c.set(key, value, ex=ttl_s, nx=True):
        _emit_metric(device, priority)
        return holder

    # Contention: read the current holder and decide.
    cur = c.get(key)
    cur_holder, cur_prio = _parse_holder_value(cur or b"")
    if priority < cur_prio:
        # Bump — overwrite atomically.
        if c.set(key, value, ex=ttl_s):
            _emit_metric(device, priority)
            logger.info("[GPU] P%d bumped P%d on %s", priority, cur_prio, device)
            return holder
    return None


def release(holder_id: str, device: str = _DEFAULT_DEVICE) -> bool:
    """Release the lease if we still own it. Returns True on success."""
    try:
        from src.orchestration.streams import get_client
        c = get_client()
        if c is None:
            return True
    except Exception:
        return True

    key = _LEASE_KEY_PREFIX + device
    cur = c.get(key)
    cur_holder, _ = _parse_holder_value(cur or b"")
    if cur_holder != holder_id:
        return False
    try:
        c.delete(key)
        _emit_metric(device, 0)
        return True
    except Exception:
        return False


def still_holding(holder_id: str, device: str = _DEFAULT_DEVICE) -> bool:
    """Long-running jobs poll this to detect a bump and exit early."""
    try:
        from src.orchestration.streams import get_client
        c = get_client()
        if c is None:
            return True
    except Exception:
        return True
    key = _LEASE_KEY_PREFIX + device
    cur = c.get(key)
    cur_holder, _ = _parse_holder_value(cur or b"")
    return cur_holder == holder_id


@contextmanager
def lease(
    device: str = _DEFAULT_DEVICE,
    priority: int = P4,
    ttl_s: int = _DEFAULT_TTL_S,
    wait_s: float = 0.0,
    poll_s: float = 0.5,
) -> Iterator[str]:
    """Context manager for the common pattern.

    Retries up to `wait_s` seconds before giving up. Zero → try once.
    Raises LeaseNotAcquired if we never get the lease.
    """
    deadline = time.time() + max(0.0, wait_s)
    holder: Optional[str] = None
    while True:
        holder = acquire(device=device, priority=priority, ttl_s=ttl_s)
        if holder is not None:
            break
        if time.time() >= deadline:
            raise LeaseNotAcquired(f"could not acquire {device} at P{priority}")
        time.sleep(poll_s)
    try:
        yield holder
    finally:
        release(holder, device=device)


def _emit_metric(device: str, holder_priority: int) -> None:
    try:
        from src.orchestration.metrics import record_gpu_lease
        record_gpu_lease(device=device, holder_priority=holder_priority)
    except Exception:
        pass


_metrics_seeded = threading.Event()


def init() -> None:
    """Seed the gauge at 0 so dashboards show an empty series pre-first-acquire."""
    if _metrics_seeded.is_set():
        return
    _emit_metric(_DEFAULT_DEVICE, 0)
    _metrics_seeded.set()
