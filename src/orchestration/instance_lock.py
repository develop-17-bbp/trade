"""Multi-instance lock — prevent accidental parallel ACT processes.

Reference: arXiv:2511.06448 — "When AI Agents Collude Online:
Financial Fraud Risks by Collaborative LLM Agents on Social
Platforms" (Nov 2025). They show that multiple agent instances
operating on the same data can amplify financial risk even without
intent to collude — through duplicate actions, racing trades, and
conflicting state writes.

ACT's risk: if an operator runs START_ALL.ps1 twice by accident (or
a cron job fires while an interactive session is live), two executor
processes would both write to `warm_store.sqlite` and both attempt to
submit orders. The second process thinks the first's state is its
own; you get double-trades, ghost positions, and corrupted audit log.

Defense: a filesystem lock on `data/act_instance.lock`. At startup
every trade-path process acquires it (non-blocking). If already held
by a living PID, the new process refuses to start with a clear error.

Pure-stdlib, cross-platform (file-exists + PID check rather than
fcntl.flock which doesn't exist on Windows).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


DEFAULT_LOCK_PATH = "data/act_instance.lock"
# If a lock is older than this AND the PID inside is dead, treat as
# orphaned and reclaim. One hour is long enough that a genuinely-paused
# process (debugger, GC stall) isn't mistakenly killed.
STALE_AFTER_S = 3600.0


class InstanceLockError(RuntimeError):
    """Raised when the instance lock can't be acquired."""


def _pid_is_alive(pid: int) -> bool:
    """Cross-platform PID liveness check."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            # Windows: OpenProcess with PROCESS_QUERY_LIMITED_INFORMATION
            # then GetExitCodeProcess. STILL_ACTIVE (259) means live.
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
            )
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return bool(ok) and exit_code.value == 259
        # POSIX: signal 0 probes without actually signaling
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return False


def _read_lock(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_lock(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def acquire(
    lock_path: Optional[str] = None,
    *,
    instance_id: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Acquire the instance lock. Raises InstanceLockError if another
    process holds it.

    `force=True` overwrites an existing lock regardless of liveness —
    intended for operator recovery ("I know my previous process
    crashed without cleaning up"). Use with care.

    Returns the lock payload that was written (includes `pid`,
    `instance_id`, `host`, `acquired_at`, `lock_path`).
    """
    path = Path(lock_path or os.getenv("ACT_INSTANCE_LOCK")
                or DEFAULT_LOCK_PATH)

    existing = _read_lock(path)
    if existing and not force:
        prev_pid = int(existing.get("pid") or 0)
        prev_ts = float(existing.get("acquired_at") or 0.0)
        alive = _pid_is_alive(prev_pid)
        stale = (time.time() - prev_ts) > STALE_AFTER_S
        if alive and not stale:
            raise InstanceLockError(
                f"ACT instance lock held by live pid={prev_pid} "
                f"(instance_id={existing.get('instance_id')!r}, "
                f"host={existing.get('host')!r}, "
                f"age={(time.time() - prev_ts):.0f}s). "
                "Refuse to start a second parallel instance. "
                "If this is wrong, delete "
                f"{path} or call acquire(force=True)."
            )
        if existing:
            logger.warning(
                "instance_lock: reclaiming stale lock (pid=%d alive=%s age=%.0fs)",
                prev_pid, alive, time.time() - prev_ts,
            )

    payload = {
        "pid": os.getpid(),
        "instance_id": (
            instance_id
            or os.getenv("ACT_INSTANCE_ID")
            or f"act-{socket.gethostname()}-{os.getpid()}"
        ),
        "host": socket.gethostname(),
        "acquired_at": time.time(),
        "lock_path": str(path),
    }
    _write_lock(path, payload)
    return payload


def release(lock_path: Optional[str] = None) -> bool:
    """Release (delete) the lock. Safe to call even if lock doesn't
    exist. Returns True if something was deleted."""
    path = Path(lock_path or os.getenv("ACT_INSTANCE_LOCK")
                or DEFAULT_LOCK_PATH)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        logger.warning("instance_lock: release failed: %s", e)
        return False


def status(lock_path: Optional[str] = None) -> Dict[str, Any]:
    """Read-only lock inspection. Returns {'held': bool, 'owner': ...}."""
    path = Path(lock_path or os.getenv("ACT_INSTANCE_LOCK")
                or DEFAULT_LOCK_PATH)
    if not path.exists():
        return {"held": False, "path": str(path)}
    payload = _read_lock(path) or {}
    prev_pid = int(payload.get("pid") or 0)
    alive = _pid_is_alive(prev_pid)
    age = time.time() - float(payload.get("acquired_at") or 0.0)
    return {
        "held": True,
        "path": str(path),
        "owner_pid": prev_pid,
        "owner_alive": alive,
        "owner_age_s": round(age, 1),
        "instance_id": payload.get("instance_id"),
        "host": payload.get("host"),
        "stale": age > STALE_AFTER_S,
    }
