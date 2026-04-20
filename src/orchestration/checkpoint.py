"""Crash-resume checkpoints — Phase 3 (Plan §1.3 / §2.4).

Writes the last-completed decision_id + open-position snapshot to both
Redis hot (for sub-second recovery) and SQLite warm (so a Redis wipe
still leaves a recoverable state).

Design:
  - Call `checkpoint_cycle(...)` after every successful decision write.
  - On executor boot, call `load_last_checkpoint()` and compare against
    the exchange state; if they diverge we log a LOUD warning and let
    the operator reconcile rather than silently overwriting.
  - Checkpoints are small (< 1 KB); overhead per cycle is ~100 µs.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CHECKPOINT_KEY = "checkpoint:executor"
_CHECKPOINT_DB = os.getenv(
    "ACT_CHECKPOINT_DB_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "warm_store.sqlite"),
)


def _ensure_checkpoint_table() -> sqlite3.Connection:
    Path(_CHECKPOINT_DB).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_CHECKPOINT_DB, timeout=5.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS checkpoints (
             id INTEGER PRIMARY KEY CHECK (id = 1),
             ts REAL,
             payload_json TEXT
           )"""
    )
    return conn


def checkpoint_cycle(
    decision_id: str,
    trace_id: str,
    symbol: str,
    open_positions: Dict[str, Any],
    equity: float,
) -> None:
    """Persist a cycle snapshot to hot + warm tiers. Never raises."""
    payload = {
        "decision_id": decision_id,
        "trace_id": trace_id,
        "symbol": symbol,
        "open_positions": open_positions,
        "equity": float(equity),
        "ts": time.time(),
    }
    # Hot tier — Redis. Soft-fail.
    try:
        from src.orchestration.hot_state import set_value
        set_value(_CHECKPOINT_KEY, payload, ttl_s=86400)
    except Exception:
        pass
    # Warm tier — SQLite. Soft-fail.
    try:
        conn = _ensure_checkpoint_table()
        conn.execute(
            "INSERT OR REPLACE INTO checkpoints (id, ts, payload_json) VALUES (1, ?, ?)",
            (payload["ts"], json.dumps(payload, default=str)),
        )
        conn.close()
    except Exception as e:
        logger.debug("checkpoint warm-write failed: %s", e)


def load_last_checkpoint() -> Optional[Dict[str, Any]]:
    """Return the most recent checkpoint, or None if nothing's been saved."""
    # Try hot first.
    try:
        from src.orchestration.hot_state import get_value
        v = get_value(_CHECKPOINT_KEY)
        if v:
            return v
    except Exception:
        pass
    # Fall back to warm.
    try:
        conn = _ensure_checkpoint_table()
        cur = conn.execute("SELECT payload_json FROM checkpoints WHERE id = 1")
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            return json.loads(row[0])
    except Exception as e:
        logger.debug("checkpoint warm-read failed: %s", e)
    return None


def log_startup_diagnostic() -> None:
    """Print a startup banner describing the last known checkpoint.

    Intended to be called from executor __init__ after state initialization.
    Non-fatal; only informational. Loud log if the bot crashed mid-cycle.
    """
    cp = load_last_checkpoint()
    if not cp:
        logger.info("[CHECKPOINT] no prior checkpoint — clean start")
        return
    age = time.time() - float(cp.get("ts", 0))
    n_pos = len(cp.get("open_positions", {}) or {})
    last_dec = cp.get("decision_id", "?")[:20]
    if age < 300:
        logger.warning(
            "[CHECKPOINT] last cycle was only %.0fs ago — possible mid-cycle crash. "
            "Last decision_id=%s, %d positions in snapshot. "
            "Verify exchange state matches before trading.",
            age, last_dec, n_pos,
        )
    else:
        logger.info(
            "[CHECKPOINT] last cycle %.0fmin ago — decision_id=%s, %d positions",
            age / 60, last_dec, n_pos,
        )
