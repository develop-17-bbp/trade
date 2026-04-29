"""SQLite WAL warm tier — Phase 3 (Plan §1.3).

The durable middle tier between Redis hot (wiped on restart) and parquet
cold (monthly archive). Holds every decision + trade outcome since install,
queryable by (symbol, regime, authority_rule). SQLite WAL mode so concurrent
reads (dashboard) don't block the writer (executor).

Why SQLite: single-file, zero-ops on a Windows rig, WAL gives us the
concurrency we need, and every analyst machine already has the CLI. The
Plan §1.3 explicitly names SQLite — don't substitute without updating
the Phase 4.5a consumer.

Threading model:
  - One writer per process, guarded by a lock. Safe to call from multiple
    threads; not safe across processes (SQLite WAL handles it but you'd
    lose batch ordering).
  - Reads are lock-free (WAL gives readers a consistent snapshot).
  - Batched flush every N writes or T seconds, whichever comes first,
    so a hot decision loop doesn't pay an fsync per cycle.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_DB = os.getenv(
    "ACT_WARM_DB_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "warm_store.sqlite"),
)
_FLUSH_EVERY_N = int(os.getenv("ACT_WARM_FLUSH_N", "16"))
_FLUSH_EVERY_S = float(os.getenv("ACT_WARM_FLUSH_S", "2.0"))


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS decisions (
        decision_id TEXT PRIMARY KEY,
        trace_id    TEXT,
        symbol      TEXT,
        ts_ns       INTEGER,
        direction   INTEGER,
        confidence  REAL,
        consensus   TEXT,
        veto        INTEGER,
        raw_signal  INTEGER,
        final_action TEXT,
        authority_violations TEXT,
        payload_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS outcomes (
        outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id TEXT,
        symbol      TEXT,
        direction   TEXT,
        entry_price REAL,
        exit_price  REAL,
        pnl_pct     REAL,
        pnl_usd     REAL,
        duration_s  REAL,
        exit_reason TEXT,
        regime      TEXT,
        entry_ts    REAL,
        exit_ts     REAL,
        payload_json TEXT,
        FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_decisions_symbol_ts ON decisions(symbol, ts_ns)",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_symbol_ts ON outcomes(symbol, exit_ts)",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_regime ON outcomes(regime)",
]


# Columns added after the initial schema shipped. Each is backward-safe:
# the ALTER runs once on startup, tolerates "duplicate column" on reruns.
# component_signals → which component suggested each leg of the decision
#                     (multi_strategy / lgbm / lora / llm_agentic / rl)
# plan_json         → the compiled TradePlan (M1) for post-hoc audit
# self_critique     → second-LLM verification written at trade close (Phase 5a)
# asset_class       → CRYPTO / STOCK / POLYMARKET — partitions warm_store
#                     so finetune corpus filter + readiness gate run per class
#                     (dual-asset extension, Phase B.2)
# venue             → 'robinhood' / 'alpaca' / 'polymarket' — concrete venue,
#                     finer-grained than asset_class for cost-modelling.
_MIGRATIONS = [
    "ALTER TABLE decisions ADD COLUMN component_signals TEXT DEFAULT '{}'",
    "ALTER TABLE decisions ADD COLUMN plan_json TEXT DEFAULT '{}'",
    "ALTER TABLE decisions ADD COLUMN self_critique TEXT DEFAULT '{}'",
    "ALTER TABLE decisions ADD COLUMN asset_class TEXT DEFAULT 'CRYPTO'",
    "ALTER TABLE decisions ADD COLUMN venue       TEXT DEFAULT 'robinhood'",
    "ALTER TABLE outcomes  ADD COLUMN asset_class TEXT DEFAULT 'CRYPTO'",
    "ALTER TABLE outcomes  ADD COLUMN venue       TEXT DEFAULT 'robinhood'",
    "CREATE INDEX IF NOT EXISTS idx_decisions_asset_class ON decisions(asset_class)",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_asset_class  ON outcomes(asset_class)",
]


class WarmStore:
    """SQLite-backed durable decision + outcome log.

    Not a singleton — callers share one via get_store(), but construction
    is cheap so tests can spin disposables on temp files.
    """

    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._pending_decisions: List[Tuple] = []
        self._pending_outcomes: List[Tuple] = []
        self._last_flush_ts = time.time()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._get_conn()
            for stmt in _SCHEMA:
                conn.execute(stmt)
            for stmt in _MIGRATIONS:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    # "duplicate column name" is expected after first run.
                    if "duplicate column" not in str(e).lower():
                        raise
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._conn = conn
        return self._conn

    # ── Public write API ───────────────────────────────────────────────

    def write_decision(self, decision: Dict[str, Any]) -> None:
        """Append a decision row. Buffered — flushes on batch size or timeout."""
        # Default asset_class/venue to 'CRYPTO'/'robinhood' for back-compat
        # with all callsites that haven't been updated to pass it explicitly.
        # Stocks executor passes 'STOCK'/'alpaca'; polymarket passes
        # 'POLYMARKET'/'polymarket'. Phase B.2 of the dual-asset rollout.
        row = (
            decision.get("decision_id"),
            decision.get("trace_id"),
            decision.get("symbol"),
            int(decision.get("ts_ns", time.time_ns())),
            int(decision.get("direction", 0) or 0),
            float(decision.get("confidence", 0.0) or 0.0),
            str(decision.get("consensus", "UNKNOWN")),
            int(bool(decision.get("veto", False))),
            int(decision.get("raw_signal", 0) or 0),
            str(decision.get("final_action", "FLAT")),
            json.dumps(decision.get("authority_violations", []) or []),
            json.dumps(decision, default=str),
            json.dumps(decision.get("component_signals") or {}, default=str),
            json.dumps(decision.get("plan") or {}, default=str),
            json.dumps(decision.get("self_critique") or {}, default=str),
            str(decision.get("asset_class") or "CRYPTO"),
            str(decision.get("venue")       or "robinhood"),
        )
        with self._lock:
            self._pending_decisions.append(row)
            self._maybe_flush_locked()

    def update_self_critique(self, decision_id: str, critique: Dict[str, Any]) -> None:
        """Write the post-close verification payload onto an existing decision row.

        Called by trade_verifier after a trade closes. No-op if the decision
        row hasn't been flushed yet — caller can retry or flush first.
        """
        with self._lock:
            self._flush_locked()
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE decisions SET self_critique=? WHERE decision_id=?",
                (json.dumps(critique, default=str), decision_id),
            )
            conn.commit()
        except Exception as e:
            logger.debug("warm_store update_self_critique failed for %s: %s", decision_id, e)

    def write_outcome(self, outcome: Dict[str, Any]) -> None:
        row = (
            outcome.get("decision_id"),
            outcome.get("symbol") or outcome.get("asset"),
            str(outcome.get("direction", "")),
            float(outcome.get("entry_price", 0.0) or 0.0),
            float(outcome.get("exit_price", 0.0) or 0.0),
            float(outcome.get("pnl_pct", 0.0) or 0.0),
            float(outcome.get("pnl_usd", 0.0) or 0.0),
            float(outcome.get("duration_s", 0.0) or 0.0),
            str(outcome.get("exit_reason", "MANUAL")),
            str(outcome.get("regime", "unknown")),
            float(outcome.get("entry_ts", 0.0) or 0.0),
            float(outcome.get("exit_ts", time.time()) or 0.0),
            json.dumps(outcome, default=str),
            str(outcome.get("asset_class") or "CRYPTO"),
            str(outcome.get("venue")       or "robinhood"),
        )
        with self._lock:
            self._pending_outcomes.append(row)
            self._maybe_flush_locked()

    def flush(self) -> None:
        """Force-drain both buffers. Safe to call from shutdown handlers."""
        with self._lock:
            self._flush_locked()

    # ── Reads (always flush first so reads see pending writes) ─────────

    def recent_outcomes(self, symbol: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        with self._lock:
            self._flush_locked()
        conn = self._get_conn()
        cur = conn.cursor()
        if symbol:
            cur.execute(
                "SELECT payload_json FROM outcomes WHERE symbol=? ORDER BY exit_ts DESC LIMIT ?",
                (symbol, limit),
            )
        else:
            cur.execute("SELECT payload_json FROM outcomes ORDER BY exit_ts DESC LIMIT ?", (limit,))
        return [json.loads(r[0]) for r in cur.fetchall()]

    def count(self, table: str) -> int:
        """Row count — used by tests + Phase 3 soak gates."""
        if table not in ("decisions", "outcomes"):
            raise ValueError(f"unknown table {table!r}")
        with self._lock:
            self._flush_locked()
        cur = self._get_conn().execute(f"SELECT COUNT(*) FROM {table}")
        return int(cur.fetchone()[0])

    def close(self) -> None:
        self.flush()
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # ── Internals ──────────────────────────────────────────────────────

    def _maybe_flush_locked(self) -> None:
        n = len(self._pending_decisions) + len(self._pending_outcomes)
        age = time.time() - self._last_flush_ts
        if n >= _FLUSH_EVERY_N or (n > 0 and age >= _FLUSH_EVERY_S):
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._pending_decisions and not self._pending_outcomes:
            return
        conn = self._get_conn()
        try:
            if self._pending_decisions:
                conn.executemany(
                    """INSERT OR REPLACE INTO decisions
                       (decision_id, trace_id, symbol, ts_ns, direction, confidence, consensus,
                        veto, raw_signal, final_action, authority_violations, payload_json,
                        component_signals, plan_json, self_critique, asset_class, venue)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    self._pending_decisions,
                )
            if self._pending_outcomes:
                conn.executemany(
                    """INSERT INTO outcomes
                       (decision_id, symbol, direction, entry_price, exit_price, pnl_pct, pnl_usd,
                        duration_s, exit_reason, regime, entry_ts, exit_ts, payload_json,
                        asset_class, venue)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    self._pending_outcomes,
                )
            conn.commit()
            self._pending_decisions.clear()
            self._pending_outcomes.clear()
            self._last_flush_ts = time.time()
        except Exception as e:
            logger.warning("warm_store flush failed: %s", e)


_store_singleton: Optional[WarmStore] = None
_store_lock = threading.Lock()


def get_store() -> WarmStore:
    """Process-wide WarmStore singleton. Creates the file on first call."""
    global _store_singleton
    with _store_lock:
        if _store_singleton is None:
            _store_singleton = WarmStore()
        return _store_singleton
