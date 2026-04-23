"""
Versioned strategy repository — persistent store for evolved strategies.

What it solves:
  genetic_strategy_engine.py produces new `StrategyDNA` every 6h. Today the
  best replaces the incumbent in-memory; older-but-good strategies are lost
  when the process restarts. This module gives every strategy a stable ID,
  persists its DNA + backtest summary, tracks live performance, and exposes
  a search/promote/demote/quarantine FSM so the autonomous loop (C4's
  Bayesian bandit) has something to sample from.

Design constraints:
  * Same SQLite WAL pattern as src/orchestration/warm_store.py — one file,
    process-wide singleton via get_repo(), cheap to instantiate for tests.
  * DNA is stored as JSON blob (StrategyDNA has no pickled-object fields).
  * Status FSM: candidate -> challenger -> champion, with -> quarantine
    branch on safety breach. No direct candidate -> champion jump; a
    strategy must survive A/B as challenger first.
  * Per-strategy aggregates are UPDATE'd, not recomputed from outcomes,
    so the bandit gets cheap O(1) reads.

Non-goals:
  * Backtesting — handled by src/backtesting/*. This module only records
    the summary.
  * Live allocation — that's autonomous_loop's Thompson sampler (C4). This
    module exposes `search(status, regime, min_sharpe)` for the sampler.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_DEFAULT_DB = os.getenv(
    "ACT_STRATEGY_REPO_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "strategy_repo.sqlite"),
)


VALID_STATUSES = ("candidate", "challenger", "champion", "quarantine", "retired")


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS strategies (
        strategy_id       TEXT PRIMARY KEY,
        name              TEXT,
        dna_json          TEXT NOT NULL,
        regime_tag        TEXT DEFAULT 'any',
        status            TEXT NOT NULL DEFAULT 'candidate',
        parent_ids        TEXT DEFAULT '[]',
        birth_ts          REAL NOT NULL,
        promoted_ts       REAL,
        backtest_summary  TEXT DEFAULT '{}',
        live_trades       INTEGER NOT NULL DEFAULT 0,
        live_wins         INTEGER NOT NULL DEFAULT 0,
        live_losses       INTEGER NOT NULL DEFAULT 0,
        live_pnl_pct_sum  REAL NOT NULL DEFAULT 0.0,
        live_pnl_sq_sum   REAL NOT NULL DEFAULT 0.0,
        live_sharpe       REAL NOT NULL DEFAULT 0.0,
        live_wr           REAL NOT NULL DEFAULT 0.0,
        last_outcome_ts   REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_strategies_status ON strategies(status)",
    "CREATE INDEX IF NOT EXISTS idx_strategies_regime ON strategies(regime_tag)",
    """
    CREATE TABLE IF NOT EXISTS ab_tests (
        ab_id            TEXT PRIMARY KEY,
        challenger_id    TEXT NOT NULL,
        champion_id      TEXT NOT NULL,
        started_ts       REAL NOT NULL,
        finished_ts      REAL,
        outcome          TEXT,   -- 'challenger_promoted' | 'champion_kept' | 'aborted'
        summary_json     TEXT DEFAULT '{}'
    )
    """,
]


@dataclass
class StrategyRecord:
    """One row of the strategies table, decoded."""
    strategy_id: str
    name: str
    dna: Dict[str, Any]
    regime_tag: str
    status: str
    parent_ids: List[str]
    birth_ts: float
    promoted_ts: Optional[float]
    backtest_summary: Dict[str, Any]
    live_trades: int
    live_wins: int
    live_losses: int
    live_pnl_pct_sum: float
    live_pnl_sq_sum: float
    live_sharpe: float
    live_wr: float
    last_outcome_ts: Optional[float]

    @classmethod
    def from_row(cls, row: tuple) -> "StrategyRecord":
        (
            sid, name, dna_json, regime, status, parent_ids_json, birth_ts, promoted_ts,
            bt_json, n, wins, losses, pnl_sum, pnl_sq, sharpe, wr, last_ts,
        ) = row
        return cls(
            strategy_id=sid, name=name or "",
            dna=json.loads(dna_json or "{}"),
            regime_tag=regime or "any",
            status=status or "candidate",
            parent_ids=json.loads(parent_ids_json or "[]"),
            birth_ts=float(birth_ts or 0.0),
            promoted_ts=(float(promoted_ts) if promoted_ts is not None else None),
            backtest_summary=json.loads(bt_json or "{}"),
            live_trades=int(n or 0), live_wins=int(wins or 0), live_losses=int(losses or 0),
            live_pnl_pct_sum=float(pnl_sum or 0.0), live_pnl_sq_sum=float(pnl_sq or 0.0),
            live_sharpe=float(sharpe or 0.0), live_wr=float(wr or 0.0),
            last_outcome_ts=(float(last_ts) if last_ts is not None else None),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "strategy_id": self.strategy_id, "name": self.name, "dna": dict(self.dna),
            "regime_tag": self.regime_tag, "status": self.status,
            "parent_ids": list(self.parent_ids), "birth_ts": self.birth_ts,
            "promoted_ts": self.promoted_ts, "backtest_summary": dict(self.backtest_summary),
            "live_trades": self.live_trades, "live_wins": self.live_wins,
            "live_losses": self.live_losses, "live_pnl_pct_sum": round(self.live_pnl_pct_sum, 4),
            "live_pnl_sq_sum": round(self.live_pnl_sq_sum, 4),
            "live_sharpe": round(self.live_sharpe, 3), "live_wr": round(self.live_wr, 3),
            "last_outcome_ts": self.last_outcome_ts,
        }
        return d


class StrategyRepository:
    """SQLite-backed versioned store. Thread-safe, not process-safe."""

    _COLUMN_LIST = (
        "strategy_id, name, dna_json, regime_tag, status, parent_ids, "
        "birth_ts, promoted_ts, backtest_summary, live_trades, live_wins, "
        "live_losses, live_pnl_pct_sum, live_pnl_sq_sum, live_sharpe, "
        "live_wr, last_outcome_ts"
    )

    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn
        return self._conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._get_conn()
            for stmt in _SCHEMA:
                conn.execute(stmt)
            conn.commit()

    # ── Registration ────────────────────────────────────────────────────

    def register(
        self,
        dna: Dict[str, Any],
        *,
        name: str = "",
        regime_tag: str = "any",
        parent_ids: Optional[List[str]] = None,
        backtest_summary: Optional[Dict[str, Any]] = None,
        strategy_id: Optional[str] = None,
    ) -> str:
        """Create a new 'candidate' strategy row. Returns its id."""
        sid = strategy_id or uuid.uuid4().hex
        row = (
            sid, name or sid[:8],
            json.dumps(dna or {}, default=str),
            regime_tag, "candidate",
            json.dumps(parent_ids or []),
            time.time(), None,
            json.dumps(backtest_summary or {}, default=str),
            0, 0, 0, 0.0, 0.0, 0.0, 0.0, None,
        )
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                f"INSERT INTO strategies ({self._COLUMN_LIST}) "
                f"VALUES ({','.join('?' * 17)})", row,
            )
            conn.commit()
        return sid

    # ── Status FSM ──────────────────────────────────────────────────────

    def set_status(self, strategy_id: str, status: str) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status {status!r}; allowed: {VALID_STATUSES}")
        with self._lock:
            conn = self._get_conn()
            params: tuple
            if status == "champion":
                conn.execute(
                    "UPDATE strategies SET status=?, promoted_ts=? WHERE strategy_id=?",
                    (status, time.time(), strategy_id),
                )
            else:
                conn.execute(
                    "UPDATE strategies SET status=? WHERE strategy_id=?",
                    (status, strategy_id),
                )
            conn.commit()

    def promote(self, strategy_id: str) -> None:
        """candidate / challenger -> champion (only one champion per regime)."""
        rec = self.get(strategy_id)
        if rec is None:
            raise KeyError(strategy_id)
        with self._lock:
            conn = self._get_conn()
            # Demote the previous champion in the same regime, if any.
            conn.execute(
                "UPDATE strategies SET status='retired' "
                "WHERE status='champion' AND regime_tag=? AND strategy_id!=?",
                (rec.regime_tag, strategy_id),
            )
            conn.execute(
                "UPDATE strategies SET status='champion', promoted_ts=? "
                "WHERE strategy_id=?",
                (time.time(), strategy_id),
            )
            conn.commit()

    def quarantine(self, strategy_id: str, reason: str = "") -> None:
        self.set_status(strategy_id, "quarantine")
        if reason:
            # Record the reason in backtest_summary for audit.
            rec = self.get(strategy_id)
            if rec is not None:
                summary = dict(rec.backtest_summary)
                summary["quarantine_reason"] = reason
                summary["quarantine_ts"] = time.time()
                with self._lock:
                    conn = self._get_conn()
                    conn.execute(
                        "UPDATE strategies SET backtest_summary=? WHERE strategy_id=?",
                        (json.dumps(summary, default=str), strategy_id),
                    )
                    conn.commit()

    # ── Outcome update (called by credit_assigner / autonomous_loop) ────

    def record_outcome(self, strategy_id: str, pnl_pct: float) -> None:
        """Increment live counters and recompute live_sharpe / live_wr."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT live_trades, live_wins, live_losses, live_pnl_pct_sum, live_pnl_sq_sum "
                "FROM strategies WHERE strategy_id=?",
                (strategy_id,),
            ).fetchone()
            if row is None:
                return
            n, wins, losses, pnl_sum, pnl_sq = row
            n += 1
            if pnl_pct > 0:
                wins += 1
            else:
                losses += 1
            pnl_sum += pnl_pct
            pnl_sq += pnl_pct * pnl_pct
            wr = wins / n if n else 0.0
            mean = pnl_sum / n if n else 0.0
            var = max(0.0, pnl_sq / n - mean * mean)
            std = var ** 0.5
            # Per-trade Sharpe — scale in consumer with sqrt(trades_per_year).
            sharpe = (mean / std) if std > 1e-9 else 0.0
            conn.execute(
                "UPDATE strategies SET live_trades=?, live_wins=?, live_losses=?, "
                "live_pnl_pct_sum=?, live_pnl_sq_sum=?, live_sharpe=?, live_wr=?, last_outcome_ts=? "
                "WHERE strategy_id=?",
                (n, wins, losses, pnl_sum, pnl_sq, sharpe, wr, time.time(), strategy_id),
            )
            conn.commit()

    # ── Reads ───────────────────────────────────────────────────────────

    def get(self, strategy_id: str) -> Optional[StrategyRecord]:
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT {self._COLUMN_LIST} FROM strategies WHERE strategy_id=?",
            (strategy_id,),
        ).fetchone()
        return StrategyRecord.from_row(row) if row else None

    def search(
        self,
        status: Optional[str] = None,
        regime: Optional[str] = None,
        min_sharpe: float = -1e9,
        limit: int = 50,
    ) -> List[StrategyRecord]:
        """List strategies matching filters, newest first."""
        q = f"SELECT {self._COLUMN_LIST} FROM strategies WHERE 1=1"
        params: List[Any] = []
        if status:
            q += " AND status=?"
            params.append(status)
        if regime:
            q += " AND (regime_tag=? OR regime_tag='any')"
            params.append(regime)
        q += " AND live_sharpe >= ?"
        params.append(float(min_sharpe))
        q += " ORDER BY birth_ts DESC LIMIT ?"
        params.append(int(limit))
        conn = self._get_conn()
        rows = conn.execute(q, tuple(params)).fetchall()
        return [StrategyRecord.from_row(r) for r in rows]

    def current_champion(self, regime: str = "any") -> Optional[StrategyRecord]:
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT {self._COLUMN_LIST} FROM strategies "
            f"WHERE status='champion' AND (regime_tag=? OR regime_tag='any') "
            f"ORDER BY promoted_ts DESC LIMIT 1",
            (regime,),
        ).fetchone()
        return StrategyRecord.from_row(row) if row else None

    def count_by_status(self) -> Dict[str, int]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM strategies GROUP BY status"
        ).fetchall()
        return {status: int(n) for status, n in rows}

    # ── A/B tests ───────────────────────────────────────────────────────

    def start_ab_test(self, challenger_id: str, champion_id: str) -> str:
        ab_id = uuid.uuid4().hex
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO ab_tests (ab_id, challenger_id, champion_id, started_ts) "
                "VALUES (?,?,?,?)",
                (ab_id, challenger_id, champion_id, time.time()),
            )
            conn.commit()
        return ab_id

    def finish_ab_test(self, ab_id: str, outcome: str, summary: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE ab_tests SET finished_ts=?, outcome=?, summary_json=? WHERE ab_id=?",
                (time.time(), outcome, json.dumps(summary or {}, default=str), ab_id),
            )
            conn.commit()

    def active_ab_tests(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT ab_id, challenger_id, champion_id, started_ts "
            "FROM ab_tests WHERE finished_ts IS NULL"
        ).fetchall()
        return [
            {"ab_id": r[0], "challenger_id": r[1], "champion_id": r[2], "started_ts": r[3]}
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None


_repo_singleton: Optional[StrategyRepository] = None
_repo_lock = threading.Lock()


def get_repo() -> StrategyRepository:
    """Process-wide StrategyRepository singleton."""
    global _repo_singleton
    with _repo_lock:
        if _repo_singleton is None:
            _repo_singleton = StrategyRepository()
        return _repo_singleton
