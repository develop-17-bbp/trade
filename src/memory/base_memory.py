"""ACT v8.0 Memory System — Abstract base class using SQLite."""

import sqlite3
import json
import time
import logging
import os
import threading
from abc import ABC

logger = logging.getLogger(__name__)


class LayerMemory(ABC):
    """Persistent, thread-safe memory layer backed by SQLite."""

    def __init__(self, layer_id: str, db_dir: str = "memory/"):
        self.layer_id = layer_id
        self.db_dir = db_dir
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, f"{layer_id}.db")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("LayerMemory initialised: %s at %s", layer_id, self.db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _create_tables(self):
        with self._lock:
            c = self._conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       REAL    NOT NULL,
                    market_regime   TEXT,
                    signal_context  TEXT,
                    action_taken    TEXT,
                    outcome_pnl     REAL,
                    outcome_label   TEXT,
                    confidence_at_entry REAL,
                    layer_id        TEXT,
                    extra_data      TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS pattern_index (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_signature TEXT UNIQUE,
                    frequency       INTEGER DEFAULT 0,
                    avg_pnl         REAL    DEFAULT 0.0,
                    win_rate        REAL    DEFAULT 0.0,
                    best_regime     TEXT,
                    last_updated    REAL
                )
            """)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def record(self, event: dict):
        """Insert a new event row."""
        try:
            with self._lock:
                self._conn.execute(
                    """INSERT INTO events
                       (timestamp, market_regime, signal_context, action_taken,
                        outcome_pnl, outcome_label, confidence_at_entry, layer_id, extra_data)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event.get("timestamp", time.time()),
                        event.get("market_regime"),
                        json.dumps(event.get("signal_context", {})),
                        event.get("action_taken"),
                        event.get("outcome_pnl", 0.0),
                        event.get("outcome_label"),
                        event.get("confidence_at_entry", 0.0),
                        event.get("layer_id", self.layer_id),
                        json.dumps(event.get("extra_data", {})),
                    ),
                )
                self._conn.commit()
        except Exception:
            logger.exception("Failed to record event in %s", self.layer_id)

    def recall(self, query: dict, top_k: int = 10) -> list[dict]:
        """Query events by market_regime, layer_id, action_taken."""
        clauses, params = [], []
        for col in ("market_regime", "layer_id", "action_taken"):
            if col in query:
                clauses.append(f"{col} = ?")
                params.append(query[col])

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM events{where} ORDER BY timestamp DESC LIMIT ?"
        params.append(top_k)

        try:
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            logger.exception("Failed to recall from %s", self.layer_id)
            return []

    def consolidate(self):
        """Group events by (market_regime, action_taken), upsert patterns with 5+ occurrences."""
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT market_regime, action_taken,
                           COUNT(*)                             AS cnt,
                           AVG(outcome_pnl)                     AS avg_pnl,
                           SUM(CASE WHEN outcome_label='WIN' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS win_rate
                    FROM events
                    GROUP BY market_regime, action_taken
                    HAVING cnt >= 5
                """).fetchall()

                now = time.time()
                for r in rows:
                    sig = f"{r['market_regime']}|{r['action_taken']}"
                    self._conn.execute("""
                        INSERT INTO pattern_index (pattern_signature, frequency, avg_pnl, win_rate, best_regime, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(pattern_signature) DO UPDATE SET
                            frequency    = excluded.frequency,
                            avg_pnl      = excluded.avg_pnl,
                            win_rate     = excluded.win_rate,
                            best_regime  = excluded.best_regime,
                            last_updated = excluded.last_updated
                    """, (sig, r["cnt"], r["avg_pnl"], r["win_rate"], r["market_regime"], now))
                self._conn.commit()
            logger.info("Consolidation done for %s — %d patterns upserted", self.layer_id, len(rows))
        except Exception:
            logger.exception("Consolidation failed for %s", self.layer_id)

    def get_stats(self) -> dict:
        """Return summary statistics."""
        try:
            with self._lock:
                total = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                wins = self._conn.execute("SELECT COUNT(*) FROM events WHERE outcome_label='WIN'").fetchone()[0]
                avg_pnl_row = self._conn.execute("SELECT AVG(outcome_pnl) FROM events").fetchone()
                patterns = self._conn.execute("SELECT COUNT(*) FROM pattern_index").fetchone()[0]
            return {
                "total_events": total,
                "win_rate": (wins / total) if total > 0 else 0.0,
                "avg_pnl": avg_pnl_row[0] if avg_pnl_row[0] is not None else 0.0,
                "patterns_extracted": patterns,
            }
        except Exception:
            logger.exception("get_stats failed for %s", self.layer_id)
            return {"total_events": 0, "win_rate": 0.0, "avg_pnl": 0.0, "patterns_extracted": 0}

    def close(self):
        with self._lock:
            self._conn.close()
