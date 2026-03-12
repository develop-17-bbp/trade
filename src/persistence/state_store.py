import os
import sqlite3
import time
from typing import Dict, List, Optional


class StateStore:
    """SQLite-based state persistence for crash recovery.

    Persists open positions, P&L / equity snapshots, circuit breaker
    states, and system metadata so the trading system can resume
    cleanly after an unexpected restart.
    """

    def __init__(self, db_path: str = "data/trading_state.db") -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    # ------------------------------------------------------------------
    # Table creation
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        cursor = self._conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS open_positions (
                asset        TEXT PRIMARY KEY,
                direction    TEXT NOT NULL,
                entry_price  REAL NOT NULL,
                size         REAL NOT NULL,
                stop_loss    REAL,
                take_profit  REAL,
                order_id     TEXT,
                timestamp    REAL NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_state (
                id             INTEGER PRIMARY KEY CHECK (id = 1),
                daily_pnl      REAL NOT NULL DEFAULT 0,
                weekly_pnl     REAL NOT NULL DEFAULT 0,
                monthly_pnl    REAL NOT NULL DEFAULT 0,
                peak_equity    REAL NOT NULL DEFAULT 0,
                current_equity REAL NOT NULL DEFAULT 0,
                is_shutdown    INTEGER NOT NULL DEFAULT 0,
                updated_at     REAL NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS circuit_breakers (
                name           TEXT PRIMARY KEY,
                is_triggered   INTEGER NOT NULL DEFAULT 0,
                current_value  REAL NOT NULL DEFAULT 0,
                threshold      REAL NOT NULL DEFAULT 0,
                last_triggered REAL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_meta (
                id            INTEGER PRIMARY KEY CHECK (id = 1),
                mode          TEXT NOT NULL DEFAULT 'paper',
                last_run_time REAL NOT NULL,
                updated_at    REAL NOT NULL
            )
            """
        )

        self._conn.commit()

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def save_position(
        self,
        asset: str,
        direction: str,
        entry_price: float,
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_id: Optional[str] = None,
    ) -> None:
        """Insert or replace an open position for *asset*."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO open_positions
                (asset, direction, entry_price, size, stop_loss, take_profit, order_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (asset, direction, entry_price, size, stop_loss, take_profit, order_id, time.time()),
        )
        self._conn.commit()

    def remove_position(self, asset: str) -> None:
        """Delete the open position for *asset*."""
        self._conn.execute(
            "DELETE FROM open_positions WHERE asset = ?",
            (asset,),
        )
        self._conn.commit()

    def get_open_positions(self) -> List[Dict]:
        """Return all open positions as a list of dicts."""
        cursor = self._conn.execute("SELECT * FROM open_positions")
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Risk state
    # ------------------------------------------------------------------

    def save_risk_state(
        self,
        daily_pnl: float,
        weekly_pnl: float,
        monthly_pnl: float,
        peak_equity: float,
        current_equity: float,
        is_shutdown: bool,
    ) -> None:
        """Upsert the single risk-state row."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO risk_state
                (id, daily_pnl, weekly_pnl, monthly_pnl, peak_equity, current_equity, is_shutdown, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """,
            (daily_pnl, weekly_pnl, monthly_pnl, peak_equity, current_equity, int(is_shutdown), time.time()),
        )
        self._conn.commit()

    def get_risk_state(self) -> Dict:
        """Return the current risk state, or an empty dict if none saved."""
        cursor = self._conn.execute("SELECT * FROM risk_state WHERE id = 1")
        row = cursor.fetchone()
        if row is None:
            return {}
        result = dict(row)
        result["is_shutdown"] = bool(result["is_shutdown"])
        return result

    # ------------------------------------------------------------------
    # Circuit breakers
    # ------------------------------------------------------------------

    def save_circuit_breaker(
        self,
        name: str,
        is_triggered: bool,
        current_value: float,
        threshold: float,
        last_triggered: Optional[float] = None,
    ) -> None:
        """Upsert a circuit breaker by *name*."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO circuit_breakers
                (name, is_triggered, current_value, threshold, last_triggered)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, int(is_triggered), current_value, threshold, last_triggered),
        )
        self._conn.commit()

    def get_circuit_breakers(self) -> List[Dict]:
        """Return all circuit breakers as a list of dicts."""
        cursor = self._conn.execute("SELECT * FROM circuit_breakers")
        rows = []
        for row in cursor.fetchall():
            d = dict(row)
            d["is_triggered"] = bool(d["is_triggered"])
            rows.append(d)
        return rows

    # ------------------------------------------------------------------
    # System metadata
    # ------------------------------------------------------------------

    def save_system_meta(self, mode: str, last_run_time: float) -> None:
        """Upsert system metadata (mode, last run time)."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO system_meta
                (id, mode, last_run_time, updated_at)
            VALUES (1, ?, ?, ?)
            """,
            (mode, last_run_time, time.time()),
        )
        self._conn.commit()

    def get_system_meta(self) -> Dict:
        """Return system metadata, or an empty dict if none saved."""
        cursor = self._conn.execute("SELECT * FROM system_meta WHERE id = 1")
        row = cursor.fetchone()
        if row is None:
            return {}
        return dict(row)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
