"""Tests for the soak-window readiness gate.

Prove that:
  - A clean warm store + no operator flag closes the gate.
  - Setting the operator flag alone is not enough.
  - Satisfying every condition opens it.
  - `format_report` is stable enough to include in a banner.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest


@pytest.fixture()
def warm_db(monkeypatch):
    """Create an empty warm_store.sqlite in a temp dir + point the gate at it."""
    tmp = tempfile.mkdtemp()
    db = Path(tmp) / "warm.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE outcomes (
            outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
            exit_ts REAL
        );
        CREATE TABLE decisions (
            decision_id TEXT PRIMARY KEY,
            ts_ns INTEGER,
            authority_violations TEXT
        );
        """
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    return db


def _seed(db: Path, n_trades: int, age_days: float, n_violations: int = 0,
          n_decisions: int = 100) -> None:
    now = time.time()
    earliest = now - age_days * 86400
    conn = sqlite3.connect(str(db))
    conn.executemany(
        "INSERT INTO outcomes (exit_ts) VALUES (?)",
        [(earliest + i * 60,) for i in range(n_trades)],
    )
    for i in range(n_decisions):
        vjson = '["rule"]' if i < n_violations else "[]"
        conn.execute(
            "INSERT INTO decisions (decision_id, ts_ns, authority_violations) VALUES (?, ?, ?)",
            (f"d{i}", int(now * 1e9) + i, vjson),
        )
    conn.commit()
    conn.close()


def test_empty_store_closes_gate(warm_db, monkeypatch):
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    from src.orchestration.readiness_gate import evaluate
    state = evaluate()
    assert state.open_ is False
    assert any("trades" in r for r in state.reasons)


def test_operator_flag_alone_is_not_enough(warm_db, monkeypatch):
    monkeypatch.setenv("ACT_REAL_CAPITAL_ENABLED", "1")
    from src.orchestration.readiness_gate import evaluate
    state = evaluate()
    assert state.open_ is False
    # No trades yet → trades-count reason must fire even with operator flag on.
    assert any("trades" in r for r in state.reasons)


def test_all_conditions_met_opens_gate(warm_db, monkeypatch):
    monkeypatch.setenv("ACT_REAL_CAPITAL_ENABLED", "1")
    monkeypatch.setenv("ACT_GATE_MIN_CREDIT_R2", "0.0")  # disable R² check for this test
    _seed(warm_db, n_trades=600, age_days=15.0, n_violations=1, n_decisions=200)
    from src.orchestration.readiness_gate import evaluate
    state = evaluate()
    assert state.open_ is True, f"expected open, reasons={state.reasons}"
    assert state.details["trades"] == 600
    assert state.details["soak_days"] >= 14.0


def test_violation_rate_closes_gate(warm_db, monkeypatch):
    monkeypatch.setenv("ACT_REAL_CAPITAL_ENABLED", "1")
    monkeypatch.setenv("ACT_GATE_MIN_CREDIT_R2", "0.0")
    _seed(warm_db, n_trades=600, age_days=15.0, n_violations=20, n_decisions=200)
    # 20/200 = 10% violation rate > 2% ceiling
    from src.orchestration.readiness_gate import evaluate
    state = evaluate()
    assert state.open_ is False
    assert any("violation" in r for r in state.reasons)


def test_format_report_is_stable(warm_db, monkeypatch):
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    from src.orchestration.readiness_gate import evaluate, format_report
    report = format_report(evaluate())
    assert "Readiness gate:" in report
    assert "CLOSED" in report
    assert "Numbers:" in report
