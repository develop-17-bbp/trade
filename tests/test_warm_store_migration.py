"""Tests for warm_store schema migration — component_signals / plan_json / self_critique."""
from __future__ import annotations

import json
import sqlite3
import time

import pytest

from src.orchestration.warm_store import WarmStore


def _decisions_columns(path: str) -> set:
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute("PRAGMA table_info(decisions)")
        return {row[1] for row in cur.fetchall()}
    finally:
        conn.close()


def test_new_columns_present_on_fresh_db(tmp_path):
    db = tmp_path / "fresh.sqlite"
    store = WarmStore(str(db))
    store.close()
    cols = _decisions_columns(str(db))
    assert {"component_signals", "plan_json", "self_critique"} <= cols


def test_migration_is_idempotent(tmp_path):
    db = tmp_path / "rerun.sqlite"
    WarmStore(str(db)).close()
    # Second open must not raise "duplicate column".
    WarmStore(str(db)).close()
    cols = _decisions_columns(str(db))
    assert {"component_signals", "plan_json", "self_critique"} <= cols


def test_migration_on_pre_existing_legacy_db(tmp_path):
    """Simulate an old DB missing the new columns — migration must add them."""
    db = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE decisions (
            decision_id TEXT PRIMARY KEY,
            trace_id TEXT, symbol TEXT, ts_ns INTEGER, direction INTEGER,
            confidence REAL, consensus TEXT, veto INTEGER, raw_signal INTEGER,
            final_action TEXT, authority_violations TEXT, payload_json TEXT
        )"""
    )
    conn.commit()
    conn.close()
    assert "component_signals" not in _decisions_columns(str(db))

    WarmStore(str(db)).close()
    cols = _decisions_columns(str(db))
    assert {"component_signals", "plan_json", "self_critique"} <= cols


def test_write_decision_with_new_fields(tmp_path):
    db = tmp_path / "write.sqlite"
    store = WarmStore(str(db))
    store.write_decision({
        "decision_id": "abc-1",
        "symbol": "BTC",
        "ts_ns": time.time_ns(),
        "direction": 1,
        "confidence": 0.72,
        "final_action": "LONG",
        "component_signals": {"top_contributor": "multi_strategy", "lgbm_conf": 0.6},
        "plan": {"plan_id": "p-1", "entry_price": 60000.0, "direction": "LONG"},
    })
    store.flush()

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT component_signals, plan_json, self_critique FROM decisions WHERE decision_id='abc-1'"
    ).fetchone()
    conn.close()
    assert row is not None
    cs = json.loads(row[0])
    pj = json.loads(row[1])
    sc = json.loads(row[2])
    assert cs["top_contributor"] == "multi_strategy"
    assert pj["entry_price"] == 60000.0
    assert sc == {}


def test_write_decision_without_new_fields_is_backward_compat(tmp_path):
    db = tmp_path / "compat.sqlite"
    store = WarmStore(str(db))
    store.write_decision({
        "decision_id": "abc-2",
        "symbol": "ETH",
        "direction": -1,
        "confidence": 0.4,
        "final_action": "FLAT",
    })
    store.flush()

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT component_signals, plan_json, self_critique FROM decisions WHERE decision_id='abc-2'"
    ).fetchone()
    conn.close()
    # Missing fields default to the JSON empty-dict sentinel.
    assert json.loads(row[0]) == {}
    assert json.loads(row[1]) == {}
    assert json.loads(row[2]) == {}


def test_update_self_critique(tmp_path):
    db = tmp_path / "critique.sqlite"
    store = WarmStore(str(db))
    store.write_decision({
        "decision_id": "abc-3",
        "symbol": "BTC",
        "direction": 1,
        "final_action": "LONG",
    })
    store.flush()

    critique = {
        "matched_thesis": True,
        "miss_reason": "",
        "updated_belief": "trend still intact",
        "confidence_calibration_delta": 0.05,
    }
    store.update_self_critique("abc-3", critique)

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT self_critique FROM decisions WHERE decision_id='abc-3'"
    ).fetchone()
    conn.close()
    assert json.loads(row[0]) == critique
