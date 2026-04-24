"""Tests for skills/weekly_brief — C20."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _load_action():
    """Load skills/weekly_brief/action.py as an anonymous module so tests
    work regardless of the registry-based loader."""
    root = Path(__file__).resolve().parents[1]
    action_path = root / "skills" / "weekly_brief" / "action.py"
    spec = importlib.util.spec_from_file_location("_wb_action", str(action_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk_warm_store(tmp_path: Path) -> Path:
    """Minimal warm_store that matches the shipped schema."""
    db = tmp_path / "warm_store.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE decisions (
            decision_id TEXT PRIMARY KEY,
            trace_id TEXT,
            symbol TEXT,
            ts_ns INTEGER,
            direction INTEGER,
            confidence REAL,
            consensus TEXT,
            veto INTEGER,
            raw_signal INTEGER,
            final_action TEXT,
            authority_violations TEXT,
            payload_json TEXT,
            component_signals TEXT,
            plan_json TEXT,
            self_critique TEXT
        )
    """)
    conn.commit()
    conn.close()
    return db


def _seed_decision(db: Path, decision_id: str, symbol: str,
                   ts_ns: int, direction: int,
                   plan_json: str = "{}", self_critique: str = "{}") -> None:
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO decisions (decision_id, symbol, ts_ns, direction, "
        "plan_json, self_critique) VALUES (?, ?, ?, ?, ?, ?)",
        (decision_id, symbol, ts_ns, direction, plan_json, self_critique),
    )
    conn.commit()
    conn.close()


def test_runs_against_empty_state(tmp_path, monkeypatch):
    """Never crashes when warm_store is missing / empty."""
    mod = _load_action()
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path, raising=True)
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(tmp_path / "does_not_exist.sqlite"))
    result = mod.run({"days": 7})
    assert result.ok is True
    assert "weekly_brief_" in result.message


def test_reads_schema_tolerant(tmp_path, monkeypatch):
    """Schema-tolerant SELECT works against the shipped ACT schema
    (which has `direction` instead of `side`, and no `tier`/`size_pct`/
    `outcome_json` top-level columns)."""
    db = _mk_warm_store(tmp_path)
    # Seed two decisions — one within window, one outside
    now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
    _seed_decision(db, "shadow-abc", "BTC", now_ns - int(1e9), direction=1,
                   plan_json=json.dumps({"entry_tier": "sniper", "size_pct": 5.0}))
    _seed_decision(db, "real-def", "ETH", now_ns - int(2e9), direction=-1,
                   plan_json=json.dumps({"entry_tier": "normal", "size_pct": 2.0}),
                   self_critique=json.dumps({"matched_thesis": True}))
    # Old, outside 7d window
    _seed_decision(db, "old-xyz", "BTC",
                   now_ns - int(30 * 86400 * 1e9), direction=1)

    mod = _load_action()
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path, raising=True)
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    result = mod.run({"days": 7})
    assert result.ok is True
    data = result.data
    assert data["total_decisions"] == 2
    assert data["shadow_plans"] == 1
    assert data["real_plans"] == 1
    assert data["critiques_total"] == 1
    assert data["critiques_matched"] == 1


def test_report_file_has_expected_sections(tmp_path, monkeypatch):
    db = _mk_warm_store(tmp_path)
    mod = _load_action()
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path, raising=True)
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    result = mod.run({"days": 7})
    assert result.ok is True
    report = Path(result.data["report_path"]).read_text(encoding="utf-8")
    for heading in (
        "# ACT Weekly Activity Brief",
        "## Executive summary",
        "## Safety-gate activity",
        "## Strategy repository",
        "## Brain activity",
        "## Learning mesh",
    ):
        assert heading in report, f"missing heading: {heading}"


def test_parses_days_override(tmp_path, monkeypatch):
    db = _mk_warm_store(tmp_path)
    mod = _load_action()
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path, raising=True)
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    result = mod.run({"days": 14})
    assert result.ok is True
    assert result.data["window_days"] == 14.0


def test_invalid_days_falls_back_to_default(tmp_path, monkeypatch):
    db = _mk_warm_store(tmp_path)
    mod = _load_action()
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path, raising=True)
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    result = mod.run({"days": "not_a_number"})
    assert result.ok is True
    # Default window is 7 days
    assert result.data["window_days"] == 7
