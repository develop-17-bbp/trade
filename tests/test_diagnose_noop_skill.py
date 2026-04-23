"""Tests for the /diagnose-noop skill."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from unittest import mock

import pytest

from skills.diagnose_noop.action import (
    _check_config,
    _check_env_flags,
    _check_warm_store_counts,
    _diagnose,
    run,
)


@pytest.fixture
def tmp_warm_store(tmp_path, monkeypatch):
    """Seed a tiny warm_store.sqlite and point ACT_WARM_DB_PATH at it."""
    db = tmp_path / "wd.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE decisions (
            decision_id TEXT PRIMARY KEY,
            trace_id TEXT, symbol TEXT, ts_ns INTEGER, direction INTEGER,
            confidence REAL, consensus TEXT, veto INTEGER, raw_signal INTEGER,
            final_action TEXT, authority_violations TEXT, payload_json TEXT,
            component_signals TEXT DEFAULT '{}',
            plan_json TEXT DEFAULT '{}',
            self_critique TEXT DEFAULT '{}'
        )
    """)
    conn.execute("""
        CREATE TABLE outcomes (
            outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT, symbol TEXT, direction TEXT,
            entry_price REAL, exit_price REAL, pnl_pct REAL, pnl_usd REAL,
            duration_s REAL, exit_reason TEXT, regime TEXT,
            entry_ts REAL, exit_ts REAL, payload_json TEXT
        )
    """)
    conn.commit()
    conn.close()
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    return db


# ── Env check ──────────────────────────────────────────────────────────


def test_env_check_shape(monkeypatch):
    monkeypatch.setenv("ACT_AGENTIC_LOOP", "1")
    out = _check_env_flags()
    assert out["ACT_AGENTIC_LOOP"] == "1"
    assert "ACT_DISABLE_AGENTIC_LOOP" in out
    assert "ACT_BRAIN_PROFILE" in out


# ── warm_store counts ──────────────────────────────────────────────────


def test_warm_store_empty_db(tmp_warm_store):
    out = _check_warm_store_counts()
    assert out["exists"] is True
    assert out["decisions_24h_total"] == 0
    assert out["decisions_24h_shadow"] == 0
    assert out["decisions_24h_real"] == 0


def test_warm_store_nonexistent_db(tmp_path, monkeypatch):
    bogus = tmp_path / "missing.sqlite"
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(bogus))
    out = _check_warm_store_counts()
    assert out["exists"] is False


def test_warm_store_with_shadow_rows(tmp_warm_store):
    conn = sqlite3.connect(str(tmp_warm_store))
    now_ns = time.time_ns()
    for i in range(7):
        conn.execute(
            "INSERT INTO decisions (decision_id, symbol, ts_ns, final_action, "
            "component_signals) VALUES (?,?,?,?,?)",
            (f"shadow-{i}", "BTC", now_ns - i * 1_000_000_000, "SHADOW_SKIP",
             json.dumps({"terminated_reason": "skip"})),
        )
    for i in range(2):
        conn.execute(
            "INSERT INTO decisions (decision_id, symbol, ts_ns, final_action) "
            "VALUES (?,?,?,?)",
            (f"real-{i}", "ETH", now_ns - i * 1_000_000_000, "FLAT"),
        )
    conn.commit()
    conn.close()

    out = _check_warm_store_counts()
    assert out["decisions_24h_total"] == 9
    assert out["decisions_24h_shadow"] == 7
    assert out["decisions_24h_real"] == 2
    assert out["shadow_terminated_reasons"].get("skip") == 7


# ── Diagnosis ──────────────────────────────────────────────────────────


def test_diagnose_kill_switch_wins():
    data = {"env": {"ACT_DISABLE_AGENTIC_LOOP": "1"}}
    assert "kill switch" in _diagnose(data)


def test_diagnose_flag_unset():
    data = {
        "env": {"ACT_AGENTIC_LOOP": "<unset>", "ACT_DISABLE_AGENTIC_LOOP": "0"},
        "config": {"agentic_loop_enabled": False},
        "ollama": {"ok": True, "required": {"x": True}},
        "warm_store": {"exists": True, "decisions_24h_total": 0},
    }
    assert "not set" in _diagnose(data) or "isn't firing" in _diagnose(data)


def test_diagnose_ollama_missing_models():
    data = {
        "env": {"ACT_AGENTIC_LOOP": "1", "ACT_DISABLE_AGENTIC_LOOP": "0"},
        "ollama": {"ok": True, "required": {"deepseek-r1:32b": False, "deepseek-r1:7b": True}},
        "warm_store": {"exists": True},
    }
    d = _diagnose(data)
    assert "missing" in d
    assert "deepseek-r1:32b" in d


def test_diagnose_ollama_unreachable():
    data = {
        "env": {"ACT_AGENTIC_LOOP": "1", "ACT_DISABLE_AGENTIC_LOOP": "0"},
        "ollama": {"ok": False, "error": "connection refused"},
        "warm_store": {"exists": True},
    }
    assert "unreachable" in _diagnose(data)


def test_diagnose_readiness_gate_closed():
    data = {
        "env": {"ACT_AGENTIC_LOOP": "1", "ACT_DISABLE_AGENTIC_LOOP": "0"},
        "ollama": {"ok": True, "required": {"m": True}},
        "warm_store": {"exists": True, "decisions_24h_shadow": 400,
                       "decisions_24h_real": 0, "decisions_24h_total": 400,
                       "outcomes_24h": 0},
        "readiness": {"open": False, "failing": ["trades 0 < 500", "soak 0d < 14d"]},
    }
    d = _diagnose(data)
    assert "Readiness gate CLOSED" in d
    assert "500" in d or "14d" in d


def test_diagnose_zero_decisions():
    data = {
        "env": {"ACT_AGENTIC_LOOP": "1", "ACT_DISABLE_AGENTIC_LOOP": "0"},
        "ollama": {"ok": True, "required": {"m": True}},
        "warm_store": {"exists": True, "decisions_24h_total": 0,
                       "decisions_24h_shadow": 0, "decisions_24h_real": 0},
        "readiness": {"open": False, "failing": []},
    }
    d = _diagnose(data)
    assert "Zero warm_store decisions" in d


def test_diagnose_stuck_positions():
    data = {
        "env": {"ACT_AGENTIC_LOOP": "1", "ACT_DISABLE_AGENTIC_LOOP": "0"},
        "ollama": {"ok": True, "required": {"m": True}},
        "warm_store": {"exists": True, "decisions_24h_total": 10,
                       "decisions_24h_shadow": 0, "decisions_24h_real": 10,
                       "outcomes_24h": 0},
        "readiness": {"open": True, "failing": []},
    }
    d = _diagnose(data)
    assert "stuck" in d.lower() or "positions may be stuck" in d.lower()


def test_diagnose_shadow_mostly_skip():
    data = {
        "env": {"ACT_AGENTIC_LOOP": "1", "ACT_DISABLE_AGENTIC_LOOP": "0"},
        "ollama": {"ok": True, "required": {"m": True}},
        "warm_store": {
            "exists": True, "decisions_24h_total": 100,
            "decisions_24h_shadow": 100, "decisions_24h_real": 0,
            "outcomes_24h": 0,
            "shadow_terminated_reasons": {"skip": 95, "plan": 5},
        },
        "readiness": {"open": True, "failing": []},
        "brain_memory": {"BTC": {"latest_scan_age_s": 60, "latest_scan_score": 40}},
    }
    d = _diagnose(data)
    # Either the "skip dominates" or the "readiness gate open" branch.
    assert "skip" in d.lower() or "conviction" in d.lower()


# ── Skill entry point ──────────────────────────────────────────────────


def test_run_returns_skill_result(tmp_warm_store, monkeypatch):
    # Make ollama check fast-fail so we don't hit the real port.
    monkeypatch.setenv("OLLAMA_REMOTE_URL", "http://127.0.0.1:1")  # unroutable
    result = run({})
    # Never raises; always produces a SkillResult with data + message.
    assert result.ok is True
    assert "Diagnosis" in result.message
    assert isinstance(result.data, dict)
    for k in ("env", "ollama", "warm_store", "readiness"):
        assert k in result.data


def test_skill_registered_in_repo():
    """The shipped repo should auto-load the diagnose-noop skill."""
    from src.skills.registry import get_registry
    reg = get_registry(refresh=True)
    assert reg.get("diagnose-noop") is not None
