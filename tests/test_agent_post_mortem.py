"""Tests for /agent-post-mortem skill — counterfactual trace chat."""
from __future__ import annotations

import json
import sqlite3
import time
from unittest import mock

import pytest

from skills.agent_post_mortem.action import (
    _load_decision,
    _summarize_trace,
    run,
)


@pytest.fixture
def tmp_warm_store(tmp_path, monkeypatch):
    db = tmp_path / "w.sqlite"
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


def _insert_decision(db, decision_id, *, with_outcome=True, pnl_pct=1.3):
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO decisions (decision_id, symbol, ts_ns, direction, "
        "confidence, final_action, authority_violations, plan_json, "
        "component_signals, self_critique) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (decision_id, "BTC", time.time_ns(), 1, 0.72, "LONG",
         json.dumps([]),
         json.dumps({"direction": "LONG", "entry_tier": "normal",
                     "size_pct": 5.0, "entry_price": 60000,
                     "sl_price": 58800, "thesis": "EMA align"}),
         json.dumps({"source": "agentic", "tool_calls": ["get_fear_greed"]}),
         json.dumps({"matched_thesis": pnl_pct > 0,
                     "miss_reason": "" if pnl_pct > 0 else "exited_early",
                     "confidence_calibration_delta": 0.05})),
    )
    if with_outcome:
        conn.execute(
            "INSERT INTO outcomes (decision_id, symbol, direction, pnl_pct, "
            "exit_reason, duration_s, exit_ts, regime, payload_json) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (decision_id, "BTC", "LONG", pnl_pct,
             "tp" if pnl_pct > 0 else "sl", 3600.0, time.time(), "trending",
             json.dumps({"pnl_pct": pnl_pct})),
        )
    conn.commit()
    conn.close()


# ── _load_decision ─────────────────────────────────────────────────────


def test_load_missing_returns_none(tmp_warm_store):
    assert _load_decision("does-not-exist") is None


def test_load_decision_with_outcome(tmp_warm_store):
    _insert_decision(tmp_warm_store, "d-1", pnl_pct=1.5)
    d = _load_decision("d-1")
    assert d is not None
    assert d["symbol"] == "BTC"
    assert d["plan"]["direction"] == "LONG"
    assert d["outcome"]["pnl_pct"] == pytest.approx(1.5)


def test_load_decision_without_outcome(tmp_warm_store):
    _insert_decision(tmp_warm_store, "d-open", with_outcome=False)
    d = _load_decision("d-open")
    assert d["outcome"] is None


# ── _summarize_trace ───────────────────────────────────────────────────


def test_summarize_includes_key_fields(tmp_warm_store):
    _insert_decision(tmp_warm_store, "d-sum")
    d = _load_decision("d-sum")
    out = _summarize_trace(d)
    assert "Decision d-sum" in out
    assert "BTC" in out
    assert "LONG" in out
    assert "Thesis" in out
    assert "Outcome" in out
    assert "Self-critique" in out


def test_summarize_without_outcome(tmp_warm_store):
    _insert_decision(tmp_warm_store, "d-no-out", with_outcome=False)
    d = _load_decision("d-no-out")
    out = _summarize_trace(d)
    assert "Decision d-no-out" in out
    # No KeyError on missing outcome.


# ── run() skill entry ──────────────────────────────────────────────────


def test_run_requires_decision_id():
    r = run({})
    assert r.ok is False
    assert "decision_id" in (r.error or "").lower()


def test_run_returns_not_found_for_missing(tmp_warm_store):
    r = run({"decision_id": "nope"})
    assert r.ok is False
    assert "not found" in (r.error or "").lower()


def test_run_happy_path_summary_only(tmp_warm_store, monkeypatch):
    _insert_decision(tmp_warm_store, "d-happy", pnl_pct=1.8)
    # Force dual_brain.analyze to be unavailable → summary-only path.
    monkeypatch.setitem(
        __import__('sys').modules, 'src.ai.dual_brain',
        mock.MagicMock(analyze=mock.MagicMock(side_effect=RuntimeError("no llm"))),
    )
    r = run({"decision_id": "d-happy"})
    assert r.ok is True
    assert "Decision d-happy" in r.message
    # Without LLM, agent_post_mortems is empty but the summary is still there.
    assert r.data["agent_post_mortems"] == {}


def test_run_llm_post_mortem_integrated(tmp_warm_store, monkeypatch):
    _insert_decision(tmp_warm_store, "d-llm", pnl_pct=-0.8)
    # Stub analyze() to return a canned answer.
    fake_resp = mock.MagicMock(ok=True, text="would skip next time due to weak macro")
    fake_module = mock.MagicMock(analyze=mock.MagicMock(return_value=fake_resp))
    monkeypatch.setitem(__import__('sys').modules, 'src.ai.dual_brain', fake_module)

    r = run({"decision_id": "d-llm", "agents": ["risk_guardian"]})
    assert r.ok is True
    assert "risk_guardian" in r.data["agent_post_mortems"]
    assert "would skip next time" in r.data["agent_post_mortems"]["risk_guardian"]


def test_run_what_if_rewind_unavailable_tolerated(tmp_warm_store):
    _insert_decision(tmp_warm_store, "d-wi")
    r = run({"decision_id": "d-wi", "what_if_seconds": 30})
    # May or may not succeed depending on brain_memory state; either way
    # the skill returns ok=True because the summary path always runs.
    assert r.ok is True


def test_run_accepts_comma_separated_agents(tmp_warm_store, monkeypatch):
    _insert_decision(tmp_warm_store, "d-multi")
    fake_resp = mock.MagicMock(ok=True, text="ok")
    fake_module = mock.MagicMock(analyze=mock.MagicMock(return_value=fake_resp))
    monkeypatch.setitem(__import__('sys').modules, 'src.ai.dual_brain', fake_module)
    r = run({"decision_id": "d-multi",
             "agents": "risk_guardian, trend_momentum, mean_reversion"})
    assert r.ok is True
    # Expected call once per listed agent (up to cap of 5).
    assert len(r.data["agent_post_mortems"]) == 3


# ── Registry integration ───────────────────────────────────────────────


def test_skill_is_registered():
    from src.skills.registry import get_registry
    reg = get_registry(refresh=True)
    assert reg.get("agent-post-mortem") is not None
