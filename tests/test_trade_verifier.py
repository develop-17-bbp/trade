"""Tests for src/ai/trade_verifier.py — post-close self-critique."""
from __future__ import annotations

import json

import pytest

from src.ai.trade_verifier import (
    DISABLE_ENV,
    SelfCritique,
    _mechanical_verdict,
    _parse_verdict,
    verify_outcome,
)


# ── Pure parser ─────────────────────────────────────────────────────────


def test_parse_verdict_happy():
    raw = json.dumps({
        "matched_thesis": True, "miss_reason": "", "updated_belief": "momo intact",
        "next_time_do": "ok", "confidence_calibration_delta": 0.15,
    })
    c = _parse_verdict(raw)
    assert c is not None
    assert c.matched_thesis is True
    assert c.confidence_calibration_delta == pytest.approx(0.15)
    assert c.verifier_source == "llm"


def test_parse_verdict_embedded_in_prose():
    raw = (
        "Here is my verdict:\n"
        '{"matched_thesis": false, "miss_reason": "exited early", '
        '"updated_belief": "trend stronger than thought", '
        '"next_time_do": "hold longer", '
        '"confidence_calibration_delta": -0.2}'
        "\nThanks."
    )
    c = _parse_verdict(raw)
    assert c is not None and c.matched_thesis is False


def test_parse_verdict_returns_none_on_garbage():
    assert _parse_verdict("no json at all") is None
    assert _parse_verdict("") is None
    assert _parse_verdict("{bad json") is None


# ── Mechanical fallback ─────────────────────────────────────────────────


def test_mechanical_match_when_long_and_positive_pnl():
    c = _mechanical_verdict(
        {"direction": "LONG"}, {"pnl_pct": 1.5, "exit_reason": "tp"}
    )
    assert c.matched_thesis is True
    assert c.verifier_source == "mechanical"
    assert c.confidence_calibration_delta > 0


def test_mechanical_miss_when_sl_hit():
    c = _mechanical_verdict(
        {"direction": "LONG"}, {"pnl_pct": -0.9, "exit_reason": "stop_loss"}
    )
    assert c.matched_thesis is False
    assert "SL" in c.miss_reason
    assert c.confidence_calibration_delta < 0


def test_mechanical_drift_exit():
    c = _mechanical_verdict(
        {"direction": "LONG"}, {"pnl_pct": -0.2, "exit_reason": "time_decay"}
    )
    assert c.matched_thesis is False
    assert "drift" in c.miss_reason.lower()


# ── End-to-end with injected LLM ────────────────────────────────────────


def test_verify_outcome_with_injected_llm_happy():
    def fake_llm(_sys, _user):
        return json.dumps({
            "matched_thesis": True,
            "miss_reason": "",
            "updated_belief": "trend persists",
            "next_time_do": "keep cadence",
            "confidence_calibration_delta": 0.1,
        })
    c = verify_outcome({"direction": "LONG"}, {"pnl_pct": 0.8, "exit_reason": "tp"}, llm_call=fake_llm)
    assert c.verifier_source == "llm"
    assert c.matched_thesis is True


def test_verify_outcome_falls_back_to_mechanical_on_unparseable():
    c = verify_outcome(
        {"direction": "LONG"},
        {"pnl_pct": -0.5, "exit_reason": "stop_loss"},
        llm_call=lambda *_: "garbage response",
    )
    assert c.verifier_source == "mechanical"
    assert c.matched_thesis is False


def test_verify_outcome_respects_disable_env(monkeypatch):
    monkeypatch.setenv(DISABLE_ENV, "1")
    called = {"n": 0}
    def fake_llm(*_):
        called["n"] += 1
        return "{}"
    c = verify_outcome({"direction": "LONG"}, {"pnl_pct": 1.0}, llm_call=fake_llm)
    assert called["n"] == 0
    assert c.verifier_source == "mechanical"


def test_verify_outcome_handles_llm_exception():
    def boom(*_):
        raise RuntimeError("net down")
    c = verify_outcome({"direction": "LONG"}, {"pnl_pct": -0.2, "exit_reason": "time_decay"}, llm_call=boom)
    assert c.verifier_source == "mechanical"


# ── Serialization ───────────────────────────────────────────────────────


def test_self_critique_to_dict_trims_fields():
    c = SelfCritique(
        matched_thesis=False,
        miss_reason="x" * 600,
        updated_belief="y" * 600,
        next_time_do="z" * 600,
        confidence_calibration_delta=0.333,
    )
    d = c.to_dict()
    assert len(d["miss_reason"]) <= 400
    assert len(d["updated_belief"]) <= 400
    assert len(d["next_time_do"]) <= 400
    assert d["confidence_calibration_delta"] == 0.333
    assert d["verifier_source"] == "llm"


# ── verify_and_persist happy path ───────────────────────────────────────


def test_verify_and_persist_writes_to_warm_store(tmp_path, monkeypatch):
    from src.orchestration.warm_store import WarmStore
    import src.orchestration.warm_store as ws_mod

    db = tmp_path / "vp.sqlite"
    store = WarmStore(str(db))
    store.write_decision({
        "decision_id": "v-1", "symbol": "BTC",
        "plan": {"direction": "LONG"},
    })
    store.flush()

    # Point module-level singleton at our temp store.
    monkeypatch.setattr(ws_mod, "_store_singleton", store, raising=False)

    from src.ai import trade_verifier as tv
    # Stub the LLM so we control the verdict.
    def stub_llm(_s, _u):
        return json.dumps({
            "matched_thesis": True, "miss_reason": "",
            "updated_belief": "ok", "next_time_do": "keep going",
            "confidence_calibration_delta": 0.05,
        })
    result = tv.verify_outcome({"direction": "LONG"}, {"pnl_pct": 0.5, "exit_reason": "tp"}, llm_call=stub_llm)
    store.update_self_critique("v-1", result.to_dict())

    # Read back and confirm.
    import sqlite3
    conn = sqlite3.connect(str(db))
    row = conn.execute("SELECT self_critique FROM decisions WHERE decision_id='v-1'").fetchone()
    conn.close()
    assert row is not None
    persisted = json.loads(row[0])
    assert persisted["matched_thesis"] is True
