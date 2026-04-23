"""Tests for the executor's C4d shadow-mode hook — _run_agentic_shadow.

Observation-only integration: the agentic bridge compiles a plan each
tick and writes it to warm_store under a shadow- decision_id. This test
isolates the method so we don't have to spin up the full executor.
"""
from __future__ import annotations

import json
from unittest import mock

import pytest


def _shadow_caller():
    """Call _run_agentic_shadow as an unbound method so we don't need a
    real TradingExecutor instance (the constructor has many deps)."""
    from src.trading.executor import TradingExecutor
    fn = TradingExecutor._run_agentic_shadow

    class _Stub:
        _ex_tag = "test"
        config = {"agentic_loop": {"enabled": True}}
        _last_regime = "TRENDING"

    return fn, _Stub()


def test_shadow_noop_when_flag_disabled(monkeypatch):
    monkeypatch.delenv("ACT_AGENTIC_LOOP", raising=False)
    monkeypatch.setenv("ACT_DISABLE_AGENTIC_LOOP", "1")
    fn, stub = _shadow_caller()
    # Kill switch forces disabled; ensure compile_agentic_plan is not called.
    with mock.patch("src.ai.agentic_bridge.compile_agentic_plan") as compile_mock:
        fn(stub, "BTC", [60000.0, 60100.0])
    compile_mock.assert_not_called()


def test_shadow_writes_to_warm_store_when_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_LOOP", raising=False)
    monkeypatch.setenv("ACT_AGENTIC_LOOP", "1")

    # Point warm_store at a tmp DB so the write is observable.
    from src.orchestration.warm_store import WarmStore
    import src.orchestration.warm_store as ws_mod
    store = WarmStore(str(tmp_path / "shadow.sqlite"))
    monkeypatch.setattr(ws_mod, "_store_singleton", store, raising=False)

    # Stub compile_agentic_plan so no real LLM is exercised.
    from src.ai.agentic_bridge import compile_agentic_plan as _real_compile  # noqa
    from src.ai.agentic_trade_loop import LoopResult
    from src.trading.trade_plan import TradePlan

    fake_plan = TradePlan(
        asset="BTC", direction="LONG", entry_tier="normal",
        entry_price=60000.0, size_pct=5.0, sl_price=58800.0,
        thesis="shadow test",
    )
    fake_result = LoopResult(
        plan=fake_plan, steps_taken=2,
        tool_calls=[{"name": "get_fear_greed"}],
        terminated_reason="plan",
    )

    fn, stub = _shadow_caller()
    with mock.patch("src.ai.agentic_bridge.compile_agentic_plan", return_value=fake_result):
        fn(stub, "BTC", [60000.0, 60100.0, 60050.0])

    store.flush()

    import sqlite3
    conn = sqlite3.connect(store.db_path)
    row = conn.execute(
        "SELECT decision_id, symbol, final_action, component_signals, plan_json "
        "FROM decisions ORDER BY ts_ns DESC LIMIT 1"
    ).fetchone()
    conn.close()

    assert row is not None
    decision_id, symbol, final_action, comp_signals, plan_json = row
    assert decision_id.startswith("shadow-")
    assert symbol == "BTC"
    assert final_action == "SHADOW_LONG"
    cs = json.loads(comp_signals)
    assert cs["source"] == "agentic_shadow"
    assert cs["terminated_reason"] == "plan"
    assert "get_fear_greed" in cs["tool_calls"]
    pj = json.loads(plan_json)
    assert pj["asset"] == "BTC"
    assert pj["direction"] == "LONG"


def test_shadow_swallows_compile_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_LOOP", raising=False)
    monkeypatch.setenv("ACT_AGENTIC_LOOP", "1")
    fn, stub = _shadow_caller()
    with mock.patch("src.ai.agentic_bridge.compile_agentic_plan", side_effect=RuntimeError("boom")):
        # Must not raise.
        fn(stub, "BTC", [60000.0])


def test_shadow_swallows_warm_store_errors(monkeypatch):
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_LOOP", raising=False)
    monkeypatch.setenv("ACT_AGENTIC_LOOP", "1")

    from src.ai.agentic_trade_loop import LoopResult
    from src.trading.trade_plan import TradePlan
    fake_result = LoopResult(
        plan=TradePlan.skip("BTC", thesis="skip"),
        steps_taken=1, terminated_reason="skip",
    )
    fn, stub = _shadow_caller()
    with mock.patch("src.ai.agentic_bridge.compile_agentic_plan", return_value=fake_result), \
         mock.patch("src.orchestration.warm_store.get_store", side_effect=RuntimeError("db down")):
        # Should not raise despite DB being unavailable.
        fn(stub, "BTC", [60000.0])
