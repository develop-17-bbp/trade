"""Tests for autonomous_loop._mesh_step — C18 learning-mesh wiring.

Goals:
  * Exercises the mesh-step code path end-to-end with stubbed inputs so
    credit_assigner + reward + safety + coevolution all get driven.
  * Confirms DSR streams are created per component, credit weights
    come back from the assigner, and processed trade-IDs are recorded
    so the same trade isn't re-fed on the next cycle.
"""

from __future__ import annotations

import pytest

from src.learning.reward import reset_singleton


@pytest.fixture
def fresh_dsr_singleton():
    reset_singleton()
    yield
    reset_singleton()


def _make_loop(tmp_path, monkeypatch):
    # Point PROJECT_ROOT at a tmp dir so state.json doesn't collide with
    # live system state.
    import src.scripts.autonomous_loop as al
    monkeypatch.setattr(al, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)
    return al.AutonomousLoop(dry_run=True)


def test_mesh_step_no_recent_trades(tmp_path, monkeypatch, fresh_dsr_singleton):
    loop = _make_loop(tmp_path, monkeypatch)
    result = loop._mesh_step({"recent_trades": [], "regime": "TRENDING"})
    assert result["enabled"] is False
    assert "no recent" in result["reason"]


def test_mesh_step_processes_trades_once(tmp_path, monkeypatch, fresh_dsr_singleton):
    loop = _make_loop(tmp_path, monkeypatch)
    trades = [
        {"trade_id": "t1", "asset": "BTC", "pnl_pct": 0.5,
         "component_actions": {"l1_rl": 1.0, "l7_lora": 0.5}},
        {"trade_id": "t2", "asset": "BTC", "pnl_pct": -0.2,
         "component_actions": {"l1_rl": -0.5, "l7_lora": 0.0}},
    ]
    metrics = {"recent_trades": trades, "regime": "RANGING"}
    r1 = loop._mesh_step(metrics)
    assert r1["enabled"] is True
    assert r1["credit_recorded"] == 2
    # DSR streams: portfolio:BTC + l1_rl:BTC + l7_lora:BTC = 3
    assert r1["dsr_streams"] == 3
    # Second call with the same trades → no new work.
    r2 = loop._mesh_step(metrics)
    assert r2["enabled"] is True
    assert r2["credit_recorded"] == 0


def test_mesh_step_weights_sum_to_reasonable(tmp_path, monkeypatch, fresh_dsr_singleton):
    loop = _make_loop(tmp_path, monkeypatch)
    # Feed enough trades for the credit assigner to produce non-trivial weights
    trades = []
    for i in range(8):
        trades.append({
            "trade_id": f"t{i}",
            "asset": "BTC",
            "pnl_pct": 0.3 if i % 2 == 0 else -0.1,
            "component_actions": {
                "l1_rl": 1.0,
                "l7_lora": 0.5 if i % 2 == 0 else -0.5,
                "authority_guardian": 1.0,
            },
        })
    result = loop._mesh_step({"recent_trades": trades, "regime": "TRENDING"})
    weights = result["weights"]
    # Weights dict is populated (cold-start uses prior → still returns something)
    assert isinstance(weights, dict)
    assert len(weights) >= 1
    # Each weight is in [0, 1]
    for v in weights.values():
        assert 0.0 <= v <= 1.0


def test_mesh_step_survives_missing_component_data(tmp_path, monkeypatch, fresh_dsr_singleton):
    loop = _make_loop(tmp_path, monkeypatch)
    # Trade with no component_actions — shouldn't crash, just records DSR
    # on the portfolio stream.
    trades = [{"trade_id": "bare", "asset": "ETH", "pnl_pct": 0.1}]
    result = loop._mesh_step({"recent_trades": trades, "regime": "VOLATILE"})
    assert result["enabled"] is True
    # Only portfolio:ETH got updated; no per-component streams.
    assert result["dsr_streams"] == 1
    # No component_actions → nothing to record into credit assigner.
    assert result["credit_recorded"] == 0
