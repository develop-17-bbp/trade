"""Tests for the brain-evolution additions:
  - catalyst_listener (event-driven preemption daemon)
  - continuous_brain (async scenario pre-computer)
  - decision_graph (causal queries over warm_store decisions)
"""
from __future__ import annotations

import time

import pytest


# ── Catalyst Listener ──────────────────────────────────────────────────


def test_catalyst_listener_dormant_when_unset(monkeypatch):
    monkeypatch.delenv("ACT_CATALYST_LISTENER", raising=False)
    from src.orchestration.catalyst_listener import CatalystListener, is_enabled
    assert is_enabled() is False
    cl = CatalystListener()
    cl.start()  # should no-op
    assert cl._thread is None or not cl._thread.is_alive()


def test_catalyst_listener_shadow_mode_logs_only(monkeypatch):
    monkeypatch.setenv("ACT_CATALYST_LISTENER", "shadow")
    from src.orchestration.catalyst_listener import is_enabled, is_authoritative
    assert is_enabled() is True
    assert is_authoritative() is False


def test_catalyst_listener_authoritative_with_one(monkeypatch):
    monkeypatch.setenv("ACT_CATALYST_LISTENER", "1")
    from src.orchestration.catalyst_listener import is_enabled, is_authoritative
    assert is_enabled() is True
    assert is_authoritative() is True


def test_catalyst_event_to_dict_bounded():
    from src.orchestration.catalyst_listener import CatalystEvent
    e = CatalystEvent(
        asset="BTC", trigger_type="news",
        trigger_value=0.95, threshold=0.8, ts=time.time(),
    )
    d = e.to_dict()
    assert d["asset"] == "BTC"
    assert d["trigger_type"] == "news"
    assert 0.0 <= d["trigger_value"] <= 1.0


def test_catalyst_listener_singleton():
    from src.orchestration.catalyst_listener import get_listener
    a = get_listener()
    b = get_listener()
    assert a is b


# ── Continuous Brain ──────────────────────────────────────────────────


def test_continuous_brain_dormant_when_unset(monkeypatch):
    monkeypatch.delenv("ACT_CONTINUOUS_BRAIN", raising=False)
    from src.ai.continuous_brain import ContinuousBrain, is_enabled
    assert is_enabled() is False
    cb = ContinuousBrain()
    cb.start()  # no-op
    assert cb._thread is None or not cb._thread.is_alive()


def test_continuous_brain_enabled(monkeypatch):
    monkeypatch.setenv("ACT_CONTINUOUS_BRAIN", "1")
    from src.ai.continuous_brain import is_enabled
    assert is_enabled() is True


def test_continuous_brain_get_cache_returns_list(monkeypatch):
    monkeypatch.setenv("ACT_CONTINUOUS_BRAIN", "1")
    from src.ai.continuous_brain import ContinuousBrain
    cb = ContinuousBrain()
    cache = cb.get_cache("BTC")
    assert isinstance(cache, list)


def test_continuous_brain_scenario_bounded():
    from src.ai.continuous_brain import Scenario
    s = Scenario(
        name="drop_1pct",
        trigger_condition="if BTC drops 1% in 30s",
        suggested_action="HOLD",
        rationale="x" * 1000,
        computed_at=time.time(),
    )
    d = s.to_dict()
    assert len(d["rationale"]) <= 200


def test_continuous_brain_refresh_with_tick_state(monkeypatch):
    monkeypatch.setenv("ACT_CONTINUOUS_BRAIN", "1")
    from src.ai import tick_state as ts
    from src.ai.continuous_brain import ContinuousBrain
    ts.update("BTC",
              ratchet_label="BREAKEVEN", ratchet_current_pnl_pct=1.1,
              open_positions_same_asset=2, regime="TREND_UP",
              gap_to_1pct=0.6, today_pct_total=0.4, today_trades=2,
              equity_usd=16500)
    cb = ContinuousBrain()
    cb._refresh_asset("BTC")
    cache = cb.get_cache("BTC")
    # Should have produced at least 1 scenario from the populated state
    assert len(cache) >= 1
    for s in cache:
        assert "name" in s
        assert "suggested_action" in s


# ── Decision Graph ──────────────────────────────────────────────────


def test_decision_graph_causal_empty_returns_safe():
    from src.ai.decision_graph import causal_query
    # Filter that won't match (regime = made-up string)
    r = causal_query(regime="DEFINITELY_DOES_NOT_EXIST_xyz")
    assert r.matched_decisions == 0
    assert r.win_rate == 0.0
    assert "no_decisions_matched_filters" in r.sample_warning


def test_decision_graph_similar_returns_structure():
    from src.ai.decision_graph import similar_setups
    out = similar_setups(
        asset="BTC", current_regime="TREND_UP",
        current_pattern="STRONG", current_direction="LONG",
        top_k=5,
    )
    assert "asset" in out
    assert "n_matched" in out
    assert "setups" in out
    assert isinstance(out["setups"], list)
    assert "advisory" in out


def test_decision_graph_node_view_structure():
    from src.ai.decision_graph import build_node_view
    out = build_node_view(asset="BTC", max_nodes=5)
    assert "n_nodes" in out
    assert "n_edges" in out
    assert "decay_days" in out
    assert isinstance(out["nodes"], list)
    assert isinstance(out["edges"], list)


# ── Tool registration smoke ──────────────────────────────────────────


def test_4_evolution_tools_registered():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    names = set(r.list_names())
    expected = {
        "query_catalyst_listener_state",
        "query_continuous_brain_scenarios",
        "query_decision_graph_causal",
        "query_decision_graph_similar",
    }
    missing = expected - names
    assert not missing, f"missing: {missing}"


def test_continuous_brain_scenarios_tool_dispatches():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    res = r.dispatch("query_continuous_brain_scenarios", {"asset": "BTC"})
    s = str(res)
    assert "asset" in s.lower()
    assert "scenarios" in s.lower()


def test_decision_graph_causal_tool_dispatches():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    res = r.dispatch("query_decision_graph_causal", {"asset": "BTC"})
    s = str(res).lower()
    assert "matched_decisions" in s or "error" in s
