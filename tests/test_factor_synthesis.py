"""Tests for factor_synthesis — single source of truth across all loops."""
from __future__ import annotations

import pytest


def test_synthesis_action_classification():
    from src.ai.factor_synthesis import _classify_action
    assert _classify_action(0.7)[1] == "strong_long"
    assert _classify_action(0.3)[1] == "mild_long"
    assert _classify_action(0.0)[1] == "neutral"
    assert _classify_action(-0.3)[1] == "mild_short"
    assert _classify_action(-0.7)[1] == "strong_short"


def test_synthesis_action_robinhood_longs_only():
    """Strong short bias on a longs-only venue → 'no_new_longs', not enter SHORT."""
    from src.ai.factor_synthesis import _classify_action
    action, _ = _classify_action(-0.8)
    assert "no_new_longs" in action


def test_synthesis_returns_unavailable_with_no_factors(monkeypatch):
    from src.ai import factor_synthesis as fs
    # Force every fetcher to fail
    def _raise(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr("src.ai.macro_overlay.fetch_macro_overlay", _raise)
    monkeypatch.setattr("src.ai.btc_dominance.fetch_btc_dominance", _raise)
    monkeypatch.setattr("src.ai.halving_cycle.get_halving_cycle", _raise)
    monkeypatch.setattr("src.data.fetcher.PriceFetcher",
                        lambda: type("X", (), {"get_recent_bars": lambda *a, **k: []}))
    fs._cache.clear()
    s = fs.compute_synthesis("BTC")
    assert s.long_bias_score == 0.0
    assert s.recommended_action == "skip"


def test_synthesis_score_bounded():
    from src.ai.factor_synthesis import compute_synthesis
    s = compute_synthesis("BTC")
    # Score must be bounded regardless of available factors
    assert -1.0 <= s.long_bias_score <= 1.0
    assert s.confidence_label in (
        "strong_long", "mild_long", "neutral",
        "mild_short", "strong_short",
    )


def test_synthesis_publishes_to_tick_state(monkeypatch):
    from src.ai import factor_synthesis as fs
    fs._cache.clear()
    fs.refresh_and_publish("BTC", force=True)
    from src.ai import tick_state as ts
    snap = ts.get("BTC")
    assert "factor_bias_score" in snap
    assert "factor_regime" in snap
    assert "factor_action" in snap


def test_synthesis_tool_registered():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    assert "query_factor_synthesis" in set(r.list_names())


def test_synthesis_format_for_brain_renders():
    """The new FACTOR_SYNTHESIS line must show in format_for_brain output."""
    from src.ai import tick_state as ts
    ts.update("BTC",
               factor_bias_score=0.42,
               factor_regime="risk_on",
               factor_action="submit_long",
               factor_confidence="mild_long",
               factor_n_available=5,
               factor_bullish="macro,halving",
               factor_bearish="",
               factor_rationale="test")
    out = ts.format_for_brain("BTC")
    assert "FACTOR_SYNTHESIS:" in out
    assert "bias_score=+0.42" in out
    assert "submit_long" in out


def test_catalyst_listener_includes_factor_triggers():
    """Catalyst listener should track factor_regime and bias_score for flips."""
    from src.orchestration.catalyst_listener import (
        CatalystListener, DEFAULT_BIAS_FLIP_THRESHOLD,
    )
    cl = CatalystListener()
    # Constants exist
    assert DEFAULT_BIAS_FLIP_THRESHOLD > 0
    # Tracking dicts initialized
    assert hasattr(cl, "_last_factor_regime")
    assert hasattr(cl, "_last_bias_score")


def test_continuous_brain_uses_synthesis():
    """ContinuousBrain._refresh_asset should generate scenarios from
    factor_bias_score when present."""
    import os
    os.environ["ACT_CONTINUOUS_BRAIN"] = "1"
    from src.ai import tick_state as ts
    from src.ai.continuous_brain import ContinuousBrain
    ts.update("BTC",
               factor_bias_score=0.7,    # strong long
               factor_regime="risk_on",
               factor_action="submit_long")
    cb = ContinuousBrain()
    cb._refresh_asset("BTC")
    cache = cb.get_cache("BTC")
    names = [s["name"] for s in cache]
    assert "factor_synthesis_strong_long" in names
