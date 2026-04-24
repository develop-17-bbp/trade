"""Tests for C26 Step 2 — unified brain tool pack."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def reset_registry():
    from src.ai.trade_tools import reset_default_registry
    reset_default_registry()
    yield
    reset_default_registry()


def test_all_nine_tools_registered():
    """Every planned unified-brain tool appears in the default registry."""
    from src.ai.trade_tools import build_default_registry
    reg = build_default_registry()
    names = set(reg.list_names())
    for expected in (
        "query_ml_ensemble", "query_multi_strategy", "find_similar_trades",
        "monte_carlo_var", "evt_tail_risk", "get_macro_bias",
        "get_economic_layer", "request_genetic_candidate", "run_full_backtest",
    ):
        assert expected in names, f"missing tool: {expected}"


def test_tools_have_input_schemas():
    """Every tool advertises an input_schema — Anthropic API needs this."""
    from src.ai.trade_tools import build_default_registry
    reg = build_default_registry()
    for name in (
        "query_ml_ensemble", "monte_carlo_var", "evt_tail_risk",
        "find_similar_trades", "run_full_backtest",
    ):
        t = reg.get(name)
        assert t is not None, name
        assert t.input_schema.get("type") == "object", name
        # Anthropic schema shape
        sch = t.anthropic_schema()
        assert "name" in sch and "description" in sch and "input_schema" in sch


def test_ml_ensemble_handler_graceful_on_missing_deps():
    """Each sub-model's absence is captured per-model, not fatal."""
    from src.ai.unified_brain_tools import _handle_ml_ensemble
    out = _handle_ml_ensemble({"asset": "BTC"})
    assert out["asset"] == "BTC"
    assert "models" in out
    assert "consensus" in out


def test_economic_layer_requires_layer_arg():
    from src.ai.unified_brain_tools import _handle_economic_layer
    err = _handle_economic_layer({})
    assert "error" in err


def test_find_similar_trades_returns_structured_output(monkeypatch):
    """Handler shape is predictable even when MemoryVault is empty."""
    from src.ai.unified_brain_tools import _handle_find_similar_trades
    # Patch MemoryVault to return empty
    import src.ai.memory_vault as mv
    class _StubVault:
        def find_similar_trades(self, **kw):
            return []
    monkeypatch.setattr(mv, "MemoryVault", _StubVault)
    out = _handle_find_similar_trades({"asset": "BTC", "regime": "CHOPPY"})
    assert out["asset"] == "BTC"
    assert out["count"] == 0
    assert isinstance(out["results"], list)


def test_tool_metadata_classifies_new_tools():
    """All 9 new tools have audit classifications for FinToolBench."""
    from src.ai.trade_tools import build_default_registry
    build_default_registry()    # triggers registration + classification
    from src.ai.tool_metadata import TOOL_CLASSIFICATIONS
    for name in (
        "query_ml_ensemble", "query_multi_strategy", "find_similar_trades",
        "monte_carlo_var", "evt_tail_risk", "get_macro_bias",
        "get_economic_layer", "request_genetic_candidate", "run_full_backtest",
    ):
        assert name in TOOL_CLASSIFICATIONS, name
        c = TOOL_CLASSIFICATIONS[name]
        assert c.is_fully_classified(), name


def test_dispatch_returns_serializable_string():
    """Dispatch always returns a string (capped), not raises."""
    from src.ai.trade_tools import build_default_registry
    reg = build_default_registry()
    # request_genetic_candidate with no deps should gracefully return
    # an {error: ...} or {available: False} dict, both fine.
    out = reg.dispatch("request_genetic_candidate", {"regime": "TRENDING"})
    assert isinstance(out, str)
    assert len(out) > 0
