"""Tests for C26 Step 3 — executor.submit_trade_plan gate stack +
paper-order routing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _fake_plan(
    asset="BTC", direction="LONG", size_pct=1.5,
    entry_price=77000.0, sl_price=75500.0,
    thesis="breakout confirmed", confidence=0.7,
    expected_pnl_pct_range=(1.0, 3.0),
    entry_tier="normal",
):
    """Return a plan object that quacks like TradePlan enough for the
    submit method."""
    plan = SimpleNamespace(
        asset=asset, direction=direction, size_pct=size_pct,
        entry_price=entry_price, sl_price=sl_price,
        thesis=thesis, confidence=confidence,
        expected_pnl_pct_range=expected_pnl_pct_range,
        entry_tier=entry_tier,
        conviction_score=5,
        tp_levels=[{"price": entry_price * 1.03}],
    )
    plan.to_dict = lambda: {
        "asset": asset, "direction": direction, "size_pct": size_pct,
        "entry_price": entry_price, "thesis": thesis,
    }
    return plan


def _fake_executor_minimal(monkeypatch, tmp_path):
    """Build a minimal object that has submit_trade_plan attached as a
    bound method, without running the full TradingExecutor __init__."""
    import importlib
    ex_mod = importlib.import_module("src.trading.executor")

    # Unbound method reference
    submit_method = ex_mod.TradingExecutor.submit_trade_plan

    # Minimal shim with just the attributes submit_trade_plan uses
    bot = SimpleNamespace()
    bot._ex_tag = "TEST"
    bot._paper = MagicMock()
    bot._paper.equity = 100000.0
    bot._paper.record_entry = MagicMock()
    bot._paper.save_state = MagicMock()

    # Bind
    def call(plan, mode="paper"):
        return submit_method(bot, plan, mode=mode)
    bot.submit = call

    # Monkey-patch warm_store so nothing actually writes
    monkeypatch.setattr(
        "src.orchestration.warm_store.get_store",
        lambda: MagicMock(write_decision=MagicMock()),
    )
    return bot


def test_kill_switch_blocks_submit(monkeypatch, tmp_path):
    monkeypatch.setenv("ACT_DISABLE_AGENTIC_SUBMIT", "1")
    bot = _fake_executor_minimal(monkeypatch, tmp_path)
    result = bot.submit(_fake_plan(), mode="paper")
    assert result["submitted"] is False
    assert result["reason"] == "disabled_by_env"


def test_skip_plan_is_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_SUBMIT", raising=False)
    bot = _fake_executor_minimal(monkeypatch, tmp_path)
    plan = _fake_plan(direction="SKIP")
    result = bot.submit(plan, mode="paper")
    assert result["submitted"] is False
    assert "no_actionable" in result["reason"]


def test_unsupported_asset_rejected(monkeypatch, tmp_path):
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_SUBMIT", raising=False)
    bot = _fake_executor_minimal(monkeypatch, tmp_path)
    plan = _fake_plan(asset="DOGE")
    result = bot.submit(plan, mode="paper")
    assert result["submitted"] is False
    assert "unsupported_asset" in result["reason"]


def test_paper_submit_fires_record_entry(monkeypatch, tmp_path):
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_SUBMIT", raising=False)
    bot = _fake_executor_minimal(monkeypatch, tmp_path)
    result = bot.submit(_fake_plan(), mode="paper")
    assert result["submitted"] is True
    assert result["asset"] == "BTC"
    assert result["direction"] == "LONG"
    bot._paper.record_entry.assert_called_once()
    call_kwargs = bot._paper.record_entry.call_args.kwargs
    assert call_kwargs["asset"] == "BTC"
    assert call_kwargs["direction"] == "LONG"


def test_real_mode_requires_real_capital_flag(monkeypatch, tmp_path):
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_SUBMIT", raising=False)
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    bot = _fake_executor_minimal(monkeypatch, tmp_path)
    result = bot.submit(_fake_plan(), mode="real")
    assert result["submitted"] is False
    assert result["reason"] == "real_capital_flag_unset"


def test_cost_gate_rejects_sub_margin_trades(monkeypatch, tmp_path):
    """A plan with tiny expected PnL gets rejected on cost grounds."""
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_SUBMIT", raising=False)
    bot = _fake_executor_minimal(monkeypatch, tmp_path)
    # expected_pnl_pct_range max = 0.5% — below Robinhood's 1.69% spread
    plan = _fake_plan(expected_pnl_pct_range=(0.1, 0.5))
    result = bot.submit(plan, mode="paper")
    assert result["submitted"] is False
    assert "cost_reject" in result["reason"] or "cost" in result["reason"].lower()


def test_warm_store_writes_non_shadow_decision(monkeypatch, tmp_path):
    """On successful submit, decision_id is 'agentic-*' not 'shadow-*'."""
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_SUBMIT", raising=False)
    captured = {}
    def _capture(args):
        captured["decision_id"] = args.get("decision_id")
        captured["source"] = args.get("component_signals", {}).get("source")
    monkeypatch.setattr(
        "src.orchestration.warm_store.get_store",
        lambda: MagicMock(write_decision=_capture),
    )

    # Rebuild bot with the new patched store
    import importlib
    ex_mod = importlib.import_module("src.trading.executor")
    submit_method = ex_mod.TradingExecutor.submit_trade_plan

    bot = SimpleNamespace()
    bot._ex_tag = "TEST"
    bot._paper = MagicMock(equity=100000.0)

    submit_method(bot, _fake_plan(), mode="paper")
    assert captured.get("decision_id", "").startswith("agentic-")
    assert captured.get("source") == "agentic_brain_submit"
