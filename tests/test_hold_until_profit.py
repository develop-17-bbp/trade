"""Tests for ACT_HOLD_UNTIL_PROFIT paper-mode guard.

When ACT_HOLD_UNTIL_PROFIT=1 in paper mode, record_exit refuses to
close a position at net-negative PnL UNLESS the reason includes a
catastrophic-trigger keyword.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def _fresh_fetcher(tmp_path: Path):
    from src.data.robinhood_fetcher import RobinhoodPaperFetcher
    cfg = {"log_dir": str(tmp_path), "initial_capital": 10000}
    f = RobinhoodPaperFetcher(config=cfg)
    return f


def _open_position(f, asset="BTC", direction="LONG", price=76000.0,
                    qty=0.001, current_price=None):
    pos = f.record_entry(
        asset=asset, direction=direction, price=price,
        score=5, quantity=qty, sl_price=price * 0.98,
        tp_price=price * 1.05,
        ml_confidence=0.5, llm_confidence=0.5, size_pct=1.0,
        reasoning="test entry",
    )
    if pos and current_price is not None:
        pos.current_price = current_price
    return pos


def test_hold_until_profit_refuses_negative_close(monkeypatch, tmp_path):
    """Default reason 'sl_hit' is NOT catastrophic → close should be refused."""
    monkeypatch.setenv("ACT_HOLD_UNTIL_PROFIT", "1")
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    f = _fresh_fetcher(tmp_path)
    p = _open_position(f, current_price=74000.0)  # underwater
    assert p is not None
    # current_pnl_pct will be negative (price dropped + spread)
    closed = f.record_exit(asset="BTC", reason="sl_hit")
    # Refused — position still open
    assert closed is None
    assert len(f.positions) == 1


def test_hold_until_profit_allows_catastrophic_close(monkeypatch, tmp_path):
    """When reason contains a catastrophic keyword, exit is allowed."""
    monkeypatch.setenv("ACT_HOLD_UNTIL_PROFIT", "1")
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    f = _fresh_fetcher(tmp_path)
    _open_position(f, current_price=74000.0)
    closed = f.record_exit(
        asset="BTC", reason="thesis_broken: macro flipped to bearish",
    )
    # Allowed — position closed
    assert closed is not None
    assert len(f.positions) == 0


def test_hold_until_profit_allows_positive_close(monkeypatch, tmp_path):
    """Net-positive closes always allowed regardless of reason."""
    monkeypatch.setenv("ACT_HOLD_UNTIL_PROFIT", "1")
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    f = _fresh_fetcher(tmp_path)
    _open_position(f, current_price=80000.0)  # well above entry + spread
    closed = f.record_exit(asset="BTC", reason="brain_take_profit")
    assert closed is not None
    assert closed.final_pnl_pct > 0


def test_hold_until_profit_inactive_when_flag_unset(monkeypatch, tmp_path):
    """No flag → existing behavior (allows negative close)."""
    monkeypatch.delenv("ACT_HOLD_UNTIL_PROFIT", raising=False)
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    f = _fresh_fetcher(tmp_path)
    _open_position(f, current_price=74000.0)
    closed = f.record_exit(asset="BTC", reason="sl_hit")
    # Allowed — flag is off
    assert closed is not None


def test_hold_until_profit_ignored_in_real_capital(monkeypatch, tmp_path):
    """Real-capital mode ignores the flag — always honors SL."""
    monkeypatch.setenv("ACT_HOLD_UNTIL_PROFIT", "1")
    monkeypatch.setenv("ACT_REAL_CAPITAL_ENABLED", "1")
    f = _fresh_fetcher(tmp_path)
    _open_position(f, current_price=74000.0)
    closed = f.record_exit(asset="BTC", reason="sl_hit")
    # Allowed — real capital mode bypasses
    assert closed is not None


def test_hold_until_profit_keywords_case_insensitive(monkeypatch, tmp_path):
    """Catastrophic keyword matching is case-insensitive."""
    monkeypatch.setenv("ACT_HOLD_UNTIL_PROFIT", "1")
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    f = _fresh_fetcher(tmp_path)
    _open_position(f, current_price=74000.0)
    closed = f.record_exit(asset="BTC", reason="REGIME_CRISIS confirmed")
    assert closed is not None


def test_hold_until_profit_writes_refusal_to_tick_state(monkeypatch, tmp_path):
    """When the guard refuses, last_refusal lands in tick_state for next tick."""
    monkeypatch.setenv("ACT_HOLD_UNTIL_PROFIT", "1")
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    f = _fresh_fetcher(tmp_path)
    _open_position(f, current_price=74000.0)
    f.record_exit(asset="BTC", reason="sl_hit")
    from src.ai import tick_state as ts
    snap = ts.get("BTC")
    assert "HOLD_UNTIL_PROFIT" in str(snap.get("last_refusal", ""))
