"""Tests for the 5 strategy additions:
  - liquidity_sweep
  - pair_trading_signal (via tool only)
  - session_bias
  - grid_chop
  - wyckoff_phase

Each must:
  1. Return bounded structured output
  2. Handle degenerate inputs gracefully (no exceptions)
  3. Anti-overfit: rule-based, not parameter-learning
  4. Default to safe / inactive when conditions don't apply
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timezone

import pytest


# ── Liquidity Sweep ─────────────────────────────────────────────────────


def _make_sweep_pattern():
    """Synthetic price series with a clear high-sweep reversal."""
    # 40 bars trending sideways, then a wick spikes ABOVE the swing
    # high and closes back inside.
    closes = [100.0] * 30 + [100.5, 100.8, 101.0, 100.7, 101.2]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    # Make bar -7 the swing high to be swept
    highs[-7] = 102.5
    # Last 3 bars: wick above swing high, then close below
    highs[-3] = 103.0  # wick poked above swing high (102.5)
    closes[-3] = 102.3  # closed back below
    closes[-1] = 101.5
    volumes = [1000.0] * 35 + [2000.0] * 5  # spike on sweep
    return highs, lows, closes, volumes


def test_liquidity_sweep_detects_high_sweep_reversal():
    from src.trading.strategies.liquidity_sweep import detect_liquidity_sweep
    h, l, c, v = _make_sweep_pattern()
    det = detect_liquidity_sweep(h, l, c, v, lookback=20)
    assert det.detected
    assert det.direction == "SHORT"
    assert det.swept_level > 0
    assert 0.0 <= det.reversal_strength <= 1.0
    assert det.confluence_count >= 2
    assert len(det.rationale) <= 300


def test_liquidity_sweep_no_pattern():
    """Flat data → no sweep detected."""
    from src.trading.strategies.liquidity_sweep import detect_liquidity_sweep
    closes = [100.0] * 50
    highs = [100.5] * 50
    lows = [99.5] * 50
    volumes = [1000.0] * 50
    det = detect_liquidity_sweep(highs, lows, closes, volumes)
    assert det.detected is False


def test_liquidity_sweep_handles_degenerate_input():
    from src.trading.strategies.liquidity_sweep import detect_liquidity_sweep
    # Too few bars
    det = detect_liquidity_sweep([1, 2], [1, 2], [1, 2], [1, 2])
    assert det.detected is False
    assert "insufficient" in det.rationale.lower()


# ── Session Bias ─────────────────────────────────────────────────────────


def test_session_bias_returns_session_for_us_hour():
    from src.trading.strategies.session_bias import current_session
    res = current_session(datetime(2026, 4, 28, 18, 0, tzinfo=timezone.utc))
    assert res["session"] == "US"
    assert res["volume_share"] >= 0.5
    assert res["conviction_multiplier"] == 1.0


def test_session_bias_multiplier_bounded_below_one():
    """Multiplier must NEVER exceed 1.0 — anti-overfit constraint."""
    from src.trading.strategies.session_bias import current_session, SESSION_RANGES
    for _name, _start, _end, _share, mult in SESSION_RANGES:
        assert 0.0 <= mult <= 1.0, f"session multiplier {mult} > 1.0 (overfit risk)"


def test_session_bias_asia_is_lowest():
    from src.trading.strategies.session_bias import current_session
    res = current_session(datetime(2026, 4, 28, 4, 0, tzinfo=timezone.utc))
    assert res["session"] == "ASIA"
    assert res["conviction_multiplier"] < 0.7


def test_session_bias_active_trading_hour():
    from src.trading.strategies.session_bias import is_active_trading_hour
    # US session
    assert is_active_trading_hour(datetime(2026, 4, 28, 18, 0, tzinfo=timezone.utc))
    # Late Asia (low volume)
    assert not is_active_trading_hour(
        datetime(2026, 4, 28, 4, 0, tzinfo=timezone.utc),
        min_volume_share=0.25,
    )


# ── Grid Chop ────────────────────────────────────────────────────────────


def test_grid_returns_empty_when_not_ranging():
    from src.trading.strategies.grid_chop import compute_grid_levels
    # is_ranging=False → empty
    levels = compute_grid_levels(
        asset="BTC", current_price=77000, atr=200,
        is_ranging=False,
    )
    assert levels == []


def test_grid_returns_empty_on_degenerate_input():
    from src.trading.strategies.grid_chop import compute_grid_levels
    assert compute_grid_levels("BTC", 0, 200, is_ranging=True) == []
    assert compute_grid_levels("BTC", 77000, 0, is_ranging=True) == []


def test_grid_levels_clear_spread():
    """Each rung's expected gain MUST clear spread × 1.5 — anti-loss
    constraint."""
    from src.trading.strategies.grid_chop import compute_grid_levels
    levels = compute_grid_levels(
        asset="BTC", current_price=77000, atr=2000, spread_pct=1.69,
        n_levels=5, is_ranging=True,
    )
    min_required = 1.69 * 1.5
    for lvl in levels:
        assert lvl["expected_pnl_pct"] >= min_required, (
            f"rung pnl {lvl['expected_pnl_pct']} below spread threshold"
        )


def test_grid_advisory_passes_through_regime():
    from src.trading.strategies.grid_chop import grid_advisory
    res = grid_advisory(
        asset="BTC", current_price=77000, atr=2000,
        regime="CHOP", hurst_value=0.40, spread_pct=1.69,
    )
    assert res["is_ranging"] is True
    assert res["regime"] == "CHOP"

    res2 = grid_advisory(
        asset="BTC", current_price=77000, atr=2000,
        regime="TREND_UP", hurst_value=0.65, spread_pct=1.69,
    )
    assert res2["is_ranging"] is False
    assert res2["levels"] == []


# ── Wyckoff Phase ────────────────────────────────────────────────────────


def test_wyckoff_unclear_on_low_sample():
    from src.trading.strategies.wyckoff_phase import detect_phase
    # Fewer than 30 bars → unclear
    v = detect_phase([100] * 10, [101] * 10, [99] * 10, [1000] * 10)
    assert v.phase == "unclear"
    assert v.confidence == 0.0


def test_wyckoff_detects_markup():
    """Synthetic uptrend with rising EMA + higher highs."""
    from src.trading.strategies.wyckoff_phase import detect_phase
    # 60 bars of clean uptrend
    closes = [100 + i * 0.3 for i in range(60)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    volumes = [1000 + i * 5 for i in range(60)]
    v = detect_phase(closes, highs, lows, volumes, lookback=50)
    assert v.phase == "markup"
    assert v.confidence > 0.5


def test_wyckoff_detects_markdown():
    from src.trading.strategies.wyckoff_phase import detect_phase
    closes = [100 - i * 0.3 for i in range(60)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    volumes = [1000 + i * 5 for i in range(60)]
    v = detect_phase(closes, highs, lows, volumes, lookback=50)
    assert v.phase == "markdown"
    assert v.confidence > 0.5


def test_wyckoff_low_sample_warning():
    from src.trading.strategies.wyckoff_phase import detect_phase
    # 35 bars (>= 30 minimum but < 50 — triggers low_sample penalty)
    closes = [100 + i * 0.3 for i in range(35)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    volumes = [1000] * 35
    v = detect_phase(closes, highs, lows, volumes, lookback=35)
    assert "low_sample_warning" in v.factors


def test_wyckoff_output_bounded():
    from src.trading.strategies.wyckoff_phase import detect_phase
    closes = [100 + math.sin(i / 5) for i in range(80)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    volumes = [1000] * 80
    v = detect_phase(closes, highs, lows, volumes)
    d = v.to_dict()
    assert len(d["factors"]) <= 8
    assert len(d["rationale"]) <= 300
    assert 0.0 <= d["confidence"] <= 1.0


# ── Tool registration smoke ─────────────────────────────────────────────


def test_all_5_strategy_tools_registered():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    names = set(r.list_names())
    expected = {
        "query_liquidity_sweep",
        "query_pair_trading_signal",
        "query_session_bias",
        "query_grid_chop",
        "query_wyckoff_phase",
    }
    missing = expected - names
    assert not missing, f"strategy tools not registered: {missing}"


def test_session_bias_tool_dispatches():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    res = r.dispatch("query_session_bias", {})
    # Returns digest string
    assert "session" in str(res).lower()
    assert "conviction_multiplier" in str(res).lower()


def test_pair_trading_tool_returns_action():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    res = r.dispatch("query_pair_trading_signal", {})
    s = str(res).lower()
    assert "action" in s
    assert "z_score" in s
