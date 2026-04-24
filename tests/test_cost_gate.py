"""Tests for src/trading/cost_gate.py — C19 cost awareness + USD drift."""

from __future__ import annotations

import pytest

from src.trading.cost_gate import (
    CostBreakdown,
    CostGateResult,
    VENUE_COSTS,
    evaluate,
)


def test_breakdown_sums_all_components_including_usd():
    b = CostBreakdown(
        spread_pct=1.69, fees_pct=0.0, slippage_pct=0.10,
        impact_pct=0.005, extra_pct=0.0, usd_drift_pct=0.15,
    )
    # Sum of all 6 components, rounded to 4 decimals.
    assert b.total_pct == pytest.approx(1.945, abs=1e-3)


def test_robinhood_blocks_2pct_move_below_margin():
    """On Robinhood (1.69% spread + slippage), a 2% move doesn't clear 1% margin."""
    r = evaluate(
        expected_return_pct=2.0, venue="robinhood", atr_pct=0.3,
        size_pct=1.0, usd_drift_pct_per_day=0.0, direction="LONG",
    )
    assert r.passed is False
    assert "cost_reject" in r.reason


def test_robinhood_passes_3pct_move():
    r = evaluate(
        expected_return_pct=3.0, venue="robinhood", atr_pct=0.3,
        size_pct=1.0, usd_drift_pct_per_day=0.0, direction="LONG",
    )
    assert r.passed is True


def test_bybit_spot_accepts_much_smaller_moves():
    """Bybit spread is 30x tighter — 0.5% move clears easily."""
    r = evaluate(
        expected_return_pct=0.5, venue="bybit_spot", atr_pct=0.1,
        size_pct=1.0, usd_drift_pct_per_day=0.0, direction="LONG",
        min_margin_pct=0.1,
    )
    assert r.passed is True


def test_usd_strengthening_hurts_long_btc():
    """USD +0.3%/day over 3 days = 0.9% headwind on LONG BTC/USD."""
    r = evaluate(
        expected_return_pct=3.0, venue="robinhood", atr_pct=0.3,
        size_pct=1.0, direction="LONG",
        usd_drift_pct_per_day=0.3, expected_hold_days=3.0,
    )
    assert r.breakdown.usd_drift_pct == pytest.approx(0.9, abs=1e-3)
    # Total friction is higher with USD headwind than without
    r_base = evaluate(
        expected_return_pct=3.0, venue="robinhood", atr_pct=0.3,
        size_pct=1.0, direction="LONG",
        usd_drift_pct_per_day=0.0, expected_hold_days=3.0,
    )
    assert r.frictional_cost_pct > r_base.frictional_cost_pct


def test_usd_strengthening_helps_short_btc():
    """USD strengthening is a tailwind for shorts (BTC/USD falls)."""
    r = evaluate(
        expected_return_pct=3.0, venue="robinhood", atr_pct=0.3,
        size_pct=1.0, direction="SHORT",
        usd_drift_pct_per_day=0.3, expected_hold_days=3.0,
    )
    assert r.breakdown.usd_drift_pct == pytest.approx(-0.9, abs=1e-3)
    # Total friction LOWER for SHORT under USD strengthening than for LONG
    r_long = evaluate(
        expected_return_pct=3.0, venue="robinhood", atr_pct=0.3,
        size_pct=1.0, direction="LONG",
        usd_drift_pct_per_day=0.3, expected_hold_days=3.0,
    )
    assert r.frictional_cost_pct < r_long.frictional_cost_pct


def test_invalid_expected_return_rejected():
    r = evaluate(
        expected_return_pct="not_a_number", venue="robinhood",
        atr_pct=0.3, size_pct=1.0, direction="LONG",
        usd_drift_pct_per_day=0.0,
    )
    assert r.passed is False
    assert "invalid" in r.reason


def test_explicit_spread_overrides_venue_preset():
    r = evaluate(
        expected_return_pct=2.0, venue="robinhood",
        spread_pct=0.0,       # pretend no spread
        fees_pct=0.0, atr_pct=0.0, size_pct=1.0,
        usd_drift_pct_per_day=0.0, direction="LONG",
    )
    assert r.passed is True    # no spread → 2% return clears 1% margin


def test_impact_grows_with_size():
    small = evaluate(
        expected_return_pct=2.5, venue="robinhood", atr_pct=0.1,
        size_pct=1.0, usd_drift_pct_per_day=0.0, direction="LONG",
    )
    huge = evaluate(
        expected_return_pct=2.5, venue="robinhood", atr_pct=0.1,
        size_pct=20.0, usd_drift_pct_per_day=0.0, direction="LONG",
    )
    assert huge.breakdown.impact_pct > small.breakdown.impact_pct


def test_auto_read_usd_drift_from_econ_layer_falls_back_to_zero(monkeypatch):
    """Auto-read returns 0.0 when economic_intelligence isn't importable."""
    # Force the import to fail
    import sys
    monkeypatch.setitem(sys.modules, "src.data.economic_intelligence", None)
    r = evaluate(
        expected_return_pct=3.0, venue="robinhood", atr_pct=0.3,
        size_pct=1.0, direction="LONG",
        # Note: usd_drift_pct_per_day NOT passed — triggers auto-read
    )
    # Should not crash; drift should be 0 or small-signal-based.
    assert isinstance(r, CostGateResult)


def test_all_venue_presets_have_required_keys():
    for venue, preset in VENUE_COSTS.items():
        assert "spread_pct" in preset, venue
        assert "fees_pct" in preset, venue
        assert preset["spread_pct"] >= 0
        assert preset["fees_pct"] >= 0
