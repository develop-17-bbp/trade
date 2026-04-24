"""Tests for src/learning/reward.py — differential Sharpe ratio (C18a)."""

from __future__ import annotations

import pytest

from src.learning.reward import (
    DEFAULT_ETA,
    DSRState,
    DSRTracker,
    dsr_to_bandit_bonus,
    dsr_to_credit_bonus,
    get_tracker,
    reset_singleton,
)


def test_dsr_warmup_returns_zero():
    """First two updates should return 0 — n<3 gate."""
    st = DSRState(eta=0.01)
    assert st.update(0.01) == 0.0
    assert st.update(0.01) == 0.0
    # n>=3 now — DSR becomes meaningful as soon as EMA variance is nonzero.
    # We don't assert a specific value; just that state advanced.
    st.update(0.01)
    assert st.n == 3


def test_dsr_positive_on_consistent_gains():
    """A stream of gains after some volatility should produce positive DSR values."""
    st = DSRState(eta=0.05)
    # Prime some variance
    for r in [0.01, -0.005, 0.008, -0.003]:
        st.update(r)
    # Now a clean positive streak. DSR is a per-trade risk-adjusted
    # increment and may spike early then decay as the EMA catches up,
    # but every step should be non-negative since every return is
    # above the running mean.
    dsrs = [st.update(r) for r in [0.02, 0.015, 0.018, 0.022]]
    assert all(d >= 0 for d in dsrs), dsrs
    assert max(dsrs) > 0


def test_dsr_negative_on_consistent_losses():
    st = DSRState(eta=0.05)
    for r in [0.01, -0.005, 0.008, -0.003]:
        st.update(r)
    dsrs = [st.update(r) for r in [-0.02, -0.015, -0.018, -0.022]]
    assert all(d <= 0 for d in dsrs[-2:]), dsrs


def test_dsr_clips_extreme_returns():
    """A gigantic return from a data bug shouldn't blow up state."""
    st = DSRState(eta=0.01)
    st.update(0.01)
    st.update(-0.01)
    # 1000x data bug
    st.update(100.0)
    snap = st.snapshot()
    assert snap["last_dsr"] <= 5.0
    assert snap["last_dsr"] >= -5.0
    # A values are clipped input so shouldn't explode either
    assert abs(snap["A"]) < 100.0


def test_dsr_nan_guard():
    st = DSRState(eta=0.01)
    st.update(0.01)
    st.update(-0.01)
    # Garbage NaN input from a division upstream
    result = st.update(float("nan"))
    assert result == 0.0 or (-5 <= result <= 5)


def test_dsr_reset():
    st = DSRState(eta=0.05)
    for r in [0.01, -0.01, 0.02]:
        st.update(r)
    assert st.n == 3
    st.reset()
    assert st.n == 0
    assert st.A == 0.0
    assert st.B == 0.0
    assert st.last_dsr == 0.0


def test_tracker_separates_by_component_and_asset():
    tr = DSRTracker(eta=0.05)
    # Different components + assets keep independent state
    tr.update("scanner", 0.01, asset="BTC")
    tr.update("scanner", 0.02, asset="ETH")
    tr.update("analyst", -0.01, asset="BTC")
    snap = tr.snapshot_all()
    assert "scanner:BTC" in snap
    assert "scanner:ETH" in snap
    assert "analyst:BTC" in snap
    # Scanner:BTC saw +0.01; scanner:ETH saw +0.02 — A should differ
    assert snap["scanner:BTC"]["A"] != snap["scanner:ETH"]["A"]


def test_tracker_get_uninitialized_returns_zero():
    tr = DSRTracker(eta=0.01)
    # Asking for a stream we've never updated is valid — returns 0 default DSR.
    assert tr.get("never_updated", asset="BTC") == 0.0


def test_singleton_reuses_instance():
    reset_singleton()
    t1 = get_tracker()
    t2 = get_tracker()
    assert t1 is t2


def test_singleton_eta_override_after_first_call():
    reset_singleton()
    t1 = get_tracker(eta=0.01)
    t2 = get_tracker(eta=0.05)
    # Same instance, but eta can be bumped on subsequent calls
    assert t1 is t2
    assert t1.eta == 0.05


def test_credit_bonus_bounded():
    # Extreme DSR still produces bounded bonus
    assert abs(dsr_to_credit_bonus(10.0, cap=0.15)) <= 0.15 + 1e-9
    assert abs(dsr_to_credit_bonus(-10.0, cap=0.15)) <= 0.15 + 1e-9
    # Zero DSR → zero bonus
    assert dsr_to_credit_bonus(0.0) == 0.0
    # NaN safe
    assert dsr_to_credit_bonus(float("nan")) == 0.0


def test_bandit_bonus_bounded():
    assert abs(dsr_to_bandit_bonus(10.0, cap=2.0)) <= 2.0 + 1e-9
    assert abs(dsr_to_bandit_bonus(-10.0, cap=2.0)) <= 2.0 + 1e-9
    assert dsr_to_bandit_bonus(0.0) == 0.0


def test_credit_bonus_sign_matches_dsr():
    assert dsr_to_credit_bonus(0.5) > 0
    assert dsr_to_credit_bonus(-0.5) < 0


def test_default_eta_sane():
    # Sanity: default eta should be small (long effective window).
    assert 0 < DEFAULT_ETA < 1.0


def test_tracker_respects_clipping_across_streams():
    tr = DSRTracker(eta=0.05)
    tr.update("s1", 0.01, "BTC")
    tr.update("s1", -0.005, "BTC")
    tr.update("s1", 1000.0, "BTC")   # data-bug scale
    st = tr.state("s1", "BTC")
    snap = st.snapshot()
    assert abs(snap["A"]) < 100.0
    assert -5.0 <= snap["last_dsr"] <= 5.0
