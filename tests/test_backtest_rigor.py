"""Tests for the 2026-frontier backtesting additions:
  - Deflated Sharpe Ratio (Bailey-López de Prado 2014)
  - Probability of Backtest Overfitting (Bailey et al. 2017)
  - Purged Walk-Forward (López de Prado 2017)
  - Realistic slippage model (size + vol + latency + session)
"""
from __future__ import annotations

import math
import random

import pytest


# ── Deflated Sharpe Ratio ────────────────────────────────────────────────


def test_dsr_handles_too_few_returns():
    from src.backtesting.overfitting_metrics import deflated_sharpe
    r = deflated_sharpe([0.01, 0.02])
    assert "too_few" in r.sample_warning
    assert r.observed_sharpe == 0.0


def test_dsr_high_n_trials_penalizes_sharpe():
    """Same returns, more trials → lower deflated_sharpe."""
    from src.backtesting.overfitting_metrics import deflated_sharpe
    rng = random.Random(42)
    returns = [rng.gauss(0.0008, 0.012) for _ in range(252)]  # 1 year daily
    r1 = deflated_sharpe(returns, n_trials=1)
    r10 = deflated_sharpe(returns, n_trials=10)
    r100 = deflated_sharpe(returns, n_trials=100)
    assert r10.deflated_sharpe < r1.deflated_sharpe
    assert r100.deflated_sharpe < r10.deflated_sharpe


def test_dsr_probability_bounded_0_to_1():
    from src.backtesting.overfitting_metrics import deflated_sharpe
    rng = random.Random(7)
    returns = [rng.gauss(0.001, 0.015) for _ in range(100)]
    r = deflated_sharpe(returns, n_trials=20)
    assert 0.0 <= r.probability_true_sharpe_positive <= 1.0


# ── Probability of Backtest Overfitting ──────────────────────────────────


def test_pbo_handles_empty_input():
    from src.backtesting.overfitting_metrics import probability_of_backtest_overfitting
    r = probability_of_backtest_overfitting([])
    assert r.pbo == 0.5
    assert "empty" in r.sample_warning


def test_pbo_handles_single_strategy():
    from src.backtesting.overfitting_metrics import probability_of_backtest_overfitting
    r = probability_of_backtest_overfitting([[0.01] * 100])
    assert "at_least_2" in r.sample_warning


def test_pbo_returns_bounded_probability():
    from src.backtesting.overfitting_metrics import probability_of_backtest_overfitting
    rng = random.Random(13)
    matrix = [
        [rng.gauss(0.0005, 0.015) for _ in range(64)]
        for _ in range(8)
    ]
    r = probability_of_backtest_overfitting(matrix)
    assert 0.0 <= r.pbo <= 1.0
    assert r.n_strategies == 8
    assert r.n_periods == 64
    assert r.n_combinations_used > 0


# ── Purged Walk-Forward ──────────────────────────────────────────────────


def test_purged_wf_handles_short_series():
    from src.backtesting.purged_walk_forward import purged_walk_forward
    r = purged_walk_forward([0.01] * 50, lambda rets: rets)
    assert "insufficient" in r.sample_warning


def test_purged_wf_runs_5_folds():
    from src.backtesting.purged_walk_forward import purged_walk_forward
    rng = random.Random(99)
    returns = [rng.gauss(0.0, 0.015) for _ in range(500)]
    # Simple strategy: identity (returns themselves)
    r = purged_walk_forward(returns, strategy_fn=lambda rets: rets, n_folds=5)
    assert r.n_folds >= 3
    assert len(r.folds) >= 3
    for f in r.folds:
        assert f.embargo_size > 0
        assert f.purged_indices >= 0


def test_purged_wf_overfit_indicator_present():
    from src.backtesting.purged_walk_forward import purged_walk_forward
    rng = random.Random(101)
    returns = [rng.gauss(0.001, 0.015) for _ in range(400)]
    r = purged_walk_forward(returns, strategy_fn=lambda rets: rets, n_folds=4)
    # Result has overfit indicator
    assert hasattr(r, "overfit_indicator")
    assert isinstance(r.aggregate_train_sharpe, float)
    assert isinstance(r.aggregate_test_sharpe, float)


# ── Realistic slippage ───────────────────────────────────────────────────


def test_slippage_floor_at_base_spread():
    """Slippage estimate must NEVER be less than the venue's base spread."""
    from src.backtesting.realistic_slippage import estimate_slippage
    r = estimate_slippage(venue="robinhood", size_pct_of_equity=0.1,
                          volatility_pct=0.1, latency_ms=10, session="US")
    assert r.expected_slippage_pct >= r.base_spread_pct


def test_slippage_size_factor_bounded():
    from src.backtesting.realistic_slippage import estimate_slippage, SIZE_FACTOR_MAX
    # Massive size — factor should saturate
    r = estimate_slippage(size_pct_of_equity=20.0)
    assert r.size_factor <= SIZE_FACTOR_MAX


def test_slippage_volatility_factor_bounded():
    from src.backtesting.realistic_slippage import estimate_slippage, VOL_FACTOR_MAX
    r = estimate_slippage(volatility_pct=50.0)
    assert r.volatility_factor <= VOL_FACTOR_MAX


def test_slippage_latency_factor_bounded():
    from src.backtesting.realistic_slippage import estimate_slippage, LATENCY_FACTOR_MAX
    r = estimate_slippage(latency_ms=2000.0)
    assert r.latency_factor <= LATENCY_FACTOR_MAX


def test_slippage_robinhood_higher_than_bybit():
    """Sanity: robinhood spread > bybit (1.69% vs 0.055%)."""
    from src.backtesting.realistic_slippage import estimate_slippage
    rh = estimate_slippage(venue="robinhood")
    by = estimate_slippage(venue="bybit")
    assert rh.expected_slippage_pct > by.expected_slippage_pct


def test_slippage_upper_bound_above_expected():
    from src.backtesting.realistic_slippage import estimate_slippage
    r = estimate_slippage(venue="robinhood")
    assert r.upper_bound_pct > r.expected_slippage_pct
    assert r.confidence_band_pct > 0


# ── Tool registration smoke ─────────────────────────────────────────────


def test_4_backtest_rigor_tools_registered():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    names = set(r.list_names())
    expected = {
        "query_deflated_sharpe",
        "query_probability_of_backtest_overfitting",
        "query_purged_walk_forward",
        "query_realistic_slippage",
    }
    missing = expected - names
    assert not missing, f"missing: {missing}"


def test_realistic_slippage_tool_dispatches():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    res = r.dispatch("query_realistic_slippage",
                     {"venue": "robinhood", "size_pct_of_equity": 2.0,
                      "volatility_pct": 1.5, "latency_ms": 200,
                      "session": "US"})
    assert "expected_slippage_pct" in str(res)


def test_dsr_tool_dispatches_with_returns():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    rng = random.Random(3)
    returns = [rng.gauss(0.0008, 0.012) for _ in range(252)]
    res = r.dispatch("query_deflated_sharpe",
                     {"returns": returns, "n_trials": 5})
    assert "deflated_sharpe" in str(res)
