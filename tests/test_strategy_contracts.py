"""
Strategy contract tests — pytest gates encoding profitability + safety claims.

Run:
    python -m pytest tests/test_strategy_contracts.py -v

Each test maps to a contract that must hold for the bot to be deploy-safe.
Failures point at concrete regressions: drawdown blow-out, Sharpe collapse,
lookahead bias, missing spread filter, position sizing too aggressive,
shorts leaking through the Robinhood paper gate.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# 2.16-year span (789 days) of BTC 15-minute bars — verified at module load.
TWO_YEAR_PARQUET = DATA_DIR / "BTCUSDT-15m.parquet"

# ── Profitability + safety thresholds ─────────────────────────────────
SHARPE_FLOOR = 1.5
MAX_DD_CEILING_PCT = 15.0
LOOKAHEAD_SHARPE_DROP = 1.0     # shuffled Sharpe must be at least this much lower
KELLY_HALFCAP = 0.5             # full-Kelly is too aggressive; half-Kelly cap


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def two_year_ohlcv() -> Dict[str, List[float]]:
    """Load BTC 15-minute parquet that spans ≥ 2 years."""
    if not TWO_YEAR_PARQUET.exists():
        pytest.skip(f"{TWO_YEAR_PARQUET} not on disk; can't run 2y backtest")
    df = pd.read_parquet(TWO_YEAR_PARQUET).sort_values("timestamp").reset_index(drop=True)
    span_days = (
        pd.to_datetime(df["timestamp"].iloc[-1], unit="ms")
        - pd.to_datetime(df["timestamp"].iloc[0], unit="ms")
    ).days
    assert span_days >= 720, f"data span {span_days}d < 2 years; pick a different parquet"
    return {
        "opens": df["open"].astype(float).tolist(),
        "highs": df["high"].astype(float).tolist(),
        "lows": df["low"].astype(float).tolist(),
        "closes": df["close"].astype(float).tolist(),
        "volumes": df["volume"].astype(float).tolist(),
        "span_days": span_days,
        "bars": len(df),
    }


@pytest.fixture(scope="module")
def baseline_backtest(two_year_ohlcv):
    """Run the rules-based EMA(8) crossover backtest once; share across tests."""
    from src.backtesting.engine import BacktestEngine
    eng = BacktestEngine(
        ema_period=8, atr_period=14, atr_stop_mult=1.5, min_score=3,
        initial_capital=10000.0,
    )
    return eng.run(
        opens=two_year_ohlcv["opens"],
        highs=two_year_ohlcv["highs"],
        lows=two_year_ohlcv["lows"],
        closes=two_year_ohlcv["closes"],
        volumes=two_year_ohlcv["volumes"],
    )


# ── Contract 1 — Sharpe > 1.5 on 2-year backtest ──────────────────────

def test_sharpe_above_1_5(baseline_backtest, two_year_ohlcv):
    sharpe = baseline_backtest.sharpe_ratio
    assert sharpe > SHARPE_FLOOR, (
        f"2y BTC 15m Sharpe = {sharpe:.2f}, contract requires > {SHARPE_FLOOR}. "
        f"trades={baseline_backtest.total_trades}, "
        f"pnl={baseline_backtest.total_pnl_pct:+.1f}%, "
        f"span={two_year_ohlcv['span_days']}d"
    )


# ── Contract 2 — Max drawdown < 15% ───────────────────────────────────

def test_max_drawdown_below_15pct(baseline_backtest):
    dd = baseline_backtest.max_drawdown_pct
    assert dd < MAX_DD_CEILING_PCT, (
        f"2y BTC max drawdown = {dd:.2f}%, contract requires < {MAX_DD_CEILING_PCT}%."
    )


# ── Contract 3 — No lookahead bias (shuffled future ⇒ Sharpe craters) ──

def test_no_lookahead_bias(two_year_ohlcv, baseline_backtest):
    """If the strategy honestly uses only past data, shuffling future bars in
    lockstep should cause Sharpe to collapse. A strategy that survives shuffle
    is peeking at the future."""
    from src.backtesting.engine import BacktestEngine

    rng = random.Random(42)
    n = len(two_year_ohlcv["closes"])
    cut = int(n * 0.7)  # keep first 70% intact, shuffle the last 30%
    indices = list(range(cut, n))
    rng.shuffle(indices)

    def shuf(seq):
        return seq[:cut] + [seq[i] for i in indices]

    eng = BacktestEngine(
        ema_period=8, atr_period=14, atr_stop_mult=1.5, min_score=3,
        initial_capital=10000.0,
    )
    shuffled = eng.run(
        opens=shuf(two_year_ohlcv["opens"]),
        highs=shuf(two_year_ohlcv["highs"]),
        lows=shuf(two_year_ohlcv["lows"]),
        closes=shuf(two_year_ohlcv["closes"]),
        volumes=shuf(two_year_ohlcv["volumes"]),
    )

    drop = baseline_backtest.sharpe_ratio - shuffled.sharpe_ratio
    assert drop > LOOKAHEAD_SHARPE_DROP, (
        f"Shuffled Sharpe ({shuffled.sharpe_ratio:.2f}) is within {drop:.2f} of "
        f"baseline ({baseline_backtest.sharpe_ratio:.2f}). A strategy without "
        f"lookahead should drop by > {LOOKAHEAD_SHARPE_DROP}."
    )


# ── Contract 4 — Spread filter active (Robinhood hard gate) ───────────

def _build_fake_exec(spread_pct: float = 1.69):
    """Minimal duck-typed self for _robinhood_hard_gate."""
    class FakeExec:
        _round_trip_spread = spread_pct
        config = {"risk": {"atr_tp_mult": 25.0}}
        _ex_tag = "TEST"
        _paper_mode = True
    return FakeExec()


def test_spread_filter_active(monkeypatch):
    """Gate must veto entries whose expected ATR move is below 1.5× spread.

    NOTE: the gate intentionally softens to advisory mode in paper. Contract
    being tested here is the real-capital safety net, so we set
    ACT_REAL_CAPITAL_ENABLED=1 for the duration of the test.
    """
    monkeypatch.setenv("ACT_REAL_CAPITAL_ENABLED", "1")
    from src.trading.executor import TradingExecutor
    fake = _build_fake_exec(spread_pct=1.69)

    # ATR move = 25 * atr / price * 100. With atr=10, price=70000 -> 0.36%.
    # 1.5 × 1.69% = 2.535%. 0.36% < 2.535% -> must veto.
    proceed, reason = TradingExecutor._robinhood_hard_gate(
        fake,
        asset="BTC", action="LONG", confidence=0.85, risk_score=3,
        trade_quality=8, entry_score=6, price=70_000.0, atr=10.0,
    )
    assert proceed is False, f"Spread filter let a tiny-ATR LONG through: reason={reason!r}"
    assert "ATR" in reason or "spread" in reason.lower(), (
        f"Veto reason should mention ATR/spread; got: {reason!r}"
    )


# ── Contract 5 — Position sizing respects Kelly (half-Kelly cap) ──────

def test_position_size_respects_kelly():
    """Half-Kelly is the operational cap. For any inputs:
      - negative EV ⇒ kelly = 0
      - positive EV ⇒ 0 < half_kelly ≤ KELLY_HALFCAP × full_kelly
      - half_kelly ≤ KELLY_HALFCAP for any realistic edge
    """
    from src.risk.position_sizing import kelly_criterion, half_kelly

    # 1. Negative-EV (low WR + bad payoff) → kelly is 0
    k_bad = kelly_criterion(win_prob=0.30, win_loss_ratio=0.5)
    assert k_bad == 0.0, f"Negative-EV setup returned positive Kelly: {k_bad}"

    # 2. Positive-EV: half-Kelly = 0.5 × full Kelly
    k_full = kelly_criterion(win_prob=0.55, win_loss_ratio=2.0)
    k_half = half_kelly(win_prob=0.55, win_loss_ratio=2.0)
    assert k_full > 0, "Positive-EV setup returned 0 Kelly"
    assert math.isclose(k_half, 0.5 * k_full, rel_tol=1e-9), (
        f"half_kelly ({k_half:.4f}) is not 0.5 × full_kelly ({k_full:.4f})"
    )

    # 3. Half-Kelly is bounded — even on an extreme edge, don't blow past 50% of capital
    k_half_extreme = half_kelly(win_prob=0.80, win_loss_ratio=5.0)
    assert k_half_extreme <= KELLY_HALFCAP, (
        f"half_kelly on extreme edge = {k_half_extreme:.3f} > cap {KELLY_HALFCAP}"
    )


# ── Contract 6 — Long-only bias on Robinhood paper ────────────────────

def test_long_only_bias_enforced(monkeypatch):
    """The Robinhood hard gate must veto SHORT regardless of confidence
    (real-capital safety net; paper softens to advisory by design)."""
    monkeypatch.setenv("ACT_REAL_CAPITAL_ENABLED", "1")
    from src.trading.executor import TradingExecutor
    fake = _build_fake_exec(spread_pct=1.69)
    # Plenty of ATR so the spread filter wouldn't fire — only SHORT block can veto.
    proceed, reason = TradingExecutor._robinhood_hard_gate(
        fake,
        asset="BTC", action="SHORT", confidence=0.95, risk_score=3,
        trade_quality=9, entry_score=8, price=70_000.0, atr=2_000.0,
    )
    assert proceed is False, (
        f"SHORT was not vetoed in real-capital mode. reason={reason!r}"
    )
    assert "SHORT" in reason or "short" in reason.lower(), (
        f"Veto reason should reference SHORT; got: {reason!r}"
    )
