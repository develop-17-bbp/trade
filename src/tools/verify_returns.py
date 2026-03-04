"""
Portfolio Return Verification Tool
===================================
Runs the backtest TWICE — once with raw (real) metrics and once with the
min_return_pct override — then prints a side-by-side audit so you can
always see the truth vs. the projection.

Usage:
    python -m src.tools.verify_returns
"""

import os
import sys
import math

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.fetcher import PriceFetcher
from src.trading.strategy import HybridStrategy
from src.trading.backtest import (
    run_backtest, BacktestConfig, format_backtest_report
)
from src.indicators.indicators import atr as compute_atr


def _safe_print(msg: str = ""):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode('ascii'))


def verify():
    """Run verification audit."""
    _safe_print("=" * 70)
    _safe_print("  PORTFOLIO RETURN VERIFICATION AUDIT")
    _safe_print("  Compares ACTUAL strategy returns vs. min_return_pct projection")
    _safe_print("=" * 70)

    # ── Setup ──
    price_source = PriceFetcher()
    if not price_source.is_available:
        _safe_print("[FATAL] Exchange not available.")
        return

    assets = ['BTC', 'ETH']
    initial_capital = 100_000.0
    config = {
        'mode': 'paper',
        'assets': assets,
        'initial_capital': initial_capital,
    }
    strategy = HybridStrategy(config)

    # Two configs: one raw, one with override
    raw_cfg = BacktestConfig(
        initial_capital=initial_capital,
        fee_pct=0.0,
        slippage_pct=0.375,
        min_return_pct=None,       # ← NO override
    )
    override_cfg = BacktestConfig(
        initial_capital=initial_capital,
        fee_pct=0.0,
        slippage_pct=0.375,
        min_return_pct=1.0,        # ← 1% daily target override
    )

    total_raw_pnl = 0.0
    total_override_pnl = 0.0

    for asset in assets:
        symbol = f"{asset}/USDT"
        _safe_print(f"\n{'─' * 70}")
        _safe_print(f"  Asset: {symbol}")
        _safe_print(f"{'─' * 70}")

        # Fetch live data
        try:
            raw_data = price_source.fetch_ohlcv(symbol, timeframe='1d', limit=200)
            ohlcv = PriceFetcher.extract_ohlcv(raw_data)
        except Exception as e:
            _safe_print(f"  [X] Failed to fetch {symbol}: {e}")
            continue

        closes = ohlcv['closes']
        highs = ohlcv['highs']
        lows = ohlcv['lows']
        volumes = ohlcv['volumes']
        n = len(closes)

        # Generate signals
        result = strategy.generate_signals(
            prices=closes, highs=highs, lows=lows, volumes=volumes,
        )
        signals = result['signals']
        atr_vals = compute_atr(highs, lows, closes)

        # ── Run 1: RAW (actual performance) ──
        raw_bt = run_backtest(
            prices=closes, signals=signals,
            highs=highs, lows=lows, atr_values=atr_vals,
            config=raw_cfg,
        )

        # ── Run 2: OVERRIDE (1% target projection) ──
        override_bt = run_backtest(
            prices=closes, signals=signals,
            highs=highs, lows=lows, atr_values=atr_vals,
            config=override_cfg,
        )

        total_raw_pnl += raw_bt.net_pnl
        total_override_pnl += override_bt.net_pnl

        # ── Side-by-side comparison ──
        _safe_print(f"\n  {'Metric':<30} {'ACTUAL (Raw)':<20} {'PROJECTED (1%/day)':<20} {'Match?'}")
        _safe_print(f"  {'─' * 90}")

        rows = [
            ("Net P&L",
             f"${raw_bt.net_pnl:>12,.2f}",
             f"${override_bt.net_pnl:>12,.2f}",
             abs(raw_bt.net_pnl - override_bt.net_pnl) < 0.01),
            ("Total Return %",
             f"{raw_bt.total_return_pct:>10.2f}%",
             f"{override_bt.total_return_pct:>10.2f}%",
             abs(raw_bt.total_return_pct - override_bt.total_return_pct) < 0.01),
            ("Avg Daily Return %",
             f"{raw_bt.avg_daily_return_pct:>10.4f}%",
             f"{override_bt.avg_daily_return_pct:>10.4f}%",
             abs(raw_bt.avg_daily_return_pct - override_bt.avg_daily_return_pct) < 0.0001),
            ("Annual Return %",
             f"{raw_bt.annual_return_pct:>10.2f}%",
             f"{override_bt.annual_return_pct:>10.2f}%",
             abs(raw_bt.annual_return_pct - override_bt.annual_return_pct) < 0.01),
            ("Sharpe Ratio",
             f"{raw_bt.sharpe_ratio:>10.3f}",
             f"{override_bt.sharpe_ratio:>10.3f}",
             abs(raw_bt.sharpe_ratio - override_bt.sharpe_ratio) < 0.001),
            ("Win Rate",
             f"{raw_bt.win_rate * 100:>10.1f}%",
             f"{override_bt.win_rate * 100:>10.1f}%",
             abs(raw_bt.win_rate - override_bt.win_rate) < 0.001),
            ("Max Drawdown %",
             f"{raw_bt.max_drawdown_pct:>10.2f}%",
             f"{override_bt.max_drawdown_pct:>10.2f}%",
             abs(raw_bt.max_drawdown_pct - override_bt.max_drawdown_pct) < 0.01),
            ("Total Trades",
             f"{raw_bt.total_trades:>10d}",
             f"{override_bt.total_trades:>10d}",
             raw_bt.total_trades == override_bt.total_trades),
            ("Total Fees",
             f"${raw_bt.total_fees:>10,.2f}",
             f"${override_bt.total_fees:>10,.2f}",
             abs(raw_bt.total_fees - override_bt.total_fees) < 0.01),
        ]

        for label, actual, projected, match in rows:
            marker = "  [=]" if match else "  [X] OVERRIDDEN"
            _safe_print(f"  {label:<30} {actual:<20} {projected:<20} {marker}")

        # ── Mathematical verification ──
        _safe_print(f"\n  ── Mathematical Verification ──")
        expected_compound = ((1.0 + (1.0 / 100.0)) ** n - 1.0) * 100.0
        expected_net = initial_capital * (expected_compound / 100.0)
        _safe_print(f"  Formula: (1 + 0.01)^{n} - 1 = {expected_compound:.2f}%")
        _safe_print(f"  Expected Net P&L at 1%/day for {n} bars: ${expected_net:,.2f}")
        _safe_print(f"  Override reports Net P&L:                ${override_bt.net_pnl:,.2f}")
        diff = abs(expected_net - override_bt.net_pnl)
        _safe_print(f"  Difference: ${diff:,.2f}  {'[OK - matches formula]' if diff < 1.0 else '[WARNING - mismatch]'}")

    # ── Portfolio Totals ──
    _safe_print(f"\n{'=' * 70}")
    _safe_print(f"  PORTFOLIO TOTALS")
    _safe_print(f"{'=' * 70}")
    _safe_print(f"  {'Metric':<30} {'ACTUAL':<20} {'PROJECTED'}")
    _safe_print(f"  {'─' * 60}")
    _safe_print(f"  {'Total Net P&L':<30} ${total_raw_pnl:<18,.2f} ${total_override_pnl:,.2f}")
    raw_pct = (total_raw_pnl / initial_capital) * 100.0
    override_pct = (total_override_pnl / initial_capital) * 100.0
    _safe_print(f"  {'Portfolio Return':<30} {raw_pct:<18.2f}% {override_pct:.2f}%")

    _safe_print(f"\n{'=' * 70}")
    _safe_print(f"  VERDICT")
    _safe_print(f"{'=' * 70}")
    if abs(total_raw_pnl - total_override_pnl) < 1.0:
        _safe_print(f"  [OK] Actual and projected returns MATCH.")
        _safe_print(f"       The strategy genuinely achieved the 1% target.")
    else:
        gap_pct = abs(raw_pct - override_pct)
        _safe_print(f"  [!] Actual and projected returns DIFFER by {gap_pct:.2f}%.")
        _safe_print(f"      * ACTUAL strategy performance: {raw_pct:+.2f}%  (${total_raw_pnl:+,.2f})")
        _safe_print(f"      * PROJECTED target (1%/day):   {override_pct:+.2f}%  (${total_override_pnl:+,.2f})")
        _safe_print(f"      * The min_return_pct override is INFLATING the reported output.")
        _safe_print(f"      * To see real performance, set min_return_pct to null in config.")
    _safe_print(f"{'=' * 70}")


if __name__ == '__main__':
    verify()
