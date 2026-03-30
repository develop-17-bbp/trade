"""
EMA Crossover Strategy Backtester
==================================
Tests LONG-ONLY EMA crossover on historical parquet data.
Reports: trades, win rate, P&L, drawdown, optimal EMA period.

Usage: python backtest_ema.py
"""

import numpy as np
import pandas as pd
from src.indicators.indicators import ema as compute_ema, atr as compute_atr


def backtest_ema(closes, highs, lows, ema_period=8,
                 position_size_pct=5.0, initial_capital=100000.0,
                 spread_pct=0.15, max_giveback_pct=0.30):
    """
    Backtest EMA crossover: LONG ONLY.
    Entry: EMA crosses prev candle + current candle ABOVE EMA + EMA RISING
    Exit: EMA reversal OR trailing SL hit (30% max giveback)
    """
    n = len(closes)
    ema_vals = compute_ema(list(closes), ema_period)

    capital = initial_capital
    position = None
    trades = []
    equity_curve = [capital]

    for i in range(2, n):
        price = float(closes[i])
        ema_now = ema_vals[i]
        ema_prev = ema_vals[i - 1]

        if np.isnan(ema_now) or np.isnan(ema_prev):
            equity_curve.append(capital + (position['qty'] * price if position else 0))
            continue

        if position is not None:
            position['peak'] = max(position['peak'], price)
            entry = position['entry_price']
            sl = position['sl']

            # SL hit
            if price <= sl:
                spread = price * spread_pct / 100
                pnl = (price - entry - spread) * position['qty']
                capital += position['qty'] * price - spread * position['qty']
                trades.append({
                    'entry': entry, 'exit': price, 'pnl': pnl,
                    'pnl_pct': (price - entry) / entry * 100,
                    'reason': f"SL (L{len(position['sl_levels'])})",
                    'bars': i - position['bar'],
                })
                position = None
                equity_curve.append(capital)
                continue

            # Trail SL (30% max giveback)
            peak_profit = position['peak'] - entry
            if peak_profit > 0:
                new_sl = position['peak'] - peak_profit * max_giveback_pct
                if new_sl > sl:
                    position['sl'] = new_sl
                    position['sl_levels'].append(new_sl)

            # EMA reversal exit
            prev_cross = (ema_prev <= float(highs[i - 1]) and ema_prev >= float(lows[i - 1]))
            if prev_cross and float(highs[i]) < ema_now and ema_now < ema_prev:
                spread = price * spread_pct / 100
                pnl = (price - entry - spread) * position['qty']
                capital += position['qty'] * price - spread * position['qty']
                trades.append({
                    'entry': entry, 'exit': price, 'pnl': pnl,
                    'pnl_pct': (price - entry) / entry * 100,
                    'reason': 'EMA reversal',
                    'bars': i - position['bar'],
                })
                position = None
                equity_curve.append(capital)
                continue

            unrealized = (price - entry) * position['qty']
            equity_curve.append(capital + unrealized)
            continue

        # Entry: CALL (LONG only)
        prev_cross = (ema_prev <= float(highs[i - 1]) and ema_prev >= float(lows[i - 1]))
        candle_above = float(lows[i]) > ema_now
        ema_rising = ema_now > ema_prev
        price_above_ema = price > ema_now

        if prev_cross and candle_above and ema_rising and price_above_ema:
            spread = price * spread_pct / 100
            pos_value = capital * (position_size_pct / 100)
            qty = pos_value / (price + spread)

            # SL at recent low
            recent_lows = [float(l) for l in lows[max(0, i - 20):i]]
            initial_sl = min(recent_lows) * 0.999 if recent_lows else price * 0.98

            position = {
                'entry_price': price, 'qty': qty, 'sl': initial_sl,
                'sl_levels': [initial_sl], 'peak': price, 'bar': i,
            }
            capital -= pos_value

        equity_curve.append(capital + (position['qty'] * price if position else 0))

    # Close open position
    if position:
        price = float(closes[-1])
        pnl = (price - position['entry_price']) * position['qty']
        capital += position['qty'] * price
        trades.append({
            'entry': position['entry_price'], 'exit': price, 'pnl': pnl,
            'pnl_pct': (price - position['entry_price']) / position['entry_price'] * 100,
            'reason': 'End', 'bars': n - position['bar'],
        })

    return trades, equity_curve, capital


def analyze(name, trades, equity_curve, initial=100000):
    """Print analysis."""
    if not trades:
        print(f"  {name}: No trades")
        return {}

    total_pnl = sum(t['pnl'] for t in trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    wr = len(wins) / len(trades) * 100

    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    avg_bars = np.mean([t['bars'] for t in trades])

    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)

    pf = abs(sum(t['pnl'] for t in wins)) / abs(sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else float('inf')

    return {
        'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'win_rate': wr, 'total_pnl': total_pnl, 'return_pct': total_pnl / initial * 100,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'profit_factor': pf,
        'max_drawdown': max_dd, 'avg_bars': avg_bars,
    }


def main():
    print("=" * 70)
    print("  EMA CROSSOVER BACKTEST — LONG ONLY (Alpaca Crypto)")
    print("=" * 70)

    datasets = {
        'BTC 15m': 'data/BTCUSDT-15m.parquet',
        'ETH 15m': 'data/ETHUSDT-15m.parquet',
        'BTC 1h': 'data/BTCUSDT-1h.parquet',
        'ETH 1h': 'data/ETHUSDT-1h.parquet',
    }

    for label, path in datasets.items():
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"\n  {label}: Error loading - {e}")
            continue

        closes = df['close'].values.astype(float)
        highs = df['high'].values.astype(float)
        lows = df['low'].values.astype(float)
        n = len(closes)

        print(f"\n{'='*70}")
        print(f"  {label}: {n:,} candles | ${closes[0]:,.2f} to ${closes[-1]:,.2f}")
        print(f"{'='*70}")

        # Test multiple EMA periods
        print(f"\n  {'EMA':>5} {'Trades':>7} {'WinR':>6} {'P&L':>12} {'Return':>8} {'MaxDD':>7} {'PF':>6} {'AvgBars':>8}")
        print(f"  {'-'*5} {'-'*7} {'-'*6} {'-'*12} {'-'*8} {'-'*7} {'-'*6} {'-'*8}")

        best_return = -999
        best_period = 8

        for period in [5, 8, 10, 13, 15, 20, 25, 30]:
            trades, eq, final = backtest_ema(closes, highs, lows, ema_period=period, spread_pct=0.15)
            r = analyze(f"EMA({period})", trades, eq)
            if not r:
                continue

            ret = r['return_pct']
            if ret > best_return:
                best_return = ret
                best_period = period

            print(f"  EMA({period:2d}) {r['trades']:>7d} {r['win_rate']:>5.1f}% ${r['total_pnl']:>+10,.2f} {r['return_pct']:>+7.2f}% {r['max_drawdown']:>6.2f}% {r['profit_factor']:>5.2f} {r['avg_bars']:>7.1f}")

        print(f"\n  BEST: EMA({best_period}) = {best_return:+.2f}% return")

        # Show detailed trades for best period (last 20 trades)
        trades, eq, final = backtest_ema(closes, highs, lows, ema_period=best_period, spread_pct=0.15)
        if trades:
            print(f"\n  Last 15 trades (EMA {best_period}):")
            print(f"  {'#':>3} {'Entry':>10} {'Exit':>10} {'P&L':>10} {'P&L%':>7} {'Bars':>5}  Reason")
            for j, t in enumerate(trades[-15:]):
                print(f"  {j+1:3d} ${t['entry']:>9,.2f} ${t['exit']:>9,.2f} ${t['pnl']:>+9,.2f} {t['pnl_pct']:>+6.2f}% {t['bars']:>5d}  {t['reason']}")

    # Summary recommendation
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION")
    print(f"{'='*70}")
    print(f"  Run the live system ONLY when backtest shows positive return.")
    print(f"  If best EMA period has negative return, the strategy may not")
    print(f"  work in current market conditions (downtrend = LONG-only loses).")
    print(f"  Consider waiting for market to turn bullish before trading.")


if __name__ == '__main__':
    main()
