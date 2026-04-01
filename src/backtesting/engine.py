"""
Backtesting Engine — EMA(8) Crossover Strategy
================================================
Simulates the exact trading logic from executor.py on historical data.
No lookahead bias: uses confirmed candles only (closes[-2]).
Computes: Sharpe, Sortino, max drawdown, win rate, profit factor.

Usage:
    python -m src.backtesting.engine --asset BTC --days 30
    python -m src.backtesting.engine --asset ETH --days 90 --ema 8 --atr-mult 1.5
"""

import json
import math
import argparse
from datetime import datetime
from typing import List, Dict, Optional

from src.indicators.indicators import ema, atr


class BacktestResult:
    """Container for backtest metrics."""
    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.params: Dict = {}

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t['pnl_pct'] > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t['pnl_pct'] <= 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0

    @property
    def total_pnl_pct(self) -> float:
        return sum(t['pnl_pct'] for t in self.trades)

    @property
    def avg_win(self) -> float:
        wins = [t['pnl_pct'] for t in self.trades if t['pnl_pct'] > 0]
        return sum(wins) / len(wins) if wins else 0

    @property
    def avg_loss(self) -> float:
        losses = [t['pnl_pct'] for t in self.trades if t['pnl_pct'] <= 0]
        return sum(losses) / len(losses) if losses else 0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t['pnl_pct'] for t in self.trades if t['pnl_pct'] > 0)
        gross_loss = abs(sum(t['pnl_pct'] for t in self.trades if t['pnl_pct'] <= 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0
        peak = self.equity_curve[0]
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe (assumes 5m bars, 252 trading days)."""
        returns = []
        for i in range(1, len(self.equity_curve)):
            r = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            returns.append(r)
        if len(returns) < 2:
            return 0
        avg_r = sum(returns) / len(returns)
        std_r = math.sqrt(sum((r - avg_r)**2 for r in returns) / (len(returns) - 1))
        if std_r == 0:
            return 0
        # Annualize: 288 5m bars per day * 252 days
        periods_per_year = 288 * 252
        return (avg_r / std_r) * math.sqrt(periods_per_year)

    @property
    def sortino_ratio(self) -> float:
        """Annualized Sortino (downside deviation only)."""
        returns = []
        for i in range(1, len(self.equity_curve)):
            r = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            returns.append(r)
        if len(returns) < 2:
            return 0
        avg_r = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return float('inf')
        downside_std = math.sqrt(sum(r**2 for r in downside) / len(downside))
        if downside_std == 0:
            return 0
        periods_per_year = 288 * 252
        return (avg_r / downside_std) * math.sqrt(periods_per_year)

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"  BACKTEST RESULTS -- {self.params.get('asset', '?')}",
            f"{'='*60}",
            f"  Period: {self.params.get('candles', 0)} candles ({self.params.get('days', 0)} days)",
            f"  EMA: {self.params.get('ema_period', 8)} | ATR mult: {self.params.get('atr_stop_mult', 1.5)}",
            f"{'-'*60}",
            f"  Total Trades:    {self.total_trades}",
            f"  Win Rate:        {self.win_rate:.1%} ({self.wins}W / {self.losses}L)",
            f"  Total P&L:       {self.total_pnl_pct:+.2f}%",
            f"  Avg Win:         {self.avg_win:+.2f}%",
            f"  Avg Loss:        {self.avg_loss:+.2f}%",
            f"  Profit Factor:   {self.profit_factor:.2f}",
            f"  Max Drawdown:    {self.max_drawdown_pct:.2f}%",
            f"  Sharpe Ratio:    {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio:   {self.sortino_ratio:.2f}",
            f"{'-'*60}",
        ]

        # Best/worst trades
        if self.trades:
            best = max(self.trades, key=lambda t: t['pnl_pct'])
            worst = min(self.trades, key=lambda t: t['pnl_pct'])
            lines.append(f"  Best Trade:      {best['pnl_pct']:+.2f}% ({best['direction']} {best['exit_reason']})")
            lines.append(f"  Worst Trade:     {worst['pnl_pct']:+.2f}% ({worst['direction']} {worst['exit_reason']})")

            # SL progression distribution
            sl_levels = [len(t.get('sl_levels', ['L1'])) for t in self.trades]
            avg_levels = sum(sl_levels) / len(sl_levels)
            max_levels = max(sl_levels)
            lines.append(f"  Avg SL Levels:   {avg_levels:.1f} (max: L{max_levels})")

        lines.append(f"{'='*60}")
        return '\n'.join(lines)


class BacktestEngine:
    """
    Simulates EMA(8) crossover with trailing SL (L1→Ln).
    Uses CONFIRMED candle logic — no lookahead.
    """

    def __init__(self, ema_period: int = 8, atr_period: int = 14,
                 atr_stop_mult: float = 1.5, min_score: int = 3,
                 initial_capital: float = 10000.0,
                 max_hold_bars: int = 72,  # 72 * 5min = 6 hours
                 overextension_pct: float = 10.0):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.min_score = min_score
        self.initial_capital = initial_capital
        self.max_hold_bars = max_hold_bars
        self.overextension_pct = overextension_pct

    def run(self, opens: List[float], highs: List[float],
            lows: List[float], closes: List[float],
            volumes: List[float]) -> BacktestResult:
        """Run backtest on OHLCV data. Returns BacktestResult."""

        result = BacktestResult()
        result.params = {
            'ema_period': self.ema_period,
            'atr_stop_mult': self.atr_stop_mult,
            'candles': len(closes),
            'days': len(closes) / 288,
        }

        if len(closes) < 30:
            return result

        ema_vals = ema(closes, self.ema_period)
        atr_vals = atr(highs, lows, closes, self.atr_period)

        equity = self.initial_capital
        result.equity_curve = [equity]

        position = None  # {direction, entry, sl, sl_levels, peak, bar_entered}

        # Start from bar 20 (need history for indicators)
        for bar in range(20, len(closes) - 1):
            # Use confirmed candle (bar) not current incomplete (bar+1)
            price = closes[bar]
            current_ema = ema_vals[bar]
            prev_ema = ema_vals[bar - 1] if bar > 0 else current_ema
            current_atr = atr_vals[bar] if bar < len(atr_vals) else price * 0.01
            ema_direction = "RISING" if current_ema > prev_ema else "FALLING"

            if current_atr <= 0:
                current_atr = price * 0.01

            # ── MANAGE EXISTING POSITION ──
            if position is not None:
                direction = position['direction']
                entry = position['entry']
                sl = position['sl']
                peak = position['peak']
                bars_held = bar - position['bar_entered']

                # Update peak
                if direction == 'LONG' and price > peak:
                    position['peak'] = price
                    peak = price
                elif direction == 'SHORT' and price < peak:
                    position['peak'] = price
                    peak = price

                # P&L
                if direction == 'LONG':
                    pnl_pct = ((price - entry) / entry) * 100
                else:
                    pnl_pct = ((entry - price) / entry) * 100

                # Hard stop -2%
                if pnl_pct <= -2.0:
                    trade = self._close(position, price, bar, "Hard stop -2%")
                    result.trades.append(trade)
                    equity *= (1 + trade['pnl_pct'] / 100)
                    position = None
                    result.equity_curve.append(equity)
                    continue

                # Time-based exit
                if bars_held >= self.max_hold_bars and pnl_pct < 1.0:
                    trade = self._close(position, price, bar, f"Time exit ({bars_held} bars)")
                    result.trades.append(trade)
                    equity *= (1 + trade['pnl_pct'] / 100)
                    position = None
                    result.equity_curve.append(equity)
                    continue

                # SL hit (confirmed candle)
                sl_hit = False
                if direction == 'LONG' and price <= sl:
                    sl_hit = True
                elif direction == 'SHORT' and price >= sl:
                    sl_hit = True

                if sl_hit:
                    trade = self._close(position, price, bar, f"SL {position['sl_levels'][-1]} hit")
                    result.trades.append(trade)
                    equity *= (1 + trade['pnl_pct'] / 100)
                    position = None
                    result.equity_curve.append(equity)
                    continue

                # Trailing SL phases
                new_sl = sl
                if direction == 'LONG':
                    if pnl_pct >= 0.5 and sl < entry:
                        new_sl = entry
                    if pnl_pct >= 1.5:
                        if pnl_pct >= 10: protect, amult = 0.70, 0.8
                        elif pnl_pct >= 5: protect, amult = 0.60, 1.0
                        elif pnl_pct >= 3: protect, amult = 0.50, 1.2
                        else: protect, amult = 0.40, 1.5
                        floor_sl = entry + (peak - entry) * protect
                        atr_sl = peak - current_atr * amult
                        best = max(floor_sl, atr_sl)
                        if best > new_sl and best < price:
                            new_sl = best
                    # Swing low tightening at 2%+
                    if pnl_pct >= 2.0:
                        lookback = min(15, bar)
                        for i in range(bar - lookback + 1, bar - 1):
                            if i > 0 and i < len(lows) - 1:
                                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                                    if lows[i] > entry and lows[i] > new_sl and lows[i] < price:
                                        new_sl = lows[i]

                else:  # SHORT
                    if pnl_pct >= 0.5 and sl > entry:
                        new_sl = entry
                    if pnl_pct >= 1.5:
                        if pnl_pct >= 10: protect, amult = 0.70, 0.8
                        elif pnl_pct >= 5: protect, amult = 0.60, 1.0
                        elif pnl_pct >= 3: protect, amult = 0.50, 1.2
                        else: protect, amult = 0.40, 1.5
                        floor_sl = entry - (entry - peak) * protect
                        atr_sl = peak + current_atr * amult
                        best = min(floor_sl, atr_sl)
                        if best < new_sl and best > price:
                            new_sl = best
                    if pnl_pct >= 2.0:
                        lookback = min(15, bar)
                        for i in range(bar - lookback + 1, bar - 1):
                            if i > 0 and i < len(highs) - 1:
                                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                                    if highs[i] < entry and highs[i] < new_sl and highs[i] > price:
                                        new_sl = highs[i]

                # SL only moves forward
                min_move = max(0.50, entry * 0.0001)
                if direction == 'LONG' and new_sl > sl + min_move:
                    position['sl'] = new_sl
                    position['sl_levels'].append(f"L{len(position['sl_levels']) + 1}")
                elif direction == 'SHORT' and new_sl < sl - min_move:
                    position['sl'] = new_sl
                    position['sl_levels'].append(f"L{len(position['sl_levels']) + 1}")

                # EMA reversal exit (E1) at 5%+ profit
                if pnl_pct >= 5.0 and bar >= 2:
                    confirmed_ema = ema_vals[bar]
                    prev_confirmed_ema = ema_vals[bar - 1]
                    if direction == 'LONG' and confirmed_ema < prev_confirmed_ema and price < confirmed_ema:
                        trade = self._close(position, price, bar, "EMA reversal E1")
                        result.trades.append(trade)
                        equity *= (1 + trade['pnl_pct'] / 100)
                        position = None
                        result.equity_curve.append(equity)
                        continue
                    elif direction == 'SHORT' and confirmed_ema > prev_confirmed_ema and price > confirmed_ema:
                        trade = self._close(position, price, bar, "EMA reversal E1")
                        result.trades.append(trade)
                        equity *= (1 + trade['pnl_pct'] / 100)
                        position = None
                        result.equity_curve.append(equity)
                        continue

                result.equity_curve.append(equity)
                continue  # Position open, no new entry

            # ── SIGNAL GENERATION (confirmed candles only) ──
            signal = "NEUTRAL"
            ema_crossed = False
            for i in range(bar, max(bar - 3, 0), -1):
                if lows[i] <= ema_vals[i] <= highs[i]:
                    ema_crossed = True
                    break

            # Price momentum override
            if bar >= 3:
                c1, c2, c3 = closes[bar], closes[bar-1], closes[bar-2]
                price_falling = (c1 < c2 and c1 < c3)
                price_rising = (c1 > c2 and c1 > c3)
            else:
                price_falling = price_rising = False

            # Overextension
            ema_sep = abs(price - current_ema) / current_ema * 100 if current_ema > 0 else 0
            if ema_sep > self.overextension_pct:
                signal = "NEUTRAL"
            elif ema_direction == "RISING" and price > current_ema and ema_crossed and not price_falling:
                signal = "BUY"
            elif ema_direction == "FALLING" and price < current_ema and ema_crossed and not price_rising:
                signal = "SELL"
            elif ema_direction == "FALLING" and price < current_ema * 0.99 and not price_rising:
                signal = "SELL"
            elif ema_direction == "RISING" and price > current_ema * 1.01 and not price_falling:
                signal = "BUY"

            if signal == "NEUTRAL":
                result.equity_curve.append(equity)
                continue

            # Pattern scoring
            score = self._compute_score(bar, ema_vals, closes, ema_direction, signal)
            if score < self.min_score:
                result.equity_curve.append(equity)
                continue

            # ── ENTRY ──
            direction = 'LONG' if signal == 'BUY' else 'SHORT'
            sl_dist = current_atr * self.atr_stop_mult
            sl_dist = max(sl_dist, price * 0.003)
            sl_dist = min(sl_dist, price * 0.02)

            if direction == 'LONG':
                sl_price = price - sl_dist
            else:
                sl_price = price + sl_dist

            position = {
                'direction': direction,
                'entry': price,
                'sl': sl_price,
                'sl_levels': ['L1'],
                'peak': price,
                'bar_entered': bar,
                'score': score,
            }

            result.equity_curve.append(equity)

        # Close any open position at end
        if position is not None:
            trade = self._close(position, closes[-1], len(closes) - 1, "End of data")
            result.trades.append(trade)
            equity *= (1 + trade['pnl_pct'] / 100)
            result.equity_curve.append(equity)

        result.params['asset'] = 'BACKTEST'
        return result

    def _close(self, position: dict, price: float, bar: int, reason: str) -> dict:
        entry = position['entry']
        direction = position['direction']
        if direction == 'LONG':
            pnl_pct = ((price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - price) / entry) * 100
        return {
            'direction': direction,
            'entry': entry,
            'exit': price,
            'pnl_pct': pnl_pct,
            'sl_levels': position['sl_levels'],
            'bars_held': bar - position['bar_entered'],
            'exit_reason': reason,
            'score': position.get('score', 0),
        }

    def _compute_score(self, bar: int, ema_vals: list, closes: list,
                       ema_direction: str, signal: str) -> int:
        score = 0
        if bar < 5 or bar >= len(ema_vals):
            return 0

        # EMA slope
        ema_slope = abs(ema_vals[bar] - ema_vals[bar-2]) / ema_vals[bar-2] * 100 if ema_vals[bar-2] > 0 else 0
        if ema_slope > 0.3: score += 3
        elif ema_slope > 0.1: score += 2
        elif ema_slope > 0.03: score += 1

        # Consecutive direction
        consec = 0
        for i in range(bar, max(0, bar - 10), -1):
            if i > 0:
                if ema_direction == "RISING" and ema_vals[i] > ema_vals[i-1]: consec += 1
                elif ema_direction == "FALLING" and ema_vals[i] < ema_vals[i-1]: consec += 1
                else: break
        if consec >= 5: score += 3
        elif consec >= 3: score += 2
        elif consec >= 2: score += 1

        # Separation
        sep = abs(closes[bar] - ema_vals[bar]) / ema_vals[bar] * 100 if ema_vals[bar] > 0 else 0
        if sep > 0.5: score += 2
        elif sep > 0.2: score += 1

        # Candle momentum
        if bar >= 3:
            if signal == "BUY" and closes[bar] > closes[bar-1] > closes[bar-2]: score += 2
            elif signal == "BUY" and closes[bar] > closes[bar-1]: score += 1
            elif signal == "SELL" and closes[bar] < closes[bar-1] < closes[bar-2]: score += 2
            elif signal == "SELL" and closes[bar] < closes[bar-1]: score += 1

        return score


def load_ohlcv_from_journal(asset: str = 'BTC') -> tuple:
    """Load OHLCV from a CSV or fetch from exchange for backtesting."""
    try:
        from src.data.fetcher import PriceFetcher
        pf = PriceFetcher(exchange_name='bybit', testnet=False)
        symbol = f"{asset}/USDT:USDT"
        raw = pf.fetch_ohlcv(symbol, timeframe='5m', limit=1000)
        ohlcv = PriceFetcher.extract_ohlcv(raw)
        return ohlcv['opens'], ohlcv['highs'], ohlcv['lows'], ohlcv['closes'], ohlcv['volumes']
    except Exception as e:
        print(f"  Failed to fetch live data: {e}")
        print("  Generating synthetic data for demonstration...")
        import random
        random.seed(42)
        n = 1000
        price = 68000.0
        opens, highs, lows, closes, volumes = [], [], [], [], []
        for _ in range(n):
            o = price
            change = random.gauss(0, 0.002) * price
            c = o + change
            h = max(o, c) + abs(random.gauss(0, 0.001) * price)
            l = min(o, c) - abs(random.gauss(0, 0.001) * price)
            v = random.uniform(50, 500)
            opens.append(o); highs.append(h); lows.append(l); closes.append(c); volumes.append(v)
            price = c
        return opens, highs, lows, closes, volumes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest EMA crossover strategy')
    parser.add_argument('--asset', default='BTC', help='Asset to backtest')
    parser.add_argument('--ema', type=int, default=8, help='EMA period')
    parser.add_argument('--atr-mult', type=float, default=1.5, help='ATR stop multiplier')
    parser.add_argument('--min-score', type=int, default=3, help='Minimum pattern score')
    parser.add_argument('--max-hold', type=int, default=72, help='Max hold bars (72=6hrs)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    args = parser.parse_args()

    print(f"Loading {args.asset} 5m data...")
    opens, highs, lows, closes, volumes = load_ohlcv_from_journal(args.asset)
    print(f"  Got {len(closes)} candles")

    engine = BacktestEngine(
        ema_period=args.ema,
        atr_stop_mult=args.atr_mult,
        min_score=args.min_score,
        initial_capital=args.capital,
        max_hold_bars=args.max_hold,
    )

    result = engine.run(opens, highs, lows, closes, volumes)
    result.params['asset'] = args.asset
    result.params['days'] = len(closes) / 288
    print(result.summary())

    # Print trade log
    print(f"\n  Trade Log ({len(result.trades)} trades):")
    for i, t in enumerate(result.trades):
        sl_chain = '->'.join(t['sl_levels'])
        print(f"  #{i+1:3d} {t['direction']:5s} entry=${t['entry']:,.2f} exit=${t['exit']:,.2f} "
              f"P&L={t['pnl_pct']:+.2f}% bars={t['bars_held']:3d} {sl_chain:20s} {t['exit_reason']}")
