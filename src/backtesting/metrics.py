"""
Backtest Metrics — Extended Analytics + CSV Export
===================================================
Computes: win rate, profit factor, Sharpe, max drawdown,
SL level distribution, exit reason breakdown, hourly analysis.
"""

import csv
import math
from datetime import datetime, timezone
from typing import List, Dict
from src.backtesting.position_manager import TradeRecord


class BacktestMetrics:
    """Extended metrics for backtest results."""

    def __init__(self, trades: List[TradeRecord], equity_curve: List[float],
                 initial_capital: float = 100000.0):
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winners(self) -> List[TradeRecord]:
        return [t for t in self.trades if t.pnl_pct > 0]

    @property
    def losers(self) -> List[TradeRecord]:
        return [t for t in self.trades if t.pnl_pct <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.winners) / self.total_trades if self.total_trades > 0 else 0

    @property
    def total_pnl_pct(self) -> float:
        return sum(t.pnl_pct for t in self.trades)

    @property
    def total_pnl_usd(self) -> float:
        return sum(t.pnl_usd for t in self.trades)

    @property
    def avg_win_pct(self) -> float:
        w = self.winners
        return sum(t.pnl_pct for t in w) / len(w) if w else 0

    @property
    def avg_loss_pct(self) -> float:
        l = self.losers
        return sum(t.pnl_pct for t in l) / len(l) if l else 0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_pct for t in self.winners)
        gross_loss = abs(sum(t.pnl_pct for t in self.losers))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    @property
    def avg_duration_min(self) -> float:
        return sum(t.duration_min for t in self.trades) / len(self.trades) if self.trades else 0

    @property
    def avg_winner_duration(self) -> float:
        w = self.winners
        return sum(t.duration_min for t in w) / len(w) if w else 0

    @property
    def avg_loser_duration(self) -> float:
        l = self.losers
        return sum(t.duration_min for t in l) / len(l) if l else 0

    def max_drawdown(self) -> float:
        """Maximum drawdown from equity curve (%)."""
        if not self.equity_curve:
            return 0
        peak = self.equity_curve[0]
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Annualized Sharpe ratio from trade returns."""
        if len(self.trades) < 2:
            return 0
        returns = [t.pnl_pct for t in self.trades]
        avg_ret = sum(returns) / len(returns)
        std_ret = (sum((r - avg_ret) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
        if std_ret == 0:
            return 0
        # Annualize: assume ~250 trading days, ~50 trades/day
        trades_per_year = max(1, len(self.trades) * 365 / max(1, self._span_days()))
        return (avg_ret - risk_free_rate) / std_ret * math.sqrt(trades_per_year)

    def sortino_ratio(self) -> float:
        """Sortino ratio (downside deviation only)."""
        if len(self.trades) < 2:
            return 0
        returns = [t.pnl_pct for t in self.trades]
        avg_ret = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return float('inf')
        downside_std = (sum(r ** 2 for r in downside) / len(downside)) ** 0.5
        if downside_std == 0:
            return 0
        trades_per_year = max(1, len(self.trades) * 365 / max(1, self._span_days()))
        return avg_ret / downside_std * math.sqrt(trades_per_year)

    def _span_days(self) -> float:
        if len(self.trades) < 2:
            return 1
        first_ts = self.trades[0].entry_ts
        last_ts = self.trades[-1].exit_ts
        return max(1, (last_ts - first_ts) / 86400000)

    def sl_level_distribution(self) -> Dict[str, int]:
        """Distribution of max SL levels reached."""
        dist = {}
        for t in self.trades:
            lvl = t.max_sl_level
            dist[lvl] = dist.get(lvl, 0) + 1
        return dict(sorted(dist.items()))

    def exit_reason_breakdown(self) -> Dict[str, dict]:
        """Breakdown by exit reason: count, win rate, avg P&L."""
        reasons = {}
        for t in self.trades:
            # Normalize reason
            reason = t.exit_reason.split('(')[0].strip()
            if reason not in reasons:
                reasons[reason] = {'count': 0, 'wins': 0, 'total_pnl': 0}
            reasons[reason]['count'] += 1
            if t.pnl_pct > 0:
                reasons[reason]['wins'] += 1
            reasons[reason]['total_pnl'] += t.pnl_pct

        for r in reasons:
            c = reasons[r]['count']
            reasons[r]['win_rate'] = reasons[r]['wins'] / c if c > 0 else 0
            reasons[r]['avg_pnl'] = reasons[r]['total_pnl'] / c if c > 0 else 0
        return reasons

    def direction_breakdown(self) -> Dict[str, dict]:
        """LONG vs SHORT performance."""
        dirs = {}
        for d in ['LONG', 'SHORT']:
            trades = [t for t in self.trades if t.direction == d]
            wins = [t for t in trades if t.pnl_pct > 0]
            dirs[d] = {
                'count': len(trades),
                'wins': len(wins),
                'win_rate': len(wins) / len(trades) if trades else 0,
                'total_pnl': sum(t.pnl_pct for t in trades),
                'avg_pnl': sum(t.pnl_pct for t in trades) / len(trades) if trades else 0,
            }
        return dirs

    def hourly_win_rate(self) -> Dict[int, dict]:
        """Win rate by UTC hour."""
        hours = {}
        for t in self.trades:
            dt = datetime.fromtimestamp(t.entry_ts / 1000, tz=timezone.utc)
            h = dt.hour
            if h not in hours:
                hours[h] = {'count': 0, 'wins': 0, 'total_pnl': 0}
            hours[h]['count'] += 1
            if t.pnl_pct > 0:
                hours[h]['wins'] += 1
            hours[h]['total_pnl'] += t.pnl_pct
        for h in hours:
            c = hours[h]['count']
            hours[h]['win_rate'] = hours[h]['wins'] / c if c > 0 else 0
        return dict(sorted(hours.items()))

    def max_consecutive_wins(self) -> int:
        max_streak = 0
        streak = 0
        for t in self.trades:
            if t.pnl_pct > 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    def max_consecutive_losses(self) -> int:
        max_streak = 0
        streak = 0
        for t in self.trades:
            if t.pnl_pct <= 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    def summary(self) -> str:
        """Console-formatted summary."""
        lines = []
        lines.append("=" * 65)
        lines.append("  BACKTEST RESULTS")
        lines.append("=" * 65)
        lines.append(f"  Total Trades:      {self.total_trades}")
        lines.append(f"  Winners:           {len(self.winners)} ({self.win_rate:.1%})")
        lines.append(f"  Losers:            {len(self.losers)} ({1-self.win_rate:.1%})")
        lines.append(f"  Total P&L:         {self.total_pnl_pct:+.2f}% (${self.total_pnl_usd:+,.2f})")
        lines.append(f"  Avg Win:           {self.avg_win_pct:+.2f}%")
        lines.append(f"  Avg Loss:          {self.avg_loss_pct:+.2f}%")
        lines.append(f"  Profit Factor:     {self.profit_factor:.2f}")
        lines.append(f"  Sharpe Ratio:      {self.sharpe_ratio():.2f}")
        lines.append(f"  Sortino Ratio:     {self.sortino_ratio():.2f}")
        lines.append(f"  Max Drawdown:      {self.max_drawdown():.2f}%")
        lines.append(f"  Avg Duration:      {self.avg_duration_min:.1f} min")
        lines.append(f"  Avg Winner Dur:    {self.avg_winner_duration:.1f} min")
        lines.append(f"  Avg Loser Dur:     {self.avg_loser_duration:.1f} min")
        lines.append(f"  Max Win Streak:    {self.max_consecutive_wins()}")
        lines.append(f"  Max Loss Streak:   {self.max_consecutive_losses()}")

        lines.append("")
        lines.append("  SL Level Distribution:")
        for lvl, count in self.sl_level_distribution().items():
            pct = count / self.total_trades * 100 if self.total_trades > 0 else 0
            bar = "#" * int(pct / 2)
            lines.append(f"    {lvl:8s}: {count:4d} ({pct:5.1f}%) {bar}")

        lines.append("")
        lines.append("  Exit Reasons:")
        for reason, stats in self.exit_reason_breakdown().items():
            lines.append(f"    {reason:30s}: {stats['count']:4d} trades | WR={stats['win_rate']:.1%} | avg={stats['avg_pnl']:+.2f}%")

        lines.append("")
        lines.append("  Direction:")
        for d, stats in self.direction_breakdown().items():
            lines.append(f"    {d:6s}: {stats['count']:4d} trades | WR={stats['win_rate']:.1%} | total={stats['total_pnl']:+.2f}%")

        lines.append("=" * 65)
        return "\n".join(lines)

    def to_csv(self, path: str):
        """Export trade-by-trade CSV."""
        if not self.trades:
            return

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'trade_id', 'direction', 'entry_time', 'exit_time',
                'entry_price', 'exit_price', 'pnl_pct', 'pnl_usd',
                'duration_min', 'max_sl_level', 'exit_reason', 'entry_score',
            ])
            for i, t in enumerate(self.trades, 1):
                entry_dt = datetime.fromtimestamp(t.entry_ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
                exit_dt = datetime.fromtimestamp(t.exit_ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
                writer.writerow([
                    i, t.direction, entry_dt, exit_dt,
                    f"{t.entry_price:.2f}", f"{t.exit_price:.2f}",
                    f"{t.pnl_pct:.4f}", f"{t.pnl_usd:.2f}",
                    f"{t.duration_min:.1f}", t.max_sl_level,
                    t.exit_reason, t.entry_score,
                ])
        print(f"  [CSV] {len(self.trades)} trades exported to {path}")
