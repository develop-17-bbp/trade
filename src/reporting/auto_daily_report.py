"""
Auto Daily Report Generator
==============================
Generates daily trading reports from the JSONL trade journal.
Run: python -m src.reporting.auto_daily_report

Outputs 3 formats in logs/daily_reports/:
  - YYYY-MM-DD_report.md   (full markdown for authorities)
  - YYYY-MM-DD_report.csv  (data for Excel)
  - YYYY-MM-DD_summary.txt (quick 1-page overview)
"""

import os
import sys
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

JOURNAL_PATH = os.path.join(PROJECT_ROOT, 'logs', 'trading_journal.jsonl')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'logs', 'daily_reports')


def load_trades(date_str: str = None) -> List[Dict]:
    """Load trades from JSONL journal, optionally filtered by date."""
    if not os.path.exists(JOURNAL_PATH):
        print(f"No journal at {JOURNAL_PATH}")
        return []

    trades = []
    with open(JOURNAL_PATH, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                if date_str:
                    ts = t.get('timestamp', t.get('time', ''))[:10]
                    if ts != date_str:
                        continue
                trades.append(t)
            except json.JSONDecodeError:
                pass
    return trades


def generate_report(date_str: str = None):
    """Generate all 3 report formats."""
    if not date_str:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')

    os.makedirs(REPORTS_DIR, exist_ok=True)
    trades = load_trades(date_str)

    if not trades:
        # Try all trades if date filter finds nothing
        trades = load_trades()
        if trades:
            print(f"No trades on {date_str}, generating report for all {len(trades)} trades")
            date_str = "ALL"
        else:
            print("No trades found in journal.")
            return

    # Calculate metrics
    total = len(trades)
    wins = sum(1 for t in trades if float(t.get('pnl_usd', t.get('pnl', 0))) > 0)
    losses = total - wins
    win_rate = wins / total if total > 0 else 0

    pnls = [float(t.get('pnl_usd', t.get('pnl', 0))) for t in trades]
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / total if total > 0 else 0

    winning_pnls = [p for p in pnls if p > 0]
    losing_pnls = [p for p in pnls if p < 0]
    avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
    avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0
    profit_factor = abs(sum(winning_pnls) / sum(losing_pnls)) if losing_pnls and sum(losing_pnls) != 0 else float('inf')

    best_trade = max(pnls) if pnls else 0
    worst_trade = min(pnls) if pnls else 0

    # Per-asset breakdown
    assets = {}
    for t in trades:
        a = t.get('asset', 'UNKNOWN')
        if a not in assets:
            assets[a] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
        assets[a]['trades'] += 1
        pnl = float(t.get('pnl_usd', t.get('pnl', 0)))
        assets[a]['pnl'] += pnl
        if pnl > 0:
            assets[a]['wins'] += 1

    # Per-direction breakdown
    longs = [t for t in trades if t.get('action', '').upper() in ('LONG', 'BUY')]
    shorts = [t for t in trades if t.get('action', '').upper() in ('SHORT', 'SELL')]

    # ── MARKDOWN REPORT ──
    md_path = os.path.join(REPORTS_DIR, f'{date_str}_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Daily Trading Report - {date_str}\n\n")
        f.write(f"## Summary\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total Trades | {total} |\n")
        f.write(f"| Wins / Losses | {wins}W / {losses}L |\n")
        f.write(f"| Win Rate | {win_rate:.1%} |\n")
        f.write(f"| Total P&L | ${total_pnl:+,.2f} |\n")
        f.write(f"| Avg P&L per Trade | ${avg_pnl:+,.2f} |\n")
        f.write(f"| Avg Win | ${avg_win:+,.2f} |\n")
        f.write(f"| Avg Loss | ${avg_loss:+,.2f} |\n")
        f.write(f"| Profit Factor | {profit_factor:.2f} |\n")
        f.write(f"| Best Trade | ${best_trade:+,.2f} |\n")
        f.write(f"| Worst Trade | ${worst_trade:+,.2f} |\n")

        f.write(f"\n## Per-Asset Breakdown\n")
        f.write(f"| Asset | Trades | Wins | Win Rate | P&L |\n")
        f.write(f"|-------|--------|------|----------|-----|\n")
        for a, stats in assets.items():
            wr = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            f.write(f"| {a} | {stats['trades']} | {stats['wins']} | {wr:.0%} | ${stats['pnl']:+,.2f} |\n")

        f.write(f"\n## Direction Breakdown\n")
        f.write(f"- LONG trades: {len(longs)} | P&L: ${sum(float(t.get('pnl_usd', t.get('pnl',0))) for t in longs):+,.2f}\n")
        f.write(f"- SHORT trades: {len(shorts)} | P&L: ${sum(float(t.get('pnl_usd', t.get('pnl',0))) for t in shorts):+,.2f}\n")

        f.write(f"\n## Trade Log\n")
        f.write(f"| Time | Asset | Direction | Entry | Exit | P&L | Reason |\n")
        f.write(f"|------|-------|-----------|-------|------|-----|--------|\n")
        for t in trades[-30:]:
            ts = t.get('timestamp', t.get('time', ''))[:19]
            asset = t.get('asset', '?')
            direction = t.get('action', '?')
            entry = float(t.get('entry_price', 0))
            exit_p = float(t.get('exit_price', 0))
            pnl = float(t.get('pnl_usd', t.get('pnl', 0)))
            reason = t.get('exit_reason', t.get('reason', ''))[:30]
            f.write(f"| {ts} | {asset} | {direction} | ${entry:,.2f} | ${exit_p:,.2f} | ${pnl:+,.2f} | {reason} |\n")

        f.write(f"\n---\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")

    # ── CSV EXPORT ──
    csv_path = os.path.join(REPORTS_DIR, f'{date_str}_report.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'asset', 'direction', 'entry_price', 'exit_price',
                         'pnl_usd', 'pnl_pct', 'exit_reason', 'confidence', 'sl_progression'])
        for t in trades:
            writer.writerow([
                t.get('timestamp', ''), t.get('asset', ''), t.get('action', ''),
                t.get('entry_price', ''), t.get('exit_price', ''),
                t.get('pnl_usd', t.get('pnl', '')), t.get('pnl_pct', ''),
                t.get('exit_reason', ''), t.get('confidence', ''),
                t.get('sl_progression', ''),
            ])

    # ── TEXT SUMMARY ──
    txt_path = os.path.join(REPORTS_DIR, f'{date_str}_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"TRADING REPORT - {date_str}\n")
        f.write(f"{'='*40}\n")
        f.write(f"Trades: {total} ({wins}W/{losses}L) | Win Rate: {win_rate:.0%}\n")
        f.write(f"Total P&L: ${total_pnl:+,.2f} | Avg: ${avg_pnl:+,.2f}/trade\n")
        f.write(f"Best: ${best_trade:+,.2f} | Worst: ${worst_trade:+,.2f}\n")
        f.write(f"Profit Factor: {profit_factor:.2f}\n")
        f.write(f"{'='*40}\n")
        for a, stats in assets.items():
            wr = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            f.write(f"{a}: {stats['trades']} trades, {wr:.0%} WR, ${stats['pnl']:+,.2f}\n")

    print(f"\n  [REPORT] Daily report generated for {date_str}:")
    print(f"    Markdown: {md_path}")
    print(f"    CSV:      {csv_path}")
    print(f"    Summary:  {txt_path}")
    print(f"    Stats: {total} trades, {win_rate:.0%} WR, ${total_pnl:+,.2f} P&L")

    return {'md': md_path, 'csv': csv_path, 'txt': txt_path}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate daily trading report')
    parser.add_argument('--date', default=None, help='Date (YYYY-MM-DD), default=today')
    parser.add_argument('--all', action='store_true', help='Report for all trades')
    args = parser.parse_args()

    if args.all:
        generate_report(date_str=None)
    else:
        generate_report(date_str=args.date)
