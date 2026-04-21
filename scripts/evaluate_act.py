"""
ACT evaluation report — one-shot CLI view of what's helping vs hurting.

Reads everything the bot has written and prints:
  * Component ON/OFF state (with exact cmd to toggle each)
  * Paper-journal totals + per-bucket attribution
  * Rolling Sharpe trend (last 10 steps)
  * Shadow-log stats (if populated)
  * Recommendations

Usage:
    python scripts/evaluate_act.py
    python scripts/evaluate_act.py --json          # machine-readable
    python scripts/evaluate_act.py --no-colors     # for logs / CI
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)


# ANSI colors (skippable via --no-colors). ASCII-only terminal formatting.
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_RED = "\033[31m"
C_YELLOW = "\033[33m"
C_DIM = "\033[2m"


def _cap(val, n: int) -> str:
    s = str(val if val is not None else "-")
    return s[:n]


def section(title: str, use_color: bool = True) -> None:
    bar = "=" * 70
    if use_color:
        print(f"\n{C_CYAN}{C_BOLD}{bar}{C_RESET}")
        print(f"{C_CYAN}{C_BOLD}  {title}{C_RESET}")
        print(f"{C_CYAN}{C_BOLD}{bar}{C_RESET}")
    else:
        print(f"\n{bar}\n  {title}\n{bar}")


def _render_components(report: dict, use_color: bool) -> None:
    section("COMPONENT STATE (current env/config)", use_color)
    print(f"{'NAME':<28} {'ENV VAR':<28} {'VALUE':<10} {'STATE':<6}")
    print("-" * 80)
    for c in report["components"]["components"]:
        name = _cap(c["name"], 27)
        env = _cap(c["env"], 27)
        val = _cap(c["value"], 9)
        state = "ON" if c["is_on"] else "OFF"
        color = C_GREEN if c["is_on"] else C_DIM
        if use_color:
            print(f"{name:<28} {env:<28} {val:<10} {color}{state:<6}{C_RESET}")
        else:
            print(f"{name:<28} {env:<28} {val:<10} {state:<6}")


def _render_totals(report: dict, use_color: bool) -> None:
    section("PAPER JOURNAL TOTALS", use_color)
    t = report["totals"]
    if not t or t.get("n", 0) == 0:
        print("  No completed trades in journal yet.")
        return
    wr = t["wr"]
    wr_color = C_GREEN if wr >= 0.55 else (C_YELLOW if wr >= 0.45 else C_RED)
    pnl_color = C_GREEN if t["total_pnl_pct"] >= 0 else C_RED
    if use_color:
        print(f"  Trades:       {t['n']}")
        print(f"  WR:           {wr_color}{wr:.1%}{C_RESET}  ({t['wins']}W / {t['losses']}L)")
        print(f"  Total PnL:    {pnl_color}{t['total_pnl_pct']:+.2f}%  (${t['total_pnl_usd']:+.2f}){C_RESET}")
        print(f"  Mean PnL:     {t['mean_pnl_pct']:+.4f}%/trade")
        print(f"  Mean held:    {t['mean_bars_held']} bars")
    else:
        print(f"  n={t['n']}  WR={wr:.1%}  PnL={t['total_pnl_pct']:+.2f}%  ${t['total_pnl_usd']:+.2f}  mean_pct={t['mean_pnl_pct']:+.4f}  bars={t['mean_bars_held']}")


def _render_attribution_table(rows: list, title: str, use_color: bool) -> None:
    section(title, use_color)
    if not rows or all((r.get("n") or 0) == 0 for r in rows):
        print("  No data.")
        return
    print(f"{'BUCKET':<22} {'N':>5} {'WR':>8} {'MEAN%':>8} {'TOTAL%':>10}")
    print("-" * 60)
    for r in rows:
        bucket = _cap(r["bucket"], 21)
        n = r.get("n") or 0
        if n == 0:
            print(f"{bucket:<22} {n:>5} {'-':>8} {'-':>8} {'-':>10}")
            continue
        wr = r["wr"] or 0.0
        wr_color = C_GREEN if wr >= 0.55 else (C_YELLOW if wr >= 0.45 else C_RED)
        mean = r["mean_pnl_pct"] or 0.0
        mean_color = C_GREEN if mean >= 0 else C_RED
        total = r["total_pnl_pct"] or 0.0
        total_color = C_GREEN if total >= 0 else C_RED
        if use_color:
            print(f"{bucket:<22} {n:>5} {wr_color}{wr:>7.1%}{C_RESET} {mean_color}{mean:>7.3f}%{C_RESET} {total_color}{total:>9.2f}%{C_RESET}")
        else:
            print(f"{bucket:<22} {n:>5} {wr:>7.1%} {mean:>7.3f}% {total:>9.2f}%")


def _render_rolling_sharpe(report: dict, use_color: bool) -> None:
    section("ROLLING SHARPE (30-trade window, last 10 steps)", use_color)
    rs = report.get("rolling_sharpe_30") or []
    if not rs:
        print(f"  Need >= 30 trades; have {report['totals'].get('n', 0)}")
        return
    tail = rs[-10:]
    print(f"{'AFTER TRADE':<14} {'N':>4} {'MEAN%':>8} {'STD%':>8} {'SHARPE':>9}")
    print("-" * 52)
    for r in tail:
        sharpe = r["sharpe"]
        color = C_GREEN if sharpe >= 1.0 else (C_YELLOW if sharpe >= 0.3 else C_RED)
        if use_color:
            print(f"{r['idx']:<14} {r['n']:>4} {r['mean']:>7.3f}% {r['std']:>7.3f}% {color}{sharpe:>8.3f}{C_RESET}")
        else:
            print(f"{r['idx']:<14} {r['n']:>4} {r['mean']:>7.3f}% {r['std']:>7.3f}% {sharpe:>8.3f}")


def _render_shadow(report: dict, use_color: bool) -> None:
    section("META SHADOW LOG", use_color)
    s = report.get("shadow", {})
    if not s.get("available"):
        print("  Shadow helper not available.")
        return
    print(f"  Records:       {s.get('total_records', 0)}")
    print(f"  Joined trades: {s.get('joined_trades', 0)}")
    if s.get("joined_trades", 0) == 0:
        print("  No completed shadow trades yet. Enable ACT_META_SHADOW_MODE=1 + ACT_DISABLE_ML=0 and let the bot run.")
        return
    combined = s.get("combined", {}) or {}
    if combined:
        print(f"  Actual WR:             {combined.get('actual_wr'):.1%}" if combined.get('actual_wr') is not None else "  Actual WR: -")
        print(f"  Meta would-VETO count: {combined.get('meta_veto_count')}")
        print(f"  Meta would-TAKE count: {combined.get('meta_take_count')}")
        print(f"  Veto precision (loss): {combined.get('veto_precision_loss')}")
        print(f"  Take precision (win):  {combined.get('take_precision_win')}")
        tot = combined.get("total_pnl_pct")
        iv = combined.get("if_vetoed_pnl_pct")
        if tot is not None and iv is not None:
            delta = iv - tot
            print(f"  Total PnL now:   {tot:+.2f}%")
            print(f"  If vetoed PnL:   {iv:+.2f}%   (delta: {delta:+.2f}pp)")


def _render_recommendations(report: dict, use_color: bool) -> None:
    section("RECOMMENDATIONS", use_color)
    for r in report["recommendations"]:
        sev = r["severity"]
        color = {"high": C_RED, "medium": C_YELLOW, "info": C_DIM}.get(sev, "")
        if use_color:
            print(f"  [{color}{sev.upper():<6}{C_RESET}] {r['area']}: {r['reason']}")
            print(f"  {' ':<9} -> {r['action']}\n")
        else:
            print(f"  [{sev.upper():<6}] {r['area']}: {r['reason']}")
            print(f"           -> {r['action']}\n")


def render(report: dict, use_color: bool = True) -> None:
    _render_components(report, use_color)
    _render_totals(report, use_color)
    _render_attribution_table(report["attribution"]["by_score"], "ATTRIBUTION BY ENTRY SCORE", use_color)
    _render_attribution_table(report["attribution"]["by_llm_conf"], "ATTRIBUTION BY LLM CONFIDENCE", use_color)
    _render_attribution_table(report["attribution"]["by_ml_conf"], "ATTRIBUTION BY ML CONFIDENCE", use_color)
    _render_attribution_table(report["attribution"]["by_spread"], "ATTRIBUTION BY SPREAD %", use_color)
    _render_attribution_table(report["attribution"]["by_direction"], "ATTRIBUTION BY DIRECTION", use_color)
    _render_attribution_table(report["attribution"]["by_asset"], "ATTRIBUTION BY ASSET", use_color)
    _render_attribution_table(report["attribution"]["by_exit_reason"], "ATTRIBUTION BY EXIT REASON", use_color)
    _render_rolling_sharpe(report, use_color)
    _render_shadow(report, use_color)
    _render_recommendations(report, use_color)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print JSON to stdout instead of pretty tables")
    ap.add_argument("--no-colors", action="store_true", help="disable ANSI colors")
    args = ap.parse_args()

    from src.evaluation.act_evaluator import build_report
    report = build_report()

    if args.json:
        # Trim the trades array for JSON output (can be huge)
        pruned = dict(report)
        pruned["trades_count"] = len(pruned.get("trades") or [])
        pruned.pop("trades", None)
        print(json.dumps(pruned, indent=2, default=str))
        return 0

    render(report, use_color=not args.no_colors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
