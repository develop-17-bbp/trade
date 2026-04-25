"""Measure Robinhood's actual round-trip spread from the bot's journal
+ a live quote, so the operator can verify whether the cost_gate's
1.69% preset is correct or stale.

Output: a recommended `ACT_ROBINHOOD_SPREAD_PCT` value plus the
evidence (live quote percent + realized-trade distribution).

Usage:
    cd C:\\Users\\admin\\trade
    python scripts/measure_spread.py [--asset BTC] [--limit 50]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
JOURNAL = REPO_ROOT / "logs" / "trading_journal.jsonl"


def _quote_live(asset: str) -> Optional[dict]:
    """Try to pull a live bid/ask from the bot's robinhood_fetcher.
    Returns {"bid": float, "ask": float, "mid": float, "spread_pct": float}
    or None if unreachable.
    """
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from src.data.robinhood_fetcher import RobinhoodFetcher  # type: ignore
    except Exception as e:
        print(f"  [skip live quote] could not import RobinhoodFetcher: {e}")
        return None
    try:
        f = RobinhoodFetcher()
        bid = f.get_bid_price(asset) if hasattr(f, "get_bid_price") else None
        ask = f.get_ask_price(asset) if hasattr(f, "get_ask_price") else None
        if bid and ask and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid * 100.0
            return {
                "bid": bid, "ask": ask, "mid": mid,
                "spread_pct": spread_pct,
                "round_trip_pct": spread_pct,  # quoted spread is per-side
                                                # but fill-to-fill round-trip
                                                # crosses both sides exactly
                                                # once, so quoted == round-trip.
            }
    except Exception as e:
        print(f"  [skip live quote] RobinhoodFetcher call raised: {e}")
    return None


def _read_journal(limit: int = 100) -> List[dict]:
    if not JOURNAL.exists():
        return []
    rows: List[dict] = []
    with JOURNAL.open(encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:] if limit > 0 else rows


def _summarize_journal(rows: List[dict], asset_filter: Optional[str]) -> dict:
    closed = []
    for r in rows:
        if r.get("exit_price") in (None, 0, 0.0):
            continue
        if asset_filter and str(r.get("asset", "")).upper() != asset_filter.upper():
            continue
        if not (isinstance(r.get("entry_price"), (int, float))
                and isinstance(r.get("exit_price"), (int, float))):
            continue
        closed.append(r)
    if not closed:
        return {"count": 0}

    pnls = [float(r.get("pnl_pct") or 0) for r in closed]
    abs_pnls = [abs(p) for p in pnls if p != 0]
    durations = [float(r.get("duration_minutes") or 0) for r in closed]

    # Compute min positive PnL — this is an upper bound on net round-trip
    # friction (any trade with |move| < friction can't be net-positive).
    smallest_winning = min((p for p in pnls if p > 0), default=None)

    # Find break-even-adjacent trades — pnl in [-1%, +1%] — these are
    # where friction matters most. Their median |pnl| is a noisy estimate
    # of the round-trip cost the executor has been paying.
    breakeven_band = [p for p in pnls if -1.5 <= p <= 1.5]
    median_abs_breakeven = (
        statistics.median([abs(p) for p in breakeven_band])
        if breakeven_band else None
    )

    return {
        "count": len(closed),
        "pnl_min": min(pnls), "pnl_max": max(pnls),
        "pnl_mean": statistics.mean(pnls), "pnl_median": statistics.median(pnls),
        "abs_pnl_median": statistics.median(abs_pnls) if abs_pnls else 0.0,
        "duration_min_median": statistics.median(durations) if durations else 0.0,
        "smallest_winning_pnl_pct": smallest_winning,
        "breakeven_band_n": len(breakeven_band),
        "median_abs_breakeven_pnl": median_abs_breakeven,
    }


def _recommend(live_pct: Optional[float],
               journal_summary: dict) -> Optional[float]:
    """Pick a recommendation. Prefer live quote (most current). Fall back
    to journal-derived. Round to 2 decimals."""
    if live_pct is not None and live_pct > 0:
        # Add a 0.1% slippage buffer for taker execution
        return round(live_pct + 0.10, 2)
    sw = journal_summary.get("smallest_winning_pnl_pct")
    if isinstance(sw, (int, float)) and sw > 0:
        # smallest_winning ~= friction + epsilon → use as upper bound
        return round(min(sw * 0.9, 1.5), 2)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--asset", default="BTC", help="asset for live-quote probe (BTC/ETH)")
    p.add_argument("--limit", type=int, default=100, help="max trades from journal")
    args = p.parse_args(argv)

    print("=" * 60)
    print(f" Robinhood spread measurement (asset={args.asset})")
    print("=" * 60)

    print("\n[1/3] Live quote via robinhood_fetcher")
    live = _quote_live(args.asset)
    if live:
        print(f"    bid:    {live['bid']:.2f}")
        print(f"    ask:    {live['ask']:.2f}")
        print(f"    mid:    {live['mid']:.2f}")
        print(f"    spread: {live['spread_pct']:.4f}% (per side)")
        print(f"    round-trip estimate: ~{live['round_trip_pct']:.2f}%")
    else:
        print("    [no live quote available]")

    print(f"\n[2/3] Trading journal: last {args.limit} closed trades")
    rows = _read_journal(args.limit)
    summary = _summarize_journal(rows, args.asset)
    if summary.get("count", 0) == 0:
        print(f"    no closed trades found in {JOURNAL}")
    else:
        print(f"    closed trades:           {summary['count']}")
        print(f"    pnl_pct range:           {summary['pnl_min']:+.2f}% .. "
              f"{summary['pnl_max']:+.2f}%")
        print(f"    pnl_pct median:          {summary['pnl_median']:+.2f}%")
        print(f"    |pnl_pct| median:        {summary['abs_pnl_median']:.2f}%")
        print(f"    duration min (median):   {summary['duration_min_median']:.1f}")
        sw = summary.get("smallest_winning_pnl_pct")
        if sw is not None:
            print(f"    smallest WINNING pnl:    +{sw:.2f}% "
                  "(upper bound on net friction)")
        mab = summary.get("median_abs_breakeven_pnl")
        if mab is not None:
            print(f"    breakeven-band median:   {mab:.2f}% "
                  f"(n={summary['breakeven_band_n']})")

    print("\n[3/3] Recommendation")
    live_rt = live.get("round_trip_pct") if live else None
    rec = _recommend(live_rt, summary)
    current = float(os.environ.get("ACT_ROBINHOOD_SPREAD_PCT") or 1.69)
    print(f"    current ACT_ROBINHOOD_SPREAD_PCT: {current:.2f}%")
    if rec is not None:
        print(f"    recommended:                     {rec:.2f}%")
        delta = current - rec
        if delta > 0.2:
            print(f"    -> set: setx ACT_ROBINHOOD_SPREAD_PCT {rec:.2f}")
            print(f"       (frees up ~{delta:.2f}% of margin per trade — more "
                  "signals will reach the authority gate)")
        elif delta < -0.2:
            print(f"    -> increase to {rec:.2f} — current is too optimistic")
        else:
            print(f"    -> within {abs(delta):.2f}% of current; no change needed")
    else:
        print("    no recommendation possible (no live quote, no journal data).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
