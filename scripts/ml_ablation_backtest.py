"""
ML-on vs ML-off ablation — reruns the backtest in both modes and prints a delta
table so you can see whether the calibration commit actually flipped ML to
positive contribution.

The pre-calibration paper journal showed ML-on lost $78 more than ML-off on
the BTC 20k-bar backtest. Target post-calibration: ML-on beats ML-off by ≥ +1pp
win rate AND by ≥ +0 total PnL.

Usage:
    python scripts/ml_ablation_backtest.py --asset BTC --bars 20000
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def run_backtest(asset: str, bars: int, ml_on: bool, out_csv: str) -> int:
    """Run the backtest harness; returns exit code."""
    env = os.environ.copy()
    env["ACT_BACKTEST_ML_ENABLED"] = "1" if ml_on else "0"
    cmd = [
        sys.executable,
        "-m", "src.scripts.run_full_backtest_direct",
        "--asset", asset,
        "--bars", str(bars),
        "--output", out_csv,
    ]
    print(f"  -> {' '.join(cmd)}  (ACT_BACKTEST_ML_ENABLED={'1' if ml_on else '0'})")
    return subprocess.call(cmd, env=env)


def read_trades(csv_path: str) -> List[Dict]:
    if not os.path.exists(csv_path):
        return []
    out: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                row["pnl_pct"] = float(row.get("pnl_pct") or 0)
                row["pnl_usd"] = float(row.get("pnl_usd") or 0)
                out.append(row)
            except Exception:
                continue
    return out


def summarize(label: str, trades: List[Dict]) -> Dict[str, float]:
    if not trades:
        return {"label": label, "n": 0}
    pnl_pcts = [t["pnl_pct"] for t in trades]
    pnl_usds = [t["pnl_usd"] for t in trades]
    wins = sum(1 for p in pnl_pcts if p > 0)
    mean_pct = sum(pnl_pcts) / len(pnl_pcts)
    stdev_pct = statistics.stdev(pnl_pcts) if len(pnl_pcts) > 1 else 0.0
    sharpe_like = (mean_pct / stdev_pct) if stdev_pct > 0 else 0.0
    return {
        "label": label,
        "n": len(trades),
        "wr": wins / len(trades),
        "mean_pct": mean_pct,
        "stdev_pct": stdev_pct,
        "sharpe_like": sharpe_like,
        "total_usd": sum(pnl_usds),
    }


def fmt(s: Dict[str, float]) -> str:
    if not s or s.get("n") == 0:
        return f"{s.get('label','?')}: no trades"
    return (f"{s['label']}: n={s['n']}  WR={s['wr']:.3f}  mean={s['mean_pct']:+.3f}%  "
            f"std={s['stdev_pct']:.3f}%  Sharpe~{s['sharpe_like']:+.3f}  total=${s['total_usd']:+.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--bars", type=int, default=20000)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ml_on_csv = os.path.join(args.out_dir, f"{args.asset.lower()}_ml_{args.bars}.csv")
    ml_off_csv = os.path.join(args.out_dir, f"{args.asset.lower()}_noml_{args.bars}.csv")

    print("\n=== Running ML-OFF (baseline) ===")
    rc1 = run_backtest(args.asset, args.bars, ml_on=False, out_csv=ml_off_csv)
    print(f"  exit={rc1}")

    print("\n=== Running ML-ON (calibrated) ===")
    rc2 = run_backtest(args.asset, args.bars, ml_on=True, out_csv=ml_on_csv)
    print(f"  exit={rc2}")

    off = summarize("ML-OFF", read_trades(ml_off_csv))
    on = summarize("ML-ON ", read_trades(ml_on_csv))

    print("\n=== Results ===")
    print(" ", fmt(off))
    print(" ", fmt(on))

    if off.get("n") and on.get("n"):
        wr_delta = (on["wr"] - off["wr"]) * 100
        pnl_delta = on["total_usd"] - off["total_usd"]
        sharpe_delta = on["sharpe_like"] - off["sharpe_like"]
        print("\n=== ML contribution (on − off) ===")
        print(f"  Δ Win rate:       {wr_delta:+.2f} pp")
        print(f"  Δ Total PnL:      ${pnl_delta:+.2f}")
        print(f"  Δ Sharpe-like:    {sharpe_delta:+.3f}")
        verdict = "ML adds value" if (wr_delta >= 1.0 and pnl_delta > 0) else "ML is not adding value"
        print(f"\n  VERDICT: {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
