"""
ML-on vs ML-off ablation using the in-process FullBacktestEngine.

The previous version shelled out to src.scripts.run_full_backtest_direct, which
has hardcoded paths (loads AAVE data regardless of --asset) and crashed on the
new 50-feature model schema. This rewrite calls FullBacktestEngine directly
with use_ml=True/False so the right model gets loaded for the right asset.

Target post-calibration: ML-on beats ML-off by >= +1pp win rate AND >= $0
total PnL on the same market window.

Usage:
    python scripts/ml_ablation_backtest.py --asset BTC --days 30
    python scripts/ml_ablation_backtest.py --asset ETH --days 14 --primary-tf 5m
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List

# Make `python scripts/<name>.py` work from any CWD + cd to repo root so model
# and cache paths resolve.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)


def summarize(label: str, metrics) -> Dict[str, float]:
    """Extract win rate / mean / stdev / sharpe-like / total PnL from BacktestMetrics."""
    trades = list(getattr(metrics, "trades", []) or [])
    if not trades:
        return {"label": label, "n": 0}
    pnl_pcts = [float(getattr(t, "pnl_pct", 0) or 0) for t in trades]
    pnl_usds = [float(getattr(t, "pnl_usd", 0) or 0) for t in trades]
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


def run_once(asset: str, days: int, primary_tf: str, use_ml: bool):
    """Fetch data + run the engine with the requested ml flag. Returns BacktestMetrics."""
    from src.backtesting.data_loader import fetch_backtest_data
    from src.backtesting.full_engine import FullBacktestEngine

    print(f"\n--- Loading {days}d of {asset} @ {primary_tf} data ---")
    data = fetch_backtest_data(
        asset=asset, days=days, primary_tf=primary_tf, local_only=False,
    )
    print(f"  Loaded {data.bar_count} bars on {primary_tf}")

    cfg = {
        "asset": asset,
        "use_ml": use_ml,
        "initial_capital": 100000.0,
        "min_entry_score": 4,
    }
    engine = FullBacktestEngine(cfg)
    print(f"--- Running engine with use_ml={use_ml} ---")
    return engine.run(data, verbose=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--days", type=int, default=30,
                    help="Days of history to fetch (default 30)")
    ap.add_argument("--primary-tf", default="5m",
                    help="Primary timeframe (default 5m)")
    args = ap.parse_args()

    # Accept --bars as a legacy alias for --days; just drop it
    try:
        off_metrics = run_once(args.asset, args.days, args.primary_tf, use_ml=False)
    except Exception as e:
        print(f"\nML-OFF run FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 2

    try:
        on_metrics = run_once(args.asset, args.days, args.primary_tf, use_ml=True)
    except Exception as e:
        print(f"\nML-ON run FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 3

    off = summarize("ML-OFF", off_metrics)
    on = summarize("ML-ON ", on_metrics)

    print("\n=== Results ===")
    print(" ", fmt(off))
    print(" ", fmt(on))

    if off.get("n") and on.get("n"):
        wr_delta = (on["wr"] - off["wr"]) * 100
        pnl_delta = on["total_usd"] - off["total_usd"]
        sharpe_delta = on["sharpe_like"] - off["sharpe_like"]
        print("\n=== ML contribution (on - off) ===")
        print(f"  Delta Win rate:       {wr_delta:+.2f} pp")
        print(f"  Delta Total PnL:      ${pnl_delta:+.2f}")
        print(f"  Delta Sharpe-like:    {sharpe_delta:+.3f}")
        verdict = "ML adds value" if (wr_delta >= 1.0 and pnl_delta > 0) else "ML is NOT adding value"
        print(f"\n  VERDICT: {verdict}")
    else:
        print("\n  No trades in one or both runs - target window / data may be too short.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
