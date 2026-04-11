"""
Autonomous Strategy Backtester
================================
Tests ALL available strategies on recent market data.
Produces strategy rankings and recommended weights.

Used by Claude Code scheduled tasks to continuously
discover which strategies work best in current conditions.

Usage:
    python -m src.scripts.strategy_backtester
    python -m src.scripts.strategy_backtester --days 30 --asset BTC
    python -m src.scripts.strategy_backtester --all --output logs/strategy_rankings.json
    python -m src.scripts.strategy_backtester --timeframe 1h --days 14
"""

import json
import math
import os
import sys
import inspect
import argparse
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root so imports work when run as script or module
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trading.sub_strategies import SubStrategy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BARS_PER_YEAR = {
    "15m": 4 * 24 * 365,   # 35_040
    "1h":  24 * 365,        # 8_760
    "4h":  6 * 365,         # 2_190
    "1d":  365,
}
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"


class StrategyBacktester:
    """
    Runs every SubStrategy subclass against historical OHLCV data,
    measures performance after accounting for Robinhood's round-trip
    spread, and ranks strategies by a composite risk-adjusted score.
    """

    def __init__(self, spread_cost_pct: float = 3.34, longs_only: bool = True):
        self.spread_cost = spread_cost_pct
        self.longs_only = longs_only
        self.strategies: Dict[str, SubStrategy] = {}

    # ------------------------------------------------------------------
    # Strategy discovery
    # ------------------------------------------------------------------
    def load_strategies(self) -> None:
        """Dynamically discover and instantiate every SubStrategy subclass."""
        import src.trading.sub_strategies as mod

        for name, cls in inspect.getmembers(mod, inspect.isclass):
            if cls is SubStrategy:
                continue
            if not issubclass(cls, SubStrategy):
                continue
            # Skip PairsStrategy -- it requires a second price series
            if name == "PairsStrategy":
                continue
            try:
                instance = cls()
                self.strategies[name] = instance
            except TypeError:
                # Constructor needs arguments we cannot auto-provide
                pass

        if not self.strategies:
            raise RuntimeError("No strategies found in src.trading.sub_strategies")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(
        self,
        asset: str = "BTC",
        timeframe: str = "4h",
        days: int = 30,
    ) -> Optional[pd.DataFrame]:
        """
        Load OHLCV from parquet.  Returns a DataFrame with columns:
        open, high, low, close, volume (and a datetime index).
        Returns None if the file does not exist or has insufficient rows.
        """
        symbol = f"{asset}USDT"
        parquet_path = DATA_DIR / f"{symbol}-{timeframe}.parquet"

        if not parquet_path.exists():
            print(f"  [WARN] Parquet not found: {parquet_path}")
            return None

        df = pd.read_parquet(parquet_path)

        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Ensure we have the required columns
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            # Try alternate column names
            rename_map = {}
            for col in df.columns:
                cl = col.lower()
                if "open" in cl and "open" not in rename_map.values():
                    rename_map[col] = "open"
                elif "high" in cl and "high" not in rename_map.values():
                    rename_map[col] = "high"
                elif "low" in cl and "low" not in rename_map.values():
                    rename_map[col] = "low"
                elif "close" in cl and "close" not in rename_map.values():
                    rename_map[col] = "close"
                elif "vol" in cl and "volume" not in rename_map.values():
                    rename_map[col] = "volume"
            df.rename(columns=rename_map, inplace=True)
            if not required.issubset(set(df.columns)):
                print(f"  [WARN] Missing columns in {parquet_path}: {required - set(df.columns)}")
                return None

        # Ensure a datetime index for slicing by days
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            # Try converting the existing index
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass

        df.sort_index(inplace=True)

        # Slice last N days
        if isinstance(df.index, pd.DatetimeIndex) and days > 0:
            cutoff = df.index.max() - pd.Timedelta(days=days)
            df = df.loc[df.index >= cutoff]

        # Drop NaN rows in price columns
        df.dropna(subset=["close", "high", "low", "volume"], inplace=True)

        if len(df) < 60:
            print(f"  [WARN] Only {len(df)} bars for {asset}-{timeframe} ({days}d) -- need >= 60")
            return None

        return df

    # ------------------------------------------------------------------
    # Per-strategy backtest
    # ------------------------------------------------------------------
    def backtest_strategy(
        self,
        strategy: SubStrategy,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
        max_hold_bars: int = 20,
        stop_loss_pct: float = -5.0,
    ) -> Dict[str, Any]:
        """
        Simulate trading a single strategy on the given price arrays.
        Longs only (matching Robinhood spot constraints).
        Deducts round-trip spread from every trade.
        """
        trades: List[Dict] = []
        position: Optional[Dict] = None
        warmup = max(60, len(closes) // 10)  # at least 60 bars warmup

        # Reset stateful strategies so prior test state does not leak
        strategy_copy = self._fresh_copy(strategy)

        for i in range(warmup, len(closes)):
            try:
                signal = strategy_copy.generate_signal(
                    list(closes[: i + 1]),
                    list(highs[: i + 1]),
                    list(lows[: i + 1]),
                    list(volumes[: i + 1]),
                )
            except Exception:
                signal = 0

            if position is None:
                # --- Look for entry ---
                if signal == 1:
                    position = {"entry": closes[i], "entry_idx": i}
                elif signal == -1 and not self.longs_only:
                    position = {"entry": closes[i], "entry_idx": i, "short": True}
            else:
                # --- Manage open position ---
                is_short = position.get("short", False)
                bars_held = i - position["entry_idx"]

                if is_short:
                    pnl_pct = (position["entry"] - closes[i]) / position["entry"] * 100
                else:
                    pnl_pct = (closes[i] - position["entry"]) / position["entry"] * 100

                # Exit conditions
                exit_reason = None
                if is_short and signal == 1:
                    exit_reason = "signal_reversal"
                elif not is_short and signal == -1:
                    exit_reason = "signal_reversal"
                elif bars_held >= max_hold_bars:
                    exit_reason = "timeout"
                elif pnl_pct <= stop_loss_pct:
                    exit_reason = "stop_loss"

                if exit_reason is not None:
                    pnl_pct -= self.spread_cost  # deduct round-trip spread
                    trades.append({
                        "entry": float(position["entry"]),
                        "exit": float(closes[i]),
                        "pnl_pct": float(pnl_pct),
                        "bars_held": int(bars_held),
                        "exit_reason": exit_reason,
                        "direction": "SHORT" if is_short else "LONG",
                    })
                    position = None

        # Close any dangling position at last bar
        if position is not None:
            is_short = position.get("short", False)
            if is_short:
                pnl_pct = (position["entry"] - closes[-1]) / position["entry"] * 100
            else:
                pnl_pct = (closes[-1] - position["entry"]) / position["entry"] * 100
            pnl_pct -= self.spread_cost
            bars_held = len(closes) - 1 - position["entry_idx"]
            trades.append({
                "entry": float(position["entry"]),
                "exit": float(closes[-1]),
                "pnl_pct": float(pnl_pct),
                "bars_held": int(bars_held),
                "exit_reason": "end_of_data",
                "direction": "SHORT" if is_short else "LONG",
            })

        return self.compute_metrics(trades)

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------
    def compute_metrics(self, trades: List[Dict], timeframe: str = "4h") -> Dict[str, Any]:
        """Compute comprehensive performance metrics from a list of trades."""
        empty = {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "avg_bars_held": 0.0,
            "expectancy": 0.0,
            "trades": [],
        }
        if not trades:
            return empty

        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_trades = len(trades)
        win_rate = len(wins) / total_trades if total_trades else 0.0
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / total_trades

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        max_win = max(pnls)
        max_loss = min(pnls)

        # Profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )

        # Expectancy = (WR * avg_win) - ((1-WR) * |avg_loss|)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

        # Sharpe ratio (annualised from per-trade returns)
        periods_per_year = BARS_PER_YEAR.get(timeframe, 2190)
        avg_bars = sum(t["bars_held"] for t in trades) / total_trades
        trades_per_year = periods_per_year / avg_bars if avg_bars > 0 else 0

        if len(pnls) >= 2:
            mean_r = np.mean(pnls)
            std_r = np.std(pnls, ddof=1)
            sharpe = (mean_r / std_r) * math.sqrt(trades_per_year) if std_r > 0 else 0.0
        else:
            sharpe = 0.0

        # Sortino
        downside = [p for p in pnls if p < 0]
        if downside and len(pnls) >= 2:
            downside_std = math.sqrt(sum(d ** 2 for d in downside) / len(downside))
            mean_r = np.mean(pnls)
            sortino = (mean_r / downside_std) * math.sqrt(trades_per_year) if downside_std > 0 else 0.0
        else:
            sortino = float("inf") if total_pnl > 0 else 0.0

        # Max drawdown (on cumulative PnL curve)
        cum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_drawdown = float(np.max(dd)) if len(dd) > 0 else 0.0

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(min(profit_factor, 999.0), 4),
            "total_pnl": round(total_pnl, 4),
            "avg_pnl": round(avg_pnl, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "max_win": round(max_win, 4),
            "max_loss": round(max_loss, 4),
            "sharpe": round(min(sharpe, 999.0), 4),
            "sortino": round(min(sortino, 999.0), 4),
            "max_drawdown": round(max_drawdown, 4),
            "avg_bars_held": round(avg_bars, 2),
            "expectancy": round(expectancy, 4),
            "trades": trades,
        }

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------
    def rank_strategies(
        self, results: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Rank strategies by a composite score:
          40% profit_factor (capped at 5)
          30% win_rate
          30% sharpe (capped at 5)
        Strategies with zero trades or negative total PnL are penalised.
        """
        scored: List[Tuple[str, Dict[str, Any], float]] = []

        for name, metrics in results.items():
            if metrics["total_trades"] == 0:
                scored.append((name, metrics, -100.0))
                continue

            pf_norm = min(metrics["profit_factor"], 5.0) / 5.0
            wr_norm = metrics["win_rate"]
            sharpe_norm = (min(max(metrics["sharpe"], -5.0), 5.0) + 5.0) / 10.0

            composite = 0.40 * pf_norm + 0.30 * wr_norm + 0.30 * sharpe_norm

            # Penalty for negative PnL
            if metrics["total_pnl"] < 0:
                composite *= 0.5

            # Bonus for high trade count (more statistically meaningful)
            if metrics["total_trades"] >= 10:
                composite *= 1.05
            elif metrics["total_trades"] < 3:
                composite *= 0.7

            metrics["composite_score"] = round(composite, 4)
            scored.append((name, metrics, composite))

        scored.sort(key=lambda x: x[2], reverse=True)
        return [(name, metrics) for name, metrics, _ in scored]

    # ------------------------------------------------------------------
    # Weight recommendation
    # ------------------------------------------------------------------
    def recommend_weights(
        self, rankings: List[Tuple[str, Dict[str, Any]]]
    ) -> Dict[str, float]:
        """
        Convert rankings into strategy allocation weights.
        Only strategies with positive PnL get a non-zero weight.
        Top strategy ~30%, then diminishing shares.
        """
        # Filter to profitable strategies
        profitable = [
            (name, m) for name, m in rankings
            if m["total_pnl"] > 0 and m["total_trades"] >= 2
        ]
        if not profitable:
            # Fall back: equal weight across top 3
            top3 = rankings[:3]
            if not top3:
                return {}
            w = round(1.0 / len(top3), 4)
            return {name: w for name, _ in top3}

        # Assign weights proportional to composite score
        scores = [m.get("composite_score", 0.0) for _, m in profitable]
        total_score = sum(scores)
        if total_score <= 0:
            w = round(1.0 / len(profitable), 4)
            return {name: w for name, _ in profitable}

        weights = {}
        for (name, m), score in zip(profitable, scores):
            weights[name] = round(score / total_score, 4)

        # Cap any single strategy at 40%
        for name in weights:
            if weights[name] > 0.40:
                excess = weights[name] - 0.40
                weights[name] = 0.40
                # Redistribute excess proportionally
                others = [n for n in weights if n != name and weights[n] > 0]
                if others:
                    per_other = excess / len(others)
                    for o in others:
                        weights[o] = round(weights[o] + per_other, 4)

        return weights

    # ------------------------------------------------------------------
    # Full analysis pipeline
    # ------------------------------------------------------------------
    def run_full_analysis(
        self,
        assets: List[str] = None,
        timeframe: str = "4h",
        days: int = 30,
        max_hold_bars: int = 20,
        stop_loss_pct: float = -5.0,
    ) -> Dict[str, Any]:
        """
        Run every strategy on each asset, collect results, rank, and
        produce recommended weights.
        """
        if assets is None:
            assets = ["BTC", "ETH"]

        if not self.strategies:
            self.load_strategies()

        report: Dict[str, Any] = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "params": {
                "assets": assets,
                "timeframe": timeframe,
                "days": days,
                "spread_cost_pct": self.spread_cost,
                "max_hold_bars": max_hold_bars,
                "stop_loss_pct": stop_loss_pct,
                "longs_only": self.longs_only,
            },
            "assets": {},
        }

        for asset in assets:
            print(f"\n{'='*60}")
            print(f"  Backtesting {asset} ({timeframe}, last {days} days)")
            print(f"{'='*60}")

            df = self.load_data(asset, timeframe, days)
            if df is None:
                report["assets"][asset] = {
                    "error": "insufficient data",
                    "rankings": [],
                    "weights": {},
                }
                continue

            closes = df["close"].values.astype(float).tolist()
            highs = df["high"].values.astype(float).tolist()
            lows = df["low"].values.astype(float).tolist()
            volumes = df["volume"].values.astype(float).tolist()

            print(f"  Loaded {len(closes)} bars")
            print(f"  Price range: ${min(closes):,.2f} - ${max(closes):,.2f}")
            print(f"  Testing {len(self.strategies)} strategies...\n")

            strategy_results: Dict[str, Dict[str, Any]] = {}

            for strat_name, strat_instance in self.strategies.items():
                try:
                    metrics = self.backtest_strategy(
                        strat_instance,
                        closes,
                        highs,
                        lows,
                        volumes,
                        max_hold_bars=max_hold_bars,
                        stop_loss_pct=stop_loss_pct,
                    )
                    # Strip individual trade details from the ranking output
                    metrics_summary = {k: v for k, v in metrics.items() if k != "trades"}
                    strategy_results[strat_name] = metrics_summary

                    status = "OK" if metrics["total_trades"] > 0 else "NO TRADES"
                    print(
                        f"  {strat_name:30s}  trades={metrics['total_trades']:3d}  "
                        f"WR={metrics['win_rate']:.0%}  PF={metrics['profit_factor']:.2f}  "
                        f"PnL={metrics['total_pnl']:+.1f}%  [{status}]"
                    )
                except Exception as exc:
                    print(f"  {strat_name:30s}  ERROR: {exc}")
                    strategy_results[strat_name] = {
                        "total_trades": 0,
                        "win_rate": 0.0,
                        "profit_factor": 0.0,
                        "total_pnl": 0.0,
                        "sharpe": 0.0,
                        "sortino": 0.0,
                        "max_drawdown": 0.0,
                        "error": str(exc),
                    }

            rankings = self.rank_strategies(strategy_results)
            weights = self.recommend_weights(rankings)

            report["assets"][asset] = {
                "bars": len(closes),
                "price_range": [round(min(closes), 2), round(max(closes), 2)],
                "rankings": [
                    {"rank": i + 1, "strategy": name, "metrics": metrics}
                    for i, (name, metrics) in enumerate(rankings)
                ],
                "weights": weights,
            }

        # Aggregate: combine per-asset weights into a global recommendation
        report["global_weights"] = self._aggregate_weights(report)

        return report

    # ------------------------------------------------------------------
    # Aggregate weights across assets
    # ------------------------------------------------------------------
    def _aggregate_weights(self, report: Dict) -> Dict[str, float]:
        """Average strategy weights across all assets."""
        combined: Dict[str, List[float]] = {}
        for asset_data in report["assets"].values():
            if isinstance(asset_data, dict) and "weights" in asset_data:
                for strat, w in asset_data["weights"].items():
                    combined.setdefault(strat, []).append(w)

        if not combined:
            return {}

        avg_weights = {
            strat: round(sum(ws) / len(ws), 4) for strat, ws in combined.items()
        }
        # Normalise to sum to 1.0
        total = sum(avg_weights.values())
        if total > 0:
            avg_weights = {s: round(w / total, 4) for s, w in avg_weights.items()}
        return avg_weights

    # ------------------------------------------------------------------
    # Report persistence
    # ------------------------------------------------------------------
    def save_report(
        self, results: Dict[str, Any], path: str = "logs/strategy_backtest_results.json"
    ) -> str:
        """
        Save results to JSON.  Also appends to a history file so we can
        track strategy performance drift over time.
        """
        out_path = PROJECT_ROOT / path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Append to rolling history (one line per run)
        history_path = out_path.parent / "strategy_backtest_history.jsonl"
        summary_line = {
            "timestamp": results.get("run_timestamp", datetime.now(timezone.utc).isoformat()),
            "global_weights": results.get("global_weights", {}),
            "per_asset": {},
        }
        for asset, data in results.get("assets", {}).items():
            if isinstance(data, dict) and "rankings" in data:
                top = data["rankings"][:3] if data["rankings"] else []
                summary_line["per_asset"][asset] = {
                    "top_strategies": [
                        {
                            "name": r["strategy"],
                            "pnl": r["metrics"].get("total_pnl", 0),
                            "wr": r["metrics"].get("win_rate", 0),
                        }
                        for r in top
                    ],
                    "weights": data.get("weights", {}),
                }
        with open(history_path, "a") as f:
            f.write(json.dumps(summary_line, default=str) + "\n")

        print(f"\n  Report saved to {out_path}")
        print(f"  History appended to {history_path}")
        return str(out_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fresh_copy(strategy: SubStrategy) -> SubStrategy:
        """
        Return a fresh instance of the strategy so that any internal
        state (e.g. EMACrossoverStrategy's trade tracking) does not
        leak between backtest runs.
        """
        try:
            return deepcopy(strategy)
        except Exception:
            # If deepcopy fails, try creating a new instance
            try:
                return strategy.__class__()
            except TypeError:
                return strategy


# ======================================================================
# CLI entry point
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Strategy Backtester -- ranks all strategies by risk-adjusted return"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of historical days to test (default: 30)",
    )
    parser.add_argument(
        "--asset", default="ALL",
        help="Asset to test: BTC, ETH, or ALL (default: ALL)",
    )
    parser.add_argument(
        "--timeframe", default="4h",
        choices=["15m", "1h", "4h", "1d"],
        help="Candle timeframe (default: 4h)",
    )
    parser.add_argument(
        "--output", default="logs/strategy_backtest_results.json",
        help="Output JSON path (default: logs/strategy_backtest_results.json)",
    )
    parser.add_argument(
        "--spread", type=float, default=3.34,
        help="Round-trip spread cost in percent (default: 3.34)",
    )
    parser.add_argument(
        "--max-hold", type=int, default=20,
        help="Max bars to hold a position (default: 20)",
    )
    parser.add_argument(
        "--stop-loss", type=float, default=-5.0,
        help="Stop loss percent (default: -5.0)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Test all available assets with parquet data",
    )
    args = parser.parse_args()

    # Determine assets
    if args.all:
        # Discover all assets from parquet files
        assets = set()
        for p in DATA_DIR.glob(f"*-{args.timeframe}.parquet"):
            symbol = p.stem.split("-")[0]
            asset = symbol.replace("USDT", "")
            assets.add(asset)
        assets = sorted(assets)
    elif args.asset.upper() == "ALL":
        assets = ["BTC", "ETH"]
    else:
        assets = [args.asset.upper()]

    print(f"Strategy Backtester")
    print(f"  Assets:     {', '.join(assets)}")
    print(f"  Timeframe:  {args.timeframe}")
    print(f"  Days:       {args.days}")
    print(f"  Spread:     {args.spread}%")
    print(f"  Max hold:   {args.max_hold} bars")
    print(f"  Stop loss:  {args.stop_loss}%")

    bt = StrategyBacktester(spread_cost_pct=args.spread)
    bt.load_strategies()
    print(f"  Strategies: {len(bt.strategies)} loaded -- {', '.join(bt.strategies.keys())}")

    results = bt.run_full_analysis(
        assets=assets,
        timeframe=args.timeframe,
        days=args.days,
        max_hold_bars=args.max_hold,
        stop_loss_pct=args.stop_loss,
    )

    bt.save_report(results, args.output)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"  STRATEGY RANKINGS SUMMARY")
    print(f"{'='*60}")

    for asset, data in results.get("assets", {}).items():
        if not isinstance(data, dict) or "rankings" not in data:
            continue
        print(f"\n  --- {asset} ---")
        for entry in data["rankings"]:
            m = entry["metrics"]
            rank = entry["rank"]
            name = entry["strategy"]
            score = m.get("composite_score", 0)
            print(
                f"  #{rank:2d} {name:30s}  "
                f"WR={m.get('win_rate', 0):.0%}  "
                f"PF={m.get('profit_factor', 0):.2f}  "
                f"PnL={m.get('total_pnl', 0):+.1f}%  "
                f"Sharpe={m.get('sharpe', 0):.2f}  "
                f"Score={score:.3f}"
            )

        weights = data.get("weights", {})
        if weights:
            print(f"\n  Recommended weights for {asset}:")
            for strat, w in sorted(weights.items(), key=lambda x: -x[1]):
                print(f"    {strat:30s}  {w:.1%}")

    # Global
    gw = results.get("global_weights", {})
    if gw:
        print(f"\n  --- GLOBAL RECOMMENDED WEIGHTS ---")
        for strat, w in sorted(gw.items(), key=lambda x: -x[1]):
            print(f"    {strat:30s}  {w:.1%}")

    print(f"\n{'='*60}")
    print(f"  Done. Results at: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
