"""
Trading Optimizer — Freqtrade-Inspired Analysis & Optimization
==============================================================
Multi-metric fitness scoring, lookahead bias detection,
parameter space search, and performance analysis.

Standard library only (+ json).
"""

import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1.  MultiMetricFitness
# ---------------------------------------------------------------------------

class MultiMetricFitness:
    """Evaluates trading performance using freqtrade-style multi-metric scoring.

    Combines: profit, drawdown, Sortino ratio, profit factor, expectancy,
    win rate, and trade count into a single fitness score.
    """

    GRADE_THRESHOLDS = [
        (100, "A+"),
        (50, "A"),
        (20, "B"),
        (5, "C"),
        (0, "D"),
    ]

    def __init__(self):
        pass

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _safe_div(a: float, b: float, default: float = 0.0) -> float:
        return a / b if b != 0 else default

    @staticmethod
    def _downside_deviation(returns: List[float], mar: float = 0.0) -> float:
        """Downside deviation (root-mean-square of negative excess returns)."""
        neg = [(r - mar) ** 2 for r in returns if r < mar]
        if not neg:
            return 0.0
        return math.sqrt(sum(neg) / len(neg))

    @staticmethod
    def _max_drawdown_pct(equity_curve: List[float]) -> float:
        """Peak-to-trough drawdown as a fraction (0-1)."""
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    # -- main API ------------------------------------------------------------

    def compute(self, trades: list) -> dict:
        """Compute all metrics from a list of trade dicts.

        Each trade: {pnl_usd, pnl_pct, duration_min (or duration_minutes),
                     entry_time (or timestamp), exit_time, asset, direction}

        Returns a dict with all metrics plus fitness_score and grade.
        """
        empty = self._empty_result()
        if not trades:
            return empty

        # --- extract returns ---
        pnl_usds = []
        pnl_pcts = []
        durations = []

        for t in trades:
            pnl = t.get("pnl_usd", 0.0)
            pct = t.get("pnl_pct", 0.0)
            dur = t.get("duration_min") or t.get("duration_minutes", 0.0)
            pnl_usds.append(float(pnl))
            pnl_pcts.append(float(pct))
            durations.append(float(dur))

        trade_count = len(pnl_usds)

        total_profit = sum(pnl_usds)
        total_profit_pct = sum(pnl_pcts)

        wins = [p for p in pnl_usds if p > 0]
        losses = [p for p in pnl_usds if p <= 0]
        gross_wins = sum(wins)
        gross_losses = sum(losses)  # negative

        win_rate = len(wins) / trade_count if trade_count else 0.0
        loss_rate = 1.0 - win_rate

        avg_win = statistics.mean(wins) if wins else 0.0
        avg_loss = abs(statistics.mean(losses)) if losses else 0.0

        # --- profit factor ---
        # Cap at 999.99 to avoid inf propagating into fitness
        pf_raw = self._safe_div(gross_wins, abs(gross_losses), default=999.99 if gross_wins > 0 else 0.0)
        profit_factor = min(pf_raw, 999.99)

        # --- expectancy ---
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        expectancy_ratio = self._safe_div(expectancy, avg_loss, default=0.0)

        # --- equity curve & drawdown ---
        equity = [0.0]
        for p in pnl_pcts:
            equity.append(equity[-1] + p)
        max_drawdown_pct = self._max_drawdown_pct([100.0 + e for e in equity])

        # --- Sortino ratio ---
        mean_return = statistics.mean(pnl_pcts) if pnl_pcts else 0.0
        dd = self._downside_deviation(pnl_pcts)
        sortino_ratio = self._safe_div(mean_return, dd)

        # --- Sharpe ratio ---
        std_dev = statistics.pstdev(pnl_pcts) if len(pnl_pcts) > 1 else 0.0
        sharpe_ratio = self._safe_div(mean_return, std_dev)

        # --- Calmar ratio ---
        # Annualise: assume avg trade ~15 min => ~35,000 trades/year
        avg_dur = statistics.mean(durations) if durations else 15.0
        trades_per_year = self._safe_div(525_600, max(avg_dur, 1.0))
        annual_return_pct = mean_return * trades_per_year
        calmar_ratio = self._safe_div(annual_return_pct, max_drawdown_pct * 100) if max_drawdown_pct > 0 else 0.0

        # --- avg duration ---
        avg_duration_min = statistics.mean(durations) if durations else 0.0

        # --- fitness score (freqtrade MultiMetricHyperOptLoss inspired) ---
        profit_draw = total_profit_pct * (1.0 - max_drawdown_pct)
        pf_log = math.log(max(profit_factor, 0.001) + 1.0)
        exp_log = math.log(min(10.0, max(expectancy, -1.0)) + 2.0)
        wr_log = math.log(1.2 + win_rate)

        fitness_score = profit_draw * pf_log * exp_log * wr_log

        # trade count penalty: linear ramp from 0.2 at 1 trade to 1.0 at 20 trades
        if trade_count < 20:
            penalty = 0.2 + 0.8 * (trade_count / 20.0)
            fitness_score *= penalty

        grade = self.grade(fitness_score)

        return {
            "total_profit": round(total_profit, 2),
            "total_profit_pct": round(total_profit_pct, 4),
            "max_drawdown_pct": round(max_drawdown_pct, 4),
            "sortino_ratio": round(sortino_ratio, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "calmar_ratio": round(calmar_ratio, 4),
            "profit_factor": round(profit_factor, 4),
            "expectancy": round(expectancy, 4),
            "expectancy_ratio": round(expectancy_ratio, 4),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "trade_count": trade_count,
            "avg_duration_min": round(avg_duration_min, 2),
            "fitness_score": round(fitness_score, 4),
            "grade": grade,
        }

    def grade(self, fitness_score: float) -> str:
        """Convert fitness score to letter grade."""
        for threshold, letter in self.GRADE_THRESHOLDS:
            if fitness_score >= threshold:
                return letter
        return "F"

    def compare(self, period1_trades: list, period2_trades: list) -> dict:
        """Compare two periods and return improvement/degradation."""
        m1 = self.compute(period1_trades)
        m2 = self.compute(period2_trades)

        delta: Dict[str, Any] = {}
        for key in m1:
            if isinstance(m1[key], (int, float)) and isinstance(m2[key], (int, float)):
                delta[key] = round(m2[key] - m1[key], 4)
            else:
                delta[key] = {"before": m1[key], "after": m2[key]}

        improved = delta.get("fitness_score", 0)
        status = "improved" if improved > 0 else ("degraded" if improved < 0 else "unchanged")

        return {
            "period1": m1,
            "period2": m2,
            "delta": delta,
            "status": status,
        }

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict:
        return {
            "total_profit": 0.0,
            "total_profit_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sortino_ratio": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "expectancy_ratio": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "trade_count": 0,
            "avg_duration_min": 0.0,
            "fitness_score": 0.0,
            "grade": "F",
        }


# ---------------------------------------------------------------------------
# 2.  LookaheadDetector
# ---------------------------------------------------------------------------

class LookaheadDetector:
    """Detects lookahead bias in indicator calculations.

    Runs indicators on progressively truncated data and checks for changes
    in values that should already be finalised.
    """

    def __init__(self):
        pass

    def check_indicator(
        self,
        compute_fn: Callable[[list], Any],
        full_data: list,
        check_points: int = 10,
    ) -> dict:
        """Check a single indicator function for lookahead bias.

        Parameters
        ----------
        compute_fn : callable
            Takes a list of numeric values, returns the indicator value
            (scalar) for the **last** element of the input list.
        full_data : list
            The complete price series.
        check_points : int
            Number of evenly-spaced points to test.

        Returns
        -------
        dict  {has_bias, biased_points, severity}
        """
        n = len(full_data)
        if n < 4:
            return {"has_bias": False, "biased_points": [], "severity": "none"}

        # pick checkpoint indices (evenly spaced, excluding the very end)
        step = max(1, (n - 2) // check_points)
        indices = list(range(max(2, n // 4), n - 1, step))[:check_points]

        biased: List[int] = []

        for idx in indices:
            # value at idx using data up to idx+1
            truncated = full_data[: idx + 1]
            try:
                val_truncated = compute_fn(truncated)
            except Exception:
                continue

            # value at idx using MORE future data
            extended = full_data[: min(n, idx + 1 + max(10, n // 10))]
            try:
                val_extended = compute_fn(extended)
            except Exception:
                continue

            # The indicator value for position idx should NOT change when we
            # add data AFTER idx.  We re-compute for the truncated-length
            # position, so we need the compute_fn to return a single value
            # for the last point.
            if val_truncated is None or val_extended is None:
                continue

            # compare with tolerance
            try:
                if abs(float(val_truncated) - float(val_extended)) > 1e-9:
                    biased.append(idx)
            except (TypeError, ValueError):
                if val_truncated != val_extended:
                    biased.append(idx)

        bias_ratio = len(biased) / max(len(indices), 1)
        if bias_ratio == 0:
            severity = "none"
        elif bias_ratio < 0.2:
            severity = "low"
        elif bias_ratio < 0.5:
            severity = "medium"
        else:
            severity = "high"

        return {
            "has_bias": len(biased) > 0,
            "biased_points": biased,
            "severity": severity,
        }

    def check_journal_leakage(self, journal_path: str) -> dict:
        """Check if trade memory/journal queries could leak future data.

        Verifies that:
        1. Timestamps are chronologically ordered.
        2. No trade references data (exit, pnl) from after its entry time
           unless it is a closing trade.
        """
        path = Path(journal_path)
        if not path.exists():
            return {"clean": True, "issues": [], "note": "journal file not found"}

        issues: List[str] = []
        prev_ts: Optional[datetime] = None
        trades_checked = 0

        with open(path, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    trade = json.loads(raw)
                except json.JSONDecodeError:
                    issues.append(f"line {line_no}: invalid JSON")
                    continue

                trades_checked += 1
                ts_str = trade.get("timestamp") or trade.get("entry_time", "")
                if not ts_str:
                    continue

                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except ValueError:
                    issues.append(f"line {line_no}: unparseable timestamp '{ts_str}'")
                    continue

                # chronological order
                if prev_ts is not None and ts < prev_ts:
                    issues.append(
                        f"line {line_no}: out-of-order timestamp "
                        f"{ts_str} < previous"
                    )
                prev_ts = ts

                # open trades should not have exit data filled
                action = trade.get("action", "")
                if "open" in action:
                    exit_price = trade.get("exit_price", 0)
                    pnl = trade.get("pnl_usd", 0)
                    if exit_price and float(exit_price) != 0:
                        issues.append(
                            f"line {line_no}: open trade has exit_price={exit_price}"
                        )
                    if pnl and float(pnl) != 0:
                        issues.append(
                            f"line {line_no}: open trade has pnl_usd={pnl}"
                        )

        return {
            "clean": len(issues) == 0,
            "issues": issues,
            "trades_checked": trades_checked,
        }


# ---------------------------------------------------------------------------
# 3.  ParameterSpace
# ---------------------------------------------------------------------------

class ParameterSpace:
    """Defines the searchable parameter space for system optimization.

    Can be used with manual grid search or optuna if available.
    """

    PARAMS: Dict[str, dict] = {
        "ema_period":          {"type": "int",   "low": 5,    "high": 21,   "default": 8},
        "min_entry_score":     {"type": "int",   "low": 3,    "high": 8,    "default": 4},
        "min_confidence":      {"type": "float", "low": 0.4,  "high": 0.8,  "default": 0.55},
        "atr_sl_mult":         {"type": "float", "low": 1.5,  "high": 4.0,  "default": 2.5},
        "breakeven_pct":       {"type": "float", "low": 0.5,  "high": 2.0,  "default": 0.8},
        "lock_15_pct":         {"type": "float", "low": 1.0,  "high": 3.0,  "default": 1.5},
        "lock_25_pct":         {"type": "float", "low": 1.5,  "high": 4.0,  "default": 2.0},
        "hard_stop_pct":       {"type": "float", "low": 2.0,  "high": 6.0,  "default": 4.0},
        "kelly_fraction":      {"type": "float", "low": 0.1,  "high": 0.5,  "default": 0.5},
        "cooldown_base_min":   {"type": "int",   "low": 5,    "high": 30,   "default": 10},
        "bear_veto_threshold": {"type": "int",   "low": 5,    "high": 9,    "default": 7},
        "roi_0":               {"type": "float", "low": 0.03, "high": 0.15, "default": 0.08},
        "roi_15":              {"type": "float", "low": 0.01, "high": 0.08, "default": 0.04},
        "roi_30":              {"type": "float", "low": 0.005,"high": 0.05, "default": 0.02},
        "roi_60":              {"type": "float", "low": 0.0,  "high": 0.02, "default": 0.005},
        "roi_120":             {"type": "float", "low": 0.0,  "high": 0.01, "default": 0.0},
        "dca_threshold_pct":   {"type": "float", "low": -5.0, "high": -1.0, "default": -2.0},
        "partial_exit_1_pct":  {"type": "float", "low": 2.0,  "high": 5.0,  "default": 3.0},
        "partial_exit_1_frac": {"type": "float", "low": 0.1,  "high": 0.4,  "default": 0.25},
    }

    def get_default_config(self) -> dict:
        """Return default parameter values."""
        return {k: v["default"] for k, v in self.PARAMS.items()}

    def generate_random(self, seed: Optional[int] = None) -> dict:
        """Generate a random parameter set within bounds."""
        rng = random.Random(seed)
        cfg: dict = {}
        for name, spec in self.PARAMS.items():
            low, high = spec["low"], spec["high"]
            if spec["type"] == "int":
                cfg[name] = rng.randint(int(low), int(high))
            else:
                cfg[name] = round(rng.uniform(low, high), 4)
        return cfg

    def grid_search_configs(
        self, params_to_vary: list, steps: int = 5
    ) -> list:
        """Generate grid search configurations for specified parameters.

        Parameters not in *params_to_vary* are fixed at their defaults.
        Returns a list of config dicts (cartesian product).
        """
        base = self.get_default_config()
        axes: List[List[Tuple[str, Any]]] = []

        for name in params_to_vary:
            if name not in self.PARAMS:
                print(f"[OPTIMIZER] WARNING: unknown param '{name}', skipping")
                continue
            spec = self.PARAMS[name]
            low, high = spec["low"], spec["high"]
            if spec["type"] == "int":
                vals = self._linspace_int(int(low), int(high), steps)
            else:
                vals = self._linspace_float(low, high, steps)
            axes.append([(name, v) for v in vals])

        if not axes:
            return [base]

        # cartesian product
        configs: list = []
        self._cartesian(axes, 0, dict(base), configs)
        return configs

    def evaluate_config(
        self, config: dict, trades: list, fitness: "MultiMetricFitness"
    ) -> dict:
        """Evaluate a parameter config against historical trades.

        Currently returns fitness metrics for the trade set.
        Future: re-simulate entries/exits with the config's parameters.
        """
        result = fitness.compute(trades)
        result["config"] = config
        return result

    def run_grid_search(
        self,
        journal_path: str,
        params_to_vary: Optional[list] = None,
        steps: int = 3,
    ) -> dict:
        """Run grid search optimisation using journal data.

        Returns {best_config, best_fitness, all_results}
        """
        trades = _load_closed_trades(journal_path)
        if not trades:
            print("[OPTIMIZER] No closed trades found in journal")
            return {"best_config": {}, "best_fitness": 0.0, "all_results": []}

        if params_to_vary is None:
            params_to_vary = ["min_confidence", "atr_sl_mult", "breakeven_pct"]

        configs = self.grid_search_configs(params_to_vary, steps)
        fitness = MultiMetricFitness()
        all_results: list = []

        print(f"[OPTIMIZER] Grid search: {len(configs)} configs, "
              f"{len(trades)} trades")

        best_score = -float("inf")
        best_config: dict = {}

        for cfg in configs:
            result = self.evaluate_config(cfg, trades, fitness)
            all_results.append(result)
            if result["fitness_score"] > best_score:
                best_score = result["fitness_score"]
                best_config = cfg

        all_results.sort(key=lambda r: r["fitness_score"], reverse=True)
        print(f"[OPTIMIZER] Best fitness: {best_score:.4f}")

        return {
            "best_config": best_config,
            "best_fitness": round(best_score, 4),
            "all_results": all_results,
        }

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _linspace_int(low: int, high: int, steps: int) -> List[int]:
        if steps <= 1 or low == high:
            return [low]
        vals = set()
        for i in range(steps):
            vals.add(round(low + (high - low) * i / (steps - 1)))
        return sorted(vals)

    @staticmethod
    def _linspace_float(low: float, high: float, steps: int) -> List[float]:
        if steps <= 1 or low == high:
            return [round(low, 4)]
        return [round(low + (high - low) * i / (steps - 1), 4) for i in range(steps)]

    @staticmethod
    def _cartesian(
        axes: List[list], depth: int, current: dict, out: list
    ) -> None:
        if depth == len(axes):
            out.append(dict(current))
            return
        for name, val in axes[depth]:
            current[name] = val
            ParameterSpace._cartesian(axes, depth + 1, current, out)


# ---------------------------------------------------------------------------
# 4.  PerformanceAnalyzer
# ---------------------------------------------------------------------------

class PerformanceAnalyzer:
    """Analyzes trading performance from journal data."""

    def __init__(self, journal_path: str = "logs/trading_journal.jsonl"):
        self.journal_path = journal_path
        self.fitness = MultiMetricFitness()

    def load_trades(self) -> list:
        """Load closed trades from journal (entries with nonzero pnl)."""
        return _load_closed_trades(self.journal_path)

    def daily_report(self) -> str:
        """Generate daily performance report.

        Today's trades, P&L, win rate, best/worst trade.
        Compare to 7-day average.
        """
        trades = self.load_trades()
        if not trades:
            return "[OPTIMIZER] No closed trades to report."

        today = datetime.utcnow().strftime("%Y-%m-%d")
        today_trades = [t for t in trades if _trade_date(t) == today]
        week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        week_trades = [t for t in trades if _trade_date(t) >= week_ago]

        today_m = self.fitness.compute(today_trades)
        week_m = self.fitness.compute(week_trades)

        lines = [
            f"=== Daily Report — {today} ===",
            f"Trades today: {today_m['trade_count']}",
            f"P&L (USD):    ${today_m['total_profit']:.2f}",
            f"P&L (%):      {today_m['total_profit_pct']:.2f}%",
            f"Win rate:     {today_m['win_rate']*100:.1f}%",
            f"Fitness:      {today_m['fitness_score']:.2f} ({today_m['grade']})",
        ]

        if today_trades:
            best = max(today_trades, key=lambda t: float(t.get("pnl_usd", 0)))
            worst = min(today_trades, key=lambda t: float(t.get("pnl_usd", 0)))
            lines.append(f"Best trade:   ${float(best.get('pnl_usd', 0)):.2f} "
                         f"({best.get('asset', '?')})")
            lines.append(f"Worst trade:  ${float(worst.get('pnl_usd', 0)):.2f} "
                         f"({worst.get('asset', '?')})")

        lines.append("")
        lines.append(f"--- 7-Day Summary ---")
        lines.append(f"Trades:       {week_m['trade_count']}")
        lines.append(f"P&L (USD):    ${week_m['total_profit']:.2f}")
        lines.append(f"Win rate:     {week_m['win_rate']*100:.1f}%")
        lines.append(f"Fitness:      {week_m['fitness_score']:.2f} ({week_m['grade']})")

        report = "\n".join(lines)
        print(f"[OPTIMIZER] Daily report generated")
        return report

    def tag_analysis(self, trades: Optional[list] = None) -> dict:
        """Analyze performance by entry/exit tags (exit_reason field)."""
        if trades is None:
            trades = self.load_trades()

        groups: Dict[str, list] = defaultdict(list)
        for t in trades:
            tag = t.get("exit_reason") or t.get("entry_tag") or "unknown"
            groups[tag].append(t)

        result: dict = {}
        for tag, group in sorted(groups.items()):
            m = self.fitness.compute(group)
            result[tag] = {
                "trade_count": m["trade_count"],
                "total_profit": m["total_profit"],
                "win_rate": m["win_rate"],
                "avg_win": m["avg_win"],
                "avg_loss": m["avg_loss"],
                "profit_factor": m["profit_factor"],
                "fitness_score": m["fitness_score"],
                "grade": m["grade"],
            }

        return result

    def regime_analysis(self, trades: Optional[list] = None) -> dict:
        """Analyze performance by market regime (direction of trade)."""
        if trades is None:
            trades = self.load_trades()

        groups: Dict[str, list] = defaultdict(list)
        for t in trades:
            action = t.get("action", "")
            if "long" in action.lower():
                regime = "long"
            elif "short" in action.lower():
                regime = "short"
            else:
                regime = t.get("direction", "unknown")
            groups[regime].append(t)

        return {
            regime: self.fitness.compute(group)
            for regime, group in sorted(groups.items())
        }

    def time_analysis(self, trades: Optional[list] = None) -> dict:
        """Analyze performance by hour of day and day of week."""
        if trades is None:
            trades = self.load_trades()

        by_hour: Dict[int, list] = defaultdict(list)
        by_dow: Dict[str, list] = defaultdict(list)
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        for t in trades:
            ts = _parse_ts(t)
            if ts is None:
                continue
            by_hour[ts.hour].append(t)
            by_dow[day_names[ts.weekday()]].append(t)

        hour_metrics = {}
        for h in sorted(by_hour):
            m = self.fitness.compute(by_hour[h])
            hour_metrics[h] = {
                "trade_count": m["trade_count"],
                "total_profit": m["total_profit"],
                "win_rate": m["win_rate"],
            }

        dow_metrics = {}
        for d in day_names:
            if d in by_dow:
                m = self.fitness.compute(by_dow[d])
                dow_metrics[d] = {
                    "trade_count": m["trade_count"],
                    "total_profit": m["total_profit"],
                    "win_rate": m["win_rate"],
                }

        return {"by_hour": hour_metrics, "by_day_of_week": dow_metrics}

    def pair_analysis(self, trades: Optional[list] = None) -> dict:
        """Analyze performance by trading pair/asset."""
        if trades is None:
            trades = self.load_trades()

        groups: Dict[str, list] = defaultdict(list)
        for t in trades:
            asset = t.get("asset") or t.get("pair") or "unknown"
            groups[asset].append(t)

        return {
            asset: self.fitness.compute(group)
            for asset, group in sorted(groups.items())
        }

    def generate_full_report(self) -> str:
        """Generate comprehensive multi-section report."""
        trades = self.load_trades()
        if not trades:
            return "[OPTIMIZER] No trades to analyze."

        sections: List[str] = []

        # Overall
        overall = self.fitness.compute(trades)
        sections.append("=" * 60)
        sections.append("  FULL PERFORMANCE REPORT")
        sections.append("=" * 60)
        sections.append("")
        sections.append(f"Total trades:    {overall['trade_count']}")
        sections.append(f"Total P&L:       ${overall['total_profit']:.2f} "
                        f"({overall['total_profit_pct']:.2f}%)")
        sections.append(f"Max drawdown:    {overall['max_drawdown_pct']*100:.2f}%")
        sections.append(f"Win rate:        {overall['win_rate']*100:.1f}%")
        sections.append(f"Profit factor:   {overall['profit_factor']:.2f}")
        sections.append(f"Expectancy:      ${overall['expectancy']:.2f}")
        sections.append(f"Sortino:         {overall['sortino_ratio']:.2f}")
        sections.append(f"Sharpe:          {overall['sharpe_ratio']:.2f}")
        sections.append(f"Calmar:          {overall['calmar_ratio']:.2f}")
        sections.append(f"Fitness:         {overall['fitness_score']:.2f} "
                        f"({overall['grade']})")
        sections.append(f"Avg duration:    {overall['avg_duration_min']:.1f} min")

        # By pair
        sections.append("")
        sections.append("-" * 40)
        sections.append("  BY ASSET")
        sections.append("-" * 40)
        for asset, m in self.pair_analysis(trades).items():
            sections.append(
                f"  {asset:>6}  trades={m['trade_count']:>3}  "
                f"P&L=${m['total_profit']:>8.2f}  "
                f"WR={m['win_rate']*100:>5.1f}%  "
                f"grade={m['grade']}"
            )

        # By exit reason
        sections.append("")
        sections.append("-" * 40)
        sections.append("  BY EXIT REASON")
        sections.append("-" * 40)
        for tag, m in self.tag_analysis(trades).items():
            sections.append(
                f"  {tag:<20}  trades={m['trade_count']:>3}  "
                f"P&L=${m['total_profit']:>8.2f}  "
                f"WR={m['win_rate']*100:>5.1f}%"
            )

        # By time
        sections.append("")
        sections.append("-" * 40)
        sections.append("  BY HOUR (UTC)")
        sections.append("-" * 40)
        time_data = self.time_analysis(trades)
        for h, m in time_data["by_hour"].items():
            sections.append(
                f"  {h:02d}:00  trades={m['trade_count']:>3}  "
                f"P&L=${m['total_profit']:>8.2f}  "
                f"WR={m['win_rate']*100:>5.1f}%"
            )

        sections.append("")
        sections.append("-" * 40)
        sections.append("  BY DAY OF WEEK")
        sections.append("-" * 40)
        for d, m in time_data["by_day_of_week"].items():
            sections.append(
                f"  {d}  trades={m['trade_count']:>3}  "
                f"P&L=${m['total_profit']:>8.2f}  "
                f"WR={m['win_rate']*100:>5.1f}%"
            )

        report = "\n".join(sections)
        print("[OPTIMIZER] Full report generated")
        return report


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_closed_trades(journal_path: str) -> list:
    """Load closed trades (nonzero pnl_usd) from a JSONL journal."""
    path = Path(journal_path)
    if not path.exists():
        print(f"[OPTIMIZER] Journal not found: {journal_path}")
        return []

    trades: list = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            pnl = entry.get("pnl_usd", 0)
            if pnl is None:
                continue
            if float(pnl) != 0:
                trades.append(entry)
    return trades


def _trade_date(trade: dict) -> str:
    """Extract YYYY-MM-DD from trade timestamp."""
    ts = trade.get("timestamp") or trade.get("entry_time", "")
    return ts[:10] if ts else ""


def _parse_ts(trade: dict) -> Optional[datetime]:
    """Parse trade timestamp to datetime."""
    ts_str = trade.get("timestamp") or trade.get("entry_time", "")
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    journal = "logs/trading_journal.jsonl"
    if len(sys.argv) > 1:
        journal = sys.argv[1]

    print("[OPTIMIZER] Running analysis...")
    analyzer = PerformanceAnalyzer(journal)
    print(analyzer.generate_full_report())

    print("\n")
    print("[OPTIMIZER] Checking journal for lookahead leakage...")
    detector = LookaheadDetector()
    leak = detector.check_journal_leakage(journal)
    if leak["clean"]:
        print("[OPTIMIZER] Journal is clean — no leakage detected")
    else:
        print(f"[OPTIMIZER] Found {len(leak['issues'])} issues:")
        for issue in leak["issues"][:20]:
            print(f"  - {issue}")

    print("\n")
    print("[OPTIMIZER] Running grid search (default params)...")
    space = ParameterSpace()
    gs = space.run_grid_search(journal)
    if gs["best_config"]:
        print(f"[OPTIMIZER] Best config: {json.dumps(gs['best_config'], indent=2)}")
