"""
Monte Carlo Backtest Simulator — ACT v8.0
Runs 10,000 path simulations from trade history to assess risk.
Outputs: probability of ruin, VaR, optimal position sizing.
Auto-runs every 20 live trades to update risk parameters.
"""
import math
import random
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class MonteCarloBacktest:
    def __init__(self, config: dict = None):
        cfg = config or {}
        self.num_simulations = cfg.get('monte_carlo_runs', 10000)
        self.trigger_every = cfg.get('monte_carlo_trigger_every_n_trades', 20)
        self._trade_count = 0
        self._last_result = None

    def run(self, trade_returns: List[float], initial_equity: float = 16000,
            max_drawdown_pct: float = 15.0) -> dict:
        """Run Monte Carlo simulation on trade history.

        Args:
            trade_returns: list of trade PnL percentages
            initial_equity: starting equity in USD
            max_drawdown_pct: ruin threshold
        """
        if len(trade_returns) < 10:
            return {'error': 'Need at least 10 trades for Monte Carlo'}

        n_trades = len(trade_returns)
        ruin_count = 0
        final_equities = []
        max_drawdowns = []
        peak_to_trough = []

        for _ in range(self.num_simulations):
            # Randomly resample trade sequence
            sampled = random.choices(trade_returns, k=n_trades)
            equity = initial_equity
            peak = equity
            max_dd = 0

            for ret_pct in sampled:
                equity *= (1 + ret_pct / 100)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            final_equities.append(equity)
            max_drawdowns.append(max_dd)
            if max_dd >= max_drawdown_pct:
                ruin_count += 1

        final_equities.sort()
        max_drawdowns.sort()

        # Percentiles
        p5 = final_equities[int(self.num_simulations * 0.05)]
        p25 = final_equities[int(self.num_simulations * 0.25)]
        p50 = final_equities[int(self.num_simulations * 0.50)]
        p75 = final_equities[int(self.num_simulations * 0.75)]
        p95 = final_equities[int(self.num_simulations * 0.95)]

        # VaR
        returns_sorted = sorted([(e - initial_equity) / initial_equity * 100 for e in final_equities])
        var_95 = returns_sorted[int(self.num_simulations * 0.05)]
        var_99 = returns_sorted[int(self.num_simulations * 0.01)]

        # Sharpe from simulations
        mean_ret = sum(returns_sorted) / len(returns_sorted)
        std_ret = math.sqrt(sum((r - mean_ret) ** 2 for r in returns_sorted) / len(returns_sorted))
        sim_sharpe = mean_ret / std_ret if std_ret > 0 else 0

        self._last_result = {
            'simulations': self.num_simulations,
            'input_trades': n_trades,
            'probability_of_ruin': round(ruin_count / self.num_simulations, 4),
            'var_95': round(var_95, 2),
            'var_99': round(var_99, 2),
            'equity_p5': round(p5, 2),
            'equity_p25': round(p25, 2),
            'equity_p50': round(p50, 2),
            'equity_p75': round(p75, 2),
            'equity_p95': round(p95, 2),
            'median_max_drawdown': round(max_drawdowns[int(len(max_drawdowns) * 0.5)], 2),
            'worst_max_drawdown': round(max_drawdowns[-1], 2),
            'simulated_sharpe': round(sim_sharpe, 3),
        }

        logger.info(f"[MONTE CARLO] P(ruin)={self._last_result['probability_of_ruin']:.1%} "
                     f"VaR95={var_95:.1f}% median_equity=${p50:.0f}")
        return self._last_result

    def get_position_sizing_recommendation(self, trade_returns: List[float],
                                            initial_equity: float = 16000,
                                            target_ruin_prob: float = 0.05) -> dict:
        """Binary search for max position size that keeps P(ruin) < target."""
        best_size = 0.5  # start at 0.5% risk
        for size_pct in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            # Scale returns by position size relative to 1%
            scaled = [r * size_pct for r in trade_returns]
            result = self.run(scaled, initial_equity)
            if result.get('probability_of_ruin', 1.0) < target_ruin_prob:
                best_size = size_pct
            else:
                break

        return {
            'recommended_risk_per_trade_pct': best_size,
            'target_ruin_probability': target_ruin_prob,
            'trades_analyzed': len(trade_returns),
        }

    def maybe_run(self, trade_returns: List[float], initial_equity: float = 16000) -> Optional[dict]:
        """Auto-trigger: runs every N trades."""
        self._trade_count += 1
        if self._trade_count >= self.trigger_every:
            self._trade_count = 0
            return self.run(trade_returns, initial_equity)
        return None

    def get_last_result(self) -> Optional[dict]:
        return self._last_result
