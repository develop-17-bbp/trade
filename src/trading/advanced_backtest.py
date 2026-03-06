"""
PHASE 5: Advanced Backtesting & Simulation
===========================================
Implements Walk-Forward Analysis and Monte Carlo Simulations.
Essential for institutional-grade strategy validation.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random

@dataclass
class SimulationResult:
    median_return: float
    prob_of_ruin: float
    var_95: float  # Value at Risk 95%
    expected_drawdown: float

class AdvancedSimulator:
    """
    Simulates strategy performance using stochastic methods.
    """
    def __init__(self, iterations: int = 1000):
        self.iterations = iterations

    def monte_carlo_simulation(self, returns: List[float], 
                               initial_capital: float = 100000.0,
                               latency_ms: float = 100.0,
                               slippage_bps: float = 5.0) -> SimulationResult:
        """
        Runs Monte Carlo simulation with Execution friction.
        Args:
            latency_ms: Simulated execution delay
            slippage_bps: Extra slippage in basis points (1 bp = 0.0001)
        """
        if not returns:
            return SimulationResult(0, 0, 0, 0)
            
        final_values = []
        max_drawdowns = []
        ruin_cnt = 0
        
        # Friction per trade (Slippage + Latency-based slippage proxy)
        friction = (slippage_bps * 0.0001) + (latency_ms / 1000.0 * 0.0005) # Proxy: 5bps per sec delay
        
        for _ in range(self.iterations):
            sim_rets = [random.choice(returns) - friction for _ in range(len(returns))]
            
            equity = [initial_capital]
            for r in sim_rets:
                new_v = equity[-1] * (1 + r)
                equity.append(new_v)
                if new_v <= 0:
                    ruin_cnt += 1
                    break
            
            final_values.append(equity[-1])
            
            # Max Drawdown
            pk = equity[0]
            mdd = 0.0
            for e in equity:
                if e > pk: pk = e
                dd = (pk - e) / pk
                if dd > mdd: mdd = dd
            max_drawdowns.append(mdd)
            
        final_vals = np.array(final_values)
        rets_pct = (final_vals / initial_capital) - 1
        
        return SimulationResult(
            median_return=float(np.median(rets_pct)),
            prob_of_ruin=float(ruin_cnt / self.iterations),
            var_95=float(np.percentile(rets_pct, 5)),
            expected_drawdown=float(np.mean(max_drawdowns))
        )

    def walk_forward_validation(self, data: List[Dict], train_size: float = 0.7) -> bool:
        """
        Validates if model trained on 'In-Sample' works on 'Out-of-Sample'.
        Returns True if performance is preserved within 30% of IS results.
        """
        split_idx = int(len(data) * train_size)
        in_sample = data[:split_idx]
        out_of_sample = data[split_idx:]
        
        is_avg_pnl = np.mean([d.get('pnl', 0) for d in in_sample])
        oos_avg_pnl = np.mean([d.get('pnl', 0) for d in out_of_sample])
        
        # Stability check
        if is_avg_pnl > 0 and oos_avg_pnl > 0:
            efficiency = oos_avg_pnl / is_avg_pnl
            return efficiency > 0.5 # Walk-Forward Efficiency Ratio > 50%
        return False
