"""
Genetic Strategy Evolution Engine — Self-Evolving Trading Strategies
=====================================================================
Creates NEW strategies by combining, mutating, and evolving existing ones.
Specifically optimized for Robinhood's 3.34% spread — only strategies
that produce 5%+ moves survive natural selection.

Architecture:
  1. GENESIS: Take top-performing strategies from backtests
  2. CROSSOVER: Combine rules from 2 parent strategies → child
  3. MUTATION: Randomly adjust parameters (periods, thresholds)
  4. SELECTION: Backtest children → keep winners, discard losers
  5. EVOLUTION: Repeat for N generations → strategies get better

This is how quant hedge funds discover alpha — automated strategy R&D
that runs 24/7 testing thousands of hypotheses on real market data.

Usage:
    python -m src.trading.genetic_strategy_engine                  # Single evolution cycle
    python -m src.trading.genetic_strategy_engine --generations 10  # 10 generations
    python -m src.trading.genetic_strategy_engine --continuous      # Evolve forever
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timezone
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.indicators.indicators import (
    sma, ema, rsi, macd, bollinger_bands, atr, stochastic,
    vwap, obv, adx, roc, williams_r, mfi, supertrend,
    choppiness_index, bb_width,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPREAD_PCT = 3.34


# ═══════════════════════════════════════════════════════════════
# DNA: Strategy genes that can be combined and mutated
# ═══════════════════════════════════════════════════════════════

# Available indicator functions and their parameter ranges
INDICATOR_GENES = {
    'ema_fast': {'func': 'ema', 'param': 'period', 'range': (5, 30), 'default': 8},
    'ema_slow': {'func': 'ema', 'param': 'period', 'range': (20, 100), 'default': 50},
    'rsi': {'func': 'rsi', 'param': 'period', 'range': (7, 28), 'default': 14},
    'rsi_oversold': {'func': 'threshold', 'param': 'value', 'range': (15, 40), 'default': 30},
    'rsi_overbought': {'func': 'threshold', 'param': 'value', 'range': (60, 85), 'default': 70},
    'bb_period': {'func': 'bollinger', 'param': 'period', 'range': (10, 40), 'default': 20},
    'bb_std': {'func': 'bollinger', 'param': 'std_dev', 'range': (1.0, 3.0), 'default': 2.0},
    'atr_period': {'func': 'atr', 'param': 'period', 'range': (7, 28), 'default': 14},
    'roc_period': {'func': 'roc', 'param': 'period', 'range': (5, 30), 'default': 10},
    'adx_threshold': {'func': 'threshold', 'param': 'value', 'range': (15, 40), 'default': 25},
    'stoch_k': {'func': 'stochastic', 'param': 'k_period', 'range': (5, 21), 'default': 14},
    'volume_mult': {'func': 'threshold', 'param': 'value', 'range': (1.0, 3.0), 'default': 1.5},
    'lookback': {'func': 'window', 'param': 'bars', 'range': (5, 50), 'default': 20},
    'min_bars_trend': {'func': 'threshold', 'param': 'value', 'range': (2, 10), 'default': 3},
}

# Entry condition templates (combine indicators into rules)
ENTRY_TEMPLATES = [
    'ema_cross',           # Fast EMA crosses above slow EMA
    'rsi_oversold_bounce', # RSI drops below threshold then reverses
    'bb_lower_touch',      # Price touches lower BB + RSI confirmation
    'momentum_surge',      # ROC + volume spike
    'trend_strength',      # ADX > threshold + EMA aligned
    'breakout_volume',     # Price breaks lookback high + volume
    'stoch_reversal',      # Stochastic crosses up from oversold
    'ema_bounce',          # Price bounces off EMA with volume
    'multi_ma_align',      # 3 EMAs aligned (fast > mid > slow)
    'volatility_squeeze',  # BB narrows then expands with direction
]

# Exit condition templates
EXIT_TEMPLATES = [
    'ema_flip',            # EMA direction reverses
    'rsi_extreme',         # RSI hits overbought
    'trailing_atr',        # Trail at N×ATR from peak
    'time_decay',          # Exit after N bars
    'profit_target',       # Fixed % profit target
    'momentum_fade',       # ROC declines below threshold
]


class StrategyDNA:
    """
    A strategy's genetic code — parameters + rules that can be evolved.
    """
    def __init__(self):
        # Indicator parameters (genes)
        self.genes: Dict[str, float] = {}
        for name, gene in INDICATOR_GENES.items():
            self.genes[name] = gene['default']

        # Entry rule (which template)
        self.entry_rule: str = random.choice(ENTRY_TEMPLATES)

        # Exit rule
        self.exit_rule: str = random.choice(EXIT_TEMPLATES)

        # Robinhood-specific: minimum move target (must clear spread)
        self.min_move_pct: float = SPREAD_PCT * 1.5  # 5% minimum

        # Fitness score (from backtesting)
        self.fitness: float = 0.0
        self.win_rate: float = 0.0
        self.total_pnl: float = 0.0
        self.trades: int = 0
        self.sharpe: float = 0.0

        # Lineage
        self.generation: int = 0
        self.parents: List[str] = []
        self.name: str = f"GEN0_{random.randint(1000,9999)}"

    def mutate(self, mutation_rate: float = 0.3):
        """Randomly adjust some genes."""
        for gene_name, gene_info in INDICATOR_GENES.items():
            if random.random() < mutation_rate:
                low, high = gene_info['range']
                if isinstance(low, float):
                    self.genes[gene_name] = round(random.uniform(low, high), 2)
                else:
                    self.genes[gene_name] = random.randint(low, high)

        # Small chance to change entry/exit rule
        if random.random() < 0.15:
            self.entry_rule = random.choice(ENTRY_TEMPLATES)
        if random.random() < 0.15:
            self.exit_rule = random.choice(EXIT_TEMPLATES)

    @staticmethod
    def crossover(parent1: 'StrategyDNA', parent2: 'StrategyDNA') -> 'StrategyDNA':
        """Create child by combining genes from two parents."""
        child = StrategyDNA()
        child.generation = max(parent1.generation, parent2.generation) + 1
        child.parents = [parent1.name, parent2.name]
        child.name = f"GEN{child.generation}_{random.randint(1000,9999)}"

        # 50/50 gene inheritance
        for gene_name in child.genes:
            if random.random() < 0.5:
                child.genes[gene_name] = parent1.genes[gene_name]
            else:
                child.genes[gene_name] = parent2.genes[gene_name]

        # Entry from fitter parent
        if parent1.fitness >= parent2.fitness:
            child.entry_rule = parent1.entry_rule
            child.exit_rule = parent2.exit_rule
        else:
            child.entry_rule = parent2.entry_rule
            child.exit_rule = parent1.exit_rule

        # Small mutation
        child.mutate(mutation_rate=0.2)
        return child

    def to_dict(self) -> dict:
        return {
            'name': self.name, 'generation': self.generation,
            'genes': self.genes, 'entry_rule': self.entry_rule,
            'exit_rule': self.exit_rule, 'fitness': self.fitness,
            'win_rate': self.win_rate, 'total_pnl': self.total_pnl,
            'trades': self.trades, 'sharpe': self.sharpe,
            'parents': self.parents,
        }


# ═══════════════════════════════════════════════════════════════
# Strategy Execution: Convert DNA into trading signals
# ═══════════════════════════════════════════════════════════════

def execute_strategy(dna: StrategyDNA, closes, highs, lows, volumes) -> int:
    """Execute a strategy DNA on market data, return signal -1/0/+1."""
    n = len(closes)
    if n < 60:
        return 0

    try:
        g = dna.genes

        # Compute indicators based on genes
        ema_f = np.array(ema(list(closes), int(g['ema_fast'])))
        ema_s = np.array(ema(list(closes), int(g['ema_slow'])))
        rsi_v = np.array(rsi(list(closes), int(g['rsi'])))
        roc_v = np.array(roc(list(closes), int(g['roc_period'])))
        atr_v = np.array(atr(list(highs), list(lows), list(closes), int(g['atr_period'])))

        i = -1  # Last bar
        price = closes[i]

        # Entry rules
        if dna.entry_rule == 'ema_cross':
            if ema_f[i] > ema_s[i] and ema_f[i-1] <= ema_s[i-1]:
                return 1
            if ema_f[i] < ema_s[i] and ema_f[i-1] >= ema_s[i-1]:
                return -1

        elif dna.entry_rule == 'rsi_oversold_bounce':
            if rsi_v[i] > g['rsi_oversold'] and rsi_v[i-1] <= g['rsi_oversold']:
                return 1
            if rsi_v[i] < g['rsi_overbought'] and rsi_v[i-1] >= g['rsi_overbought']:
                return -1

        elif dna.entry_rule == 'bb_lower_touch':
            bb_u, bb_m, bb_l = bollinger_bands(list(closes), int(g['bb_period']), g['bb_std'])
            if price <= bb_l[i] and rsi_v[i] < g['rsi_oversold'] + 5:
                return 1
            if price >= bb_u[i] and rsi_v[i] > g['rsi_overbought'] - 5:
                return -1

        elif dna.entry_rule == 'momentum_surge':
            if roc_v[i] > 1.0 and volumes[i] > np.mean(volumes[-20:]) * g['volume_mult']:
                return 1
            if roc_v[i] < -1.0 and volumes[i] > np.mean(volumes[-20:]) * g['volume_mult']:
                return -1

        elif dna.entry_rule == 'trend_strength':
            adx_vals = adx(list(highs), list(lows), list(closes), 14)
            if isinstance(adx_vals, tuple):
                adx_line = adx_vals[0]
            else:
                adx_line = adx_vals
            adx_v = np.array(adx_line)
            if adx_v[i] > g['adx_threshold'] and ema_f[i] > ema_s[i]:
                return 1

        elif dna.entry_rule == 'breakout_volume':
            lookback = int(g['lookback'])
            if price > max(highs[-lookback-1:-1]) and volumes[i] > np.mean(volumes[-lookback:]) * g['volume_mult']:
                return 1

        elif dna.entry_rule == 'stoch_reversal':
            k_vals, d_vals = stochastic(list(highs), list(lows), list(closes), int(g['stoch_k']), 3)
            if k_vals[i] > d_vals[i] and k_vals[i-1] <= d_vals[i-1] and k_vals[i] < 30:
                return 1

        elif dna.entry_rule == 'ema_bounce':
            ema_dist = abs(price - ema_f[i]) / price * 100
            if price > ema_f[i] and ema_dist < 0.5 and ema_f[i] > ema_f[i-1] and volumes[i] > np.mean(volumes[-10:]):
                return 1

        elif dna.entry_rule == 'multi_ma_align':
            ema_mid = np.array(ema(list(closes), int((g['ema_fast'] + g['ema_slow']) / 2)))
            if ema_f[i] > ema_mid[i] > ema_s[i] and ema_f[i] > ema_f[i-1]:
                return 1
            if ema_f[i] < ema_mid[i] < ema_s[i] and ema_f[i] < ema_f[i-1]:
                return -1

        elif dna.entry_rule == 'volatility_squeeze':
            bw = np.array(bb_width(list(closes), int(g['bb_period'])))
            if bw[i] > bw[i-1] and bw[i-1] < np.mean(bw[-20:]) * 0.8 and ema_f[i] > ema_s[i]:
                return 1

    except Exception:
        pass

    return 0


# ═══════════════════════════════════════════════════════════════
# Backtesting: Evaluate fitness of a strategy DNA
# ═══════════════════════════════════════════════════════════════

def backtest_dna(dna: StrategyDNA, closes, highs, lows, volumes,
                 spread_pct: float = SPREAD_PCT, max_hold: int = 30) -> Dict:
    """Backtest a strategy DNA and compute fitness metrics."""
    n = len(closes)
    if n < 100:
        return {'fitness': 0, 'trades': 0}

    trades = []
    position = None

    for i in range(60, n - max_hold):
        c_slice = closes[:i+1]
        h_slice = highs[:i+1]
        l_slice = lows[:i+1]
        v_slice = volumes[:i+1]

        if position is None:
            sig = execute_strategy(dna, c_slice, h_slice, l_slice, v_slice)
            if sig == 1:  # LONG only for Robinhood
                position = {'entry': closes[i], 'entry_idx': i}

        elif position is not None:
            bars_held = i - position['entry_idx']
            pnl_pct = (closes[i] - position['entry']) / position['entry'] * 100

            # Exit conditions based on DNA exit rule
            should_exit = False

            if dna.exit_rule == 'ema_flip':
                ema_f = ema(list(c_slice), int(dna.genes['ema_fast']))
                if len(ema_f) >= 2 and ema_f[-1] < ema_f[-2]:
                    should_exit = pnl_pct - spread_pct > 0  # Only exit if profitable after spread

            elif dna.exit_rule == 'rsi_extreme':
                rsi_v = rsi(list(c_slice), int(dna.genes['rsi']))
                if rsi_v[-1] > dna.genes['rsi_overbought']:
                    should_exit = True

            elif dna.exit_rule == 'trailing_atr':
                atr_v = atr(list(h_slice), list(l_slice), list(c_slice), int(dna.genes['atr_period']))
                peak = max(closes[position['entry_idx']:i+1])
                trail = peak - atr_v[-1] * 2
                if closes[i] < trail and pnl_pct - spread_pct > 0:
                    should_exit = True

            elif dna.exit_rule == 'profit_target':
                if pnl_pct >= dna.min_move_pct + spread_pct:
                    should_exit = True

            elif dna.exit_rule == 'momentum_fade':
                roc_v = roc(list(c_slice), int(dna.genes['roc_period']))
                if roc_v[-1] < 0 and pnl_pct - spread_pct > 0:
                    should_exit = True

            # Time-based exits
            if bars_held >= max_hold:
                should_exit = True
            if pnl_pct <= -5.0:  # Hard stop
                should_exit = True

            if should_exit:
                net_pnl = pnl_pct - spread_pct
                trades.append({
                    'pnl_pct': net_pnl,
                    'bars_held': bars_held,
                    'won': net_pnl > 0,
                })
                position = None

    # Compute fitness
    if not trades:
        return {'fitness': 0, 'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'sharpe': 0}

    wins = [t for t in trades if t['won']]
    pnls = [t['pnl_pct'] for t in trades]
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(trades)
    avg_pnl = np.mean(pnls)
    std_pnl = np.std(pnls) if len(pnls) > 1 else 1.0
    sharpe = avg_pnl / (std_pnl + 1e-9)

    # Fitness = weighted combination (Robinhood-optimized)
    # Heavy weight on POSITIVE PnL (must clear 3.34% spread)
    fitness = (
        0.40 * min(1.0, max(0, total_pnl / 20))  # PnL (capped at 20%)
        + 0.30 * win_rate                           # Win rate
        + 0.20 * min(1.0, max(0, sharpe / 3))      # Sharpe (capped at 3)
        + 0.10 * min(1.0, len(trades) / 20)        # Trade frequency
    )

    # Penalty for negative PnL
    if total_pnl < 0:
        fitness *= 0.3

    return {
        'fitness': round(fitness, 4),
        'trades': len(trades),
        'wins': len(wins),
        'win_rate': round(win_rate, 4),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 2),
        'sharpe': round(sharpe, 3),
    }


# ═══════════════════════════════════════════════════════════════
# Evolution Engine
# ═══════════════════════════════════════════════════════════════

class GeneticStrategyEngine:
    """
    Evolves trading strategies through genetic algorithms.

    Usage:
        engine = GeneticStrategyEngine()
        engine.load_market_data('BTC')
        winners = engine.evolve(generations=10, population_size=50)
    """

    def __init__(self, spread_pct: float = SPREAD_PCT):
        self.spread_pct = spread_pct
        self.population: List[StrategyDNA] = []
        self.hall_of_fame: List[StrategyDNA] = []  # Best ever strategies
        self.generation = 0
        self.closes = None
        self.highs = None
        self.lows = None
        self.volumes = None

    def load_market_data(self, asset: str = 'BTC', timeframe: str = '4h'):
        """Load OHLCV data for backtesting."""
        try:
            import pandas as pd
            path = os.path.join(PROJECT_ROOT, f'data/{asset}USDT-{timeframe}.parquet')
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            self.closes = df['close'].values.astype(float)
            self.highs = df['high'].values.astype(float)
            self.lows = df['low'].values.astype(float)
            self.volumes = df['volume'].values.astype(float)
            print(f"  Loaded {asset} {timeframe}: {len(self.closes)} bars, last ${self.closes[-1]:,.2f}")
            return True
        except Exception as e:
            logger.error(f"Data load failed: {e}")
            return False

    def initialize_population(self, size: int = 50):
        """Create initial random population."""
        self.population = []
        for _ in range(size):
            dna = StrategyDNA()
            dna.mutate(mutation_rate=0.8)  # Heavy mutation for diversity
            self.population.append(dna)
        print(f"  Initialized {size} random strategies")

    def evaluate_population(self):
        """Backtest every strategy in the population."""
        for dna in self.population:
            result = backtest_dna(dna, self.closes, self.highs, self.lows, self.volumes, self.spread_pct)
            dna.fitness = result['fitness']
            dna.win_rate = result['win_rate']
            dna.total_pnl = result['total_pnl']
            dna.trades = result['trades']
            dna.sharpe = result['sharpe']

        # Sort by fitness
        self.population.sort(key=lambda d: d.fitness, reverse=True)

    def select_parents(self, tournament_size: int = 5) -> Tuple[StrategyDNA, StrategyDNA]:
        """Tournament selection — pick best from random subset."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        tournament.sort(key=lambda d: d.fitness, reverse=True)
        parent1 = tournament[0]
        parent2 = tournament[1] if len(tournament) > 1 else tournament[0]
        return parent1, parent2

    def evolve_generation(self, elite_count: int = 5, mutation_rate: float = 0.3):
        """Create next generation through selection, crossover, and mutation."""
        new_population = []

        # Elitism: keep top N unchanged
        elite = self.population[:elite_count]
        new_population.extend(deepcopy(elite))

        # Fill rest with children
        while len(new_population) < len(self.population):
            parent1, parent2 = self.select_parents()
            child = StrategyDNA.crossover(parent1, parent2)
            child.mutate(mutation_rate)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def update_hall_of_fame(self, max_size: int = 10):
        """Track best strategies across all generations."""
        for dna in self.population:
            if dna.fitness > 0 and dna.total_pnl > 0:
                self.hall_of_fame.append(deepcopy(dna))

        # Keep only top N
        self.hall_of_fame.sort(key=lambda d: d.fitness, reverse=True)
        self.hall_of_fame = self.hall_of_fame[:max_size]

    def evolve(self, generations: int = 10, population_size: int = 50) -> List[StrategyDNA]:
        """Run full evolution loop."""
        print(f"\n{'='*60}")
        print(f"  GENETIC STRATEGY EVOLUTION — {generations} generations × {population_size} strategies")
        print(f"  Spread: {self.spread_pct}% | Only LONG | Target: 5%+ net profit per trade")
        print(f"{'='*60}")

        self.initialize_population(population_size)

        for gen in range(generations):
            self.evaluate_population()
            best = self.population[0]

            # Stats
            avg_fitness = np.mean([d.fitness for d in self.population])
            profitable = sum(1 for d in self.population if d.total_pnl > 0)

            print(f"\n  Gen {gen+1}/{generations}: best={best.name} fitness={best.fitness:.4f} "
                  f"PnL={best.total_pnl:+.1f}% WR={best.win_rate:.0%} trades={best.trades} "
                  f"| avg_fitness={avg_fitness:.4f} | profitable={profitable}/{population_size}")

            if best.total_pnl > 0:
                print(f"    WINNER: {best.entry_rule} + {best.exit_rule} "
                      f"| ema_fast={int(best.genes['ema_fast'])} ema_slow={int(best.genes['ema_slow'])} "
                      f"rsi={int(best.genes['rsi'])} lookback={int(best.genes['lookback'])}")

            self.update_hall_of_fame()
            self.evolve_generation()

        # Final evaluation
        self.evaluate_population()
        self.update_hall_of_fame()

        print(f"\n{'='*60}")
        print(f"  EVOLUTION COMPLETE — {generations} generations")
        print(f"{'='*60}")

        if self.hall_of_fame:
            print(f"\n  HALL OF FAME (profitable after {self.spread_pct}% spread):")
            for i, dna in enumerate(self.hall_of_fame[:5], 1):
                print(f"    #{i} {dna.name}: PnL={dna.total_pnl:+.1f}% WR={dna.win_rate:.0%} "
                      f"trades={dna.trades} sharpe={dna.sharpe:.2f}")
                print(f"       Entry: {dna.entry_rule} | Exit: {dna.exit_rule} "
                      f"| EMA {int(dna.genes['ema_fast'])}/{int(dna.genes['ema_slow'])} "
                      f"RSI {int(dna.genes['rsi'])} ({int(dna.genes['rsi_oversold'])}/{int(dna.genes['rsi_overbought'])})")
        else:
            print(f"\n  No strategies found profitable after {self.spread_pct}% spread")
            print(f"  This confirms: Robinhood's spread is extremely punishing")

        # Save results
        self._save_results()

        return self.hall_of_fame

    def _save_results(self):
        """Save evolution results."""
        results = {
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'generations': self.generation,
            'spread_pct': self.spread_pct,
            'population_size': len(self.population),
            'hall_of_fame': [dna.to_dict() for dna in self.hall_of_fame],
            'best_fitness': self.hall_of_fame[0].fitness if self.hall_of_fame else 0,
            'best_pnl': self.hall_of_fame[0].total_pnl if self.hall_of_fame else 0,
        }

        path = os.path.join(PROJECT_ROOT, 'logs/genetic_evolution_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Also append to history
        hist_path = os.path.join(PROJECT_ROOT, 'logs/genetic_evolution_history.jsonl')
        with open(hist_path, 'a') as f:
            f.write(json.dumps({
                'timestamp': results['timestamp'],
                'generations': self.generation,
                'best_pnl': results['best_pnl'],
                'best_fitness': results['best_fitness'],
                'hall_of_fame_count': len(self.hall_of_fame),
            }, default=str) + '\n')

        print(f"\n  Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='Genetic Strategy Evolution Engine')
    parser.add_argument('--generations', type=int, default=10)
    parser.add_argument('--population', type=int, default=50)
    parser.add_argument('--asset', default='BTC')
    parser.add_argument('--timeframe', default='4h')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--interval', type=float, default=2.0, help='Hours between evolution cycles')
    args = parser.parse_args()

    engine = GeneticStrategyEngine()

    if not engine.load_market_data(args.asset, args.timeframe):
        print("Failed to load market data")
        return

    if args.continuous:
        print(f"Continuous evolution mode (every {args.interval}h)...")
        while True:
            engine.load_market_data(args.asset, args.timeframe)  # Refresh data
            engine.evolve(args.generations, args.population)
            print(f"\nSleeping {args.interval}h...")
            time.sleep(args.interval * 3600)
    else:
        engine.evolve(args.generations, args.population)


if __name__ == '__main__':
    main()
