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

Enhanced with:
  - Adaptive mutation rates (stagnation-aware)
  - Advanced crossover operators (uniform, single-point, arithmetic, blend)
  - Advanced selection (tournament, roulette, rank-based)
  - Diversity preservation via fitness sharing (niching)
  - Multi-objective optimization (NSGA-II Pareto front)
  - Island model for regime-specialized populations
  - MultiMetricFitness integration
  - Comprehensive evolution logging

Usage:
    python -m src.trading.genetic_strategy_engine                  # Single evolution cycle
    python -m src.trading.genetic_strategy_engine --generations 10  # 10 generations
    python -m src.trading.genetic_strategy_engine --multiobjective  # NSGA-II Pareto
    python -m src.trading.genetic_strategy_engine --islands          # Regime islands
    python -m src.trading.genetic_strategy_engine --continuous       # Evolve forever
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
from collections import Counter
import math

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

ENTRY_TEMPLATES = [
    'ema_cross', 'rsi_oversold_bounce', 'bb_lower_touch', 'momentum_surge',
    'trend_strength', 'breakout_volume', 'stoch_reversal', 'ema_bounce',
    'multi_ma_align', 'volatility_squeeze',
]

EXIT_TEMPLATES = [
    'ema_flip', 'rsi_extreme', 'trailing_atr', 'time_decay',
    'profit_target', 'momentum_fade',
]


def _clamp_gene(name: str, value: float) -> float:
    """Clamp a gene value to its valid range."""
    gene_info = INDICATOR_GENES.get(name)
    if not gene_info:
        return value
    lo, hi = gene_info['range']
    return max(lo, min(hi, value))


class StrategyDNA:
    """A strategy's genetic code — parameters + rules that can be evolved."""

    def __init__(self):
        self.genes: Dict[str, float] = {}
        for name, gene in INDICATOR_GENES.items():
            self.genes[name] = gene['default']
        self.entry_rule: str = random.choice(ENTRY_TEMPLATES)
        self.exit_rule: str = random.choice(EXIT_TEMPLATES)
        self.min_move_pct: float = SPREAD_PCT * 1.5
        self.fitness: float = 0.0
        self.win_rate: float = 0.0
        self.total_pnl: float = 0.0
        self.trades: int = 0
        self.sharpe: float = 0.0
        self.metrics: dict = {}  # Full metrics from MultiMetricFitness
        self.generation: int = 0
        self.parents: List[str] = []
        self.name: str = f"GEN0_{random.randint(1000, 9999)}"

    def mutate(self, mutation_rate: float = 0.3):
        """Randomly adjust some genes."""
        for gene_name, gene_info in INDICATOR_GENES.items():
            if random.random() < mutation_rate:
                low, high = gene_info['range']
                if isinstance(low, float):
                    self.genes[gene_name] = round(random.uniform(low, high), 2)
                else:
                    self.genes[gene_name] = random.randint(low, high)
        if random.random() < 0.15:
            self.entry_rule = random.choice(ENTRY_TEMPLATES)
        if random.random() < 0.15:
            self.exit_rule = random.choice(EXIT_TEMPLATES)

    # ── Crossover Operators ──────────────────────────────────────

    @staticmethod
    def crossover(parent1: 'StrategyDNA', parent2: 'StrategyDNA') -> 'StrategyDNA':
        """Create child using a randomly selected crossover operator."""
        op = random.choice(['uniform', 'single_point', 'arithmetic', 'blend'])
        if op == 'single_point':
            child = StrategyDNA.crossover_single_point(parent1, parent2)
        elif op == 'arithmetic':
            child = StrategyDNA.crossover_arithmetic(parent1, parent2)
        elif op == 'blend':
            child = StrategyDNA.crossover_blend(parent1, parent2)
        else:
            child = StrategyDNA._crossover_uniform(parent1, parent2)
        child.generation = max(parent1.generation, parent2.generation) + 1
        child.parents = [parent1.name, parent2.name]
        child.name = f"GEN{child.generation}_{random.randint(1000, 9999)}"
        child.mutate(mutation_rate=0.2)
        return child

    @staticmethod
    def _crossover_uniform(parent1: 'StrategyDNA', parent2: 'StrategyDNA') -> 'StrategyDNA':
        """Original uniform crossover: 50/50 gene inheritance."""
        child = StrategyDNA()
        for gene_name in child.genes:
            p1_val = parent1.genes.get(gene_name, child.genes[gene_name])
            p2_val = parent2.genes.get(gene_name, child.genes[gene_name])
            child.genes[gene_name] = p1_val if random.random() < 0.5 else p2_val
        if parent1.fitness >= parent2.fitness:
            child.entry_rule = parent1.entry_rule
            child.exit_rule = parent2.exit_rule
        else:
            child.entry_rule = parent2.entry_rule
            child.exit_rule = parent1.exit_rule
        return child

    @staticmethod
    def crossover_single_point(parent1: 'StrategyDNA', parent2: 'StrategyDNA') -> 'StrategyDNA':
        """Single-point crossover: genes before cut from p1, after from p2."""
        child = StrategyDNA()
        keys = list(child.genes.keys())
        cut = random.randint(1, max(len(keys) - 1, 1))
        for i, k in enumerate(keys):
            p1_val = parent1.genes.get(k, child.genes[k])
            p2_val = parent2.genes.get(k, child.genes[k])
            child.genes[k] = p1_val if i < cut else p2_val
        child.entry_rule = parent1.entry_rule if parent1.fitness >= parent2.fitness else parent2.entry_rule
        child.exit_rule = parent2.exit_rule if parent1.fitness >= parent2.fitness else parent1.exit_rule
        return child

    @staticmethod
    def crossover_arithmetic(parent1: 'StrategyDNA', parent2: 'StrategyDNA', alpha: float = 0.5) -> 'StrategyDNA':
        """Arithmetic crossover: weighted average of numeric genes."""
        child = StrategyDNA()
        for k in child.genes:
            p1_val = parent1.genes.get(k, child.genes[k])
            p2_val = parent2.genes.get(k, child.genes[k])
            val = alpha * p1_val + (1 - alpha) * p2_val
            child.genes[k] = _clamp_gene(k, val)
        child.entry_rule = parent1.entry_rule if random.random() < alpha else parent2.entry_rule
        child.exit_rule = parent2.exit_rule if random.random() < alpha else parent1.exit_rule
        return child

    @staticmethod
    def crossover_blend(parent1: 'StrategyDNA', parent2: 'StrategyDNA', alpha: float = 0.3) -> 'StrategyDNA':
        """BLX-alpha: sample from expanded range [min-alpha*d, max+alpha*d]."""
        child = StrategyDNA()
        for k in child.genes:
            p1_val = parent1.genes.get(k, child.genes[k])
            p2_val = parent2.genes.get(k, child.genes[k])
            lo = min(p1_val, p2_val)
            hi = max(p1_val, p2_val)
            d = hi - lo
            val = random.uniform(lo - alpha * d, hi + alpha * d)
            child.genes[k] = _clamp_gene(k, val)
        child.entry_rule = random.choice([parent1.entry_rule, parent2.entry_rule])
        child.exit_rule = random.choice([parent1.exit_rule, parent2.exit_rule])
        return child

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            'name': self.name, 'generation': self.generation,
            'genes': self.genes, 'entry_rule': self.entry_rule,
            'exit_rule': self.exit_rule, 'fitness': self.fitness,
            'win_rate': self.win_rate, 'total_pnl': self.total_pnl,
            'trades': self.trades, 'sharpe': self.sharpe,
            'parents': self.parents, 'metrics': self.metrics,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'StrategyDNA':
        """Reconstruct a StrategyDNA from a dict (round-trip with to_dict)."""
        dna = cls()
        dna.genes = d.get('genes', dna.genes)
        dna.entry_rule = d.get('entry_rule', dna.entry_rule)
        dna.exit_rule = d.get('exit_rule', dna.exit_rule)
        dna.min_move_pct = d.get('min_move_pct', dna.min_move_pct)
        dna.fitness = d.get('fitness', 0.0)
        dna.win_rate = d.get('win_rate', 0.0)
        dna.total_pnl = d.get('total_pnl', 0.0)
        dna.trades = d.get('trades', 0)
        dna.sharpe = d.get('sharpe', 0.0)
        dna.metrics = d.get('metrics', {})
        dna.generation = d.get('generation', 0)
        dna.parents = d.get('parents', [])
        dna.name = d.get('name', dna.name)
        return dna


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
        ema_f = np.array(ema(list(closes), int(g['ema_fast'])))
        ema_s = np.array(ema(list(closes), int(g['ema_slow'])))
        rsi_v = np.array(rsi(list(closes), int(g['rsi'])))
        roc_v = np.array(roc(list(closes), int(g['roc_period'])))
        atr_v = np.array(atr(list(highs), list(lows), list(closes), int(g['atr_period'])))
        i = -1
        price = closes[i]

        if dna.entry_rule == 'ema_cross':
            if ema_f[i] > ema_s[i] and ema_f[i-1] <= ema_s[i-1]: return 1
            if ema_f[i] < ema_s[i] and ema_f[i-1] >= ema_s[i-1]: return -1
        elif dna.entry_rule == 'rsi_oversold_bounce':
            if rsi_v[i] > g['rsi_oversold'] and rsi_v[i-1] <= g['rsi_oversold']: return 1
            if rsi_v[i] < g['rsi_overbought'] and rsi_v[i-1] >= g['rsi_overbought']: return -1
        elif dna.entry_rule == 'bb_lower_touch':
            bb_u, bb_m, bb_l = bollinger_bands(list(closes), int(g['bb_period']), g['bb_std'])
            if price <= bb_l[i] and rsi_v[i] < g['rsi_oversold'] + 5: return 1
            if price >= bb_u[i] and rsi_v[i] > g['rsi_overbought'] - 5: return -1
        elif dna.entry_rule == 'momentum_surge':
            if roc_v[i] > 1.0 and volumes[i] > np.mean(volumes[-20:]) * g['volume_mult']: return 1
            if roc_v[i] < -1.0 and volumes[i] > np.mean(volumes[-20:]) * g['volume_mult']: return -1
        elif dna.entry_rule == 'trend_strength':
            adx_vals = adx(list(highs), list(lows), list(closes), 14)
            adx_line = adx_vals[0] if isinstance(adx_vals, tuple) else adx_vals
            adx_v = np.array(adx_line)
            if adx_v[i] > g['adx_threshold'] and ema_f[i] > ema_s[i]: return 1
        elif dna.entry_rule == 'breakout_volume':
            lookback = int(g['lookback'])
            if price > max(highs[-lookback-1:-1]) and volumes[i] > np.mean(volumes[-lookback:]) * g['volume_mult']: return 1
        elif dna.entry_rule == 'stoch_reversal':
            k_vals, d_vals = stochastic(list(highs), list(lows), list(closes), int(g['stoch_k']), 3)
            if k_vals[i] > d_vals[i] and k_vals[i-1] <= d_vals[i-1] and k_vals[i] < 30: return 1
        elif dna.entry_rule == 'ema_bounce':
            ema_dist = abs(price - ema_f[i]) / price * 100
            if price > ema_f[i] and ema_dist < 0.5 and ema_f[i] > ema_f[i-1] and volumes[i] > np.mean(volumes[-10:]): return 1
        elif dna.entry_rule == 'multi_ma_align':
            ema_mid = np.array(ema(list(closes), int((g['ema_fast'] + g['ema_slow']) / 2)))
            if ema_f[i] > ema_mid[i] > ema_s[i] and ema_f[i] > ema_f[i-1]: return 1
            if ema_f[i] < ema_mid[i] < ema_s[i] and ema_f[i] < ema_f[i-1]: return -1
        elif dna.entry_rule == 'volatility_squeeze':
            bw = np.array(bb_width(list(closes), int(g['bb_period'])))
            if bw[i] > bw[i-1] and bw[i-1] < np.mean(bw[-20:]) * 0.8 and ema_f[i] > ema_s[i]: return 1
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
        return {'fitness': 0, 'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'sharpe': 0}

    trades = []
    position = None

    for i in range(60, n - max_hold):
        c_slice = closes[:i+1]
        h_slice = highs[:i+1]
        l_slice = lows[:i+1]
        v_slice = volumes[:i+1]

        if position is None:
            sig = execute_strategy(dna, c_slice, h_slice, l_slice, v_slice)
            if sig == 1:
                position = {'entry': closes[i], 'entry_idx': i}
        elif position is not None:
            bars_held = i - position['entry_idx']
            pnl_pct = (closes[i] - position['entry']) / position['entry'] * 100
            should_exit = False

            if dna.exit_rule == 'ema_flip':
                ema_f = ema(list(c_slice), int(dna.genes['ema_fast']))
                if len(ema_f) >= 2 and ema_f[-1] < ema_f[-2]:
                    should_exit = pnl_pct - spread_pct > 0
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

            if bars_held >= max_hold: should_exit = True
            if pnl_pct <= -5.0: should_exit = True

            if should_exit:
                net_pnl = pnl_pct - spread_pct
                trades.append({
                    'pnl_pct': net_pnl,
                    'pnl_usd': net_pnl * 10,  # Approximate for MultiMetricFitness
                    'bars_held': bars_held,
                    'duration_min': bars_held * 240,  # 4h bars = 240 min
                    'won': net_pnl > 0,
                })
                position = None

    if not trades:
        return {'fitness': 0, 'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'sharpe': 0}

    wins = [t for t in trades if t['won']]
    pnls = [t['pnl_pct'] for t in trades]
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(trades)
    avg_pnl = np.mean(pnls)
    std_pnl = np.std(pnls) if len(pnls) > 1 else 1.0
    sharpe = avg_pnl / (std_pnl + 1e-9)

    # Try MultiMetricFitness first, fallback to simple formula
    fitness = 0.0
    metrics = {}
    try:
        from src.trading.optimizer import MultiMetricFitness
        mmf = MultiMetricFitness()
        metrics = mmf.compute(trades)
        fitness = metrics.get('fitness_score', 0)
    except Exception:
        # Fallback: original hand-coded fitness
        fitness = (
            0.40 * min(1.0, max(0, total_pnl / 20))
            + 0.30 * win_rate
            + 0.20 * min(1.0, max(0, sharpe / 3))
            + 0.10 * min(1.0, len(trades) / 20)
        )
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
        'metrics': metrics,
    }


# ═══════════════════════════════════════════════════════════════
# Multi-Objective: Pareto Front (NSGA-II)
# ═══════════════════════════════════════════════════════════════

class ParetoFront:
    """Maintains non-dominated solutions across multiple objectives."""

    OBJECTIVES = ['total_pnl', 'sharpe', 'win_rate']

    def __init__(self):
        self.front: List[StrategyDNA] = []

    def dominates(self, a: StrategyDNA, b: StrategyDNA) -> bool:
        vals_a = [getattr(a, obj, 0) for obj in self.OBJECTIVES]
        vals_b = [getattr(b, obj, 0) for obj in self.OBJECTIVES]
        return all(va >= vb for va, vb in zip(vals_a, vals_b)) and any(va > vb for va, vb in zip(vals_a, vals_b))

    def update(self, population: list):
        candidates = self.front + list(population)
        new_front = []
        for ind in candidates:
            if not any(self.dominates(other, ind) for other in candidates if other is not ind):
                new_front.append(ind)
        self.front = new_front[:50]  # Cap size

    def to_list(self) -> list:
        return [dna.to_dict() for dna in self.front]


def fast_nondominated_sort(population: list) -> List[List[int]]:
    """NSGA-II fast non-dominated sort. Returns list of fronts (index lists)."""
    objectives = ParetoFront.OBJECTIVES
    n = len(population)
    domination_count = [0] * n
    dominated_set: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vi = [getattr(population[i], o, 0) for o in objectives]
            vj = [getattr(population[j], o, 0) for o in objectives]
            if all(a >= b for a, b in zip(vi, vj)) and any(a > b for a, b in zip(vi, vj)):
                dominated_set[i].append(j)
            elif all(b >= a for a, b in zip(vi, vj)) and any(b > a for a, b in zip(vi, vj)):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    fi = 0
    while fronts[fi]:
        nxt = []
        for i in fronts[fi]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    nxt.append(j)
        fi += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def crowding_distance(population: list, front_indices: list) -> Dict[int, float]:
    """Compute crowding distance for individuals in a front."""
    objectives = ParetoFront.OBJECTIVES
    distances = {i: 0.0 for i in front_indices}
    if len(front_indices) <= 2:
        for i in front_indices:
            distances[i] = float('inf')
        return distances
    for obj in objectives:
        sorted_idx = sorted(front_indices, key=lambda i: getattr(population[i], obj, 0))
        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')
        obj_range = getattr(population[sorted_idx[-1]], obj, 0) - getattr(population[sorted_idx[0]], obj, 0)
        if obj_range == 0:
            continue
        for k in range(1, len(sorted_idx) - 1):
            distances[sorted_idx[k]] += (
                getattr(population[sorted_idx[k+1]], obj, 0) - getattr(population[sorted_idx[k-1]], obj, 0)
            ) / obj_range
    return distances


# ═══════════════════════════════════════════════════════════════
# Island Model: Regime-Specialized Populations
# ═══════════════════════════════════════════════════════════════

def label_regime_bars(closes: list, window: int = 50) -> list:
    """Label each bar with a regime: BULL, BEAR, SIDEWAYS, VOLATILE."""
    closes_arr = np.array(closes, dtype=float)
    labels = []
    for i in range(len(closes)):
        if i < window:
            labels.append('SIDEWAYS')
            continue
        segment = closes_arr[i-window:i]
        returns = np.diff(segment) / segment[:-1]
        trend = (segment[-1] - segment[0]) / segment[0]
        volatility = np.std(returns)
        if volatility > np.percentile(np.abs(returns), 90) * 2:
            labels.append('VOLATILE')
        elif trend > 0.03:
            labels.append('BULL')
        elif trend < -0.03:
            labels.append('BEAR')
        else:
            labels.append('SIDEWAYS')
    return labels


class IslandModel:
    """Parallel populations evolving on regime-filtered data."""

    REGIMES = ['BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE']

    def __init__(self, spread_pct: float = SPREAD_PCT, migration_interval: int = 3, migration_count: int = 2):
        self.islands: Dict[str, GeneticStrategyEngine] = {}
        self.migration_interval = migration_interval
        self.migration_count = migration_count
        for regime in self.REGIMES:
            self.islands[regime] = GeneticStrategyEngine(spread_pct=spread_pct)

    def _filter_data(self, closes, highs, lows, volumes, labels, target):
        indices = [i for i, l in enumerate(labels) if l == target]
        if len(indices) < 100:
            return closes, highs, lows, volumes
        return ([closes[i] for i in indices], [highs[i] for i in indices],
                [lows[i] for i in indices], [volumes[i] for i in indices])

    def migrate(self):
        for src_regime, src_eng in self.islands.items():
            if not src_eng.population:
                continue
            top = sorted(src_eng.population, key=lambda x: x.fitness, reverse=True)[:self.migration_count]
            for tgt_regime, tgt_eng in self.islands.items():
                if tgt_regime == src_regime or not tgt_eng.population:
                    continue
                tgt_eng.population.sort(key=lambda x: x.fitness)
                for i, migrant in enumerate(top):
                    if i < len(tgt_eng.population):
                        clone = deepcopy(migrant)
                        clone.name = f"MIG_{src_regime[:3]}_{clone.name}"
                        tgt_eng.population[i] = clone

    def evolve_islands(self, closes, highs, lows, volumes, generations=10, population_size=25):
        labels = label_regime_bars(closes)
        counts = Counter(labels)
        print(f"  [ISLANDS] Regime distribution: {dict(counts)}")

        for regime, engine in self.islands.items():
            fc, fh, fl, fv = self._filter_data(closes, highs, lows, volumes, labels, regime)
            engine.closes, engine.highs, engine.lows, engine.volumes = fc, fh, fl, fv
            engine.initialize_population(population_size)

        for gen in range(generations):
            for regime, engine in self.islands.items():
                engine.evaluate_population()
                engine.evolve_generation()
                best = max(engine.population, key=lambda x: x.fitness) if engine.population else None
                if best:
                    print(f"  [ISLAND:{regime}] Gen {gen}: best={best.fitness:.4f} entry={best.entry_rule}")
            if (gen + 1) % self.migration_interval == 0:
                self.migrate()
                print(f"  [ISLANDS] Migration at gen {gen+1}")

        combined_hof = []
        for regime, engine in self.islands.items():
            engine.evaluate_population()
            engine.update_hall_of_fame()
            for entry in engine.hall_of_fame[:3]:
                d = entry.to_dict()
                d['home_regime'] = regime
                combined_hof.append(d)
        combined_hof.sort(key=lambda x: x.get('fitness', 0), reverse=True)
        return {'hall_of_fame': combined_hof[:10], 'regime_counts': dict(counts)}


# ═══════════════════════════════════════════════════════════════
# Evolution Engine
# ═══════════════════════════════════════════════════════════════

class GeneticStrategyEngine:
    """Evolves trading strategies through genetic algorithms."""

    def __init__(self, spread_pct: float = SPREAD_PCT):
        self.spread_pct = spread_pct
        self.population: List[StrategyDNA] = []
        self.hall_of_fame: List[StrategyDNA] = []
        self.pareto_front = ParetoFront()
        self.generation = 0
        self.closes = None
        self.highs = None
        self.lows = None
        self.volumes = None
        # Adaptive mutation tracking
        self._fitness_history: List[float] = []
        self._stagnation_counter: int = 0
        self._current_mutation_rate: float = 0.15
        self._diversity_history: List[dict] = []
        self._total_generations: int = 0

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
            dna.mutate(mutation_rate=0.8)
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
            dna.metrics = result.get('metrics', {})

        # Fitness sharing (niching): penalize overcrowded niches
        niche_counts = Counter((ind.entry_rule, ind.exit_rule) for ind in self.population)
        for ind in self.population:
            niche_size = niche_counts[(ind.entry_rule, ind.exit_rule)]
            if niche_size > 1:
                ind.fitness = ind.fitness / (niche_size ** 0.5)

        self.population.sort(key=lambda d: d.fitness, reverse=True)

    # ── Adaptive Mutation ────────────────────────────────────────

    def _compute_adaptive_mutation_rate(self) -> float:
        """Adapt mutation rate based on fitness stagnation."""
        if len(self._fitness_history) >= 2:
            prev = self._fitness_history[-2]
            curr = self._fitness_history[-1]
            if curr <= prev * 1.001:
                self._stagnation_counter += 1
            else:
                self._stagnation_counter = max(0, self._stagnation_counter - 2)
        stagnation_bonus = min(0.5, self._stagnation_counter * 0.1)
        self._current_mutation_rate = min(0.8, 0.15 + stagnation_bonus)
        return self._current_mutation_rate

    # ── Selection Methods ────────────────────────────────────────

    def select_tournament(self, tournament_size: int = 5) -> Tuple[StrategyDNA, StrategyDNA]:
        """Tournament selection — pick best from random subset."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        tournament.sort(key=lambda d: d.fitness, reverse=True)
        return tournament[0], tournament[1] if len(tournament) > 1 else tournament[0]

    def select_roulette(self) -> Tuple[StrategyDNA, StrategyDNA]:
        """Fitness-proportional selection."""
        total = sum(max(0.001, ind.fitness) for ind in self.population)
        parents = []
        for _ in range(2):
            pick = random.uniform(0, total)
            current = 0
            for ind in self.population:
                current += max(0.001, ind.fitness)
                if current >= pick:
                    parents.append(ind)
                    break
            else:
                parents.append(self.population[-1])
        return parents[0], parents[1]

    def select_rank_based(self) -> Tuple[StrategyDNA, StrategyDNA]:
        """Rank-based linear selection."""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness)
        n = len(sorted_pop)
        weights = [i + 1 for i in range(n)]
        selected = random.choices(sorted_pop, weights=weights, k=2)
        return selected[0], selected[1]

    def select_parents(self, tournament_size: int = 5) -> Tuple[StrategyDNA, StrategyDNA]:
        """Adaptive selection: varies method by generation phase."""
        gen = self._total_generations
        if gen < 4:
            return self.select_tournament(tournament_size)
        elif gen < 8:
            return self.select_rank_based()
        else:
            return self.select_roulette()

    # ── Diversity Metrics ────────────────────────────────────────

    def compute_diversity_metrics(self) -> dict:
        """Compute population diversity metrics."""
        if not self.population:
            return {'gene_diversity': 0, 'entry_entropy': 0, 'exit_entropy': 0, 'phenotype_count': 0}

        # Collect all gene keys across the population (not just the first individual)
        all_keys = set()
        for ind in self.population:
            all_keys.update(ind.genes.keys())
        gene_arrays = {k: [] for k in all_keys}
        for ind in self.population:
            for k in all_keys:
                if k in ind.genes:
                    gene_arrays[k].append(ind.genes[k])
        gene_diversity = float(np.mean([np.std(vals) for vals in gene_arrays.values() if len(vals) > 1]) if gene_arrays else 0)

        n = len(self.population)
        entry_counts = Counter(ind.entry_rule for ind in self.population)
        exit_counts = Counter(ind.exit_rule for ind in self.population)
        entry_entropy = -sum((c/n) * math.log2(c/n + 1e-10) for c in entry_counts.values())
        exit_entropy = -sum((c/n) * math.log2(c/n + 1e-10) for c in exit_counts.values())
        phenotypes = len(set((ind.entry_rule, ind.exit_rule) for ind in self.population))

        return {
            'gene_diversity': round(gene_diversity, 4),
            'entry_entropy': round(entry_entropy, 3),
            'exit_entropy': round(exit_entropy, 3),
            'phenotype_count': phenotypes,
            'population_size': n,
        }

    # ── Generation Step ──────────────────────────────────────────

    def evolve_generation(self, elite_count: int = 5, mutation_rate: float = None):
        """Create next generation with adaptive mutation and advanced operators."""
        rate = mutation_rate if mutation_rate is not None else self._current_mutation_rate
        new_population = []

        elite = self.population[:elite_count]
        new_population.extend(deepcopy(elite))

        while len(new_population) < len(self.population):
            parent1, parent2 = self.select_parents()
            child = StrategyDNA.crossover(parent1, parent2)
            child.mutate(rate)
            new_population.append(child)

        self.population = new_population
        self.generation += 1
        self._total_generations += 1

    def update_hall_of_fame(self, max_size: int = 10):
        for dna in self.population:
            if dna.fitness > 0 and dna.total_pnl > 0:
                self.hall_of_fame.append(deepcopy(dna))
        self.hall_of_fame.sort(key=lambda d: d.fitness, reverse=True)
        self.hall_of_fame = self.hall_of_fame[:max_size]

    # ── Per-Generation Logging ───────────────────────────────────

    def _log_generation(self, generation: int):
        """Append per-generation metrics to JSONL log."""
        if not self.population:
            return
        fitnesses = [ind.fitness for ind in self.population]
        pnls = [ind.total_pnl for ind in self.population]
        diversity = self.compute_diversity_metrics()

        entry = {
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'generation': generation,
            'population_size': len(self.population),
            'best_fitness': max(fitnesses),
            'avg_fitness': round(sum(fitnesses) / len(fitnesses), 4),
            'worst_fitness': min(fitnesses),
            'best_pnl': max(pnls),
            'avg_pnl': round(sum(pnls) / len(pnls), 2),
            'best_strategy': max(self.population, key=lambda x: x.fitness).name,
            'mutation_rate': self._current_mutation_rate,
            'stagnation_counter': self._stagnation_counter,
            'convergence': round(sum(1 for f in fitnesses if f >= max(fitnesses) * 0.9) / len(fitnesses), 3),
            'profitable_pct': round(sum(1 for p in pnls if p > 0) / len(pnls), 3),
            'hall_of_fame_size': len(self.hall_of_fame),
            'diversity': diversity,
        }

        log_path = os.path.join(PROJECT_ROOT, 'logs', 'genetic_evolution_history.jsonl')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry, default=str) + '\n')
        except Exception as e:
            logger.debug(f"Gen log write failed: {e}")

    # ── Evolution Summary (for dashboard/LLM) ────────────────────

    def get_evolution_summary(self) -> dict:
        """Structured summary for dashboard API or LLM context."""
        summary = {
            'total_generations_run': self._total_generations,
            'hall_of_fame_count': len(self.hall_of_fame),
            'best_evolved_fitness': 0,
            'best_evolved_strategy': None,
            'population_size': len(self.population),
            'current_mutation_rate': self._current_mutation_rate,
            'stagnation_generations': self._stagnation_counter,
            'pareto_front_size': len(self.pareto_front.front),
        }
        if self.hall_of_fame:
            best = self.hall_of_fame[0]
            summary['best_evolved_fitness'] = best.fitness
            summary['best_evolved_strategy'] = {
                'name': best.name, 'entry_rule': best.entry_rule, 'exit_rule': best.exit_rule,
                'fitness': best.fitness, 'win_rate': best.win_rate, 'total_pnl': best.total_pnl,
            }
        return summary

    # ── Main Evolution Loops ─────────────────────────────────────

    def evolve(self, generations: int = 10, population_size: int = 50) -> List[StrategyDNA]:
        """Run full single-objective evolution loop."""
        print(f"\n{'='*60}")
        print(f"  GENETIC STRATEGY EVOLUTION — {generations} gen × {population_size} strategies")
        print(f"  Spread: {self.spread_pct}% | Only LONG | Adaptive mutation + niching")
        print(f"{'='*60}")

        self.initialize_population(population_size)

        for gen in range(generations):
            self.evaluate_population()
            best = self.population[0]
            self._fitness_history.append(best.fitness)
            self._compute_adaptive_mutation_rate()
            diversity = self.compute_diversity_metrics()
            self._diversity_history.append(diversity)

            avg_fitness = np.mean([d.fitness for d in self.population])
            profitable = sum(1 for d in self.population if d.total_pnl > 0)

            print(f"\n  Gen {gen+1}/{generations}: best={best.name} fitness={best.fitness:.4f} "
                  f"PnL={best.total_pnl:+.1f}% WR={best.win_rate:.0%} trades={best.trades} "
                  f"| avg={avg_fitness:.4f} | profit={profitable}/{population_size} "
                  f"| mut={self._current_mutation_rate:.2f} pheno={diversity['phenotype_count']}")

            if best.total_pnl > 0:
                print(f"    WINNER: {best.entry_rule} + {best.exit_rule}")

            self.update_hall_of_fame()
            self._log_generation(gen + 1)
            self.evolve_generation()

        self.evaluate_population()
        self.update_hall_of_fame()
        self._print_results(generations)
        self._save_results()
        return self.hall_of_fame

    def evolve_multiobjective(self, generations: int = 10, population_size: int = 50) -> List[StrategyDNA]:
        """Run NSGA-II multi-objective evolution."""
        print(f"\n{'='*60}")
        print(f"  NSGA-II MULTI-OBJECTIVE EVOLUTION — {generations} gen × {population_size}")
        print(f"  Objectives: PnL, Sharpe, Win Rate | Spread: {self.spread_pct}%")
        print(f"{'='*60}")

        self.initialize_population(population_size)

        for gen in range(generations):
            self.evaluate_population()
            self.pareto_front.update(self.population)
            self._fitness_history.append(self.population[0].fitness if self.population else 0)
            self._compute_adaptive_mutation_rate()

            fronts = fast_nondominated_sort(self.population)
            all_distances = {}
            for front in fronts:
                all_distances.update(crowding_distance(self.population, front))

            # NSGA-II selection: prefer lower front rank, then higher crowding distance
            ranked = []
            for fi, front in enumerate(fronts):
                for idx in front:
                    ranked.append((fi, -all_distances.get(idx, 0), idx))
            ranked.sort()

            # Select top half as parents
            parent_indices = [r[2] for r in ranked[:population_size // 2]]
            new_pop = [deepcopy(self.population[i]) for i in parent_indices[:5]]  # Elite

            while len(new_pop) < population_size:
                i1, i2 = random.sample(parent_indices, 2)
                child = StrategyDNA.crossover(self.population[i1], self.population[i2])
                child.mutate(self._current_mutation_rate)
                new_pop.append(child)

            self.population = new_pop
            self._total_generations += 1

            print(f"  Gen {gen+1}: fronts={len(fronts)} pareto={len(self.pareto_front.front)} "
                  f"mut={self._current_mutation_rate:.2f}")
            self._log_generation(gen + 1)

        self.evaluate_population()
        self.update_hall_of_fame()
        self.pareto_front.update(self.population)
        self._print_results(generations)
        self._save_results()
        return self.hall_of_fame

    def run_quick_evolution(self, market_data: dict, generations: int = 5,
                            population_size: int = 30, seed_dna: list = None) -> dict:
        """Short evolution cycle for continuous_adapt pipeline integration."""
        self.closes = market_data.get('closes', self.closes)
        self.highs = market_data.get('highs', self.highs)
        self.lows = market_data.get('lows', self.lows)
        self.volumes = market_data.get('volumes', self.volumes)

        if self.closes is None or len(self.closes) < 100:
            return {'hall_of_fame': [], 'best_fitness': 0, 'generations_run': 0}

        self.population = []
        if seed_dna:
            for sd in seed_dna[:population_size // 2]:
                dna = StrategyDNA.from_dict(sd)
                dna.mutate(0.3)
                self.population.append(dna)

        while len(self.population) < population_size:
            dna = StrategyDNA()
            dna.mutate(0.8)
            self.population.append(dna)

        for gen in range(generations):
            self.evaluate_population()
            self._fitness_history.append(self.population[0].fitness if self.population else 0)
            self._compute_adaptive_mutation_rate()
            self.update_hall_of_fame()
            self._log_generation(gen + 1)
            self.evolve_generation()

        self.evaluate_population()
        self.update_hall_of_fame()
        self._save_results()

        return {
            'hall_of_fame': [d.to_dict() for d in self.hall_of_fame],
            'best_fitness': self.hall_of_fame[0].fitness if self.hall_of_fame else 0,
            'generations_run': generations,
        }

    # ── Output Helpers ───────────────────────────────────────────

    def _print_results(self, generations: int):
        print(f"\n{'='*60}")
        print(f"  EVOLUTION COMPLETE — {generations} generations")
        print(f"{'='*60}")
        if self.hall_of_fame:
            print(f"\n  HALL OF FAME (profitable after {self.spread_pct}% spread):")
            for i, dna in enumerate(self.hall_of_fame[:5], 1):
                print(f"    #{i} {dna.name}: PnL={dna.total_pnl:+.1f}% WR={dna.win_rate:.0%} "
                      f"trades={dna.trades} sharpe={dna.sharpe:.2f}")
                print(f"       Entry: {dna.entry_rule} | Exit: {dna.exit_rule}")
        else:
            print(f"\n  No strategies found profitable after {self.spread_pct}% spread")
        if self.pareto_front.front:
            print(f"\n  PARETO FRONT: {len(self.pareto_front.front)} non-dominated strategies")

    def _save_results(self):
        """Save evolution results + Pareto front + evolution summary."""
        results = {
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'generations': self.generation,
            'spread_pct': self.spread_pct,
            'population_size': len(self.population),
            'hall_of_fame': [dna.to_dict() for dna in self.hall_of_fame],
            'best_fitness': self.hall_of_fame[0].fitness if self.hall_of_fame else 0,
            'best_pnl': self.hall_of_fame[0].total_pnl if self.hall_of_fame else 0,
            'pareto_front': self.pareto_front.to_list(),
            'evolution_summary': self.get_evolution_summary(),
        }

        path = os.path.join(PROJECT_ROOT, 'logs/genetic_evolution_results.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Write evolution summary to adaptation context for LLM enrichment
        try:
            ctx_path = os.path.join(PROJECT_ROOT, 'data', 'adaptation_context.json')
            ctx = {}
            if os.path.exists(ctx_path):
                with open(ctx_path) as f:
                    ctx = json.load(f)
            ctx['genetic_evolution'] = self.get_evolution_summary()
            with open(ctx_path, 'w') as f:
                json.dump(ctx, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"Failed to write evolution context: {e}")

        print(f"\n  Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='Genetic Strategy Evolution Engine')
    parser.add_argument('--generations', type=int, default=10)
    parser.add_argument('--population', type=int, default=50)
    parser.add_argument('--asset', default='BTC')
    parser.add_argument('--timeframe', default='4h')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--interval', type=float, default=2.0, help='Hours between evolution cycles')
    parser.add_argument('--multiobjective', action='store_true', help='Use NSGA-II multi-objective optimization')
    parser.add_argument('--islands', action='store_true', help='Use island model with regime-specialized populations')
    args = parser.parse_args()

    if args.islands:
        # Island model
        engine = GeneticStrategyEngine()
        if not engine.load_market_data(args.asset, args.timeframe):
            print("Failed to load market data")
            return
        model = IslandModel(spread_pct=engine.spread_pct)
        results = model.evolve_islands(engine.closes, engine.highs, engine.lows, engine.volumes,
                                        generations=args.generations, population_size=args.population)
        # Save island results
        path = os.path.join(PROJECT_ROOT, 'logs/genetic_evolution_results.json')
        with open(path, 'w') as f:
            json.dump({
                'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                'mode': 'islands',
                'hall_of_fame': results['hall_of_fame'],
                'regime_counts': results['regime_counts'],
            }, f, indent=2, default=str)
        print(f"\n  Island results saved to {path}")
        return

    engine = GeneticStrategyEngine()
    if not engine.load_market_data(args.asset, args.timeframe):
        print("Failed to load market data")
        return

    if args.continuous:
        print(f"Continuous evolution mode (every {args.interval}h)...")
        while True:
            engine.load_market_data(args.asset, args.timeframe)
            if args.multiobjective:
                engine.evolve_multiobjective(args.generations, args.population)
            else:
                engine.evolve(args.generations, args.population)
            print(f"\nSleeping {args.interval}h...")
            time.sleep(args.interval * 3600)
    else:
        if args.multiobjective:
            engine.evolve_multiobjective(args.generations, args.population)
        else:
            engine.evolve(args.generations, args.population)


if __name__ == '__main__':
    main()
