"""
Grammatical Evolution Branch (P3 of genetic-loop audit)
========================================================
Context-free grammar (BNF) that produces strategy boolean expressions,
with integer-genome → derivation mapping per O'Neill & Ryan (2003).

The standard genetic engine in `genetic_strategy_engine.py` evolves a
fixed catalog of `entry_rule` templates. This module evolves the
*expression itself* — `(ema_fast > ema_slow) and (rsi < 35)` — by
deriving it from a grammar with a variable-length integer genome.

This unlocks combinatorial diversity well beyond the 22+9 templates,
which is the standard 2025-2026 evolutionary-trading approach for
strategy-rule discovery (Brabazon, O'Neill, et al.).

Grammar (subset, sufficient for tradable rules):

    <rule>       ::= <expr>
    <expr>       ::= <comp> | (<expr> <bool_op> <expr>)
    <bool_op>    ::= and | or
    <comp>       ::= <indicator> <cmp_op> <indicator>
                  | <indicator> <cmp_op> <constant>
    <indicator>  ::= ema_fast | ema_slow | rsi | roc | atr | stoch_k | close
    <cmp_op>     ::= > | <
    <constant>   ::= 0 | 1 | 30 | 50 | 70

Anti-overfit + safety:
  * Hard depth cap (default 6) — prevents pathological deep trees.
  * Wraparound cap (default 3 passes) — bounded compute per individual.
  * Zero-trades penalty: individuals that don't fire trades get fitness 0.
  * Returns serializable expression strings for audit + reproducibility.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.indicators.indicators import ema, rsi, roc, atr, stochastic

logger = logging.getLogger(__name__)


# ── Grammar definition ────────────────────────────────────────────────────


GRAMMAR: Dict[str, List[List[str]]] = {
    "<rule>": [["<expr>"]],
    "<expr>": [
        ["<comp>"],
        ["(", "<expr>", "<bool_op>", "<expr>", ")"],
    ],
    "<bool_op>": [["and"], ["or"]],
    "<comp>": [
        ["<indicator>", "<cmp_op>", "<indicator>"],
        ["<indicator>", "<cmp_op>", "<constant>"],
    ],
    "<indicator>": [
        ["ema_fast"], ["ema_slow"], ["rsi"], ["roc"],
        ["atr"], ["stoch_k"], ["close"],
    ],
    "<cmp_op>": [[">"], ["<"]],
    "<constant>": [["0"], ["1"], ["30"], ["50"], ["70"]],
}

START_SYMBOL = "<rule>"
NON_TERMINALS = set(GRAMMAR.keys())
INDICATORS = {"ema_fast", "ema_slow", "rsi", "roc", "atr", "stoch_k", "close"}


# ── Genome → expression derivation ────────────────────────────────────────


@dataclass
class GEDerivation:
    expression: str
    tokens: List[str]
    depth_reached: int
    n_codons_used: int
    n_wraparounds: int
    overflowed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expression": self.expression,
            "depth_reached": self.depth_reached,
            "n_codons_used": self.n_codons_used,
            "n_wraparounds": self.n_wraparounds,
            "overflowed": self.overflowed,
        }


def derive(
    genome: List[int],
    max_depth: int = 6,
    max_wraparounds: int = 3,
) -> GEDerivation:
    """Map an integer genome to a derived terminal-only expression."""
    tokens: List[str] = [START_SYMBOL]
    n_codons_used = 0
    n_wraparounds = 0
    depth_reached = 0
    overflowed = False
    codon_idx = 0

    # We track depth via a parallel stack mapping each NT to its current depth.
    nt_depth: List[int] = [0]

    while True:
        # Find the leftmost non-terminal.
        nt_pos = None
        for i, t in enumerate(tokens):
            if t in NON_TERMINALS:
                nt_pos = i
                break
        if nt_pos is None:
            break  # all terminals — derivation done

        nt = tokens[nt_pos]
        cur_depth = nt_depth[nt_pos]
        depth_reached = max(depth_reached, cur_depth)

        productions = GRAMMAR[nt]
        # If max depth reached, force a terminal-or-shallow choice when
        # such an option exists (e.g. <expr> → <comp> instead of recursive).
        if cur_depth >= max_depth:
            shallow_options = []
            for idx, prod in enumerate(productions):
                if not any(tok in NON_TERMINALS for tok in prod):
                    shallow_options.append(idx)
                else:
                    # Allow if expansion only leads to one extra level.
                    nt_count = sum(1 for tok in prod if tok in NON_TERMINALS)
                    if nt_count <= 1 and not any(tok == nt for tok in prod):
                        shallow_options.append(idx)
            if shallow_options:
                # Use codon to pick among shallow options.
                if codon_idx >= len(genome):
                    n_wraparounds += 1
                    if n_wraparounds > max_wraparounds:
                        overflowed = True
                        break
                    codon_idx = 0
                choice = shallow_options[genome[codon_idx] % len(shallow_options)]
            else:
                # Fallback: pick first option even if recursive.
                if codon_idx >= len(genome):
                    n_wraparounds += 1
                    if n_wraparounds > max_wraparounds:
                        overflowed = True
                        break
                    codon_idx = 0
                choice = genome[codon_idx] % len(productions)
        else:
            if codon_idx >= len(genome):
                n_wraparounds += 1
                if n_wraparounds > max_wraparounds:
                    overflowed = True
                    break
                codon_idx = 0
            choice = genome[codon_idx] % len(productions)

        codon_idx += 1
        n_codons_used += 1
        production = productions[choice]

        # Replace the NT with its production tokens, propagating depth.
        new_depths = []
        for tok in production:
            if tok in NON_TERMINALS:
                new_depths.append(cur_depth + 1)
            else:
                new_depths.append(cur_depth)
        tokens = tokens[:nt_pos] + production + tokens[nt_pos + 1:]
        nt_depth = nt_depth[:nt_pos] + new_depths + nt_depth[nt_pos + 1:]

    expression = " ".join(tokens) if not overflowed else ""
    return GEDerivation(
        expression=expression, tokens=list(tokens),
        depth_reached=depth_reached, n_codons_used=n_codons_used,
        n_wraparounds=n_wraparounds, overflowed=overflowed,
    )


# ── Indicator evaluation cache ─────────────────────────────────────────────


def _build_indicator_cache(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        "ema_fast": np.array(ema(list(closes), 8)),
        "ema_slow": np.array(ema(list(closes), 50)),
        "rsi": np.array(rsi(list(closes), 14)),
        "roc": np.array(roc(list(closes), 10)),
        "atr": np.array(atr(list(highs), list(lows), list(closes), 14)),
        "stoch_k": np.array(stochastic(list(highs), list(lows),
                                        list(closes), 14, 3)[0]),
        "close": closes.astype(float),
    }


def _eval_at_bar(
    expression: str, cache: Dict[str, np.ndarray], i: int,
) -> Optional[bool]:
    """Evaluate a derived expression at bar index `i`. None = invalid."""
    if not expression:
        return None
    # Substitute symbols → their values at bar i.
    py_expr = expression
    # Replace indicator names (longest-first to avoid partial replacement).
    for ind in sorted(INDICATORS, key=len, reverse=True):
        if ind in cache and i < len(cache[ind]):
            val = cache[ind][i]
            if not np.isfinite(val):
                return None
            py_expr = py_expr.replace(ind, f"({val})")
        else:
            return None
    # Sanitize: only allow whitelisted operators + digits + parens.
    # Whitelist: digits, ., +, -, *, /, (, ), >, <, =, space, "and", "or"
    allowed = set("0123456789.+-*/() <>=eE")
    test = py_expr.replace("and", "").replace("or", "")
    if not all(c in allowed for c in test):
        return None
    try:
        # eval with no builtins → pure boolean expression on numbers.
        return bool(eval(py_expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


# ── Backtest / fitness ─────────────────────────────────────────────────────


def evaluate_ge_individual(
    derivation: GEDerivation,
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    spread_pct: float = 1.69,
    max_hold: int = 30,
) -> Dict[str, Any]:
    """Backtest a derived GE expression as an entry rule.

    Exits use simple ATR trailing + max_hold. Returns metrics dict.
    """
    if derivation.overflowed or not derivation.expression:
        return {"fitness": 0.0, "trades": 0, "win_rate": 0.0,
                "total_pnl": 0.0, "sharpe": 0.0, "rejected": "overflowed"}

    n = len(closes)
    if n < 100:
        return {"fitness": 0.0, "trades": 0, "win_rate": 0.0,
                "total_pnl": 0.0, "sharpe": 0.0,
                "rejected": "insufficient_bars"}

    cache = _build_indicator_cache(closes, highs, lows)
    atr_arr = cache["atr"]

    trades = []
    position = None
    for i in range(60, n - max_hold):
        if position is None:
            sig = _eval_at_bar(derivation.expression, cache, i)
            if sig is True:
                position = {"entry": float(closes[i]), "entry_idx": i}
        else:
            bars_held = i - position["entry_idx"]
            pnl_pct = (closes[i] - position["entry"]) / position["entry"] * 100
            should_exit = False
            if not np.isfinite(atr_arr[i]):
                pass
            else:
                # ATR trailing: price below peak - 2*ATR while in profit.
                peak = float(np.max(closes[position["entry_idx"]:i + 1]))
                trail = peak - 2 * atr_arr[i]
                if closes[i] < trail and pnl_pct - spread_pct > 0:
                    should_exit = True
            if bars_held >= max_hold:
                should_exit = True
            if pnl_pct <= -5.0:
                should_exit = True
            if should_exit:
                net_pnl = pnl_pct - spread_pct
                trades.append({
                    "pnl_pct": net_pnl,
                    "won": net_pnl > 0,
                    "bars_held": bars_held,
                })
                position = None

    if not trades:
        return {"fitness": 0.0, "trades": 0, "win_rate": 0.0,
                "total_pnl": 0.0, "sharpe": 0.0,
                "rejected": "no_trades"}

    pnls = [t["pnl_pct"] for t in trades]
    total_pnl = float(sum(pnls))
    wins = sum(1 for t in trades if t["won"])
    win_rate = wins / len(trades)
    avg = float(np.mean(pnls))
    std = float(np.std(pnls)) or 1.0
    sharpe = avg / std

    fitness = (
        0.40 * min(1.0, max(0.0, total_pnl / 20))
        + 0.30 * win_rate
        + 0.20 * min(1.0, max(0.0, sharpe / 3))
        + 0.10 * min(1.0, len(trades) / 20)
    )
    if total_pnl < 0:
        fitness *= 0.3

    return {
        "fitness": round(float(fitness), 4),
        "trades": len(trades), "wins": wins,
        "win_rate": round(float(win_rate), 4),
        "total_pnl": round(total_pnl, 2),
        "sharpe": round(sharpe, 3),
    }


# ── Evolution loop ────────────────────────────────────────────────────────


@dataclass
class GEIndividual:
    genome: List[int]
    derivation: GEDerivation
    fitness: float = 0.0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    trades: int = 0
    sharpe: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_len": len(self.genome),
            **self.derivation.to_dict(),
            "fitness": round(self.fitness, 4),
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "trades": self.trades,
            "sharpe": round(self.sharpe, 3),
        }


def _random_genome(length: int = 30, max_int: int = 256, rng=None) -> List[int]:
    rng = rng or random.Random()
    return [rng.randint(0, max_int) for _ in range(length)]


def _crossover_genomes(g1: List[int], g2: List[int], rng=None) -> List[int]:
    rng = rng or random.Random()
    if len(g1) < 2 or len(g2) < 2:
        return list(g1)
    cut = rng.randint(1, min(len(g1), len(g2)) - 1)
    return list(g1[:cut]) + list(g2[cut:])


def _mutate_genome(g: List[int], rate: float = 0.1, max_int: int = 256, rng=None) -> List[int]:
    rng = rng or random.Random()
    out = []
    for codon in g:
        if rng.random() < rate:
            out.append(rng.randint(0, max_int))
        else:
            out.append(codon)
    return out


def evolve_grammatical(
    closes: List[float], highs: List[float], lows: List[float],
    spread_pct: float = 1.69,
    population_size: int = 40,
    generations: int = 8,
    genome_length: int = 30,
    elite_count: int = 4,
    seed: Optional[int] = None,
) -> List[GEIndividual]:
    """Run a Grammatical-Evolution loop returning the top individuals."""
    rng = random.Random(seed)
    np_closes = np.array(closes, dtype=float)
    np_highs = np.array(highs, dtype=float)
    np_lows = np.array(lows, dtype=float)

    population: List[GEIndividual] = []
    for _ in range(population_size):
        g = _random_genome(genome_length, rng=rng)
        d = derive(g)
        population.append(GEIndividual(genome=g, derivation=d))

    for gen in range(generations):
        for ind in population:
            res = evaluate_ge_individual(
                ind.derivation, np_closes, np_highs, np_lows,
                spread_pct=spread_pct,
            )
            ind.fitness = res.get("fitness", 0.0)
            ind.win_rate = res.get("win_rate", 0.0)
            ind.total_pnl = res.get("total_pnl", 0.0)
            ind.trades = res.get("trades", 0)
            ind.sharpe = res.get("sharpe", 0.0)

        population.sort(key=lambda x: x.fitness, reverse=True)
        best = population[0]
        logger.info(
            "[GE] gen %d/%d  best_fit=%.4f  expr=%s",
            gen + 1, generations, best.fitness,
            best.derivation.expression[:80],
        )

        # Next gen
        new_pop: List[GEIndividual] = list(population[:elite_count])
        while len(new_pop) < population_size:
            p1 = rng.choices(
                population[: max(1, population_size // 2)], k=1,
            )[0]
            p2 = rng.choices(
                population[: max(1, population_size // 2)], k=1,
            )[0]
            child_g = _crossover_genomes(p1.genome, p2.genome, rng=rng)
            child_g = _mutate_genome(child_g, rate=0.10, rng=rng)
            child_d = derive(child_g)
            new_pop.append(GEIndividual(genome=child_g, derivation=child_d))
        population = new_pop

    # Final eval
    for ind in population:
        if ind.fitness == 0:
            res = evaluate_ge_individual(
                ind.derivation, np_closes, np_highs, np_lows,
                spread_pct=spread_pct,
            )
            ind.fitness = res.get("fitness", 0.0)
            ind.win_rate = res.get("win_rate", 0.0)
            ind.total_pnl = res.get("total_pnl", 0.0)
            ind.trades = res.get("trades", 0)
            ind.sharpe = res.get("sharpe", 0.0)
    population.sort(key=lambda x: x.fitness, reverse=True)
    return population[: max(elite_count, 10)]


__all__ = [
    "GRAMMAR",
    "GEDerivation",
    "GEIndividual",
    "derive",
    "evaluate_ge_individual",
    "evolve_grammatical",
]
