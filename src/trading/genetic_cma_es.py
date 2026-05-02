"""
CMA-ES Local Search for Numeric Genes (P1 of genetic-loop audit)
=================================================================
Covariance Matrix Adaptation Evolution Strategy — a derivative-free
continuous optimizer that adapts a multivariate Gaussian sampling
distribution from successful candidates (Hansen 2001/2016).

ACT's existing genetic engine handles structural search well
(entry_rule, exit_rule combinatorics) via NSGA-II + crossover. But
its numeric genes (ema_fast, rsi_oversold, bb_std, etc.) are searched
with random mutation, which has poor sample efficiency in continuous
spaces. CMA-ES gives a 10-100x sample-efficiency improvement on the
numeric portion.

Hybrid usage:
    1. Run normal genetic evolution (entry_rule + exit_rule + numeric).
    2. Take top-K Hall-of-Fame entries.
    3. Freeze each one's structural genes (entry/exit).
    4. Run CMA-ES on the numeric portion to refine that strategy.
    5. Re-insert refined version into Hall-of-Fame.

This module implements a minimal CMA-ES (no external `cma` package
dependency) operating on the numeric subset of INDICATOR_GENES.

References:
  Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial."
  https://arxiv.org/abs/1604.00772
"""
from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.trading.genetic_strategy_engine import (
    INDICATOR_GENES,
    StrategyDNA,
    backtest_dna,
)

logger = logging.getLogger(__name__)


# Genes that CMA-ES should optimize. Booleans + categorical-ish are excluded.
_BOOL_GENES = {"retest_allowed"}
_INTEGER_GENES = {
    "ema_fast", "ema_slow", "rsi", "bb_period", "atr_period", "roc_period",
    "stoch_k", "lookback", "min_bars_trend", "ema400_period", "signal_candles",
}


def _numeric_gene_names() -> List[str]:
    """Names of genes CMA-ES will tune (excludes booleans)."""
    return sorted(name for name in INDICATOR_GENES if name not in _BOOL_GENES)


def _genes_to_vector(genes: Dict[str, float], names: List[str]) -> np.ndarray:
    """Map gene dict → normalized [0, 1] vector for CMA-ES."""
    v = np.zeros(len(names))
    for i, name in enumerate(names):
        lo, hi = INDICATOR_GENES[name]["range"]
        rng = hi - lo if hi > lo else 1.0
        v[i] = (float(genes.get(name, INDICATOR_GENES[name]["default"])) - lo) / rng
    return np.clip(v, 0.0, 1.0)


def _vector_to_genes(v: np.ndarray, names: List[str]) -> Dict[str, float]:
    """Map normalized [0,1] vector → gene dict (clamped to ranges)."""
    out: Dict[str, float] = {}
    for i, name in enumerate(names):
        lo, hi = INDICATOR_GENES[name]["range"]
        val = lo + float(np.clip(v[i], 0.0, 1.0)) * (hi - lo)
        if name in _INTEGER_GENES or isinstance(lo, int):
            val = int(round(val))
        else:
            val = round(val, 4)
        out[name] = val
    return out


@dataclass
class CMAESResult:
    initial_fitness: float
    final_fitness: float
    improvement: float
    n_evaluations: int
    converged: bool
    refined_genes: Dict[str, float]
    history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_fitness": round(self.initial_fitness, 4),
            "final_fitness": round(self.final_fitness, 4),
            "improvement": round(self.improvement, 4),
            "n_evaluations": self.n_evaluations,
            "converged": self.converged,
            "refined_genes": self.refined_genes,
            "history_len": len(self.history),
        }


def _fitness_fn_factory(
    structural: StrategyDNA,
    closes: List[float], highs: List[float],
    lows: List[float], volumes: List[float],
    spread_pct: float,
) -> Callable[[Dict[str, float]], float]:
    """Return a function that backtests a candidate genome."""
    def evaluate(genes: Dict[str, float]) -> float:
        candidate = StrategyDNA()
        candidate.entry_rule = structural.entry_rule
        candidate.exit_rule = structural.exit_rule
        candidate.min_move_pct = structural.min_move_pct
        candidate.genes.update(genes)
        result = backtest_dna(candidate, closes, highs, lows, volumes, spread_pct)
        return float(result.get("fitness", 0.0))
    return evaluate


def cma_es_refine(
    seed_dna: StrategyDNA,
    closes: List[float], highs: List[float],
    lows: List[float], volumes: List[float],
    spread_pct: float = 1.69,
    max_generations: int = 12,
    population_size: Optional[int] = None,
    sigma_init: float = 0.25,
    seed: Optional[int] = None,
) -> Tuple[StrategyDNA, CMAESResult]:
    """Run CMA-ES on the numeric genes of `seed_dna` (entry/exit frozen).

    Args:
      seed_dna: DNA whose structural genes are kept; numeric genes are
        the CMA-ES initial mean.
      closes/highs/lows/volumes: market data slice.
      spread_pct: round-trip spread.
      max_generations: CMA-ES iterations.
      population_size: λ (lambda) — defaults to 4 + 3*ln(n) per Hansen.
      sigma_init: initial step size in normalized [0,1] space.
      seed: RNG seed for reproducibility.

    Returns:
      (refined_dna, CMAESResult)
    """
    rng = np.random.default_rng(seed)
    names = _numeric_gene_names()
    n = len(names)

    # CMA-ES parameters (Hansen defaults).
    if population_size is None:
        population_size = max(8, 4 + int(3 * math.log(n)))
    mu = population_size // 2
    weights_raw = np.array([math.log((population_size + 1) / 2) - math.log(i + 1)
                             for i in range(mu)])
    weights = weights_raw / weights_raw.sum()
    mu_eff = 1.0 / (weights ** 2).sum()

    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1,
               2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

    # State
    mean = _genes_to_vector(seed_dna.genes, names)
    sigma = float(sigma_init)
    C = np.eye(n)
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)

    fitness_fn = _fitness_fn_factory(
        seed_dna, closes, highs, lows, volumes, spread_pct,
    )
    initial_fitness = fitness_fn(seed_dna.genes)

    best_x = mean.copy()
    best_fitness = initial_fitness
    history = [initial_fitness]
    n_evals = 1
    converged = False

    expected_norm = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    for gen in range(max_generations):
        # Sample λ candidates: x_i = mean + sigma * BD * z_i
        try:
            BD = np.linalg.cholesky(C + 1e-9 * np.eye(n))
        except np.linalg.LinAlgError:
            # Recovery: regularize C
            C = 0.5 * (C + C.T)
            C += np.eye(n) * 1e-6
            try:
                BD = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                break

        z = rng.standard_normal((population_size, n))
        candidates = mean + sigma * (z @ BD.T)
        candidates = np.clip(candidates, 0.0, 1.0)

        # Evaluate fitness
        results = []
        for x in candidates:
            genes = _vector_to_genes(x, names)
            fit = fitness_fn(genes)
            results.append((fit, x))
            n_evals += 1
            if fit > best_fitness:
                best_fitness = fit
                best_x = x.copy()

        # Sort by fitness descending (max), take μ best
        results.sort(key=lambda r: r[0], reverse=True)
        selected = np.array([r[1] for r in results[:mu]])

        # Update mean (recombination)
        old_mean = mean.copy()
        mean = (selected * weights[:, None]).sum(axis=0)

        # Update evolution paths
        try:
            C_inv_sqrt = np.linalg.pinv(np.linalg.cholesky(C + 1e-9 * np.eye(n)))
        except np.linalg.LinAlgError:
            C_inv_sqrt = np.eye(n)
        p_sigma = ((1 - c_sigma) * p_sigma
                   + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
                   * C_inv_sqrt @ (mean - old_mean) / max(sigma, 1e-12))

        h_sigma = (np.linalg.norm(p_sigma)
                   / math.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
                   < (1.4 + 2 / (n + 1)) * expected_norm)
        h_sigma_int = 1.0 if h_sigma else 0.0

        p_c = ((1 - c_c) * p_c
               + h_sigma_int * math.sqrt(c_c * (2 - c_c) * mu_eff)
               * (mean - old_mean) / max(sigma, 1e-12))

        # Update covariance
        artmp = (selected - old_mean) / max(sigma, 1e-12)
        rank_one = np.outer(p_c, p_c)
        rank_mu = np.zeros((n, n))
        for i in range(mu):
            rank_mu += weights[i] * np.outer(artmp[i], artmp[i])
        delta_h = (1 - h_sigma_int) * c_c * (2 - c_c)
        C = ((1 - c_1 - c_mu) * C
             + c_1 * (rank_one + delta_h * C)
             + c_mu * rank_mu)

        # Update step size
        sigma = sigma * math.exp((c_sigma / d_sigma)
                                  * (np.linalg.norm(p_sigma) / expected_norm - 1))
        sigma = float(np.clip(sigma, 1e-6, 0.5))

        history.append(best_fitness)

        # Convergence: sigma collapses or improvement plateau
        if sigma < 1e-4:
            converged = True
            break
        if (gen >= 4
            and abs(history[-1] - history[-5]) < 1e-5
            and history[-1] >= initial_fitness):
            converged = True
            break

    # Build refined DNA
    refined = deepcopy(seed_dna)
    refined.genes.update(_vector_to_genes(best_x, names))
    refined.name = f"CMAES_{seed_dna.name}"
    refined.parents = [seed_dna.name]

    return refined, CMAESResult(
        initial_fitness=initial_fitness,
        final_fitness=best_fitness,
        improvement=best_fitness - initial_fitness,
        n_evaluations=n_evals,
        converged=converged,
        refined_genes=refined.genes,
        history=history,
    )


def hybrid_cma_es_refine_top_k(
    engine: Any,
    k: int = 5,
    max_generations: int = 12,
    sigma_init: float = 0.25,
    seed: Optional[int] = None,
) -> List[CMAESResult]:
    """Refine the top-K Hall-of-Fame entries with CMA-ES.

    Modifies engine.hall_of_fame in place: each refined DNA is appended
    if its fitness exceeds the seed; the list is re-sorted afterwards.
    """
    if not engine.hall_of_fame or engine.closes is None:
        return []

    closes = list(engine.closes)
    highs = list(engine.highs)
    lows = list(engine.lows)
    volumes = list(engine.volumes)

    results: List[CMAESResult] = []
    for seed_dna in list(engine.hall_of_fame[:k]):
        try:
            refined, res = cma_es_refine(
                seed_dna, closes, highs, lows, volumes,
                spread_pct=engine.spread_pct,
                max_generations=max_generations,
                sigma_init=sigma_init,
                seed=seed,
            )
            results.append(res)
            if res.improvement > 0:
                refined.fitness = res.final_fitness
                refined.total_pnl = seed_dna.total_pnl  # carry forward
                engine.hall_of_fame.append(refined)
                logger.info(
                    "[CMA-ES] %s: %.4f -> %.4f (+%.4f) in %d evals",
                    seed_dna.name, res.initial_fitness, res.final_fitness,
                    res.improvement, res.n_evaluations,
                )
        except Exception as exc:
            logger.warning("CMA-ES refine failed for %s: %s",
                          getattr(seed_dna, "name", "?"), exc)

    engine.hall_of_fame.sort(key=lambda d: getattr(d, "fitness", 0.0),
                              reverse=True)
    engine.hall_of_fame = engine.hall_of_fame[:max(k * 2, 10)]
    return results


__all__ = [
    "cma_es_refine",
    "hybrid_cma_es_refine_top_k",
    "CMAESResult",
]
