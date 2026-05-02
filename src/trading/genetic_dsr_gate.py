"""
Deflated-Sharpe Fitness Gate (P0 of genetic-loop audit)
========================================================
Selection-bias correction for the genetic loop. When N > 20 strategies
are evaluated in a single generation, the in-sample Sharpe of the
"winner" is heavily biased upward by trial multiplicity (Bailey-de
Prado 2014). This module applies Deflated Sharpe Ratio correction so
the post-gate fitness reflects honest selection statistics.

Usage:
    from src.trading.genetic_dsr_gate import apply_dsr_gate
    apply_dsr_gate(engine.population, n_trials=N, threshold_p=0.55)

After the call, each DNA's `fitness` is shrunk toward the deflated
estimate; DNAs whose probability(true Sharpe > 0) falls below the
threshold are penalised by `fail_penalty` (default 0.3×).

Anti-overfit design:
  * Pure function — no learned parameters.
  * Synthesizes per-trade returns from sharpe + avg_pnl when raw trade
    list is unavailable (genetic engine doesn't keep them).
  * Threshold is conservative (0.55, not 0.50) — strategies on the
    edge of indistinguishability from noise are downweighted, not
    promoted.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from src.backtesting.overfitting_metrics import deflated_sharpe

logger = logging.getLogger(__name__)


def _synthesize_returns(dna: Any, n_synth: int = 30) -> List[float]:
    """Build a deterministic gaussian return series matching dna's
    avg_pnl + sharpe. Used when raw trade list isn't retained."""
    sharpe = float(getattr(dna, "sharpe", 0.0) or 0.0)
    metrics = getattr(dna, "metrics", {}) or {}
    avg_pct = metrics.get("avg_profit_pct", None)
    if avg_pct is None:
        n_trades = max(1, int(getattr(dna, "trades", 1) or 1))
        avg_pct = float(getattr(dna, "total_pnl", 0.0) or 0.0) / n_trades
    avg = float(avg_pct) / 100.0  # decimal
    if sharpe == 0:
        std = max(abs(avg), 0.001)
    else:
        std = max(abs(avg) / max(0.001, abs(sharpe)), 0.0001)
    seed = abs(hash(getattr(dna, "name", "unnamed"))) % (2**32)
    rng = np.random.default_rng(seed=seed)
    return list(rng.normal(loc=avg, scale=std, size=n_synth))


def apply_dsr_gate(
    population: Iterable[Any],
    n_trials: int,
    threshold_p: float = 0.55,
    fail_penalty: float = 0.3,
    annualization_factor: float = math.sqrt(252),
    only_when_above: int = 20,
) -> Dict[str, Any]:
    """Apply Deflated-Sharpe correction to a genetic population.

    Args:
      population: iterable of StrategyDNA (must have .fitness, .sharpe,
        .total_pnl, .trades, .name).
      n_trials: number of strategies tested in this cycle (gen × pop).
      threshold_p: minimum probability(true Sharpe>0) required to keep
        full fitness. Below threshold, fitness is multiplied by
        `fail_penalty`.
      fail_penalty: shrink factor for strategies that fail DSR.
      only_when_above: skip the gate entirely if n_trials < this.

    Returns:
      dict with `applied`, `n_dna_failed`, `median_p_positive`, etc.
    """
    pop_list = list(population)
    if not pop_list:
        return {
            "applied": False, "reason": "empty_population",
            "n_trials": int(n_trials),
        }
    if n_trials < only_when_above:
        return {
            "applied": False, "reason": "below_trial_threshold",
            "n_trials": int(n_trials), "threshold": int(only_when_above),
        }

    n_failed = 0
    p_positives: List[float] = []
    deflated_sharpes: List[float] = []
    for dna in pop_list:
        # Only test DNAs that produced trades — zero-trade strategies
        # already have fitness 0 and don't need DSR.
        n_trades = int(getattr(dna, "trades", 0) or 0)
        if n_trades < 3:
            continue
        returns = _synthesize_returns(dna, n_synth=max(30, n_trades))
        result = deflated_sharpe(
            returns,
            n_trials=int(n_trials),
            annualization_factor=annualization_factor,
        )
        p_pos = result.probability_true_sharpe_positive
        p_positives.append(p_pos)
        deflated_sharpes.append(result.deflated_sharpe)

        # Annotate DNA for traceability.
        dna.dsr_p_positive = round(float(p_pos), 4)
        dna.dsr_deflated = round(float(result.deflated_sharpe), 4)

        if p_pos < threshold_p:
            old_fitness = float(getattr(dna, "fitness", 0.0) or 0.0)
            dna.fitness = old_fitness * fail_penalty
            dna.dsr_penalty_applied = True
            n_failed += 1
        else:
            dna.dsr_penalty_applied = False

    median_p = (
        sorted(p_positives)[len(p_positives) // 2]
        if p_positives else 0.0
    )
    median_dsr = (
        sorted(deflated_sharpes)[len(deflated_sharpes) // 2]
        if deflated_sharpes else 0.0
    )
    return {
        "applied": True,
        "n_trials": int(n_trials),
        "n_dna_tested": len(p_positives),
        "n_dna_failed": int(n_failed),
        "median_p_positive": round(float(median_p), 4),
        "median_deflated_sharpe": round(float(median_dsr), 4),
        "threshold_p": float(threshold_p),
        "fail_penalty": float(fail_penalty),
    }


__all__ = ["apply_dsr_gate"]
