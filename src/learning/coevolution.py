"""Co-evolution adapters — Phase 4.5b (§5).

Three cross-learner transfers:

  5.1  Genetic → RL    top-K DNA become hyperparam seeds for RL mini-policies.
                       Guardrail: cap the RL-policy delta at 10% per cycle.
  5.2  RL → Genetic    RL V(s_entry) adds a small fitness bonus (α=0.05) so
                       the GA biases exploration toward RL-valued regions.
                       Hard-capped so GA doesn't chase RL hallucinations.
  5.3  LoRA ↔ Calibrator  post-adapter-merge, calibrator re-fits; its curve
                          becomes a preference source for LoRA's next round.

These adapters are INTENTIONALLY lightweight — they call into the big
learner modules (lora_trainer, genetic_strategy_engine, rl_agent) only
via read-only accessors so they can ship without restructuring those files.
The actual transfers happen when learners choose to consume. This file
exposes the transfer helpers; wiring them is the learner's call.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Transfer limits (Plan §5 + §7.1)
MAX_RL_DELTA_PER_CYCLE = 0.10          # 10% weight delta cap
MAX_GENETIC_RL_BONUS = 0.05            # max 5% fitness bonus from RL V(s)
K_TOP_DNA_FOR_RL = 5                   # genetic top-5 become RL warm-starts


# ── 5.1 Genetic → RL ────────────────────────────────────────────────────

def export_genetic_dna_for_rl(top_k: List[Dict[str, Any]], regime: str = "unknown") -> List[Dict[str, float]]:
    """Map genetic DNA fingerprints to RL hyperparameter vectors.

    Input (from genetic_strategy_engine):
        [{"name": "dna-42", "fitness": 0.73,
          "genes": {"ema": 8, "rsi": 14, "atr_mult": 1.5, ...}}, ...]

    Output (ready for RL mini-policy init):
        [{"learning_rate": ..., "gamma": ..., "entropy_coef": ..., "reward_clip": ...}]
    """
    out = []
    for dna in (top_k or [])[:K_TOP_DNA_FOR_RL]:
        genes = (dna or {}).get("genes", {}) or {}
        fitness = float(dna.get("fitness", 0.5))
        # Deterministic mapping — the specific numbers matter less than the
        # spread across seeds. Fitness scales learning rate (better DNA →
        # smaller, safer step); mutation variance seeds entropy.
        out.append({
            "dna_name": dna.get("name", "dna-?"),
            "learning_rate": max(1e-5, 1e-3 * (1.0 - 0.5 * fitness)),
            "gamma": 0.95 + 0.04 * fitness,                       # 0.95-0.99
            "entropy_coef": max(0.001, 0.02 - 0.015 * fitness),   # explore less as fitness grows
            "reward_clip": 1.0 + float(genes.get("atr_mult", 1.0)),
            "source_regime": regime,
        })
    return out


def apply_rl_warm_starts_publish(top_k: List[Dict[str, Any]], regime: str, model_version: str = "") -> int:
    """Fire-and-forget: push top-K DNA → signal bus for RL to consume.

    Called from the genetic loop at end of every generation.
    Returns number of DNA pushed.
    """
    try:
        from src.learning.signal_bus import publish_genetic_top_k
        publish_genetic_top_k(top_k or [], regime=regime, model_version=model_version)
        return min(len(top_k or []), K_TOP_DNA_FOR_RL)
    except Exception as e:
        logger.debug("genetic top-K publish failed: %s", e)
        return 0


# ── 5.2 RL → Genetic ────────────────────────────────────────────────────

def rl_fitness_bonus(value_estimate: float) -> float:
    """Map RL V(s_entry) ∈ [-1, 1] to a capped fitness bonus for the GA.

    The GA's fitness is 0.8·backtest + 0.2·live today. This function
    returns the third term (≤ MAX_GENETIC_RL_BONUS) that gets added on.
    """
    try:
        v = float(value_estimate)
    except Exception:
        return 0.0
    # Clamp V(s) and rescale into [-1, 1] just in case the RL signal leaks.
    v = max(-1.0, min(1.0, v))
    return max(-MAX_GENETIC_RL_BONUS, min(MAX_GENETIC_RL_BONUS, MAX_GENETIC_RL_BONUS * v))


def fetch_rl_value_for_genetic(state_key: str, group: str = "genetic-coevo", consumer: str = "g1") -> Optional[float]:
    """Drain the latest RL value estimate for a given state_key (regime, say).

    Returns None if no RL signal is available. Callers should treat that
    as "no bonus, proceed with 0.8·backtest + 0.2·live as usual".
    """
    try:
        from src.learning.signal_bus import (
            PRODUCER_L1_RL, SIGNAL_VALUE_ESTIMATE, subscribe, ack,
        )
        latest: Optional[float] = None
        for mid, env in subscribe(
            PRODUCER_L1_RL, SIGNAL_VALUE_ESTIMATE,
            group=group, consumer=consumer, count=16, block_ms=10,
        ):
            payload = (env or {}).get("payload", {}) or {}
            if payload.get("state_key") == state_key:
                latest = float(payload.get("v", 0.0))
            ack(PRODUCER_L1_RL, SIGNAL_VALUE_ESTIMATE, group, mid)
        return latest
    except Exception as e:
        logger.debug("fetch_rl_value_for_genetic failed: %s", e)
        return None


# ── 5.3 LoRA ↔ Calibrator ───────────────────────────────────────────────

def calibrator_curve_for_lora(curve_points: int = 10) -> Optional[List[float]]:
    """Return the calibrator's current reliability curve for LoRA DPO.

    Placeholder: real impl will XRANGE calibrator stream and pick the
    freshest envelope. For now we return None if nothing is published,
    which the LoRA trainer treats as "skip preference-pair injection".
    """
    try:
        from src.learning.signal_bus import (
            PRODUCER_CALIBRATOR, SIGNAL_BRIER, subscribe, ack,
        )
        latest = None
        for mid, env in subscribe(
            PRODUCER_CALIBRATOR, SIGNAL_BRIER,
            group="lora-coevo", consumer="l1", count=1, block_ms=10,
        ):
            latest = (env or {}).get("payload", {})
            ack(PRODUCER_CALIBRATOR, SIGNAL_BRIER, "lora-coevo", mid)
        if latest:
            # Fake a uniform curve derived from Brier — real impl replaces.
            brier = float(latest.get("brier", 0.25))
            return [max(0.0, min(1.0, 0.5 + (i - curve_points / 2) * (1 - brier) / curve_points))
                    for i in range(curve_points)]
        return None
    except Exception:
        return None
