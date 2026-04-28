"""Dual-path reasoning (FS-ReasoningAgent pattern).

Splits the analyst's per-tick reasoning into two constrained paths:
  * FACT path        — prices, indicators, ML predictions, on-chain,
                       institutional flows. Objective signals only.
  * SUBJECTIVITY     — news, social sentiment, polymarket implied
                       probabilities, fear-greed. Opinion signals only.

A reflection agent reads recent outcomes from warm_store and computes
the historical accuracy of each path. The synthesis combines the two
verdicts using these weights.

Why split: research (FS-ReasoningAgent, FinAgent) reports measurably
better in-context reasoning when the brain considers facts and
subjectivity in separate constrained passes rather than fusing them.
The reflection agent learns "in CRISIS regime trust facts 80/20; in
calm sentiment-driven phases trust subjectivity 60/40."

Anti-overfit design:
  * Weights computed from rolling window with min-sample threshold
    (DEFAULT_MIN_SAMPLES = 20). Below that → 50/50 prior.
  * Bayesian update on accuracy: Beta(alpha, beta) where alpha = wins+1
    and beta = losses+1. Avoids zero-divide and small-sample swings.
  * Output digest size capped to prevent prompt bloat.
  * Time-decayed memory: only last 60 days of outcomes.

Activation:
  ACT_DUAL_PATH unset or "0"   → module dormant, no compute, no calls
  ACT_DUAL_PATH = "shadow"     → runs alongside existing flow, logs to
                                 logs/dual_path_shadow.jsonl, never
                                 affects TradePlan output
  ACT_DUAL_PATH = "1"          → output is the TradePlan source
                                 (caller in agentic_bridge swaps in)

Existing ACT flow is untouched when env is unset. The module is
called from agentic_bridge ONLY when the env flag is set, and only
in shadow mode by default in any rollout.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MIN_SAMPLES = 20
DEFAULT_DECAY_DAYS = 60
SHADOW_LOG_PATH = "logs/dual_path_shadow.jsonl"


@dataclass
class PathVerdict:
    """One path's reasoning output. Bounded in size to prevent prompt bloat."""
    path: str                    # "fact" | "subjectivity"
    direction: str               # "LONG" | "SHORT" | "FLAT" | "SKIP"
    confidence: float            # 0.0 - 1.0
    rationale: str = ""          # capped at 400 chars
    inputs_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "direction": self.direction,
            "confidence": round(float(self.confidence), 3),
            "rationale": self.rationale[:400],
            "inputs_used": self.inputs_used[:20],
        }


@dataclass
class DualPathSynthesis:
    """Combined verdict + per-path weights + provenance."""
    fact: PathVerdict
    subjectivity: PathVerdict
    fact_weight: float           # 0.0 - 1.0; subjectivity_weight = 1 - this
    subjectivity_weight: float
    direction: str               # synthesized
    confidence: float            # synthesized
    sample_size: int             # outcomes considered for weighting
    weight_source: str           # "prior" | "rolling_outcomes"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact": self.fact.to_dict(),
            "subjectivity": self.subjectivity.to_dict(),
            "fact_weight": round(float(self.fact_weight), 3),
            "subjectivity_weight": round(float(self.subjectivity_weight), 3),
            "direction": self.direction,
            "confidence": round(float(self.confidence), 3),
            "sample_size": int(self.sample_size),
            "weight_source": self.weight_source,
        }


def is_enabled() -> bool:
    """Module compute fires only when env flag is set."""
    val = (os.environ.get("ACT_DUAL_PATH") or "").strip().lower()
    return val in ("shadow", "1", "true", "on")


def is_authoritative() -> bool:
    """Output is the TradePlan source only when explicitly turned on."""
    val = (os.environ.get("ACT_DUAL_PATH") or "").strip().lower()
    return val in ("1", "true", "on")


def _read_recent_outcomes(min_samples: int = DEFAULT_MIN_SAMPLES,
                          decay_days: int = DEFAULT_DECAY_DAYS) -> List[Dict[str, Any]]:
    """Pull recent path-tagged outcomes from warm_store. Returns [] on
    any error so a cold-start doesn't block the loop."""
    try:
        import sqlite3
        from src.orchestration.warm_store import get_store
        store = get_store()
        cutoff_ns = int((time.time() - decay_days * 86400) * 1e9)
        conn = sqlite3.connect(store.db_path, timeout=2.0)
        try:
            rows = conn.execute(
                "SELECT plan_json, self_critique FROM decisions "
                "WHERE ts_ns >= ? AND self_critique != '{}' AND self_critique IS NOT NULL "
                "ORDER BY ts_ns DESC LIMIT ?",
                (cutoff_ns, max(min_samples * 5, 100)),
            ).fetchall()
        finally:
            conn.close()
        out = []
        for plan_raw, crit_raw in rows:
            try:
                plan = json.loads(plan_raw or "{}")
                crit = json.loads(crit_raw or "{}")
            except Exception:
                continue
            tag = plan.get("dual_path_tag")  # "fact_won" / "subj_won" / "agreed"
            pnl = crit.get("realized_pnl_pct")
            if tag is None or pnl is None:
                continue
            out.append({"tag": str(tag), "pnl_pct": float(pnl)})
        return out
    except Exception as e:
        logger.debug("dual_path read_outcomes failed: %s", e)
        return []


def compute_weights(min_samples: int = DEFAULT_MIN_SAMPLES) -> Tuple[float, float, int, str]:
    """Bayesian weight computation: how often did the fact path's
    direction match the realized winner vs the subjectivity path?

    Returns (fact_weight, subjectivity_weight, sample_size, source).
    Below min_samples → 50/50 prior, source="prior".
    """
    outcomes = _read_recent_outcomes(min_samples=min_samples)
    if len(outcomes) < min_samples:
        return 0.5, 0.5, len(outcomes), "prior"

    fact_wins = sum(1 for o in outcomes if o["tag"] == "fact_won" and o["pnl_pct"] > 0)
    subj_wins = sum(1 for o in outcomes if o["tag"] == "subj_won" and o["pnl_pct"] > 0)
    fact_losses = sum(1 for o in outcomes if o["tag"] == "fact_won" and o["pnl_pct"] <= 0)
    subj_losses = sum(1 for o in outcomes if o["tag"] == "subj_won" and o["pnl_pct"] <= 0)

    # Beta(alpha, beta) posterior means with priors of 1/1.
    fact_post = (fact_wins + 1) / (fact_wins + fact_losses + 2)
    subj_post = (subj_wins + 1) / (subj_wins + subj_losses + 2)
    total = fact_post + subj_post
    if total <= 0:
        return 0.5, 0.5, len(outcomes), "prior"
    return fact_post / total, subj_post / total, len(outcomes), "rolling_outcomes"


def synthesize(fact: PathVerdict, subjectivity: PathVerdict,
               min_samples: int = DEFAULT_MIN_SAMPLES) -> DualPathSynthesis:
    """Combine the two path verdicts using the reflection-agent's weights.

    Synthesis rule:
      * Direction = weighted-vote winner (fact_weight × fact_dir_score
        + subj_weight × subj_dir_score).
      * Confidence = weighted average of path confidences, scaled by
        agreement bonus when both paths agree.
    """
    fw, sw, n, src = compute_weights(min_samples=min_samples)

    def _dir_score(d: str) -> int:
        d = (d or "").upper()
        if d in ("LONG", "BUY"):
            return 1
        if d in ("SHORT", "SELL"):
            return -1
        return 0

    fact_score = _dir_score(fact.direction)
    subj_score = _dir_score(subjectivity.direction)
    blended = fw * fact.confidence * fact_score + sw * subjectivity.confidence * subj_score

    if blended > 0.15:
        direction = "LONG"
    elif blended < -0.15:
        direction = "SHORT"
    else:
        direction = "SKIP"

    base_conf = fw * fact.confidence + sw * subjectivity.confidence
    agreement_bonus = 0.1 if fact_score == subj_score and fact_score != 0 else 0.0
    confidence = max(0.0, min(1.0, base_conf + agreement_bonus))

    return DualPathSynthesis(
        fact=fact, subjectivity=subjectivity,
        fact_weight=fw, subjectivity_weight=sw,
        direction=direction, confidence=confidence,
        sample_size=n, weight_source=src,
    )


def log_shadow(asset: str, synthesis: DualPathSynthesis,
               existing_direction: str, existing_confidence: float) -> None:
    """Append a shadow-mode comparison row so the operator can decide
    when dual-path is ready to promote. Never raises."""
    try:
        path = Path(SHADOW_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.time(),
            "asset": asset,
            "dual_path": synthesis.to_dict(),
            "existing_direction": existing_direction,
            "existing_confidence": round(float(existing_confidence), 3),
            "agree": synthesis.direction == existing_direction,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception as e:
        logger.debug("dual_path shadow log failed: %s", e)


def fact_path_prompt_block() -> str:
    """Constrained system addendum for the fact path. Caller appends
    to the analyst system prompt when running fact-path inference."""
    return (
        "\n\n## DUAL-PATH MODE: FACT-ONLY\n"
        "For this reasoning pass, consider ONLY objective signals:\n"
        "  - real-time price, EMA, ATR, RSI, OHLCV bars, swing levels\n"
        "  - ML ensemble outputs (LGBM, LSTM, PatchTST, RL, META-CTRL)\n"
        "  - quant models (Hurst, Kalman, GARCH, HMM, Hawkes, OU)\n"
        "  - multi-strategy + 242-strategy universe consensus\n"
        "  - on-chain metrics (query_on_chain_signals)\n"
        "  - institutional flows (query_institutional_flows)\n"
        "  - portfolio + ratchet state, cost/spread\n"
        "EXPLICITLY IGNORE: news_digest, fear-greed, sentiment scores,\n"
        "polymarket probabilities, social, knowledge_graph subjectivity edges.\n"
        "Output: direction + confidence + 3-5 inputs cited."
    )


def subjectivity_path_prompt_block() -> str:
    """Constrained system addendum for the subjectivity path."""
    return (
        "\n\n## DUAL-PATH MODE: SUBJECTIVITY-ONLY\n"
        "For this reasoning pass, consider ONLY opinion/narrative signals:\n"
        "  - news_digest, news_risk_classifier verdicts\n"
        "  - sentiment scores (FinBERT, fear-greed)\n"
        "  - polymarket implied probabilities\n"
        "  - knowledge_graph narrative edges (institutional, social)\n"
        "  - macro narrative (FOMC tone, CPI surprise direction)\n"
        "EXPLICITLY IGNORE: prices, indicators, ML predictions, quant\n"
        "models, on-chain numerics, multi-strategy votes.\n"
        "Output: direction + confidence + 3-5 inputs cited."
    )


def disagreement_resolver_note(synthesis: DualPathSynthesis) -> str:
    """When the two paths disagree, surface that to the analyst with
    the historical-accuracy weights so it knows which to trust."""
    if synthesis.fact.direction == synthesis.subjectivity.direction:
        return ""
    return (
        f"\n\n## DUAL-PATH DISAGREEMENT\n"
        f"Fact path says {synthesis.fact.direction} "
        f"(conf={synthesis.fact.confidence:.2f}); Subjectivity path says "
        f"{synthesis.subjectivity.direction} (conf={synthesis.subjectivity.confidence:.2f}). "
        f"Reflection-agent weight: fact={synthesis.fact_weight:.2f} "
        f"subjectivity={synthesis.subjectivity_weight:.2f} "
        f"(based on {synthesis.sample_size} outcomes; source={synthesis.weight_source}). "
        f"If sample_size < {DEFAULT_MIN_SAMPLES}, treat weights as priors only."
    )
