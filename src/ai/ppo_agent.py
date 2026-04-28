"""PPO RL agent scaffolding (modern upgrade from Q-learning).

ACT's existing RL agent is Q-learning over a discrete action space.
2026 trading-RL research consistently uses PPO (Proximal Policy
Optimization) instead — clipped policy updates prevent destructive
weight changes, supports continuous action spaces (size_pct), and
has well-known reward-shaping practices.

This module is a SCAFFOLD with:
  * `PPOActionSpace` definition (the actions the agent can take)
  * `PPOReward` definition (how to score each trade outcome)
  * `infer_action()` — inference wrapper with graceful fallback
  * Stub for training (delegated to a separate offline pipeline)

Anti-overfit / anti-noise / anti-reward-hacking:
  * Reward function clipped: per-trade reward bounded at ±5% to
    prevent the policy from optimizing toward outlier wins
  * Action space INCLUDES "skip" — agent can decline to act, removing
    the "force-trade" reward-hacking risk
  * Policy entropy bonus during training (stochastic enough to keep
    exploring; not implemented in this scaffold but documented)
  * Champion gate ≥2% improvement BEFORE any new policy goes live
  * Stays in shadow mode by default (ACT_PPO_AGENT=shadow logs only)

VRAM: a typical PPO actor-critic for 50-feature input + 4-action
output is ~2 MB GPU. Negligible vs current ~25 GB live serving.

This module deliberately defers the actual PPO TRAINING to a
separate offline pipeline (similar to the LoRA fine-tune cycle).
Online inference is what's wired here; training requires a
500+ sample buffer which the existing experience_replay collects.

Activation:
  ACT_PPO_AGENT unset / "0"  → module dormant; existing Q-learning
                                stays authoritative
  ACT_PPO_AGENT = "shadow"   → PPO inference runs alongside Q-
                                learning; logs comparison; never
                                authoritative
  ACT_PPO_AGENT = "1"        → PPO inference is authoritative;
                                Q-learning becomes advisory only

Champion gate is enforced at promotion: a new PPO checkpoint must
beat the incumbent (Q-learning OR previous PPO) by ≥2% Sharpe over
the same evaluation window before this env value can be flipped to
"1" without operator intervention.
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

# Action space (discrete, but each action carries a sizing scalar)
PPO_ACTIONS = (
    "SKIP",                # no trade
    "ENTER_LONG_SMALL",    # 1% size
    "ENTER_LONG_NORMAL",   # 2.5% size
    "ENTER_LONG_SNIPER",   # 5% size
    "EXIT_FULL",           # close all on this asset
    "EXIT_PARTIAL_50",     # close half
    "MODIFY_SL_TIGHTER",   # raise SL by 0.5x ATR
)

# Reward bounds — anti reward-hacking
REWARD_CLIP_PCT = 5.0     # per-trade reward clipped at ±5%
SHADOW_LOG_PATH = "logs/ppo_shadow.jsonl"


@dataclass
class PPOInferenceResult:
    method: str               # "ppo" | "fallback_qlearning_proxy"
    action: str
    action_idx: int
    log_prob: float           # policy log-probability of chosen action
    value_estimate: float     # critic's value estimate
    confidence: float         # 0-1 derived from policy entropy
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "action": self.action,
            "action_idx": int(self.action_idx),
            "log_prob": round(float(self.log_prob), 4),
            "value_estimate": round(float(self.value_estimate), 4),
            "confidence": round(float(self.confidence), 3),
            "rationale": self.rationale[:200],
        }


def is_enabled() -> bool:
    val = (os.environ.get("ACT_PPO_AGENT") or "").strip().lower()
    return val in ("shadow", "1", "true", "on")


def is_authoritative() -> bool:
    val = (os.environ.get("ACT_PPO_AGENT") or "").strip().lower()
    return val in ("1", "true", "on")


def clip_reward(raw_pnl_pct: float) -> float:
    """Bound per-trade reward to prevent the policy from optimizing
    toward outlier wins (reward hacking). The clipped reward is what
    the policy gradient sees; the original PnL is still recorded for
    audit."""
    return max(-REWARD_CLIP_PCT, min(REWARD_CLIP_PCT, float(raw_pnl_pct)))


def _try_ppo_inference(
    state_features: List[float],
    asset: str,
) -> Optional[PPOInferenceResult]:
    """Attempt PPO checkpoint inference. Returns None when checkpoint
    doesn't exist or import fails."""
    try:
        ckpt = Path(f"models/ppo_{asset.lower()}.pkl")
        if not ckpt.exists():
            return None
        import torch
    except (ImportError, Exception):
        return None
    try:
        # Cache loaded model
        global _ppo_model_cache
        if "_ppo_model_cache" not in globals():
            _ppo_model_cache = {}
        cache_key = asset.lower()
        if cache_key not in _ppo_model_cache:
            # Lazy: in real impl this loads a TorchScript or pickled
            # actor-critic. For scaffolding we skip the actual load
            # and let the caller fall back. Once an actual checkpoint
            # is trained, replace this stub with the real load logic.
            _ppo_model_cache[cache_key] = None
        # Stub returns None so caller falls back. Real impl here.
        return None
    except Exception as e:
        logger.debug("PPO inference failed: %s", e)
        return None


def _fallback_qlearning_proxy(state_features: List[float], asset: str) -> PPOInferenceResult:
    """When PPO checkpoint isn't trained yet, return a deterministic
    rule-based action so the caller always gets a result. Mirrors
    the existing Q-learning agent's behavior at a high level."""
    if not state_features:
        return PPOInferenceResult(
            method="fallback_qlearning_proxy",
            action="SKIP", action_idx=0,
            log_prob=0.0, value_estimate=0.0, confidence=0.0,
            rationale="empty_state",
        )
    # Very simple proxy: feature[0] = trend strength (positive = up)
    # feature[1] = volatility (high = caution)
    # feature[2] = portfolio exposure (high = caution)
    trend = state_features[0] if len(state_features) > 0 else 0.0
    vol = state_features[1] if len(state_features) > 1 else 0.5
    exposure = state_features[2] if len(state_features) > 2 else 0.0

    if exposure > 0.05 and vol > 0.7:
        action, idx = "EXIT_PARTIAL_50", 5
        rationale = "high exposure + high vol → reduce"
    elif trend > 0.6 and vol < 0.5:
        action, idx = "ENTER_LONG_SNIPER", 3
        rationale = "strong trend + low vol → sniper-tier entry"
    elif trend > 0.3:
        action, idx = "ENTER_LONG_NORMAL", 2
        rationale = "moderate trend → normal entry"
    elif trend > 0.1:
        action, idx = "ENTER_LONG_SMALL", 1
        rationale = "weak trend → small entry"
    else:
        action, idx = "SKIP", 0
        rationale = "no clear trend"

    return PPOInferenceResult(
        method="fallback_qlearning_proxy",
        action=action, action_idx=idx,
        log_prob=0.0,    # not a real policy
        value_estimate=trend,
        confidence=min(1.0, abs(trend)),
        rationale=rationale,
    )


def infer_action(
    state_features: List[float],
    asset: str = "BTC",
) -> PPOInferenceResult:
    """Get a PPO action recommendation. Returns the rule-based fallback
    when a trained PPO checkpoint isn't available."""
    if not is_enabled():
        return PPOInferenceResult(
            method="disabled",
            action="SKIP", action_idx=0,
            log_prob=0.0, value_estimate=0.0, confidence=0.0,
            rationale="ACT_PPO_AGENT not set",
        )

    ppo = _try_ppo_inference(state_features, asset)
    if ppo is not None:
        return ppo

    return _fallback_qlearning_proxy(state_features, asset)


def log_shadow(asset: str, ppo_result: PPOInferenceResult,
               qlearning_action: str) -> None:
    """Shadow comparison log: PPO vs Q-learning verdicts."""
    try:
        path = Path(SHADOW_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.time(),
            "asset": asset,
            "ppo": ppo_result.to_dict(),
            "qlearning_action": str(qlearning_action),
            "agree": ppo_result.action == qlearning_action,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception as e:
        logger.debug("PPO shadow log failed: %s", e)
