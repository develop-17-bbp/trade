"""
Regime Intelligence Agent
=========================
Synthesises HMM regime detection, GARCH volatility modelling, and
transition-matrix forecasting into a macro regime view.

Outputs a directional vote plus a recommended_strategy tag:
  crisis    -> 'scalping'
  bull      -> 'trend_following'
  bear      -> 'trend_following' (short-biased)
  sideways  -> 'mean_reversion'
"""

from typing import Dict
from src.agents.base_agent import BaseAgent, AgentVote


class RegimeIntelligenceAgent(BaseAgent):
    """Macro regime classifier using HMM + GARCH volatility."""

    def __init__(self, name: str = 'regime_intelligence', config: Dict = None):
        super().__init__(name=name, config=config)

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        hmm = quant_state.get('hmm_regime', {})
        vol = quant_state.get('volatility', {})
        data_quality = context.get('data_quality', 1.0)
        if not isinstance(data_quality, (int, float)) or data_quality <= 0:
            data_quality = 1.0

        regime = hmm.get('regime', 'sideways')
        crisis_prob = hmm.get('crisis_prob', 0.0)
        stability = hmm.get('stability', 0.5)
        transition_matrix = hmm.get('transition_matrix', None)
        probs = hmm.get('probs', {})

        vol_regime = vol.get('vol_regime', 'normal')
        garch_vol = vol.get('garch_vol', 0.0)
        vol_percentile = vol.get('vol_percentile', 50)

        direction = 0
        confidence = 0.0
        position_scale = 1.0
        strategy = 'mean_reversion'
        reasoning_parts = []

        # --- Crisis override (highest priority) ---
        if regime == 'crisis' or crisis_prob > 0.5:
            direction = 0
            position_scale = 0.0
            confidence = min(1.0, max(crisis_prob, 0.8)) * data_quality
            strategy = 'scalping'
            reasoning_parts.append(
                f"[REGIME={regime}] [CRISIS_PROB={crisis_prob:.2f}] crisis detected -> "
                f"FLAT, zero sizing"
            )
            return AgentVote(
                direction=direction,
                confidence=round(confidence, 4),
                position_scale=0.0,
                reasoning=" | ".join(reasoning_parts),
                metadata={
                    'regime': regime,
                    'recommended_strategy': strategy,
                    'crisis_prob': crisis_prob,
                    'stability': stability,
                },
            )

        # --- Check bear->bull transition probability ---
        bear_to_bull_boost = 0.0
        if transition_matrix is not None and isinstance(transition_matrix, (list, tuple)):
            try:
                # Assume row ordering: [bull, bear, sideways] or similar
                # Bear row index = 1, bull col index = 0
                if len(transition_matrix) > 1 and len(transition_matrix[1]) > 0:
                    bear_to_bull_prob = transition_matrix[1][0]
                    if bear_to_bull_prob > 0.3:
                        bear_to_bull_boost = bear_to_bull_prob * 0.3
                        reasoning_parts.append(
                            f"[BEAR_TO_BULL_PROB={bear_to_bull_prob:.2f}] transition "
                            f"likely, reducing bearish bias"
                        )
            except (IndexError, TypeError):
                pass

        # --- Bull regime ---
        if regime == 'bull':
            strategy = 'trend_following'
            vol_moderate = vol_regime in ('normal', 'low') or vol_percentile < 75
            if stability > 0.7 and vol_moderate:
                direction = 1
                position_scale = min(1.0, stability)
                confidence = stability * 0.9 * data_quality
                reasoning_parts.append(
                    f"[REGIME=bull] [STABILITY={stability:.2f}] [VOL_REGIME={vol_regime}] "
                    f"stable bull, moderate vol -> BUY full"
                )
            elif stability > 0.7 and not vol_moderate:
                direction = 1
                position_scale = 0.5
                confidence = stability * 0.6 * data_quality
                reasoning_parts.append(
                    f"[REGIME=bull] [STABILITY={stability:.2f}] [VOL_REGIME={vol_regime}] "
                    f"bull but high vol -> BUY reduced"
                )
            else:
                direction = 1
                position_scale = 0.3
                confidence = 0.4 * data_quality
                reasoning_parts.append(
                    f"[REGIME=bull] [STABILITY={stability:.2f}] unstable bull -> BUY small"
                )

        # --- Bear regime ---
        elif regime == 'bear':
            strategy = 'trend_following'
            if stability > 0.7:
                direction = -1
                position_scale = min(1.0, stability) - bear_to_bull_boost
                position_scale = max(0.1, position_scale)
                confidence = stability * 0.85 * data_quality
                reasoning_parts.append(
                    f"[REGIME=bear] [STABILITY={stability:.2f}] stable bear -> SELL"
                )
            else:
                direction = -1
                position_scale = max(0.1, 0.4 - bear_to_bull_boost)
                confidence = 0.4 * data_quality
                reasoning_parts.append(
                    f"[REGIME=bear] [STABILITY={stability:.2f}] unstable bear -> SELL small"
                )

        # --- Sideways regime ---
        else:
            strategy = 'mean_reversion'
            if stability > 0.85:
                direction = 0
                position_scale = 0.3
                confidence = 0.35 * data_quality
                reasoning_parts.append(
                    f"[REGIME=sideways] [STABILITY={stability:.2f}] very stable range -> "
                    f"FLAT (defer to mean-reversion agents)"
                )
            else:
                direction = 0
                position_scale = 0.0
                confidence = 0.5 * data_quality
                reasoning_parts.append(
                    f"[REGIME=sideways] [STABILITY={stability:.2f}] -> FLAT"
                )

        confidence = max(0.0, min(1.0, confidence))

        return AgentVote(
            direction=direction,
            confidence=round(confidence, 4),
            position_scale=round(max(0.0, min(1.0, position_scale)), 4),
            reasoning=" | ".join(reasoning_parts),
            metadata={
                'regime': regime,
                'recommended_strategy': strategy,
                'crisis_prob': crisis_prob,
                'stability': stability,
                'vol_regime': vol_regime,
                'garch_vol': garch_vol,
            },
        )
