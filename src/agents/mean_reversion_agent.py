"""
Mean Reversion Agent
====================
Activates only when the Ornstein-Uhlenbeck process confirms stationarity
(ADF p-value < 0.05). Combines OU z-score, Bollinger band position, and
RSI to detect overbought/oversold extremes worth fading.

Key thresholds:
  z_score >  2.0 + RSI > 70 + bb_position > 0.9  -> SELL (overbought)
  z_score < -2.0 + RSI < 30 + bb_position < 0.1  -> BUY  (oversold)
"""

from typing import Dict
from src.agents.base_agent import BaseAgent, AgentVote


class MeanReversionAgent(BaseAgent):
    """Fades statistical extremes when mean-reversion conditions hold."""

    def __init__(self, name: str = 'mean_reversion', config: Dict = None):
        super().__init__(name=name, config=config)

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        ou = quant_state.get('ou_process', {})
        trend = quant_state.get('trend', {})
        data_quality = context.get('data_quality', 1.0)
        if not isinstance(data_quality, (int, float)) or data_quality <= 0:
            data_quality = 1.0

        is_stationary = ou.get('is_stationary', False)
        adf_pvalue = ou.get('adf_pvalue', 1.0)
        z_score = ou.get('z_score', 0.0)
        kappa = ou.get('kappa', 0.0)
        half_life = ou.get('half_life', 100.0)

        rsi = trend.get('rsi_14', 50.0)
        bb_position = trend.get('bb_position', 0.5)

        direction = 0
        confidence = 0.0
        position_scale = 0.0
        reasoning_parts = []

        # --- Stationarity gate ---
        if not is_stationary or adf_pvalue >= 0.05:
            return AgentVote(
                direction=0,
                confidence=round(0.15 * data_quality, 4),
                position_scale=0.0,
                reasoning=(
                    f"[IS_STATIONARY={is_stationary}] [ADF_P={adf_pvalue:.4f}] "
                    f"not stationary, mean-reversion inactive -> FLAT"
                ),
                metadata={
                    'stationary': False,
                    'adf_pvalue': adf_pvalue,
                },
            )

        # --- Overbought: SELL ---
        if z_score > 2.0 and rsi > 70 and bb_position > 0.9:
            direction = -1
            reasoning_parts.append(
                f"[Z_SCORE={z_score:.2f}] [RSI={rsi:.1f}] [BB_POS={bb_position:.2f}] "
                f"overbought trifecta -> SELL"
            )
        # --- Oversold: BUY ---
        elif z_score < -2.0 and rsi < 30 and bb_position < 0.1:
            direction = 1
            reasoning_parts.append(
                f"[Z_SCORE={z_score:.2f}] [RSI={rsi:.1f}] [BB_POS={bb_position:.2f}] "
                f"oversold trifecta -> BUY"
            )
        # --- Moderate signals (relaxed thresholds) ---
        elif z_score > 1.5 and rsi > 65:
            direction = -1
            reasoning_parts.append(
                f"[Z_SCORE={z_score:.2f}] [RSI={rsi:.1f}] moderate overbought -> SELL (weak)"
            )
        elif z_score < -1.5 and rsi < 35:
            direction = 1
            reasoning_parts.append(
                f"[Z_SCORE={z_score:.2f}] [RSI={rsi:.1f}] moderate oversold -> BUY (weak)"
            )
        else:
            direction = 0
            reasoning_parts.append(
                f"[Z_SCORE={z_score:.2f}] [RSI={rsi:.1f}] [BB_POS={bb_position:.2f}] "
                f"no extreme detected -> FLAT"
            )

        # --- Position scale: proportional to z-score extremity ---
        position_scale = min(1.0, abs(z_score) / 3.0) if direction != 0 else 0.0

        # --- Confidence: z-score extremity * kappa (faster reversion = higher conf) ---
        z_conf = min(1.0, abs(z_score) / 4.0)
        kappa_conf = min(1.0, kappa * 10.0) if kappa > 0 else 0.3
        confidence = z_conf * kappa_conf * data_quality
        if direction == 0:
            confidence = min(confidence, 0.3)

        confidence = max(0.0, min(1.0, confidence))

        return AgentVote(
            direction=direction,
            confidence=round(confidence, 4),
            position_scale=round(position_scale, 4),
            reasoning=" | ".join(reasoning_parts),
            metadata={
                'stationary': True,
                'adf_pvalue': adf_pvalue,
                'z_score': z_score,
                'kappa': kappa,
                'expected_hold_bars': round(half_life, 1),
                'half_life': half_life,
            },
        )
