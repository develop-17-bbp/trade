"""
Market Structure Agent
======================
Combines Hurst exponent regime detection, Kalman filter signal quality,
and order book microstructure to determine market structure state.

Regimes:
  - Trending (Hurst > 0.55): follow Kalman slope direction
  - Mean-reverting (Hurst < 0.45): fade Kalman residual spikes
  - Random walk: stay flat
"""

from typing import Dict
from src.agents.base_agent import BaseAgent, AgentVote


class MarketStructureAgent(BaseAgent):
    """Detects market microstructure regime and aligns trades accordingly."""

    def __init__(self, name: str = 'market_structure', config: Dict = None):
        super().__init__(name=name, config=config)

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        hurst_data = quant_state.get('hurst', {})
        kalman_data = quant_state.get('kalman', {})
        ext_feats = context.get('ext_feats', {})
        data_quality = context.get('data_quality', 1.0)
        if not isinstance(data_quality, (int, float)) or data_quality <= 0:
            data_quality = 1.0

        hurst_val = hurst_data.get('hurst', 0.5)
        hurst_regime = hurst_data.get('regime', 'random')
        hurst_conf = hurst_data.get('confidence', 0.5)

        kalman_slope = kalman_data.get('slope', 0.0)
        kalman_snr = kalman_data.get('snr', 0.0)
        kalman_residual = kalman_data.get('residual', 0.0)

        bid_ask_imbalance = ext_feats.get('bid_ask_imbalance', 0.0)
        if bid_ask_imbalance is None:
            bid_ask_imbalance = 0.0

        direction = 0
        confidence = 0.0
        position_scale = 1.0
        reasoning_parts = []

        # --- Trending regime ---
        if hurst_val > 0.55 and kalman_snr > 2.0:
            if kalman_slope > 0:
                direction = 1
                reasoning_parts.append(
                    f"[HURST={hurst_val:.3f}] trending + [KALMAN_SLOPE={kalman_slope:.4f}] positive "
                    f"+ [SNR={kalman_snr:.2f}] strong signal -> BUY"
                )
            elif kalman_slope < 0:
                direction = -1
                reasoning_parts.append(
                    f"[HURST={hurst_val:.3f}] trending + [KALMAN_SLOPE={kalman_slope:.4f}] negative "
                    f"+ [SNR={kalman_snr:.2f}] strong signal -> SELL"
                )
            else:
                direction = 0
                reasoning_parts.append(
                    f"[HURST={hurst_val:.3f}] trending but [KALMAN_SLOPE=0] flat -> FLAT"
                )
            confidence = min(hurst_conf, min(kalman_snr / 5.0, 1.0)) * data_quality

        # --- Mean-reverting regime ---
        elif hurst_val < 0.45 and abs(kalman_residual) > 2.0:
            if kalman_residual > 2.0:
                direction = -1  # fade the spike: sell when residual is high
                reasoning_parts.append(
                    f"[HURST={hurst_val:.3f}] mean-reverting + [RESIDUAL={kalman_residual:.2f}] "
                    f"spike above +2 std -> mean-reversion SELL"
                )
            else:
                direction = 1  # fade the dip: buy when residual is low
                reasoning_parts.append(
                    f"[HURST={hurst_val:.3f}] mean-reverting + [RESIDUAL={kalman_residual:.2f}] "
                    f"spike below -2 std -> mean-reversion BUY"
                )
            confidence = min(hurst_conf, min(abs(kalman_residual) / 4.0, 1.0)) * data_quality
            position_scale = min(1.0, abs(kalman_residual) / 3.0)

        # --- Random walk / low signal ---
        else:
            direction = 0
            confidence = 0.2 * data_quality
            reasoning_parts.append(
                f"[HURST={hurst_val:.3f}] {hurst_regime} + [SNR={kalman_snr:.2f}] "
                f"low signal quality -> FLAT"
            )

        # --- Order book imbalance confirmation / override ---
        if bid_ask_imbalance > 0.3 and direction >= 0:
            if direction == 0:
                direction = 1
                confidence = max(confidence, 0.3) * data_quality
            confidence = min(1.0, confidence * 1.15)
            reasoning_parts.append(
                f"[BID_ASK_IMBALANCE={bid_ask_imbalance:.2f}] bullish order flow confirms"
            )
        elif bid_ask_imbalance < -0.3 and direction <= 0:
            if direction == 0:
                direction = -1
                confidence = max(confidence, 0.3) * data_quality
            confidence = min(1.0, confidence * 1.15)
            reasoning_parts.append(
                f"[BID_ASK_IMBALANCE={bid_ask_imbalance:.2f}] bearish order flow confirms"
            )
        elif abs(bid_ask_imbalance) > 0.3 and (
            (bid_ask_imbalance > 0 and direction == -1) or
            (bid_ask_imbalance < 0 and direction == 1)
        ):
            confidence *= 0.7
            reasoning_parts.append(
                f"[BID_ASK_IMBALANCE={bid_ask_imbalance:.2f}] contradicts direction, "
                f"reducing confidence"
            )

        confidence = max(0.0, min(1.0, confidence))

        return AgentVote(
            direction=direction,
            confidence=round(confidence, 4),
            position_scale=round(position_scale, 4),
            reasoning=" | ".join(reasoning_parts),
            metadata={
                'hurst_regime': hurst_regime,
                'hurst_value': hurst_val,
                'kalman_snr': kalman_snr,
                'bid_ask_imbalance': bid_ask_imbalance,
            },
        )
