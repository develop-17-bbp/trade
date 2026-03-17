"""
Trend Momentum Agent
====================
Multi-factor trend confirmation engine combining:
  - ADX (trend strength)
  - MACD histogram (momentum direction)
  - EMA/SMA alignment (golden/death cross)
  - Kalman slope (filtered trend)
  - Hurst exponent (trending regime)
  - Fractional differentiation momentum (optional)

Core rule: ADX > 25 + Hurst trending + aligned Kalman & MACD -> trade.
"""

from typing import Dict
from src.agents.base_agent import BaseAgent, AgentVote


class TrendMomentumAgent(BaseAgent):
    """Multi-indicator trend/momentum confirmation agent."""

    def __init__(self, name: str = 'trend_momentum', config: Dict = None):
        super().__init__(name=name, config=config)

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        trend = quant_state.get('trend', {})
        kalman = quant_state.get('kalman', {})
        hurst_data = quant_state.get('hurst', {})
        fracdiff = quant_state.get('fracdiff', {})
        data_quality = context.get('data_quality', 1.0)
        if not isinstance(data_quality, (int, float)) or data_quality <= 0:
            data_quality = 1.0

        adx = trend.get('adx', 0.0)
        macd_hist = trend.get('macd_hist', 0.0)
        ema_10 = trend.get('ema_10', 0.0)
        sma_20 = trend.get('sma_20', 0.0)
        sma_50 = trend.get('sma_50', 0.0)

        kalman_slope = kalman.get('slope', 0.0)

        hurst_val = hurst_data.get('hurst', 0.5)
        hurst_conf = hurst_data.get('confidence', 0.5)
        hurst_trending = hurst_val > 0.55

        fracdiff_momentum = fracdiff.get('fracdiff_momentum', 0.0)

        direction = 0
        confidence = 0.0
        position_scale = 1.0
        reasoning_parts = []

        # --- No trend: ADX too low ---
        if adx < 20:
            return AgentVote(
                direction=0,
                confidence=round(0.2 * data_quality, 4),
                position_scale=0.0,
                reasoning=(
                    f"[ADX={adx:.1f}] below 20, no meaningful trend -> FLAT"
                ),
                metadata={
                    'adx': adx,
                    'hurst_trending': hurst_trending,
                    'trend_strength': 'none',
                },
            )

        # --- Bullish trend confirmation ---
        bull_signals = 0
        bear_signals = 0

        if kalman_slope > 0:
            bull_signals += 1
        elif kalman_slope < 0:
            bear_signals += 1

        if macd_hist > 0:
            bull_signals += 1
        elif macd_hist < 0:
            bear_signals += 1

        if hurst_trending:
            # Hurst confirms trend exists; direction from other indicators
            bull_signals += 0  # neutral boost
            bear_signals += 0
        else:
            # Weak trend environment even if ADX > 20
            confidence *= 0.7

        # --- Golden/Death cross ---
        golden_cross = (ema_10 > sma_20 > sma_50) and all(
            v > 0 for v in [ema_10, sma_20, sma_50]
        )
        death_cross = (ema_10 < sma_20 < sma_50) and all(
            v > 0 for v in [ema_10, sma_20, sma_50]
        )

        if golden_cross:
            bull_signals += 1
            reasoning_parts.append(
                f"[EMA10={ema_10:.2f}]>[SMA20={sma_20:.2f}]>[SMA50={sma_50:.2f}] golden cross"
            )
        elif death_cross:
            bear_signals += 1
            reasoning_parts.append(
                f"[EMA10={ema_10:.2f}]<[SMA20={sma_20:.2f}]<[SMA50={sma_50:.2f}] death cross"
            )

        # --- FracDiff momentum ---
        if fracdiff_momentum > 0.01:
            bull_signals += 1
            reasoning_parts.append(f"[FRACDIFF_MOM={fracdiff_momentum:.4f}] bullish")
        elif fracdiff_momentum < -0.01:
            bear_signals += 1
            reasoning_parts.append(f"[FRACDIFF_MOM={fracdiff_momentum:.4f}] bearish")

        # --- Strong trend: ADX > 25 + Hurst trending + aligned signals ---
        strong_trend = adx > 25 and hurst_trending

        if strong_trend and bull_signals >= 2 and bull_signals > bear_signals:
            direction = 1
            reasoning_parts.insert(0,
                f"[ADX={adx:.1f}] [HURST={hurst_val:.3f}] trending "
                f"[KALMAN_SLOPE={kalman_slope:.4f}] [MACD_HIST={macd_hist:.4f}] "
                f"bull signals={bull_signals} -> BUY"
            )
        elif strong_trend and bear_signals >= 2 and bear_signals > bull_signals:
            direction = -1
            reasoning_parts.insert(0,
                f"[ADX={adx:.1f}] [HURST={hurst_val:.3f}] trending "
                f"[KALMAN_SLOPE={kalman_slope:.4f}] [MACD_HIST={macd_hist:.4f}] "
                f"bear signals={bear_signals} -> SELL"
            )
        elif adx > 25:
            # ADX strong but signals conflict or Hurst not trending
            if bull_signals > bear_signals:
                direction = 1
                position_scale = 0.5
            elif bear_signals > bull_signals:
                direction = -1
                position_scale = 0.5
            else:
                direction = 0
                position_scale = 0.0
            reasoning_parts.insert(0,
                f"[ADX={adx:.1f}] moderate trend, mixed signals "
                f"bull={bull_signals} bear={bear_signals} -> "
                f"{'BUY' if direction > 0 else 'SELL' if direction < 0 else 'FLAT'} (reduced)"
            )
        else:
            # ADX 20-25: weak trend
            direction = 0
            position_scale = 0.0
            reasoning_parts.insert(0,
                f"[ADX={adx:.1f}] weak trend zone (20-25) -> FLAT"
            )

        # --- Confidence: normalized ADX * hurst_confidence ---
        adx_norm = min(1.0, adx / 50.0)
        confidence = adx_norm * hurst_conf * data_quality
        if not hurst_trending:
            confidence *= 0.7
        if direction == 0:
            confidence = min(confidence, 0.25)

        confidence = max(0.0, min(1.0, confidence))

        return AgentVote(
            direction=direction,
            confidence=round(confidence, 4),
            position_scale=round(max(0.0, min(1.0, position_scale)), 4),
            reasoning=" | ".join(reasoning_parts),
            metadata={
                'adx': adx,
                'hurst_trending': hurst_trending,
                'bull_signals': bull_signals,
                'bear_signals': bear_signals,
                'golden_cross': golden_cross,
                'death_cross': death_cross,
                'fracdiff_momentum': fracdiff_momentum,
                'trend_strength': 'strong' if adx > 40 else 'moderate' if adx > 25 else 'weak',
            },
        )
