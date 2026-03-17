"""
Trade Timing Agent
===================
Determines optimal entry/exit timing by combining Hawkes process intensity,
alpha decay freshness, OU mean-reversion z-score, RSI, and Kalman residuals.
Outputs an action tag (ENTER / WAIT / EXIT) alongside the directional vote.
"""

from typing import Dict, Any
from src.agents.base_agent import BaseAgent, AgentVote


class TradeTimingAgent(BaseAgent):
    """Decides WHEN to trade, not just which direction."""

    def __init__(self, name: str = 'trade_timing', config: Dict = None):
        super().__init__(name=name, config=config)

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        # --- Extract inputs ---
        hawkes = quant_state.get('hawkes', {})
        hawkes_intensity = hawkes.get('intensity', 0.0)
        hawkes_regime = hawkes.get('regime', 'unknown')
        trade_allowed = hawkes.get('trade_allowed', True)

        alpha = quant_state.get('alpha_decay', {})
        freshness = alpha.get('freshness', 0.5)
        should_exit = alpha.get('should_exit', False)

        ou = quant_state.get('ou_process', {})
        z_score = ou.get('z_score', 0.0)

        trend = quant_state.get('trend', {})
        rsi = trend.get('rsi_14', 50.0)

        kalman = quant_state.get('kalman', {})
        kalman_residual = abs(kalman.get('residual', 0.0))

        raw_signal = context.get('raw_signal', 0)

        # --- Determine action: ENTER / WAIT / EXIT ---
        action = 'WAIT'
        direction = 0
        position_scale = 0.0
        reasons = []
        alignment_count = 0
        total_factors = 4  # hawkes, alpha, OU, kalman

        # Factor 1: Hawkes regime
        hawkes_calm = hawkes_regime in ('calm', 'normal') and trade_allowed
        if hawkes_calm:
            alignment_count += 1
        else:
            reasons.append(f"Hawkes clustering [REGIME={hawkes_regime}] [INTENSITY={hawkes_intensity:.2f}]")

        # Factor 2: Alpha freshness
        alpha_fresh = freshness > 0.7
        alpha_stale = freshness < 0.3 and should_exit
        if alpha_fresh:
            alignment_count += 1
        elif alpha_stale:
            reasons.append(f"Alpha stale [FRESHNESS={freshness:.2f}] should_exit")

        # Factor 3: OU z-score extreme
        ou_extreme = abs(z_score) > 1.5
        if ou_extreme:
            alignment_count += 1
            ou_dir = -1 if z_score > 1.5 else 1  # mean reversion
        else:
            ou_dir = 0

        # Factor 4: Kalman residual breakout
        kalman_breakout = kalman_residual > 2.0
        if kalman_breakout:
            alignment_count += 1
            reasons.append(f"Kalman breakout [RESIDUAL={kalman_residual:.2f}]")

        # --- Decision logic ---
        if hawkes_calm and alpha_fresh and ou_extreme:
            # Strong entry signal
            action = 'ENTER'
            direction = ou_dir if raw_signal == 0 else raw_signal
            position_scale = 0.8
            reasons.append(f"ENTER: Hawkes calm + alpha fresh + [OU_Z={z_score:.2f}]")

        elif hawkes_calm and kalman_breakout and alpha_fresh:
            # Breakout entry
            action = 'ENTER'
            direction = raw_signal if raw_signal != 0 else (1 if kalman.get('slope', 0) > 0 else -1)
            position_scale = 0.7
            reasons.append(f"ENTER via Kalman breakout [SLOPE={kalman.get('slope', 0):.4f}]")

        elif alpha_stale:
            # Exit signal: alpha has decayed
            action = 'EXIT'
            direction = 0
            position_scale = 0.0
            reasons.append(f"EXIT: alpha decayed [FRESHNESS={freshness:.2f}]")

        elif not trade_allowed or hawkes_regime == 'clustering':
            # Wait: market microstructure unfavorable
            action = 'WAIT'
            direction = 0
            position_scale = 0.1
            reasons.append("WAIT: Hawkes clustering / trade not allowed")

        else:
            # Default: weak wait
            action = 'WAIT'
            direction = raw_signal
            position_scale = 0.3
            reasons.append(f"WAIT: insufficient timing alignment ({alignment_count}/{total_factors})")

        # RSI confirmation / override
        if action == 'ENTER':
            if (direction > 0 and rsi > 75) or (direction < 0 and rsi < 25):
                position_scale *= 0.5
                reasons.append(f"RSI overextended [RSI={rsi:.1f}] reducing scale")

        # --- Confidence ---
        confidence = alignment_count / total_factors
        confidence = max(0.05, min(1.0, confidence))

        reasoning = "; ".join(reasons) if reasons else "No timing signal"

        return AgentVote(
            direction=direction,
            confidence=round(confidence, 4),
            position_scale=round(max(0.0, min(1.0, position_scale)), 4),
            reasoning=reasoning,
            metadata={
                'action': action,
                'alignment_count': alignment_count,
                'hawkes_regime': hawkes_regime,
                'freshness': freshness,
                'ou_z_score': z_score,
                'kalman_residual': kalman_residual,
            },
        )
