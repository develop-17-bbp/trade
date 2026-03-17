"""
Portfolio Optimizer Agent
==========================
Manages position sizing, diversification, and capital allocation using
Monte Carlo profit probabilities, Kelly criterion, open-position counts,
correlation checks, and daily PnL targets.
"""

from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent, AgentVote


class PortfolioOptimizerAgent(BaseAgent):
    """Optimizes portfolio-level risk and position sizing."""

    def __init__(self, name: str = 'portfolio_optimizer', config: Dict = None):
        super().__init__(name=name, config=config)
        self.max_positions = config.get('max_positions', 3) if config else 3
        self.max_capital_pct = config.get('max_capital_pct', 0.30) if config else 0.30
        self.correlation_threshold = 0.85

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        # --- Extract inputs ---
        mc = quant_state.get('monte_carlo_risk', {})
        prob_profit = mc.get('prob_profit', 0.5)
        mc_risk_score = mc.get('risk_score', 0.5)
        mc_position_scale = mc.get('position_scale', 1.0)

        account_balance = context.get('account_balance', 10000.0)
        open_positions: List[Dict] = context.get('open_positions', [])
        daily_pnl = context.get('daily_pnl', 0.0)
        raw_signal = context.get('raw_signal', 0)
        ext_feats = context.get('ext_feats', {})

        num_open = len(open_positions)
        reasons = []

        # --- Kelly criterion sizing ---
        kelly = 0.0
        if prob_profit > 0.0:
            # Simplified Kelly: f* = p - q/b where b=1 (even payoff assumed)
            kelly = prob_profit - (1.0 - prob_profit) / 2.0
            kelly = max(0.0, min(0.5, kelly))
            reasons.append(f"Kelly fraction [KELLY={kelly:.3f}] from [PROB_PROFIT={prob_profit:.2f}]")

        # --- Position count check ---
        position_scale = kelly if kelly > 0 else mc_position_scale
        confidence_scale = 1.0

        if num_open >= self.max_positions:
            confidence_scale *= 0.5
            position_scale *= 0.3
            reasons.append(f"Max positions reached [OPEN={num_open}] scaling down")

        # --- Capital concentration check ---
        if account_balance > 0 and num_open > 0:
            avg_position_pct = 1.0 / max(num_open, 1)
            if avg_position_pct > self.max_capital_pct:
                position_scale *= 0.7
                reasons.append(f"Concentration risk [AVG_PCT={avg_position_pct:.1%}] > {self.max_capital_pct:.0%}")

        # --- Correlation check ---
        correlation_penalty = False
        for pos in open_positions:
            pos_corr = pos.get('correlation', 0.0)
            if abs(pos_corr) > self.correlation_threshold:
                position_scale *= 0.5
                correlation_penalty = True
                reasons.append(f"High correlation [CORR={pos_corr:.2f}] with open position, halving scale")
                break  # One penalty is enough

        # --- Portfolio action ---
        if num_open >= self.max_positions:
            portfolio_action = 'FLAT'
            direction = 0
            reasons.append("FLAT: too many open positions")
        elif daily_pnl > 2.0:
            portfolio_action = 'REDUCE'
            direction = 0
            position_scale *= 0.4
            reasons.append(f"REDUCE: [DAILY_PNL={daily_pnl:.2f}%] near target, locking profits")
        elif daily_pnl < 0.5 and num_open < self.max_positions:
            portfolio_action = 'ADD_EXPOSURE'
            direction = raw_signal
            reasons.append(f"ADD_EXPOSURE: room for new position [OPEN={num_open}] [PNL={daily_pnl:.2f}%]")
        else:
            portfolio_action = 'HOLD'
            direction = raw_signal
            reasons.append(f"HOLD: portfolio stable [OPEN={num_open}] [PNL={daily_pnl:.2f}%]")

        # --- Diversification-based confidence ---
        if num_open == 0:
            diversification_score = 1.0
        elif num_open == 1:
            diversification_score = 0.8
        elif num_open == 2:
            diversification_score = 0.6 if not correlation_penalty else 0.4
        else:
            diversification_score = 0.3

        confidence = diversification_score * confidence_scale * min(1.0, prob_profit + 0.2)
        confidence = max(0.05, min(1.0, confidence))

        # Clamp
        position_scale = max(0.0, min(1.0, position_scale))

        reasoning = "; ".join(reasons) if reasons else "Portfolio neutral"

        return AgentVote(
            direction=direction,
            confidence=round(confidence, 4),
            position_scale=round(position_scale, 4),
            reasoning=reasoning,
            metadata={
                'portfolio_action': portfolio_action,
                'kelly_fraction': round(kelly, 4),
                'num_open_positions': num_open,
                'diversification_score': round(diversification_score, 3),
                'correlation_penalty': correlation_penalty,
                'mc_prob_profit': prob_profit,
            },
        )
