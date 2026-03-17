"""
Agent Combiner — Bayesian Weighted Consensus with Regime-Adaptive Multipliers
=============================================================================
Combines 10 analysis agent votes into a single direction + confidence + position_scale.

Formula:
  S_dir = SUM(w_i * r_i * c_i) for agents voting that direction
  P_dir = S_dir / (S_long + S_short + S_flat)
  If P_dir > consensus_threshold => trade that direction

Bayesian weight update after each trade:
  w_i(t+1) = w_i(t) * (alpha * accuracy_i + (1 - alpha))
  Normalized and clamped to [0.3, 3.0]
"""

from typing import Dict, Optional
from src.agents.base_agent import AgentVote, EnhancedDecision


# Regime-adaptive multipliers for each agent type
REGIME_MULTIPLIERS = {
    'market_structure': {'crisis': 0.5, 'bull': 1.0, 'bear': 1.0, 'sideways': 1.2},
    'regime_intelligence': {'crisis': 2.0, 'bull': 0.8, 'bear': 1.5, 'sideways': 1.0},
    'mean_reversion': {'crisis': 0.3, 'bull': 0.5, 'bear': 1.3, 'sideways': 1.5},
    'trend_momentum': {'crisis': 0.3, 'bull': 1.5, 'bear': 0.7, 'sideways': 0.5},
    'risk_guardian': {'crisis': 2.0, 'bull': 0.8, 'bear': 1.5, 'sideways': 1.0},
    'sentiment_decoder': {'crisis': 1.5, 'bull': 1.0, 'bear': 1.5, 'sideways': 0.8},
    'trade_timing': {'crisis': 0.5, 'bull': 1.0, 'bear': 1.0, 'sideways': 1.5},
    'portfolio_optimizer': {'crisis': 1.5, 'bull': 1.0, 'bear': 1.5, 'sideways': 1.0},
    'pattern_matcher': {'crisis': 0.5, 'bull': 1.2, 'bear': 1.0, 'sideways': 1.2},
    'loss_prevention': {'crisis': 2.5, 'bull': 0.8, 'bear': 2.0, 'sideways': 1.0},
    'polymarket_arb': {'crisis': 1.5, 'bull': 1.0, 'bear': 1.0, 'sideways': 1.2},
}


class AgentCombiner:
    """Combines 10 analysis agent votes via confidence-weighted consensus."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.consensus_threshold = cfg.get('consensus_threshold', 0.55)
        self.min_agents_for_trade = cfg.get('min_agents_for_trade', 4)

    def combine(self, votes: Dict[str, AgentVote], agents: Dict,
                regime: str = 'sideways',
                loss_guardian_vote: Optional[AgentVote] = None) -> EnhancedDecision:
        """
        Combine all analysis agent votes into a single EnhancedDecision.

        Args:
            votes: Dict of agent_name -> AgentVote (10 analysis agents)
            agents: Dict of agent_name -> BaseAgent (for getting dynamic weights)
            regime: Current HMM regime (bull/bear/crisis/sideways)
            loss_guardian_vote: LossPreventionGuardian's vote (checked for VETO)

        Returns:
            EnhancedDecision with combined direction, confidence, position_scale
        """
        # Check for Loss Prevention Guardian VETO first
        if loss_guardian_vote and loss_guardian_vote.veto:
            return EnhancedDecision(
                direction=0,
                confidence=1.0,
                position_scale=0.0,
                consensus_level='VETOED',
                daily_pnl_mode=loss_guardian_vote.metadata.get('mode', 'HALT'),
                agent_votes=votes,
                veto=True,
                risk_params={'reason': 'Loss Prevention Guardian VETO'},
            )

        regime_key = regime.lower() if regime else 'sideways'
        if regime_key not in ('crisis', 'bull', 'bear', 'sideways'):
            regime_key = 'sideways'

        # Compute weighted scores per direction
        s_long = 0.0
        s_short = 0.0
        s_flat = 0.0
        position_scales_weighted = []
        total_weight = 0.0
        directional_count = 0
        strategy_rec = ''

        for name, vote in votes.items():
            if name in ('data_integrity', 'decision_auditor'):
                continue  # Gate agents don't vote directionally

            # Dynamic Bayesian weight from agent
            agent = agents.get(name)
            w_i = agent.get_weight() if agent else 1.0

            # Regime-adaptive multiplier
            r_i = REGIME_MULTIPLIERS.get(name, {}).get(regime_key, 1.0)

            weighted_conf = w_i * r_i * vote.confidence

            if vote.direction > 0:
                s_long += weighted_conf
                directional_count += 1
            elif vote.direction < 0:
                s_short += weighted_conf
                directional_count += 1
            else:
                s_flat += weighted_conf

            # Accumulate weighted position scales
            position_scales_weighted.append((w_i * r_i, vote.position_scale))
            total_weight += w_i * r_i

            # Capture strategy recommendation from regime agent
            if name == 'regime_intelligence' and vote.metadata.get('recommended_strategy'):
                strategy_rec = vote.metadata['recommended_strategy']

        # Normalize direction probabilities
        total_score = s_long + s_short + s_flat
        if total_score < 1e-8:
            return EnhancedDecision(
                direction=0, confidence=0.0, position_scale=0.0,
                consensus_level='CONFLICT', agent_votes=votes,
            )

        p_long = s_long / total_score
        p_short = s_short / total_score
        p_flat = s_flat / total_score

        # Determine direction by consensus threshold
        if p_long > self.consensus_threshold:
            direction = 1
            confidence = p_long
        elif p_short > self.consensus_threshold:
            direction = -1
            confidence = p_short
        else:
            direction = 0
            confidence = p_flat

        # Determine consensus level
        max_p = max(p_long, p_short, p_flat)
        if max_p > 0.75:
            consensus_level = 'STRONG'
        elif max_p > 0.60:
            consensus_level = 'MODERATE'
        elif max_p > self.consensus_threshold:
            consensus_level = 'WEAK'
        else:
            consensus_level = 'CONFLICT'

        # Compute weighted average position scale
        if total_weight > 0:
            position_scale = sum(w * ps for w, ps in position_scales_weighted) / total_weight
        else:
            position_scale = 0.0

        # Apply Loss Prevention Guardian's allowed scale
        if loss_guardian_vote:
            lp_scale = loss_guardian_vote.position_scale
            position_scale = min(position_scale, lp_scale)
            daily_mode = loss_guardian_vote.metadata.get('mode', 'NORMAL')

            # Enforce minimum agent agreement based on mode
            min_required = loss_guardian_vote.metadata.get('min_agents_required', self.min_agents_for_trade)
            if directional_count < min_required and direction != 0:
                direction = 0
                confidence = 0.0
                position_scale = 0.0
                consensus_level = 'CONFLICT'
        else:
            daily_mode = 'NORMAL'

        # If direction is flat, zero out position
        if direction == 0:
            position_scale = 0.0

        return EnhancedDecision(
            direction=direction,
            confidence=min(1.0, confidence),
            position_scale=min(1.0, max(0.0, position_scale)),
            strategy_recommendation=strategy_rec,
            agent_votes=votes,
            consensus_level=consensus_level,
            daily_pnl_mode=daily_mode,
            risk_params={
                'p_long': round(p_long, 4),
                'p_short': round(p_short, 4),
                'p_flat': round(p_flat, 4),
                'directional_agents': directional_count,
            },
        )
