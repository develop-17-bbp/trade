"""
Agent Debate Engine — Multi-Round Adversarial Deliberation
===========================================================
Instead of agents voting independently, they now debate:

Round 1: Initial votes (existing system — parallel analysis)
Round 2: Challengers critique opposing votes citing specific metrics
Round 3: Defenders respond, may update conviction or flip
Final:   Post-debate votes with conviction multipliers feed into combiner

The debate surfaces hidden contradictions and strengthens real setups:
- Bull agents must justify against bear arguments (and vice versa)
- Agents that survive cross-examination get a conviction bonus
- Agents that flip under scrutiny lose weight for this decision
- Unanimous post-debate consensus gets a strong bonus

This replaces blind averaging with adversarial stress-testing.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from src.agents.base_agent import AgentVote

logger = logging.getLogger(__name__)

# Agents grouped by natural opposition — these pairs challenge each other
DEBATE_MATCHUPS = {
    # Bull-leaning vs Bear-leaning pairs
    'trend_momentum': 'mean_reversion',
    'mean_reversion': 'trend_momentum',
    # Opportunity vs Risk pairs
    'market_structure': 'risk_guardian',
    'risk_guardian': 'market_structure',
    # Sentiment vs Quant pairs
    'sentiment_decoder': 'pattern_matcher',
    'pattern_matcher': 'sentiment_decoder',
    # Timing vs Portfolio pairs
    'trade_timing': 'portfolio_optimizer',
    'portfolio_optimizer': 'trade_timing',
    # External vs Internal pairs
    'polymarket_arb': 'regime_intelligence',
    'regime_intelligence': 'polymarket_arb',
}

# Metric keys each agent is an authority on (used to evaluate challenges)
AGENT_EXPERTISE = {
    'trend_momentum': ['adx', 'macd_hist', 'hurst', 'kalman_slope', 'ema_alignment'],
    'mean_reversion': ['zscore', 'bollinger_pct', 'kalman_residual', 'hurst'],
    'market_structure': ['hurst', 'kalman_snr', 'ob_imbalance', 'liquidity_regime'],
    'risk_guardian': ['mc_var_risk', 'evt_tail_risk', 'hawkes_intensity', 'vpin', 'drawdown_ratio'],
    'sentiment_decoder': ['sentiment_score', 'news_momentum', 'fear_greed'],
    'pattern_matcher': ['rsi', 'macd_hist', 'bollinger_pct', 'volume_ratio'],
    'trade_timing': ['hawkes_intensity', 'session_quality', 'spread_score'],
    'portfolio_optimizer': ['correlation', 'portfolio_heat', 'max_allocation'],
    'polymarket_arb': ['prediction_market_prob', 'divergence', 'pm_volume'],
    'regime_intelligence': ['hmm_regime', 'regime_confidence', 'crisis_prob'],
}


@dataclass
class DebateRound:
    """Record of a single debate exchange."""
    challenger: str
    defender: str
    challenge: str           # The critique
    defense: str             # The response
    challenger_metric: str   # Key metric cited by challenger
    defender_held: bool      # Did the defender maintain their position?
    conviction_delta: float  # Change in defender's conviction (-0.3 to +0.2)


@dataclass
class DebateResult:
    """Full debate outcome for one trade decision."""
    rounds: List[DebateRound] = field(default_factory=list)
    post_debate_votes: Dict[str, AgentVote] = field(default_factory=dict)
    conviction_multipliers: Dict[str, float] = field(default_factory=dict)
    flipped_agents: List[str] = field(default_factory=list)
    strengthened_agents: List[str] = field(default_factory=list)
    debate_summary: str = ""
    consensus_shift: str = ""  # "STRENGTHENED", "WEAKENED", "REVERSED", "UNCHANGED"


class DebateEngine:
    """
    Orchestrates adversarial debate between agents before final vote combination.

    The debate protocol:
    1. Each agent with a directional vote gets challenged by its natural opponent
    2. Challenges cite specific metrics that contradict the voter's position
    3. Defenders evaluate challenges against their own analysis
    4. Conviction is adjusted: survive challenge → bonus, fail → penalty
    5. Agents may flip direction if the challenge is strong enough
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        debate_cfg = cfg.get('debate', cfg.get('agents', {}))

        # Conviction adjustments
        self.survive_bonus = debate_cfg.get('survive_bonus', 0.15)
        self.flip_penalty = debate_cfg.get('flip_penalty', 0.30)
        self.partial_penalty = debate_cfg.get('partial_penalty', 0.10)

        # Thresholds
        self.challenge_strength_threshold = debate_cfg.get('challenge_threshold', 0.4)
        self.flip_threshold = debate_cfg.get('flip_threshold', 0.7)
        self.min_confidence_to_debate = debate_cfg.get('min_confidence_to_debate', 0.15)

        # Unanimous consensus bonus
        self.unanimity_bonus = debate_cfg.get('unanimity_bonus', 0.20)

    def run_debate(self, votes: Dict[str, AgentVote],
                   quant_state: Dict, context: Dict) -> DebateResult:
        """
        Execute the full debate protocol.

        Args:
            votes: Initial agent votes from Round 1 (parallel analysis)
            quant_state: The sanitized quant state (metrics for challenge evaluation)
            context: Full context dict

        Returns:
            DebateResult with post-debate votes and conviction multipliers
        """
        result = DebateResult()
        conviction = {name: 1.0 for name in votes}
        post_votes = {name: AgentVote(
            direction=v.direction,
            confidence=v.confidence,
            position_scale=v.position_scale,
            reasoning=v.reasoning,
            veto=v.veto,
            metadata=dict(v.metadata),
        ) for name, v in votes.items()}

        # Skip gate agents from debate
        debatable = {n: v for n, v in votes.items()
                     if n not in ('data_integrity', 'decision_auditor', 'loss_prevention')}

        # Identify the two camps
        bulls = {n: v for n, v in debatable.items() if v.direction > 0}
        bears = {n: v for n, v in debatable.items() if v.direction < 0}
        flats = {n: v for n, v in debatable.items() if v.direction == 0}

        # ── ROUND 2: Challenges ──
        # Each directional agent gets challenged by its natural opponent
        for agent_name, vote in debatable.items():
            if vote.direction == 0 and vote.confidence < self.min_confidence_to_debate:
                continue  # Flat low-confidence agents don't need to defend

            opponent_name = DEBATE_MATCHUPS.get(agent_name)
            if not opponent_name or opponent_name not in debatable:
                continue

            opponent_vote = debatable[opponent_name]

            # Only debate if they disagree or one is flat
            if vote.direction == opponent_vote.direction and vote.direction != 0:
                # They agree — both get a small conviction boost
                conviction[agent_name] = min(1.0 + self.survive_bonus, conviction[agent_name] + 0.05)
                continue

            # Generate challenge from opponent's perspective
            challenge_strength, challenge_metric, challenge_text = self._generate_challenge(
                challenger_name=opponent_name,
                challenger_vote=opponent_vote,
                defender_name=agent_name,
                defender_vote=vote,
                quant_state=quant_state,
                context=context,
            )

            # ── ROUND 3: Defense ──
            defended, defense_text, conviction_delta = self._evaluate_defense(
                defender_name=agent_name,
                defender_vote=vote,
                challenge_strength=challenge_strength,
                challenge_metric=challenge_metric,
                quant_state=quant_state,
            )

            debate_round = DebateRound(
                challenger=opponent_name,
                defender=agent_name,
                challenge=challenge_text,
                defense=defense_text,
                challenger_metric=challenge_metric,
                defender_held=defended,
                conviction_delta=conviction_delta,
            )
            result.rounds.append(debate_round)

            # Apply conviction adjustment
            conviction[agent_name] += conviction_delta

            # Check if agent should flip
            if not defended and challenge_strength > self.flip_threshold:
                # Agent flips direction under strong challenge
                old_dir = post_votes[agent_name].direction
                new_dir = opponent_vote.direction
                if old_dir != new_dir:
                    post_votes[agent_name].direction = new_dir
                    post_votes[agent_name].confidence *= (1.0 - self.flip_penalty)
                    post_votes[agent_name].reasoning += f" [DEBATE-FLIP: {challenge_text}]"
                    result.flipped_agents.append(agent_name)
                    conviction[agent_name] *= (1.0 - self.flip_penalty)
                    logger.info(f"[DEBATE] {agent_name} FLIPPED {old_dir:+d} -> {new_dir:+d} "
                                f"under challenge from {opponent_name}")

            elif defended:
                # Survived challenge — conviction boost
                result.strengthened_agents.append(agent_name)
                post_votes[agent_name].reasoning += f" [DEBATE-HELD: survived {opponent_name} challenge]"
                logger.debug(f"[DEBATE] {agent_name} HELD position against {opponent_name}")

        # ── FINAL: Apply conviction multipliers to confidence ──
        for name in post_votes:
            conv = max(0.3, min(1.5, conviction.get(name, 1.0)))
            result.conviction_multipliers[name] = round(conv, 3)
            post_votes[name].confidence = min(1.0, post_votes[name].confidence * conv)

        # Unanimity bonus: if all directional agents agree post-debate
        post_dirs = [v.direction for v in post_votes.values()
                     if v.direction != 0 and v.confidence > 0.1]
        if post_dirs and all(d == post_dirs[0] for d in post_dirs) and len(post_dirs) >= 4:
            for name in post_votes:
                if post_votes[name].direction == post_dirs[0]:
                    post_votes[name].confidence = min(1.0,
                        post_votes[name].confidence + self.unanimity_bonus)
            result.consensus_shift = "STRENGTHENED"
            logger.info(f"[DEBATE] Unanimous consensus ({post_dirs[0]:+d}) — "
                        f"applying {self.unanimity_bonus:.0%} bonus")
        elif len(result.flipped_agents) > 2:
            result.consensus_shift = "REVERSED"
        elif len(result.flipped_agents) > 0:
            result.consensus_shift = "WEAKENED"
        else:
            result.consensus_shift = "UNCHANGED"

        result.post_debate_votes = post_votes
        result.debate_summary = self._build_summary(result, bulls, bears, flats)

        return result

    def _generate_challenge(self, challenger_name: str, challenger_vote: AgentVote,
                            defender_name: str, defender_vote: AgentVote,
                            quant_state: Dict, context: Dict,
                            ) -> Tuple[float, str, str]:
        """
        Generate a metric-based challenge from one agent to another.

        Returns:
            (challenge_strength: 0-1, key_metric: str, challenge_text: str)
        """
        # Find the strongest metric the challenger can cite against the defender
        challenger_metrics = AGENT_EXPERTISE.get(challenger_name, [])
        best_strength = 0.0
        best_metric = ""
        best_text = ""

        for metric_key in challenger_metrics:
            value = self._extract_metric(quant_state, context, metric_key)
            if value is None:
                continue

            strength, text = self._evaluate_metric_against_position(
                metric_key, value, defender_vote.direction, defender_vote.confidence
            )

            if strength > best_strength:
                best_strength = strength
                best_metric = metric_key
                best_text = text

        if not best_text:
            best_text = (f"{challenger_name} challenges {defender_name}'s "
                         f"{'LONG' if defender_vote.direction > 0 else 'SHORT' if defender_vote.direction < 0 else 'FLAT'} "
                         f"with conf={challenger_vote.confidence:.2f}")
            best_strength = challenger_vote.confidence * 0.5

        return best_strength, best_metric, best_text

    def _evaluate_defense(self, defender_name: str, defender_vote: AgentVote,
                          challenge_strength: float, challenge_metric: str,
                          quant_state: Dict) -> Tuple[bool, str, float]:
        """
        Evaluate whether the defender's position survives the challenge.

        Returns:
            (defended: bool, defense_text: str, conviction_delta: float)
        """
        defender_metrics = AGENT_EXPERTISE.get(defender_name, [])

        # Count how many of the defender's own metrics support their position
        supporting_count = 0
        total_checked = 0

        for metric_key in defender_metrics:
            value = self._extract_metric(quant_state, {}, metric_key)
            if value is None:
                continue
            total_checked += 1
            if self._metric_supports_direction(metric_key, value, defender_vote.direction):
                supporting_count += 1

        if total_checked == 0:
            # No metrics to defend with — partial penalty
            return False, f"{defender_name} has no supporting metrics", -self.partial_penalty

        support_ratio = supporting_count / total_checked

        # Defense succeeds if majority of own metrics support AND confidence is decent
        defended = (support_ratio >= 0.5 and
                    defender_vote.confidence > 0.3 and
                    challenge_strength < 0.65)

        if defended:
            conviction_delta = self.survive_bonus * support_ratio
            defense_text = (f"{defender_name} defended: {supporting_count}/{total_checked} "
                            f"metrics support position")
        else:
            # Failed defense — penalty proportional to challenge strength
            conviction_delta = -(self.partial_penalty + challenge_strength * 0.15)
            defense_text = (f"{defender_name} weakened: only {supporting_count}/{total_checked} "
                            f"metrics support vs {challenge_metric}={challenge_strength:.2f}")

        return defended, defense_text, conviction_delta

    def _extract_metric(self, quant_state: Dict, context: Dict, key: str):
        """Extract a metric value from quant_state or context, searching nested dicts."""
        # Direct lookup
        if key in quant_state:
            val = quant_state[key]
            if isinstance(val, dict):
                return val
            return val

        # Search nested structures
        for section in ['trend', 'kalman', 'hurst', 'fracdiff', 'monte_carlo_risk',
                        'evt_tail_risk', 'hawkes', 'vpin', 'drawdown',
                        'bollinger', 'rsi_data', 'volume_profile']:
            sub = quant_state.get(section, {})
            if isinstance(sub, dict) and key in sub:
                return sub[key]

        # Context fallbacks
        ext = context.get('ext_feats', {})
        if key in ext:
            return ext[key]

        sentiment = context.get('sentiment_data', {})
        if key in sentiment:
            return sentiment[key]

        return None

    def _evaluate_metric_against_position(self, metric_key: str, value,
                                           direction: int, confidence: float
                                           ) -> Tuple[float, str]:
        """
        Evaluate how strongly a metric contradicts a given direction.

        Returns:
            (contradiction_strength: 0-1, description: str)
        """
        if isinstance(value, dict):
            # Handle dict-type metrics (e.g., hmm_regime)
            return self._evaluate_dict_metric(metric_key, value, direction)

        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0, ""

        # Metric-specific contradiction rules
        contradiction = 0.0
        text = ""

        if metric_key == 'adx':
            if v < 20 and direction != 0:
                contradiction = 0.7
                text = f"[ADX={v:.1f}] No trend — directional trade unjustified"
            elif v > 40 and direction == 0:
                contradiction = 0.5
                text = f"[ADX={v:.1f}] Strong trend exists — FLAT is leaving money on table"

        elif metric_key in ('macd_hist',):
            if (v > 0 and direction < 0) or (v < 0 and direction > 0):
                contradiction = min(1.0, abs(v) * 5)
                dir_label = 'bullish' if v > 0 else 'bearish'
                text = f"[MACD_HIST={v:.4f}] Histogram is {dir_label} — contradicts signal"

        elif metric_key == 'hurst':
            if v < 0.45 and direction != 0:
                contradiction = 0.6
                text = f"[HURST={v:.3f}] Mean-reverting regime — trend following risky"
            elif v > 0.55 and direction == 0:
                contradiction = 0.5
                text = f"[HURST={v:.3f}] Trending regime — missing opportunity"

        elif metric_key == 'zscore':
            if v > 2.0 and direction > 0:
                contradiction = min(1.0, v / 3.0)
                text = f"[ZSCORE={v:.2f}] Overbought — LONG entry is chasing"
            elif v < -2.0 and direction < 0:
                contradiction = min(1.0, abs(v) / 3.0)
                text = f"[ZSCORE={v:.2f}] Oversold — SHORT is selling the bottom"

        elif metric_key in ('mc_var_risk', 'evt_tail_risk'):
            if v > 0.6 and direction != 0:
                contradiction = v
                text = f"[{metric_key.upper()}={v:.2f}] High risk — trade endangers capital"

        elif metric_key == 'vpin':
            if v > 0.7 and direction != 0:
                contradiction = v
                text = f"[VPIN={v:.2f}] Toxic flow — adverse selection likely"

        elif metric_key == 'hawkes_intensity':
            if v > 0.5 and direction != 0:
                contradiction = min(1.0, v)
                text = f"[HAWKES={v:.2f}] Event clustering — unstable for entry"

        elif metric_key == 'kalman_slope':
            if (v > 0 and direction < 0) or (v < 0 and direction > 0):
                contradiction = min(1.0, abs(v) * 3)
                slope_dir = 'up' if v > 0 else 'down'
                text = f"[KALMAN_SLOPE={v:.4f}] Kalman trending {slope_dir} — contradicts"

        elif metric_key == 'kalman_snr':
            if v < 1.0 and direction != 0:
                contradiction = 0.5
                text = f"[KALMAN_SNR={v:.2f}] Low signal-to-noise — signal unreliable"

        elif metric_key == 'rsi':
            if v > 70 and direction > 0:
                contradiction = min(1.0, (v - 70) / 30)
                text = f"[RSI={v:.1f}] Overbought — LONG is chasing"
            elif v < 30 and direction < 0:
                contradiction = min(1.0, (30 - v) / 30)
                text = f"[RSI={v:.1f}] Oversold — SHORT is late"

        elif metric_key == 'drawdown_ratio':
            if v > 0.5 and direction != 0:
                contradiction = v
                text = f"[DRAWDOWN={v:.2f}] Deep drawdown — preserve capital"

        elif metric_key == 'sentiment_score':
            if (v > 0.3 and direction < 0) or (v < -0.3 and direction > 0):
                contradiction = min(1.0, abs(v))
                sent_dir = 'bullish' if v > 0 else 'bearish'
                text = f"[SENTIMENT={v:.2f}] Market sentiment is {sent_dir}"

        elif metric_key == 'ob_imbalance':
            if (v > 0.3 and direction < 0) or (v < -0.3 and direction > 0):
                contradiction = min(1.0, abs(v))
                text = f"[OB_IMBALANCE={v:.2f}] Order book pressure contradicts"

        elif metric_key == 'bollinger_pct':
            if v > 0.95 and direction > 0:
                contradiction = 0.6
                text = f"[BOLL%={v:.2f}] At upper band — breakout or reversal?"
            elif v < 0.05 and direction < 0:
                contradiction = 0.6
                text = f"[BOLL%={v:.2f}] At lower band — bounce likely"

        # Scale by defender's confidence (harder to challenge high-conviction)
        contradiction *= max(0.5, 1.0 - confidence * 0.3)

        return contradiction, text

    def _evaluate_dict_metric(self, key: str, value: Dict,
                               direction: int) -> Tuple[float, str]:
        """Handle dict-type metrics like hmm_regime."""
        if key == 'hmm_regime':
            regime = value.get('regime', 'sideways')
            crisis_prob = float(value.get('crisis_prob', 0.0))
            if crisis_prob > 0.5 and direction != 0:
                return crisis_prob, f"[CRISIS_PROB={crisis_prob:.2f}] Crisis regime — all entries dangerous"
            if regime == 'bear' and direction > 0:
                return 0.5, f"[REGIME=bear] Bear market — LONG is fighting the trend"
            if regime == 'bull' and direction < 0:
                return 0.5, f"[REGIME=bull] Bull market — SHORT is fighting the trend"
        return 0.0, ""

    def _metric_supports_direction(self, metric_key: str, value,
                                    direction: int) -> bool:
        """Check if a metric value supports the given direction."""
        if isinstance(value, dict):
            if metric_key == 'hmm_regime':
                regime = value.get('regime', 'sideways')
                if direction > 0:
                    return regime in ('bull', 'sideways')
                elif direction < 0:
                    return regime in ('bear', 'crisis')
            return True

        try:
            v = float(value)
        except (TypeError, ValueError):
            return True  # Can't evaluate — assume supporting

        if metric_key in ('macd_hist', 'kalman_slope', 'sentiment_score', 'ob_imbalance'):
            return (v > 0 and direction > 0) or (v < 0 and direction < 0) or direction == 0

        if metric_key == 'adx':
            return (v > 20 and direction != 0) or (v < 20 and direction == 0)

        if metric_key == 'hurst':
            return (v > 0.50 and direction != 0) or (v < 0.50 and direction == 0)

        if metric_key in ('mc_var_risk', 'evt_tail_risk', 'vpin', 'hawkes_intensity'):
            return v < 0.5 or direction == 0

        if metric_key == 'rsi':
            if direction > 0:
                return v < 70
            elif direction < 0:
                return v > 30
            return True

        if metric_key == 'zscore':
            if direction > 0:
                return v < 2.0
            elif direction < 0:
                return v > -2.0
            return True

        return True  # Unknown metric — assume supporting

    def _build_summary(self, result: DebateResult,
                       bulls: Dict, bears: Dict, flats: Dict) -> str:
        """Build a human-readable debate summary."""
        lines = []
        pre_bull = len(bulls)
        pre_bear = len(bears)
        pre_flat = len(flats)

        post_dirs = {}
        for name, v in result.post_debate_votes.items():
            d = 'LONG' if v.direction > 0 else ('SHORT' if v.direction < 0 else 'FLAT')
            post_dirs.setdefault(d, []).append(name)

        post_bull = len(post_dirs.get('LONG', []))
        post_bear = len(post_dirs.get('SHORT', []))
        post_flat = len(post_dirs.get('FLAT', []))

        lines.append(f"Pre-debate: {pre_bull} LONG / {pre_bear} SHORT / {pre_flat} FLAT")
        lines.append(f"Post-debate: {post_bull} LONG / {post_bear} SHORT / {post_flat} FLAT")

        if result.flipped_agents:
            lines.append(f"Flipped: {', '.join(result.flipped_agents)}")
        if result.strengthened_agents:
            lines.append(f"Strengthened: {', '.join(result.strengthened_agents[:5])}")

        lines.append(f"Consensus: {result.consensus_shift}")

        return " | ".join(lines)
