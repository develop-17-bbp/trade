"""
Decision Auditor Agent (POST-COMBINATION AUDIT)
==================================================
Receives the EnhancedDecision from context and cross-checks it against
raw quant data, agent votes, and historical performance.
"""

from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent, AgentVote, DataIntegrityReport, AuditResult


class DecisionAuditor(BaseAgent):
    """Post-combination auditor that validates final trading decisions."""

    def __init__(self, name: str = 'decision_auditor', config: Dict = None):
        super().__init__(name=name, config=config)

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        decision = context.get('enhanced_decision', {})
        agent_votes = context.get('agent_votes', {})
        trade_history = context.get('trade_history', [])

        if isinstance(decision, dict):
            dec_direction = decision.get('direction', 0)
            dec_confidence = decision.get('confidence', 0.0)
            dec_pos_scale = decision.get('position_scale', 1.0)
        else:
            # Assume dataclass-like object
            dec_direction = getattr(decision, 'direction', 0)
            dec_confidence = getattr(decision, 'confidence', 0.0)
            dec_pos_scale = getattr(decision, 'position_scale', 1.0)

        audit = AuditResult()
        audit.adjusted_confidence = dec_confidence
        audit.adjusted_position_scale = dec_pos_scale
        flags: List[str] = []
        contradiction_count = 0

        # --- 1. Decision-data alignment ---
        support, oppose = self._count_indicator_alignment(quant_state, dec_direction)
        total = support + oppose
        alignment_score = support / total if total > 0 else 0.5
        audit.data_alignment_score = round(alignment_score, 4)

        if alignment_score < 0.4:
            flags.append(f"WEAK_ALIGNMENT: only {support}/{total} indicators support direction={dec_direction}")
            audit.adjusted_confidence *= 0.8

        # --- 2. Agent contradictions ---
        contradiction_count = self._check_agent_contradictions(
            agent_votes, quant_state, flags
        )
        audit.contradiction_count = contradiction_count

        if contradiction_count >= 3:
            flags.append(f"HIGH_CONTRADICTIONS: {contradiction_count} agent conflicts detected")
            audit.adjusted_confidence *= 0.85

        # --- 3. RegimeIntelligence crisis + BUY → BLOCK ---
        regime_vote = agent_votes.get('RegimeIntelligence', agent_votes.get('regime_intelligence', {}))
        regime_meta = self._get_vote_metadata(regime_vote)
        regime_label = regime_meta.get('regime', '').lower() if regime_meta else ''

        hmm_regime = self._safe_get(quant_state, 'hmm_regime', 'regime', default='')
        crisis_prob = self._safe_get(quant_state, 'hmm_regime', 'crisis_prob', default=0.0)

        if (str(regime_label) == 'crisis' or str(hmm_regime).lower() == 'crisis' or crisis_prob > 0.7) \
                and dec_direction > 0:
            flags.append("BLOCK: RegimeIntelligence=CRISIS but decision=BUY")
            audit.recommendation = "BLOCK"
            audit.adjusted_position_scale = 0.0
            audit.approved = False

        # --- 4. TimingAgent says WAIT but decision is to trade ---
        timing_vote = agent_votes.get('TimingAgent', agent_votes.get('timing_agent', {}))
        timing_meta = self._get_vote_metadata(timing_vote)
        timing_action = timing_meta.get('action', '').lower() if timing_meta else ''

        if timing_action == 'wait' and dec_direction != 0:
            flags.append("DEFER: TimingAgent says WAIT but decision is to trade")
            if audit.recommendation not in ("BLOCK",):
                audit.recommendation = "DEFER"
                audit.adjusted_confidence *= 0.7

        # --- 5. Overconfidence check ---
        risk_vote = agent_votes.get('RiskGuardianAgent', agent_votes.get('risk_guardian', {}))
        risk_meta = self._get_vote_metadata(risk_vote)
        risk_level = risk_meta.get('risk_level', '').upper() if risk_meta else ''

        if dec_confidence > 0.9 and risk_level in ('HIGH', 'EXTREME'):
            flags.append(f"OVERCONFIDENCE: conf={dec_confidence:.2f} but RiskGuardian={risk_level}, capping at 0.7")
            audit.adjusted_confidence = min(audit.adjusted_confidence, 0.7)

        # --- 6. Historical losing streak check ---
        losing_streak = self._get_losing_streak(trade_history)
        if losing_streak >= 3:
            required_conf = 0.6 + (losing_streak * 0.05)
            required_conf = min(required_conf, 0.9)
            if audit.adjusted_confidence < required_conf:
                flags.append(
                    f"LOSING_STREAK: {losing_streak} consecutive losses, "
                    f"need conf>={required_conf:.2f} but have {audit.adjusted_confidence:.2f}"
                )
                if audit.recommendation not in ("BLOCK",):
                    audit.recommendation = "REDUCE"
                    audit.adjusted_position_scale *= 0.6

        audit.historical_win_rate = self._calc_win_rate(trade_history)

        # --- Final recommendation ---
        audit.audit_flags = flags
        audit.adjusted_confidence = round(max(0.0, min(1.0, audit.adjusted_confidence)), 4)
        audit.adjusted_position_scale = round(max(0.0, min(1.0, audit.adjusted_position_scale)), 4)

        if audit.recommendation == "EXECUTE" and not flags:
            audit.approved = True
        elif audit.recommendation == "EXECUTE" and flags:
            # Flags present but no override → REDUCE
            if len(flags) >= 2:
                audit.recommendation = "REDUCE"
                audit.adjusted_position_scale *= 0.85
                audit.adjusted_position_scale = round(max(0.0, audit.adjusted_position_scale), 4)
            audit.approved = True
        elif audit.recommendation in ("REDUCE", "DEFER"):
            audit.approved = True
        # BLOCK keeps approved=False

        reasoning_parts = [
            f"[RECOMMENDATION={audit.recommendation}]",
            f"[ALIGNMENT={alignment_score:.2f}]",
            f"[CONTRADICTIONS={contradiction_count}]",
            f"[ADJ_CONF={audit.adjusted_confidence:.2f}]",
            f"[ADJ_SCALE={audit.adjusted_position_scale:.2f}]",
        ]
        if flags:
            reasoning_parts.append(f"[FLAGS={len(flags)}]")

        return AgentVote(
            direction=dec_direction if audit.approved else 0,
            confidence=audit.adjusted_confidence,
            position_scale=audit.adjusted_position_scale,
            reasoning=" ".join(reasoning_parts),
            veto=(audit.recommendation == "BLOCK"),
            metadata={"audit_result": audit.__dict__},
        )

    # ------------------------------------------------------------------
    def _count_indicator_alignment(self, quant_state: Dict, direction: int) -> tuple:
        """Count indicators that support vs oppose the given direction."""
        if direction == 0:
            return (1, 0)

        support = 0
        oppose = 0

        # RSI
        rsi = self._safe_get(quant_state, 'trend', 'rsi_14', default=50.0)
        if direction > 0 and rsi < 70:
            support += 1
        elif direction < 0 and rsi > 30:
            support += 1
        else:
            oppose += 1

        # MACD histogram
        macd_hist = self._safe_get(quant_state, 'trend', 'macd_hist', default=0.0)
        if (direction > 0 and macd_hist > 0) or (direction < 0 and macd_hist < 0):
            support += 1
        else:
            oppose += 1

        # Kalman slope
        slope = self._safe_get(quant_state, 'kalman', 'slope', default=0.0)
        if (direction > 0 and slope > 0) or (direction < 0 and slope < 0):
            support += 1
        else:
            oppose += 1

        # OU z_score (mean reversion)
        z = self._safe_get(quant_state, 'ou_process', 'z_score', default=0.0)
        if (direction > 0 and z < -1) or (direction < 0 and z > 1):
            support += 1
        elif abs(z) > 1:
            oppose += 1

        # Sentiment
        sent = self._safe_get(quant_state, 'sentiment', 'score', default=0.0)
        if (direction > 0 and sent > 0) or (direction < 0 and sent < 0):
            support += 1
        elif abs(sent) > 0.1:
            oppose += 1

        # Hurst regime
        hurst_regime = str(self._safe_get(quant_state, 'hurst', 'regime', default='')).lower()
        if direction != 0 and hurst_regime == 'trending':
            support += 1
        elif hurst_regime == 'mean_reverting':
            oppose += 1

        return (support, oppose)

    def _check_agent_contradictions(self, agent_votes: Dict, quant_state: Dict,
                                     flags: List[str]) -> int:
        """Check for contradictions between agent votes."""
        contradictions = 0

        trend_vote = self._get_vote_direction(agent_votes, 'TrendMomentumAgent', 'trend_momentum')
        mr_vote = self._get_vote_direction(agent_votes, 'MeanReversionAgent', 'mean_reversion')

        if trend_vote != 0 and mr_vote != 0 and trend_vote != mr_vote:
            contradictions += 1
            hurst = self._safe_get(quant_state, 'hurst', 'hurst', default=0.5)
            if hurst > 0.55:
                flags.append(f"RESOLVED: Trend vs MeanReversion conflict → Hurst={hurst:.2f} favors Trend")
            else:
                flags.append(f"RESOLVED: Trend vs MeanReversion conflict → Hurst={hurst:.2f} favors MeanReversion")

        # Count general direction disagreements among all votes
        directions = {}
        for name, vote in agent_votes.items():
            d = self._get_vote_direction_raw(vote)
            if d != 0:
                directions[name] = d

        if directions:
            buy_count = sum(1 for d in directions.values() if d > 0)
            sell_count = sum(1 for d in directions.values() if d < 0)
            contradictions += min(buy_count, sell_count)

        return contradictions

    def _get_vote_metadata(self, vote) -> Dict:
        if isinstance(vote, dict):
            return vote.get('metadata', {})
        return getattr(vote, 'metadata', {}) if vote else {}

    def _get_vote_direction(self, votes: Dict, *keys) -> int:
        for k in keys:
            v = votes.get(k)
            if v is not None:
                return self._get_vote_direction_raw(v)
        return 0

    def _get_vote_direction_raw(self, vote) -> int:
        if isinstance(vote, dict):
            return vote.get('direction', 0)
        return getattr(vote, 'direction', 0) if vote else 0

    def _get_losing_streak(self, trade_history: List) -> int:
        """Count consecutive recent losses."""
        streak = 0
        for trade in reversed(trade_history):
            if not isinstance(trade, dict):
                break
            pnl = trade.get('pnl', trade.get('profit', 0.0))
            if pnl is not None and pnl < 0:
                streak += 1
            else:
                break
        return streak

    def _calc_win_rate(self, trade_history: List) -> float:
        if not trade_history:
            return 0.5
        wins = 0
        total = 0
        for trade in trade_history:
            if not isinstance(trade, dict):
                continue
            pnl = trade.get('pnl', trade.get('profit', None))
            if pnl is not None:
                total += 1
                if pnl > 0:
                    wins += 1
        return wins / total if total > 0 else 0.5
