"""
Agent Orchestrator — 4-Step Pipeline for Multi-Agent Trading Intelligence
=========================================================================
Step 1: DataIntegrityValidator (pre-gate) — sanitize & validate quant data
Step 2: 10 analysis agents in parallel — each produces AgentVote
Step 3: AgentCombiner — Bayesian weighted consensus
Step 4: DecisionAuditor (post-gate) — cross-check for contradictions
"""

import json
import os
import time
import logging
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.base_agent import (
    BaseAgent, AgentVote, DataIntegrityReport, AuditResult, EnhancedDecision
)
from src.agents.combiner import AgentCombiner
from src.agents.debate_engine import DebateEngine

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Central coordinator that runs all 12 agents in the correct order
    and produces an EnhancedDecision for the MetaController.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        agent_cfg = self.config.get('agents', {})
        self.enabled = agent_cfg.get('enabled', True)
        self.blend_weight = agent_cfg.get('blend_weight', 0.60)

        # Fix #9: Store last votes per asset for outcome-based weight updates
        self._last_votes: Dict[str, Dict[str, AgentVote]] = {}

        # Initialize combiner and debate engine
        self.combiner = AgentCombiner(agent_cfg)
        self.debate_engine = DebateEngine(self.config)

        # Initialize all 12 agents
        self.agents: Dict[str, BaseAgent] = {}
        self._init_agents(agent_cfg)

        # Load persisted weights
        self._load_all_states()

    def _init_agents(self, cfg: Dict):
        """Instantiate all 12 specialized agents with graceful degradation.

        If any individual agent fails to initialize, a warning is logged and
        the system continues with the remaining agents (degraded-but-alive).
        """
        agent_specs = []

        try:
            from src.agents.data_integrity_validator import DataIntegrityValidator
            agent_specs.append(('data_integrity', DataIntegrityValidator))
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] data_integrity agent import failed: {e}")

        try:
            from src.agents.market_structure_agent import MarketStructureAgent
            agent_specs.append(('market_structure', MarketStructureAgent))
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] market_structure agent import failed: {e}")

        try:
            from src.agents.regime_intelligence_agent import RegimeIntelligenceAgent
            agent_specs.append(('regime_intelligence', RegimeIntelligenceAgent))
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] regime_intelligence agent import failed: {e}")

        try:
            from src.agents.mean_reversion_agent import MeanReversionAgent
            agent_specs.append(('mean_reversion', MeanReversionAgent))
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] mean_reversion agent import failed: {e}")

        try:
            from src.agents.trend_momentum_agent import TrendMomentumAgent
            agent_specs.append(('trend_momentum', TrendMomentumAgent))
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] trend_momentum agent import failed: {e}")

        try:
            from src.agents.risk_guardian_agent import RiskGuardianAgent
            agent_specs.append(('risk_guardian', RiskGuardianAgent))
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] risk_guardian agent import failed: {e}")

        try:
            from src.agents.sentiment_decoder_agent import SentimentDecoderAgent
            agent_specs.append(('sentiment_decoder', SentimentDecoderAgent))
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] sentiment_decoder agent import failed: {e}")

        try:
            from src.agents.trade_timing_agent import TradeTimingAgent
            agent_specs.append(('trade_timing', TradeTimingAgent))
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] trade_timing agent import failed: {e}")

        try:
            from src.agents.portfolio_optimizer_agent import PortfolioOptimizerAgent
            agent_specs.append(('portfolio_optimizer', PortfolioOptimizerAgent))
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] portfolio_optimizer agent import failed: {e}")

        try:
            from src.agents.pattern_matcher_agent import PatternMatcherAgent
            agent_specs.append(('pattern_matcher', PatternMatcherAgent))
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] pattern_matcher agent import failed: {e}")

        try:
            from src.agents.loss_prevention_guardian import LossPreventionGuardian
            agent_specs.append(('loss_prevention', LossPreventionGuardian))
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] loss_prevention agent import failed: {e}")

        try:
            from src.agents.polymarket_agent import PolymarketArbitrageAgent
            agent_specs.append(('polymarket_arb', PolymarketArbitrageAgent))
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] polymarket_arb agent import failed: {e}")

        try:
            from src.agents.decision_auditor import DecisionAuditor
            agent_specs.append(('decision_auditor', DecisionAuditor))
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] decision_auditor agent import failed: {e}")

        self.agents = {}
        for agent_name, agent_cls in agent_specs:
            try:
                self.agents[agent_name] = agent_cls(agent_name, cfg)
                logger.debug(f"[ORCHESTRATOR] Agent '{agent_name}' initialized")
            except Exception as e:
                logger.warning(
                    f"[ORCHESTRATOR] Agent '{agent_name}' failed to initialize (degraded): {e}"
                )

        initialized = len(self.agents)
        total = len(agent_specs)
        if initialized < total:
            logger.warning(
                f"[ORCHESTRATOR] {initialized}/{total} agents initialized — "
                f"system running in degraded mode"
            )
        else:
            logger.info(f"[ORCHESTRATOR] All {initialized} agents initialized")

    def run_cycle(self, quant_state: Dict, raw_signal: int = 0,
                  raw_confidence: float = 0.0,
                  ext_feats: Optional[Dict] = None,
                  on_chain: Optional[Dict] = None,
                  sentiment_data: Optional[Dict] = None,
                  ohlcv_data: Optional[Dict] = None,
                  asset: str = 'BTC',
                  daily_pnl: float = 0.0,
                  account_balance: float = 100_000.0,
                  open_positions: Optional[List] = None,
                  trade_history: Optional[List] = None) -> EnhancedDecision:
        """
        Execute the full 4-step agent pipeline.

        Returns:
            EnhancedDecision with direction, confidence, position_scale, etc.
        """
        if not self.enabled:
            return EnhancedDecision(
                direction=raw_signal,
                confidence=raw_confidence,
                position_scale=1.0 if raw_signal != 0 else 0.0,
                consensus_level='DISABLED',
            )

        start_time = time.time()

        context = {
            'raw_signal': raw_signal,
            'raw_confidence': raw_confidence,
            'ext_feats': ext_feats or {},
            'on_chain': on_chain or {},
            'sentiment_data': sentiment_data or {},
            'ohlcv_data': ohlcv_data or {},
            'asset': asset,
            'daily_pnl': daily_pnl,
            'account_balance': account_balance,
            'open_positions': open_positions or [],
            'trade_history': trade_history or [],
        }

        # ── STEP 1: Data Integrity Validation (pre-gate) ──
        validator = self.agents['data_integrity']
        integrity_vote = validator.analyze(quant_state, context)
        integrity_report = integrity_vote.metadata.get('report', DataIntegrityReport())

        if isinstance(integrity_report, dict):
            quality_score = integrity_report.get('quality_score', 1.0)
            recommendation = integrity_report.get('recommendation', 'PROCEED')
            sanitized_state = integrity_report.get('sanitized_state', quant_state)
            confidence_adjustments = integrity_report.get('confidence_adjustments', {})
        else:
            quality_score = integrity_report.quality_score
            recommendation = integrity_report.recommendation
            sanitized_state = integrity_report.sanitized_state
            confidence_adjustments = integrity_report.confidence_adjustments

        # Halt on bad data
        if recommendation == 'HALT_BAD_DATA' or quality_score < 0.3:
            logger.warning(f"[AgentOrchestrator] HALT: bad data quality={quality_score:.2f}")
            return EnhancedDecision(
                direction=0, confidence=0.0, position_scale=0.0,
                consensus_level='VETOED', data_quality=quality_score,
                risk_params={'reason': f'Bad data quality: {quality_score:.2f}'},
                veto=True,
            )

        context['confidence_adjustments'] = confidence_adjustments
        context['data_quality'] = quality_score

        # ── STEP 2: Run 10 analysis agents in parallel ──
        analysis_agents = [
            'market_structure', 'regime_intelligence', 'mean_reversion',
            'trend_momentum', 'risk_guardian', 'sentiment_decoder',
            'trade_timing', 'portfolio_optimizer', 'pattern_matcher',
            'loss_prevention', 'polymarket_arb',
        ]

        votes: Dict[str, AgentVote] = {}

        with ThreadPoolExecutor(max_workers=11) as executor:
            futures = {}
            for name in analysis_agents:
                agent = self.agents.get(name)
                if agent:
                    futures[executor.submit(self._safe_analyze, agent, sanitized_state, context)] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    vote = future.result(timeout=5.0)
                    # Apply data quality confidence adjustment
                    if recommendation == 'PROCEED_WITH_CAUTION':
                        vote.confidence *= quality_score
                    # Apply per-model confidence adjustments
                    adj = confidence_adjustments.get(name, 1.0)
                    vote.confidence *= adj
                    votes[name] = vote
                except Exception as e:
                    logger.warning(f"[AgentOrchestrator] Agent {name} failed: {e}")
                    votes[name] = AgentVote(direction=0, confidence=0.0, reasoning=f"Error: {e}")

        # Fix #9: Save votes for outcome-based weight update
        self._last_votes[asset] = dict(votes)

        # ── STEP 2.5: ADVERSARIAL DEBATE ──
        # Agents challenge each other's positions with specific metrics.
        # Agents that survive cross-examination get conviction bonuses;
        # agents that fail may flip direction or lose confidence.
        debate_result = None
        try:
            debate_result = self.debate_engine.run_debate(
                votes=votes,
                quant_state=sanitized_state,
                context=context,
            )
            # Use post-debate votes (with conviction adjustments) for combining
            debate_votes = debate_result.post_debate_votes
            logger.info(
                f"[DEBATE] {asset}: {debate_result.debate_summary} "
                f"| flipped={len(debate_result.flipped_agents)} "
                f"| strengthened={len(debate_result.strengthened_agents)}"
            )
        except Exception as e:
            logger.warning(f"[DEBATE] Debate failed, using raw votes: {e}")
            debate_votes = votes

        # ── STEP 3: Combine votes via Bayesian weighted consensus ──
        regime = sanitized_state.get('hmm_regime', {}).get('regime', 'sideways') \
            if isinstance(sanitized_state.get('hmm_regime'), dict) \
            else sanitized_state.get('hmm_regime', 'sideways')

        loss_vote = debate_votes.get('loss_prevention')
        enhanced = self.combiner.combine(
            votes=debate_votes,
            agents=self.agents,
            regime=regime,
            loss_guardian_vote=loss_vote,
        )
        enhanced.data_quality = quality_score

        # Attach debate metadata for dashboard visibility
        if debate_result:
            enhanced.risk_params['debate_summary'] = debate_result.debate_summary
            enhanced.risk_params['debate_flipped'] = debate_result.flipped_agents
            enhanced.risk_params['debate_strengthened'] = debate_result.strengthened_agents
            enhanced.risk_params['debate_consensus_shift'] = debate_result.consensus_shift
            enhanced.risk_params['debate_conviction'] = debate_result.conviction_multipliers

        # ── STEP 4: Decision Auditor (post-gate) ──
        auditor = self.agents['decision_auditor']
        audit_context = {
            **context,
            'enhanced_decision': enhanced,
            'agent_votes': votes,
            'integrity_report': integrity_report,
        }
        audit_vote = auditor.analyze(sanitized_state, audit_context)
        audit_result = audit_vote.metadata.get('audit_result', AuditResult())

        if isinstance(audit_result, dict):
            audit_rec = audit_result.get('recommendation', 'EXECUTE')
            adj_conf = audit_result.get('adjusted_confidence', enhanced.confidence)
            adj_scale = audit_result.get('adjusted_position_scale', enhanced.position_scale)
        else:
            audit_rec = audit_result.recommendation
            adj_conf = audit_result.adjusted_confidence
            adj_scale = audit_result.adjusted_position_scale

        # Apply audit adjustments (auditor can only DOWNGRADE, never upgrade)
        enhanced.confidence = min(enhanced.confidence, adj_conf)
        enhanced.position_scale = min(enhanced.position_scale, adj_scale)
        enhanced.audit_result = audit_result

        if audit_rec == 'BLOCK':
            enhanced.direction = 0
            enhanced.position_scale = 0.0
            enhanced.consensus_level = 'VETOED'
            enhanced.veto = True
        elif audit_rec == 'DEFER':
            enhanced.direction = 0
            enhanced.position_scale = 0.0
            enhanced.consensus_level = 'CONFLICT'

        elapsed = time.time() - start_time
        logger.info(
            f"[AgentOrchestrator] Cycle complete in {elapsed:.2f}s: "
            f"dir={enhanced.direction} conf={enhanced.confidence:.3f} "
            f"scale={enhanced.position_scale:.3f} consensus={enhanced.consensus_level} "
            f"data_quality={quality_score:.2f} audit={audit_rec}"
        )

        return enhanced

    def _safe_analyze(self, agent: BaseAgent, quant_state: Dict, context: Dict) -> AgentVote:
        """Run an agent with exception safety."""
        try:
            return agent.analyze(quant_state, context)
        except Exception as e:
            logger.warning(f"[{agent.name}] Analysis failed: {e}")
            return AgentVote(direction=0, confidence=0.0, reasoning=f"Error: {e}")

    def post_trade_feedback(self, trade_result: Dict):
        """
        Update all agent accuracies after a trade resolves.
        Called by TradingExecutor after trade closes.
        """
        was_profitable = trade_result.get('pnl', 0) > 0
        agent_votes = trade_result.get('agent_votes', {})

        for name, agent in self.agents.items():
            if name in ('data_integrity', 'decision_auditor'):
                continue  # Gate agents tracked differently
            vote = agent_votes.get(name)
            if vote:
                predicted_dir = vote.direction if isinstance(vote, AgentVote) else vote.get('direction', 0)
                agent.update_accuracy(predicted_dir, was_profitable)

        # Save all updated states
        self._save_all_states()

        logger.info(
            f"[AgentOrchestrator] Feedback: profitable={was_profitable}, "
            f"updated {len(agent_votes)} agent weights"
        )

    def record_outcome(self, asset: str, direction: int, pnl: float):
        """
        Fix #9: Call after a trade closes. Updates agent weights based on whether
        each agent's vote aligned with the profitable direction.
        """
        if asset not in self._last_votes:
            return

        was_profitable = pnl > 0
        alpha = self.config.get('agents', {}).get('weight_update_alpha', 0.15)

        for agent_name, vote in self._last_votes[asset].items():
            agent = self.agents.get(agent_name)
            if agent is None:
                continue
            # Did this agent agree with the actual direction?
            agent_agreed = (vote.direction == direction)
            # Was the trade profitable?
            correct = agent_agreed == was_profitable

            # Update weight: increase if correct, decrease if wrong
            current_weight = getattr(agent, '_current_weight', 1.0)
            if correct:
                new_weight = current_weight + alpha * (1.0 - current_weight)
            else:
                new_weight = current_weight - alpha * current_weight
            new_weight = max(0.1, min(2.0, new_weight))  # clamp [0.1, 2.0]
            agent._current_weight = new_weight
            agent._total_calls = getattr(agent, '_total_calls', 0) + 1

        # Persist updated weights
        self._save_all_states()
        logger.info(f"[AgentOrchestrator] Outcome recorded for {asset}: pnl={pnl:.2f}, updated weights")

    def _save_all_states(self):
        """Persist all agent weights to disk."""
        states = {}
        for name, agent in self.agents.items():
            states[name] = {
                'weight': agent.get_weight(),
                'accuracy': agent.get_accuracy(),
                'total_calls': agent._total_calls,
            }
        # Use os.path for cross-platform compatibility (Windows + Unix)
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "memory", "agent_weights.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, 'w') as f:
                json.dump(states, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save agent weights: {e}")

    def _load_all_states(self):
        """Load persisted agent weights."""
        # Use os.path for cross-platform compatibility (Windows + Unix)
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "memory", "agent_weights.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r') as f:
                states = json.load(f)
            for name, state in states.items():
                agent = self.agents.get(name)
                if agent:
                    agent._current_weight = state.get('weight', 1.0)
                    agent._total_calls = state.get('total_calls', 0)
            logger.info(f"[AgentOrchestrator] Loaded weights for {len(states)} agents")
        except Exception as e:
            logger.warning(f"Failed to load agent weights: {e}")

    def get_agent_stats(self) -> Dict:
        """Return all agent performance stats for dashboard."""
        stats = {}
        for name, agent in self.agents.items():
            stats[name] = {
                'weight': round(agent.get_weight(), 4),
                'accuracy': round(agent.get_accuracy(), 4),
                'total_calls': agent._total_calls,
            }
        return stats
