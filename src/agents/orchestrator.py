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

        # Initialize combiner
        self.combiner = AgentCombiner(agent_cfg)

        # Initialize all 12 agents
        self.agents: Dict[str, BaseAgent] = {}
        self._init_agents(agent_cfg)

        # Load persisted weights
        self._load_all_states()

    def _init_agents(self, cfg: Dict):
        """Instantiate all 12 specialized agents."""
        from src.agents.data_integrity_validator import DataIntegrityValidator
        from src.agents.market_structure_agent import MarketStructureAgent
        from src.agents.regime_intelligence_agent import RegimeIntelligenceAgent
        from src.agents.mean_reversion_agent import MeanReversionAgent
        from src.agents.trend_momentum_agent import TrendMomentumAgent
        from src.agents.risk_guardian_agent import RiskGuardianAgent
        from src.agents.sentiment_decoder_agent import SentimentDecoderAgent
        from src.agents.trade_timing_agent import TradeTimingAgent
        from src.agents.portfolio_optimizer_agent import PortfolioOptimizerAgent
        from src.agents.pattern_matcher_agent import PatternMatcherAgent
        from src.agents.loss_prevention_guardian import LossPreventionGuardian
        from src.agents.decision_auditor import DecisionAuditor
        from src.agents.polymarket_agent import PolymarketArbitrageAgent

        self.agents = {
            # Pre-gate
            'data_integrity': DataIntegrityValidator('data_integrity', cfg),
            # 11 analysis agents
            'market_structure': MarketStructureAgent('market_structure', cfg),
            'regime_intelligence': RegimeIntelligenceAgent('regime_intelligence', cfg),
            'mean_reversion': MeanReversionAgent('mean_reversion', cfg),
            'trend_momentum': TrendMomentumAgent('trend_momentum', cfg),
            'risk_guardian': RiskGuardianAgent('risk_guardian', cfg),
            'sentiment_decoder': SentimentDecoderAgent('sentiment_decoder', cfg),
            'trade_timing': TradeTimingAgent('trade_timing', cfg),
            'portfolio_optimizer': PortfolioOptimizerAgent('portfolio_optimizer', cfg),
            'pattern_matcher': PatternMatcherAgent('pattern_matcher', cfg),
            'loss_prevention': LossPreventionGuardian('loss_prevention', cfg),
            'polymarket_arb': PolymarketArbitrageAgent('polymarket_arb', cfg),
            # Post-gate
            'decision_auditor': DecisionAuditor('decision_auditor', cfg),
        }

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

        # ── STEP 3: Combine votes via Bayesian weighted consensus ──
        regime = sanitized_state.get('hmm_regime', {}).get('regime', 'sideways') \
            if isinstance(sanitized_state.get('hmm_regime'), dict) \
            else sanitized_state.get('hmm_regime', 'sideways')

        loss_vote = votes.get('loss_prevention')
        enhanced = self.combiner.combine(
            votes=votes,
            agents=self.agents,
            regime=regime,
            loss_guardian_vote=loss_vote,
        )
        enhanced.data_quality = quality_score

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

    def _save_all_states(self):
        """Persist all agent weights to disk."""
        states = {}
        for name, agent in self.agents.items():
            states[name] = {
                'weight': agent.get_weight(),
                'accuracy': agent.get_accuracy(),
                'total_calls': agent._total_calls,
            }
        path = "/c/Users/convo/trade/memory/agent_weights.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, 'w') as f:
                json.dump(states, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save agent weights: {e}")

    def _load_all_states(self):
        """Load persisted agent weights."""
        path = "/c/Users/convo/trade/memory/agent_weights.json"
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
