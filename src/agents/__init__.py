"""
Multi-Agent Intelligence Overlay for Autonomous Trading
========================================================
12 specialized agents that enhance the existing L1/L2/L3 signal pipeline.

Architecture:
  Step 1: DataIntegrityValidator (pre-gate)
  Step 2: 10 analysis agents in parallel
  Step 3: AgentCombiner (Bayesian weighted consensus)
  Step 4: DecisionAuditor (post-gate)
"""

from src.agents.base_agent import BaseAgent, AgentVote, DataIntegrityReport, AuditResult, EnhancedDecision
from src.agents.orchestrator import AgentOrchestrator
from src.agents.combiner import AgentCombiner

__all__ = [
    'BaseAgent', 'AgentVote', 'DataIntegrityReport', 'AuditResult',
    'EnhancedDecision', 'AgentOrchestrator', 'AgentCombiner',
]
