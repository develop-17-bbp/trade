"""
Base Agent & Data Structures for Multi-Agent Trading System
============================================================
All agents inherit from BaseAgent which provides:
- Bayesian accuracy tracking with EMA updates
- State persistence (save/load weights)
- Standardized AgentVote output format
"""

import json
import os
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque


@dataclass
class AgentVote:
    """Structured output from each analysis agent."""
    direction: int = 0             # -1 (sell), 0 (flat), +1 (buy)
    confidence: float = 0.0        # 0.0 - 1.0
    position_scale: float = 1.0    # 0.0 - 1.0 sizing multiplier
    reasoning: str = ""            # Must cite [METRIC=VALUE]
    veto: bool = False             # Only LossPreventionGuardian uses this
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataIntegrityReport:
    """Output from DataIntegrityValidator (pre-gate)."""
    is_valid: bool = True
    quality_score: float = 1.0              # 0.0-1.0 (1.0 = perfect)
    sanitized_state: Dict = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    inconsistencies: List[Dict] = field(default_factory=list)
    confidence_adjustments: Dict[str, float] = field(default_factory=dict)
    recommendation: str = "PROCEED"         # PROCEED / PROCEED_WITH_CAUTION / HALT_BAD_DATA


@dataclass
class AuditResult:
    """Output from DecisionAuditor (post-gate)."""
    approved: bool = True
    adjusted_confidence: float = 0.0
    adjusted_position_scale: float = 1.0
    audit_flags: List[str] = field(default_factory=list)
    contradiction_count: int = 0
    data_alignment_score: float = 1.0       # 0-1: how well decision matches raw data
    historical_win_rate: float = 0.5
    recommendation: str = "EXECUTE"         # EXECUTE / REDUCE / DEFER / BLOCK


@dataclass
class EnhancedDecision:
    """Final output from the AgentOrchestrator after all 4 steps."""
    direction: int = 0
    confidence: float = 0.0
    position_scale: float = 0.0
    strategy_recommendation: str = ""
    risk_params: Dict = field(default_factory=dict)
    agent_votes: Dict[str, AgentVote] = field(default_factory=dict)
    consensus_level: str = "CONFLICT"       # STRONG/MODERATE/WEAK/CONFLICT/VETOED
    daily_pnl_mode: str = "NORMAL"
    data_quality: float = 1.0
    audit_result: Optional[AuditResult] = None
    veto: bool = False


class BaseAgent(ABC):
    """Abstract base for all 12 specialized agents."""

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}

        # Bayesian accuracy tracking
        self._accuracy_history: deque = deque(maxlen=200)
        self._current_weight: float = 1.0
        self._total_calls: int = 0
        self._correct_calls: int = 0
        self._alpha: float = self.config.get('weight_update_alpha', 0.15)

    @abstractmethod
    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        """
        Each agent implements its specialized analysis.

        Args:
            quant_state: Sanitized output from MathInjector.compute_full_state()
                         Contains ALL 18 quant model outputs.
            context: Additional context dict with:
                - raw_signal (int): existing L1+L2+L3 combined signal
                - raw_confidence (float): existing pipeline confidence
                - ext_feats (dict): external features (funding, OI, etc.)
                - on_chain (dict): whale, liquidation, exchange flow
                - sentiment_data (dict): FinBERT aggregate
                - ohlcv_data (dict): prices, highs, lows, volumes
                - asset (str): trading asset name
                - daily_pnl (float): current day PnL percentage
                - account_balance (float): current balance
                - open_positions (list): currently open positions
                - trade_history (list): recent trade results

        Returns:
            AgentVote with direction, confidence, position_scale, reasoning
        """
        pass

    def update_accuracy(self, predicted_dir: int, was_profitable: bool):
        """
        Bayesian EMA update of agent accuracy after trade resolution.
        Called by AgentOrchestrator.post_trade_feedback().
        """
        self._total_calls += 1
        correct = (predicted_dir > 0 and was_profitable) or \
                  (predicted_dir < 0 and not was_profitable) or \
                  (predicted_dir == 0)

        if correct:
            self._correct_calls += 1

        self._accuracy_history.append(1.0 if correct else 0.0)

        # Bayesian weight update: w(t+1) = w(t) * (alpha * accuracy + (1-alpha))
        accuracy = self.get_accuracy()
        self._current_weight *= (self._alpha * accuracy + (1 - self._alpha))

        # Clamp to [0.3, 3.0] to prevent any single agent from dominating
        self._current_weight = max(0.3, min(3.0, self._current_weight))

    def get_accuracy(self) -> float:
        """Return rolling accuracy percentage (0-1)."""
        if not self._accuracy_history:
            return 0.5  # Prior: assume 50% accuracy
        return sum(self._accuracy_history) / len(self._accuracy_history)

    def get_weight(self) -> float:
        """Return current dynamic Bayesian weight."""
        return self._current_weight

    def save_state(self, path: str = ""):
        """Persist agent state to JSON."""
        if not path:
            path = f"/c/Users/convo/trade/memory/agent_{self.name}_state.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'name': self.name,
            'weight': self._current_weight,
            'total_calls': self._total_calls,
            'correct_calls': self._correct_calls,
            'accuracy_history': list(self._accuracy_history),
        }
        try:
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def load_state(self, path: str = ""):
        """Load persisted agent state."""
        if not path:
            path = f"/c/Users/convo/trade/memory/agent_{self.name}_state.json"
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            self._current_weight = state.get('weight', 1.0)
            self._total_calls = state.get('total_calls', 0)
            self._correct_calls = state.get('correct_calls', 0)
            hist = state.get('accuracy_history', [])
            self._accuracy_history = deque(hist, maxlen=200)
        except Exception:
            pass

    def _safe_get(self, d: Dict, *keys, default=0.0):
        """Safely traverse nested dicts."""
        current = d
        for k in keys:
            if isinstance(current, dict):
                current = current.get(k, default)
            else:
                return default
        return current if current is not None else default
