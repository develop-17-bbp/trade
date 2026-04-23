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

        # Per-agent episodic memory (C12b) — each agent keeps its own
        # recent (state_digest, vote, outcome) tuples so it can answer
        # "when I saw this kind of setup before, how did I vote and was
        # I right?" Populated by record_episode() after trade close,
        # queried by get_similar_episodes() before vote.
        self._episode_buffer: deque = deque(
            maxlen=int(self.config.get('episode_buffer_size', 100))
        )

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
            # C12b — persist episodic memory so restarts don't lose it.
            'episode_buffer': list(self._episode_buffer),
        }
        try:
            with open(path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
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
            # C12b — restore episodic memory if present (backward-compat
            # with old state files that don't carry this key).
            eps = state.get('episode_buffer', []) or []
            self._episode_buffer = deque(
                eps, maxlen=int(self.config.get('episode_buffer_size', 100)),
            )
        except Exception:
            pass

    # ── Per-agent episodic memory (C12b) ───────────────────────────────

    # Quant-state keys worth keeping in the episode "fingerprint" — tight,
    # comparable across time. Subclasses can extend via extra_keys arg.
    _EPISODE_DEFAULT_KEYS = (
        "ema_slope_pct", "hurst", "bollinger_pct", "zscore",
        "rsi", "macd_hist", "atr_pct", "regime", "vol_regime",
        "funding_rate", "fear_greed_index",
    )

    def _episode_fingerprint(
        self, quant_state: Dict, extra_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract a compact snapshot of the current quant_state. Only
        numeric/string scalars — no arrays — so comparison is cheap and
        persistence stays small."""
        keys = list(self._EPISODE_DEFAULT_KEYS) + list(extra_keys or [])
        snap: Dict[str, Any] = {}
        for k in keys:
            v = quant_state.get(k)
            if isinstance(v, (int, float, str, bool)) or v is None:
                snap[k] = v
        return snap

    def record_episode(
        self,
        quant_state: Dict,
        vote: "AgentVote",
        outcome: Optional[Dict] = None,
    ) -> None:
        """Append one episode to this agent's personal buffer.

        Called from orchestrator.post_trade_feedback (or agentic_trade_loop
        close callback). `outcome` carries pnl_pct + was_profitable once
        the trade closes; pass None at open-time to record just the vote.
        """
        try:
            self._episode_buffer.append({
                "ts": __import__("time").time(),
                "state": self._episode_fingerprint(quant_state),
                "vote": {
                    "direction": int(getattr(vote, "direction", 0)),
                    "confidence": float(getattr(vote, "confidence", 0.0)),
                },
                "outcome": (dict(outcome) if isinstance(outcome, dict) else None),
            })
        except Exception:
            # Never let memory bookkeeping break the vote path.
            pass

    def get_similar_episodes(
        self,
        current_state: Dict,
        k: int = 5,
        min_numeric_keys: int = 2,
    ) -> List[Dict[str, Any]]:
        """Return the top-k past episodes whose fingerprint is closest
        to `current_state`. Similarity = inverse L1 distance over shared
        numeric keys, ignoring missing/non-numeric fields. If the agent
        has < min_numeric_keys overlap, returns an empty list rather
        than a noisy match.
        """
        if not self._episode_buffer:
            return []
        current = self._episode_fingerprint(current_state)
        current_nums = {k: v for k, v in current.items()
                        if isinstance(v, (int, float)) and v is not None}
        if len(current_nums) < min_numeric_keys:
            return []

        scored: List[tuple] = []
        for ep in self._episode_buffer:
            st = ep.get("state") or {}
            shared = [
                kk for kk in current_nums.keys()
                if isinstance(st.get(kk), (int, float))
            ]
            if len(shared) < min_numeric_keys:
                continue
            dist = sum(abs(float(current_nums[kk]) - float(st[kk])) for kk in shared)
            scored.append((dist, ep))
        scored.sort(key=lambda x: x[0])
        return [ep for _, ep in scored[: max(1, int(k))]]

    def episodic_memory_size(self) -> int:
        """For dashboards / /regime-check. No I/O."""
        return len(self._episode_buffer)

    def _safe_get(self, d: Dict, *keys, default=0.0):
        """Safely traverse nested dicts."""
        current = d
        for k in keys:
            if isinstance(current, dict):
                current = current.get(k, default)
            else:
                return default
        return current if current is not None else default
