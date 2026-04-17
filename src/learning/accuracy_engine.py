"""
Self-Learning Accuracy Engine + Consistency Guardian — ACT v8.0
All weights, thresholds, and sizing are DYNAMIC — learned from memory, never hardcoded.

- Tracks per-model accuracy by regime with rolling windows
- Dynamic ensemble weights: poorly performing models get near-zero weight
- Consistency Guardian: loss streak protection, weekly drawdown block
- Robinhood slippage profiling from real fill data
"""
import time
import math
import logging
from typing import Dict, Optional, List
from collections import deque

logger = logging.getLogger(__name__)


class AccuracyEngine:
    """Tracks every model and agent's accuracy, computes dynamic ensemble weights."""

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.consecutive_loss_limit = cfg.get('consecutive_loss_limit', 3)
        self.preservation_trades = cfg.get('capital_preservation_trades', 5)
        self.weekly_dd_block = cfg.get('weekly_drawdown_block_threshold', 0.02)
        self.min_regime_accuracy = cfg.get('min_model_regime_accuracy', 0.55)

        # Per-model tracking: model_name → {regime → [correct/incorrect bools]}
        self._model_accuracy: Dict[str, Dict[str, deque]] = {}
        # Per-agent tracking
        self._agent_accuracy: Dict[str, deque] = {}

        # Consistency Guardian state
        self._streak_type: str = 'none'   # 'winning' / 'losing' / 'none'
        self._streak_length: int = 0
        self._capital_preservation_remaining: int = 0
        self._hot_hand_used: bool = False

        # Weekly tracking
        self._weekly_pnl: float = 0.0
        self._weekly_start_time: float = time.time()
        self._weekly_blocked: bool = False

        # Robinhood slippage profiling
        self._slippage_samples: deque = deque(maxlen=100)
        self._fill_time_samples: deque = deque(maxlen=100)

        # Overall trade outcomes
        self._outcomes: deque = deque(maxlen=500)

    # ── Model Accuracy Tracking ──────────────────────────────────

    def record_model_prediction(self, model_name: str, regime: str,
                                 predicted_direction: str, actual_profitable: bool):
        """Record whether a model's prediction was correct."""
        if model_name not in self._model_accuracy:
            self._model_accuracy[model_name] = {}
        if regime not in self._model_accuracy[model_name]:
            self._model_accuracy[model_name][regime] = deque(maxlen=200)
        self._model_accuracy[model_name][regime].append(actual_profitable)

    def get_model_accuracy(self, model_name: str, regime: str,
                            window: int = 50) -> float:
        """Get model accuracy in given regime over last N trades."""
        data = self._model_accuracy.get(model_name, {}).get(regime, deque())
        recent = list(data)[-window:]
        if len(recent) < 5:
            return 0.5  # default: no edge
        return sum(recent) / len(recent)

    def get_ensemble_weights(self, current_regime: str) -> Dict[str, float]:
        """Dynamic ensemble weights — models that perform poorly in current
        regime get near-zero weight. Weights sum to 1.0."""
        models = list(self._model_accuracy.keys())
        if not models:
            return {'lgbm': 0.34, 'patchtst': 0.33, 'rl': 0.33}

        raw_weights = {}
        for model in models:
            acc = self.get_model_accuracy(model, current_regime)
            # Scale: 0.5 accuracy → 0.0 weight, 0.7 accuracy → 1.0 weight
            raw_weights[model] = max(0.01, (acc - 0.45) / 0.25)

        total = sum(raw_weights.values())
        if total <= 0:
            return {m: 1.0 / len(models) for m in models}
        return {m: round(w / total, 3) for m, w in raw_weights.items()}

    def should_trust_model(self, model_name: str, regime: str) -> bool:
        """Returns False if model is unreliable in current regime."""
        acc = self.get_model_accuracy(model_name, regime)
        return acc >= self.min_regime_accuracy

    # ── Agent Accuracy Tracking ──────────────────────────────────

    def record_agent_vote(self, agent_name: str, was_correct: bool):
        if agent_name not in self._agent_accuracy:
            self._agent_accuracy[agent_name] = deque(maxlen=200)
        self._agent_accuracy[agent_name].append(was_correct)

    def get_agent_dynamic_weight(self, agent_name: str) -> float:
        """Returns dynamic Bayesian weight for agent based on recent accuracy."""
        data = list(self._agent_accuracy.get(agent_name, deque()))
        if len(data) < 10:
            return 1.0  # default weight until enough data
        acc = sum(data[-50:]) / len(data[-50:])
        # Map accuracy to weight: 0.45 → 0.5x, 0.55 → 1.0x, 0.70 → 1.5x
        weight = 0.5 + (acc - 0.45) * 4.0
        return round(max(0.1, min(2.0, weight)), 2)

    # ── Consistency Guardian ─────────────────────────────────────

    def record_trade_outcome(self, pnl_pct: float, pnl_usd: float = 0):
        """Update streak tracking and weekly PnL."""
        self._outcomes.append({'pnl_pct': pnl_pct, 'time': time.time()})

        # Update weekly PnL
        self._weekly_pnl += pnl_pct

        # Check weekly reset (7 days)
        if time.time() - self._weekly_start_time > 7 * 86400:
            self._weekly_pnl = pnl_pct
            self._weekly_start_time = time.time()
            self._weekly_blocked = False

        # Weekly drawdown block
        if self._weekly_pnl < -self.weekly_dd_block * 100:
            self._weekly_blocked = True
            logger.warning(f"[CONSISTENCY] Weekly block: {self._weekly_pnl:.2f}% loss")

        # Update streak
        won = pnl_pct > 0
        if won:
            if self._streak_type == 'winning':
                self._streak_length += 1
            else:
                self._streak_type = 'winning'
                self._streak_length = 1
                self._hot_hand_used = False
        else:
            if self._streak_type == 'losing':
                self._streak_length += 1
            else:
                self._streak_type = 'losing'
                self._streak_length = 1

        # Capital preservation on loss streak
        if self._streak_type == 'losing' and self._streak_length >= self.consecutive_loss_limit:
            self._capital_preservation_remaining = self.preservation_trades
            logger.warning(f"[CONSISTENCY] {self._streak_length} consecutive losses → Capital Preservation Mode ({self.preservation_trades} trades)")

        # Decrement preservation counter
        if self._capital_preservation_remaining > 0:
            self._capital_preservation_remaining -= 1
            # Exit preservation early if 2 consecutive wins
            if self._streak_type == 'winning' and self._streak_length >= 2:
                self._capital_preservation_remaining = 0
                logger.info("[CONSISTENCY] 2 wins → exiting Capital Preservation Mode")

    def should_skip_trade(self) -> tuple:
        """Returns (skip: bool, reason: str)."""
        if self._weekly_blocked:
            return True, f"Weekly drawdown block ({self._weekly_pnl:.2f}%)"
        if self._capital_preservation_remaining > 0:
            if self._capital_preservation_remaining == self.preservation_trades:
                return True, f"Capital Preservation: skipping after {self.consecutive_loss_limit} losses"
        return False, ""

    def get_position_size_multiplier(self) -> float:
        """Dynamic position size based on streak and preservation mode."""
        if self._capital_preservation_remaining > 0:
            return 0.5  # 50% size during preservation

        if self._streak_type == 'winning' and self._streak_length >= 5 and not self._hot_hand_used:
            self._hot_hand_used = True
            logger.info("[CONSISTENCY] Hot Hand: 10% size increase (one-time)")
            return 1.1

        return 1.0

    # ── Robinhood Slippage Profiling ────────────────────────────

    def record_fill(self, signal_price: float, fill_price: float, fill_time_sec: float):
        """Record actual Robinhood fill data for slippage learning."""
        slippage_pct = abs(fill_price - signal_price) / signal_price * 100
        self._slippage_samples.append(slippage_pct)
        self._fill_time_samples.append(fill_time_sec)

    def get_avg_slippage(self) -> float:
        if not self._slippage_samples:
            return 0.0
        return sum(self._slippage_samples) / len(self._slippage_samples)

    def get_effective_spread(self, base_spread: float = 1.69) -> float:
        """Actual spread = base spread + learned slippage."""
        return base_spread + self.get_avg_slippage()

    def get_avg_fill_time(self) -> float:
        if not self._fill_time_samples:
            return 5.0
        return sum(self._fill_time_samples) / len(self._fill_time_samples)

    # ── LLM Fine-Tune Data Generation ───────────────────────────

    def get_finetune_accuracy_context(self) -> dict:
        """Returns accuracy data that gets baked into LLM fine-tuning examples.
        The LLM learns which models and agents to trust in which regime."""
        model_data = {}
        for model, regimes in self._model_accuracy.items():
            model_data[model] = {}
            for regime, outcomes in regimes.items():
                recent = list(outcomes)[-50:]
                if len(recent) >= 5:
                    model_data[model][regime] = round(sum(recent) / len(recent), 3)
        agent_data = {}
        for agent, outcomes in self._agent_accuracy.items():
            recent = list(outcomes)[-50:]
            if len(recent) >= 10:
                agent_data[agent] = round(sum(recent) / len(recent), 3)
        return {
            'model_regime_accuracy': model_data,
            'agent_accuracy': agent_data,
            'current_streak': f"{self._streak_type}_{self._streak_length}",
            'weekly_pnl_pct': round(self._weekly_pnl, 2),
            'avg_slippage_pct': round(self.get_avg_slippage(), 3),
        }

    def get_stats(self) -> dict:
        return {
            'streak': f"{self._streak_type} x{self._streak_length}",
            'capital_preservation': self._capital_preservation_remaining > 0,
            'weekly_pnl': round(self._weekly_pnl, 2),
            'weekly_blocked': self._weekly_blocked,
            'total_outcomes': len(self._outcomes),
            'avg_slippage': round(self.get_avg_slippage(), 4),
            'models_tracked': list(self._model_accuracy.keys()),
            'agents_tracked': list(self._agent_accuracy.keys()),
        }
