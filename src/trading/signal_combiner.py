"""
Three-Layer Hybrid Signal Combiner
=====================================
Fuses L1 (Quantitative), L2 (Sentiment), and L3 (Risk) signals
according to the architecture specification:

  L1 — 50% weight  (quantitative engine)
  L2 — 30% weight  (sentiment layer)
  L3 — 20% weight  + VETO authority (risk engine)

Final Signal:
  S_final = w1·S_L1 + w2·S_L2 + w3·S_L3

  If L3 issues VETO → S_final = 0 (no trade), regardless of L1/L2.

Staleness decay for L2:
  If sentiment hasn't been updated in T seconds, its weight decays:
    w2_effective = w2 · e^{-λ·T}
    Remaining weight redistributed to L1.
"""

import math
import time
from typing import Dict, Optional, List, Tuple

from src.risk.manager import RiskAction


class SignalCombiner:
    """
    Three-layer hybrid signal combiner with dynamic weight adjustment.

    The combiner takes:
      - L1 composite signal ∈ [-1, +1]
      - L2 aggregate sentiment ∈ [-1, +1] with confidence and freshness
      - L3 risk evaluation with possible VETO

    And produces a final trade decision.
    """

    def __init__(self, config: Optional[Dict] = None,
                 accuracy_engine=None, economic_intelligence=None,
                 sharpe_optimizer=None):
        cfg = config or {}
        # Base weights (fallback — overridden by accuracy_engine if available)
        self.w_l1 = cfg.get('l1_weight', 0.50)
        self.w_l2 = cfg.get('l2_weight', 0.30)
        self.w_l3 = cfg.get('l3_weight', 0.20)

        # v8.0 dynamic systems
        self._accuracy_engine = accuracy_engine
        self._economic_intelligence = economic_intelligence
        self._sharpe_optimizer = sharpe_optimizer

        # Track layer performance for dynamic weight learning
        self._l1_outcomes: List[bool] = []  # was L1 direction correct?
        self._l2_outcomes: List[bool] = []
        self._l3_outcomes: List[bool] = []

        # L2 staleness decay rate — Fix A: faster decay so stale sentiment penalized harder
        self.l2_decay_rate = cfg.get('l2_decay_rate', 0.005)  # per second (was 0.002)
        self.l2_max_age = cfg.get('l2_max_age', 600)  # 10 minutes max

        # Signal thresholds for final decision
        self.entry_threshold = cfg.get('entry_threshold', 0.20)
        self.exit_threshold = cfg.get('exit_threshold', 0.05)

        # Agreement bonus: if L1 and L2 agree, boost signal
        self.agreement_bonus = cfg.get('agreement_bonus', 0.10)

    def record_layer_outcome(self, l1_correct: bool, l2_correct: bool, l3_correct: bool):
        """Record whether each layer's signal was correct after trade close.
        Used to dynamically rebalance L1/L2/L3 weights."""
        self._l1_outcomes.append(l1_correct)
        self._l2_outcomes.append(l2_correct)
        self._l3_outcomes.append(l3_correct)
        # Keep last 100
        self._l1_outcomes = self._l1_outcomes[-100:]
        self._l2_outcomes = self._l2_outcomes[-100:]
        self._l3_outcomes = self._l3_outcomes[-100:]

    def _get_dynamic_weights(self) -> Tuple[float, float, float]:
        """Compute L1/L2/L3 weights from actual accuracy. No hardcoded values.
        Layers that have been more accurate recently get higher weight."""
        min_samples = 10
        if (len(self._l1_outcomes) < min_samples or
            len(self._l2_outcomes) < min_samples or
            len(self._l3_outcomes) < min_samples):
            return self.w_l1, self.w_l2, self.w_l3  # not enough data yet

        l1_acc = sum(self._l1_outcomes[-50:]) / len(self._l1_outcomes[-50:])
        l2_acc = sum(self._l2_outcomes[-50:]) / len(self._l2_outcomes[-50:])
        l3_acc = sum(self._l3_outcomes[-50:]) / len(self._l3_outcomes[-50:])

        # Convert accuracy to raw weight (0.4 acc -> 0.0, 0.7 acc -> 1.0)
        raw_l1 = max(0.05, (l1_acc - 0.35) / 0.35)
        raw_l2 = max(0.05, (l2_acc - 0.35) / 0.35)
        raw_l3 = max(0.05, (l3_acc - 0.35) / 0.35)

        total = raw_l1 + raw_l2 + raw_l3
        return raw_l1 / total, raw_l2 / total, raw_l3 / total

    def combine(self,
                l1_signal: float,
                l2_sentiment: Dict,
                l3_evaluation: Dict,
                l2_timestamp: Optional[float] = None,
                current_time: Optional[float] = None,
                ) -> Dict:
        """
        Combine all three layers into a final trading decision.

        Args:
            l1_signal: L1 composite signal ∈ [-1, +1]
            l2_sentiment: dict from SentimentPipeline.aggregate_sentiment()
            l3_evaluation: dict from RiskManager.evaluate_trade()
            l2_timestamp: when L2 was last updated (for staleness)
            current_time: current time (for staleness calc)

        Returns:
            {
                'final_signal': float [-1, +1],
                'action': str ('buy', 'sell', 'hold'),
                'confidence': float [0, 1],
                'position_size': float,
                'stop_loss': float,
                'take_profit': float,
                'breakdown': dict with L1/L2/L3 weighted contributions,
                'risk_status': str,
            }
        """
        now = current_time or time.time()

        # ---- L3 VETO check ----
        l3_action = l3_evaluation.get('action', RiskAction.ALLOW)
        if isinstance(l3_action, RiskAction):
            l3_action_val = l3_action
        else:
            l3_action_val = RiskAction(l3_action) if isinstance(l3_action, str) else RiskAction.ALLOW

        if l3_action_val in (RiskAction.VETO, RiskAction.EMERGENCY_EXIT):
            return self._vetoed_result(l3_evaluation)

        # ---- v8.0: Use DYNAMIC weights learned from accuracy ----
        dyn_l1, dyn_l2, dyn_l3 = self._get_dynamic_weights()

        # ---- Compute effective L2 weight with staleness decay ----
        l2_score = l2_sentiment.get('aggregate_score', 0.0)
        l2_confidence = l2_sentiment.get('confidence', 0.0)
        l2_freshness = l2_sentiment.get('freshness', 1.0)

        if l2_timestamp is not None:
            age = now - l2_timestamp
            if age > self.l2_max_age:
                l2_decay = 0.0  # fully stale
            else:
                l2_decay = math.exp(-self.l2_decay_rate * age)
        else:
            l2_decay = l2_freshness

        # Effective L2 weight = DYNAMIC weight * decay * confidence
        w_l2_effective = dyn_l2 * l2_decay * l2_confidence

        # Redistribute decayed L2 weight to L1
        w_l1_effective = dyn_l1 + (dyn_l2 - w_l2_effective)
        w_l3_effective = dyn_l3

        # ---- v8.0: Macro Intelligence overlay ----
        macro_adjustment = 0.0
        macro_info = ''
        if self._economic_intelligence:
            try:
                macro = self._economic_intelligence.get_macro_summary()
                if macro.get('crisis'):
                    return self._vetoed_result({'reason': 'CRISIS: macro intelligence triggered halt',
                                                'action': RiskAction.VETO, 'risk_score': 1.0})
                if macro.get('composite_signal') == 'BULLISH':
                    macro_adjustment = 0.05  # slight bullish boost
                elif macro.get('composite_signal') == 'BEARISH':
                    macro_adjustment = -0.05  # slight bearish drag
                macro_info = f"macro={macro.get('composite_signal', '?')}"
            except Exception:
                pass

        # ---- v8.0: Sharpe quality gate ----
        sharpe_size_mult = 1.0
        if self._sharpe_optimizer:
            try:
                adj = self._sharpe_optimizer.get_filter_adjustments()
                sharpe_size_mult = adj.get('position_size_mult', 1.0)
            except Exception:
                pass

        # ---- L3 risk signal ----
        risk_score = l3_evaluation.get('risk_score', 0.0)
        l3_signal = -risk_score  # high risk → negative contribution

        # ---- Weighted combination ----
        raw_signal = (w_l1_effective * l1_signal
                      + w_l2_effective * l2_score
                      + w_l3_effective * l3_signal
                      + macro_adjustment)  # v8.0: macro intelligence overlay

        # ---- Agreement bonus ----
        if l1_signal * l2_score > 0 and l2_decay > 0.5:
            # L1 and L2 agree on direction — boost confidence
            agreement_boost = self.agreement_bonus * min(abs(l1_signal), abs(l2_score))
            if l1_signal > 0:
                raw_signal += agreement_boost
            else:
                raw_signal -= agreement_boost

        # Clamp to [-1, +1]
        final_signal = max(-1.0, min(1.0, raw_signal))

        # ---- Determine action ----
        if l3_action_val == RiskAction.BLOCK:
            action = 'hold'
            final_signal = 0.0
        elif final_signal > self.entry_threshold:
            action = 'buy'
        elif final_signal < -self.entry_threshold:
            action = 'sell'
        else:
            action = 'hold'

        # ---- L3 position sizing adjustments ----
        adjusted_size = l3_evaluation.get('adjusted_size', 0.0)
        if l3_action_val == RiskAction.REDUCE:
            adjusted_size *= 0.5
        # v8.0: Sharpe-adjusted sizing
        adjusted_size *= sharpe_size_mult
        # v8.0: Macro pre-event reduction
        if self._economic_intelligence:
            try:
                macro = self._economic_intelligence.get_macro_summary()
                if macro.get('pre_event_flag'):
                    adjusted_size *= 0.6  # reduce 40% before high-impact events
            except Exception:
                pass

        # ---- Confidence (composite) ----
        # Fix B: weighted average instead of multiplicative (prevents collapse compounding)
        signal_contrib  = abs(final_signal)                    # 0-1
        risk_contrib    = 1.0 - risk_score                     # high risk = lower
        sentiment_boost = 0.5 + 0.5 * l2_decay                # fresh sentiment = boost
        confidence = 0.50 * signal_contrib + 0.30 * risk_contrib + 0.20 * sentiment_boost
        confidence = max(0.0, min(1.0, confidence))

        return {
            'final_signal': final_signal,
            'action': action,
            'confidence': min(1.0, confidence),
            'position_size': adjusted_size,
            'stop_loss': l3_evaluation.get('stop_loss', 0.0),
            'take_profit': l3_evaluation.get('take_profit', 0.0),
            'breakdown': {
                'l1_signal': l1_signal,
                'l1_weight': w_l1_effective,
                'l1_contribution': w_l1_effective * l1_signal,
                'l2_signal': l2_score,
                'l2_weight': w_l2_effective,
                'l2_contribution': w_l2_effective * l2_score,
                'l2_freshness': l2_decay,
                'l3_signal': l3_signal,
                'l3_weight': w_l3_effective,
                'l3_contribution': w_l3_effective * l3_signal,
                'risk_score': risk_score,
            },
            'risk_status': l3_evaluation.get('reason', 'OK'),
            'dynamic_weights': {'l1': round(dyn_l1, 3), 'l2': round(dyn_l2, 3), 'l3': round(dyn_l3, 3)},
            'sharpe_size_mult': sharpe_size_mult,
            'macro_adjustment': macro_adjustment,
        }

    def _vetoed_result(self, l3_evaluation: Dict) -> Dict:
        return {
            'final_signal': 0.0,
            'action': 'hold',
            'confidence': 0.0,
            'position_size': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'breakdown': {
                'l1_signal': 0, 'l1_weight': 0, 'l1_contribution': 0,
                'l2_signal': 0, 'l2_weight': 0, 'l2_contribution': 0,
                'l2_freshness': 0,
                'l3_signal': -1, 'l3_weight': 1.0, 'l3_contribution': -1,
                'risk_score': 1.0,
            },
            'risk_status': f"VETO: {l3_evaluation.get('reason', 'unknown')}",
        }


class MultiAssetCombiner:
    """Manages signal combining across multiple assets with correlation awareness."""

    def __init__(self, config: Optional[Dict] = None):
        self.combiners: Dict[str, SignalCombiner] = {}
        self.config = config or {}
        self.max_long_positions = self.config.get('max_long_positions', 3)
        self.max_short_positions = self.config.get('max_short_positions', 2)

    def get_combiner(self, asset: str) -> SignalCombiner:
        if asset not in self.combiners:
            self.combiners[asset] = SignalCombiner(self.config)
        return self.combiners[asset]

    def rank_signals(self, signals: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """
        Rank assets by signal strength and select top N for trading.
        Prevents over-concentration.
        """
        ranked = sorted(
            signals.items(),
            key=lambda x: abs(x[1].get('final_signal', 0)),
            reverse=True
        )

        long_count = 0
        short_count = 0
        selected: List[Tuple[str, Dict]] = []

        for asset, sig in ranked:
            action = sig.get('action', 'hold')
            if action == 'buy':
                if long_count < self.max_long_positions:
                    selected.append((asset, sig))
                    long_count += 1
            elif action == 'sell':
                if short_count < self.max_short_positions:
                    selected.append((asset, sig))
                    short_count += 1
            else:
                selected.append((asset, sig))

        return selected
