"""
Adaptive Feedback Loop — Closed-Loop Learning from Every Trade
================================================================
Every trade outcome feeds back into EVERY layer of the system:

  Trade closes → AdaptiveFeedbackLoop.record_outcome()
    ↓
    ├─ Strategy weights update (winning strategy gets boosted)
    ├─ Agent weights update (accurate agents get more vote power)
    ├─ ML confidence calibration (model accuracy tracked per bucket)
    ├─ LLM context enrichment (winner/loser DNA extracted)
    ├─ Risk parameter tuning (SL/TP auto-adjusted)
    ├─ Regime-performance tracking (which regime makes money?)
    └─ Performance metrics update (rolling win rate, PF, Sharpe)

The goal: the system gets SMARTER after every single trade.
Not just ML retraining — every component adapts.
"""

import json
import os
import time
import math
import logging
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TradeOutcome:
    """Complete record of a closed trade for learning."""
    __slots__ = (
        'asset', 'direction', 'entry_price', 'exit_price',
        'pnl_pct', 'pnl_usd', 'duration_min', 'entry_score',
        'strategy_used', 'strategy_signals', 'strategy_weights',
        'agent_votes', 'llm_confidence', 'risk_score', 'trade_quality',
        'regime', 'hurst', 'volatility_regime', 'sl_level',
        'exit_reason', 'timeframe', 'timestamp',
        'multi_strategy_details', 'spread_cost_pct',
    )

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot, None))


class AdaptiveFeedbackLoop:
    """
    Central nervous system — distributes learning signals from every trade.

    Usage:
        feedback = AdaptiveFeedbackLoop(config)

        # After every trade closes:
        feedback.record_outcome(TradeOutcome(...))

        # Before every new trade:
        context = feedback.get_adaptive_context(asset, regime)
        # Returns: adjusted weights, confidence multiplier, regime stats
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._lock = threading.Lock()

        # ── Rolling performance windows ──
        self._recent_trades: deque = deque(maxlen=100)  # Last 100 trades
        self._trades_by_asset: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._trades_by_regime: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._trades_by_strategy: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

        # ── Adaptive weights (learned from outcomes) ──
        self.strategy_performance: Dict[str, Dict] = {}  # strategy → {wins, losses, total_pnl, avg_pnl}
        self.agent_accuracy: Dict[str, Dict] = {}         # agent → {correct, total, weight_mult}
        self.regime_profitability: Dict[str, Dict] = {}    # regime → {wins, losses, avg_pnl}
        self.confidence_calibration: Dict[str, Dict] = {}  # confidence_bucket → {actual_win_rate}

        # ── Winner/Loser DNA ──
        self.winner_dna: Dict[str, Any] = {}  # Common traits of winning trades
        self.loser_dna: Dict[str, Any] = {}   # Common traits of losing trades

        # ── Adaptive parameters ──
        self.adaptive_params = {
            'confidence_multiplier': 1.0,    # Adjusted based on recent accuracy
            'size_multiplier': 1.0,          # Adjusted based on recent drawdown
            'regime_override': None,         # If a regime consistently loses, block it
            'strategy_blacklist': set(),     # Strategies that are currently losing
            'min_score_adjustment': 0,       # Dynamic entry score adjustment
        }

        # Load saved state
        self._state_file = os.path.join(PROJECT_ROOT, 'data', 'adaptive_state.json')
        self._load_state()

        print(f"  [ADAPTIVE] Feedback loop ACTIVE — learning from every trade outcome")

    def record_outcome(self, outcome: TradeOutcome):
        """
        Process a closed trade and distribute learning to all layers.
        Call this from executor._close_position().
        """
        with self._lock:
            won = (outcome.pnl_pct or 0) > 0
            pnl = outcome.pnl_pct or 0

            # 1. Store in rolling windows
            trade_data = {
                'won': won,
                'pnl_pct': pnl,
                'pnl_usd': outcome.pnl_usd or 0,
                'asset': outcome.asset,
                'regime': outcome.regime or 'UNKNOWN',
                'strategy': outcome.strategy_used or 'ema_trend',
                'hurst': outcome.hurst or 0.5,
                'entry_score': outcome.entry_score or 0,
                'llm_confidence': outcome.llm_confidence or 0,
                'risk_score': outcome.risk_score or 0,
                'trade_quality': outcome.trade_quality or 0,
                'sl_level': outcome.sl_level or 1,
                'duration_min': outcome.duration_min or 0,
                'exit_reason': outcome.exit_reason or '',
                'timestamp': time.time(),
            }

            self._recent_trades.append(trade_data)
            self._trades_by_asset[outcome.asset].append(trade_data)
            self._trades_by_regime[outcome.regime or 'UNKNOWN'].append(trade_data)

            # 2. Update strategy performance
            self._update_strategy_performance(outcome, won, pnl)

            # 3. Update agent accuracy
            self._update_agent_accuracy(outcome, won)

            # 4. Update regime profitability
            self._update_regime_profitability(outcome, won, pnl)

            # 5. Update confidence calibration
            self._update_confidence_calibration(outcome, won)

            # 6. Extract winner/loser DNA
            self._update_trade_dna(trade_data, won)

            # 7. Compute adaptive parameters
            self._recompute_adaptive_params()

            # 8. Save state
            self._save_state()

            # Log
            tag = "WIN" if won else "LOSS"
            logger.info(
                f"[ADAPTIVE] {tag} {outcome.asset} {pnl:+.2f}% | "
                f"strategy={outcome.strategy_used} regime={outcome.regime} "
                f"conf={outcome.llm_confidence:.2f} score={outcome.entry_score} "
                f"→ conf_mult={self.adaptive_params['confidence_multiplier']:.2f} "
                f"size_mult={self.adaptive_params['size_multiplier']:.2f}"
            )

    def _update_strategy_performance(self, outcome: TradeOutcome, won: bool, pnl: float):
        """Track per-strategy win rate and PnL."""
        # Update for the primary strategy
        strategy = outcome.strategy_used or 'ema_trend'
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': 0}
        sp = self.strategy_performance[strategy]
        sp['trades'] += 1
        sp['total_pnl'] += pnl
        if won:
            sp['wins'] += 1
        else:
            sp['losses'] += 1

        # Also update multi-strategy weights if available
        if outcome.multi_strategy_details:
            for strat_name, detail in outcome.multi_strategy_details.items():
                if strat_name.startswith('_'):
                    continue
                if strat_name not in self.strategy_performance:
                    self.strategy_performance[strat_name] = {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': 0}
                sig = detail.get('signal', 0)
                direction = 1 if outcome.direction == 'LONG' else -1
                if sig == direction and won:
                    self.strategy_performance[strat_name]['wins'] += 1
                elif sig == direction and not won:
                    self.strategy_performance[strat_name]['losses'] += 1

    def _update_agent_accuracy(self, outcome: TradeOutcome, won: bool):
        """Track which agents predicted correctly."""
        if not outcome.agent_votes:
            return
        direction = 1 if outcome.direction == 'LONG' else -1
        for agent_name, vote in outcome.agent_votes.items():
            if agent_name not in self.agent_accuracy:
                self.agent_accuracy[agent_name] = {'correct': 0, 'total': 0, 'weight_mult': 1.0}
            aa = self.agent_accuracy[agent_name]
            aa['total'] += 1
            vote_dir = vote.get('direction', 0) if isinstance(vote, dict) else 0
            # Agent was correct if: voted same direction AND trade won, OR voted against AND trade lost
            if (vote_dir == direction and won) or (vote_dir != direction and not won):
                aa['correct'] += 1
            # Update weight multiplier (agents with >60% accuracy get boosted)
            if aa['total'] >= 5:
                accuracy = aa['correct'] / aa['total']
                aa['weight_mult'] = 0.5 + accuracy  # 0.5-1.5x range

    def _update_regime_profitability(self, outcome: TradeOutcome, won: bool, pnl: float):
        """Track which regimes are profitable."""
        regime = outcome.regime or 'UNKNOWN'
        if regime not in self.regime_profitability:
            self.regime_profitability[regime] = {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': 0}
        rp = self.regime_profitability[regime]
        rp['trades'] += 1
        rp['total_pnl'] += pnl
        if won:
            rp['wins'] += 1
        else:
            rp['losses'] += 1

    def _update_confidence_calibration(self, outcome: TradeOutcome, won: bool):
        """Track actual win rate per confidence bucket for calibration."""
        conf = outcome.llm_confidence or 0
        bucket = f"{int(conf * 10) / 10:.1f}"  # 0.0, 0.1, ..., 0.9, 1.0
        if bucket not in self.confidence_calibration:
            self.confidence_calibration[bucket] = {'wins': 0, 'total': 0}
        cc = self.confidence_calibration[bucket]
        cc['total'] += 1
        if won:
            cc['wins'] += 1

    def _update_trade_dna(self, trade_data: Dict, won: bool):
        """Extract common traits of winners vs losers."""
        target = self.winner_dna if won else self.loser_dna

        # Accumulate averages
        for key in ['entry_score', 'llm_confidence', 'risk_score', 'trade_quality', 'hurst', 'duration_min', 'sl_level']:
            if key not in target:
                target[key] = {'sum': 0, 'count': 0}
            target[key]['sum'] += trade_data.get(key, 0) or 0
            target[key]['count'] += 1

        # Track regime distribution
        regime = trade_data.get('regime', 'UNKNOWN')
        if 'regimes' not in target:
            target['regimes'] = {}
        target['regimes'][regime] = target['regimes'].get(regime, 0) + 1

        # Track strategy distribution
        strategy = trade_data.get('strategy', 'unknown')
        if 'strategies' not in target:
            target['strategies'] = {}
        target['strategies'][strategy] = target['strategies'].get(strategy, 0) + 1

    def _recompute_adaptive_params(self):
        """Adjust system parameters based on recent performance."""
        recent = list(self._recent_trades)
        if len(recent) < 3:
            return

        # Rolling win rate (last 20 trades)
        last_n = recent[-20:]
        wins = sum(1 for t in last_n if t['won'])
        wr = wins / len(last_n)

        # Rolling PnL
        total_pnl = sum(t['pnl_pct'] for t in last_n)

        # ── Confidence multiplier ──
        # If winning > 50%: boost confidence requirements slightly (we're doing well, stay selective)
        # If winning < 30%: reduce confidence (we're being too aggressive)
        if wr > 0.55:
            self.adaptive_params['confidence_multiplier'] = 1.05  # Slightly raise the bar
        elif wr < 0.30:
            self.adaptive_params['confidence_multiplier'] = 1.20  # Much higher bar — being too loose
        else:
            self.adaptive_params['confidence_multiplier'] = 1.0

        # ── Size multiplier (drawdown protection) ──
        # If on a losing streak (3+ losses in a row): reduce size
        streak = 0
        for t in reversed(last_n):
            if not t['won']:
                streak += 1
            else:
                break
        if streak >= 5:
            self.adaptive_params['size_multiplier'] = 0.3  # 70% reduction
        elif streak >= 3:
            self.adaptive_params['size_multiplier'] = 0.5  # 50% reduction
        elif wr > 0.6 and total_pnl > 0:
            self.adaptive_params['size_multiplier'] = 1.2  # Slight boost when winning
        else:
            self.adaptive_params['size_multiplier'] = 1.0

        # ── Regime blocking ──
        # If a regime has lost 5+ trades with <20% WR, suggest avoiding it
        for regime, rp in self.regime_profitability.items():
            if rp['trades'] >= 5:
                regime_wr = rp['wins'] / rp['trades']
                if regime_wr < 0.20:
                    self.adaptive_params['regime_override'] = regime
                    logger.warning(f"[ADAPTIVE] REGIME WARNING: {regime} has {regime_wr:.0%} WR — suggesting avoidance")

        # ── Strategy blacklisting ──
        # Temporarily disable strategies that are losing badly
        blacklist = set()
        for strat, sp in self.strategy_performance.items():
            if sp['trades'] >= 5:
                strat_wr = sp['wins'] / sp['trades']
                if strat_wr < 0.15 and sp['total_pnl'] < -5:
                    blacklist.add(strat)
                    logger.warning(f"[ADAPTIVE] STRATEGY BLACKLISTED: {strat} ({strat_wr:.0%} WR, {sp['total_pnl']:+.1f}% PnL)")
        self.adaptive_params['strategy_blacklist'] = blacklist

        # ── Entry score adjustment ──
        # If average winning score is much higher than losing score, raise the bar
        if self.winner_dna.get('entry_score', {}).get('count', 0) >= 3 and \
           self.loser_dna.get('entry_score', {}).get('count', 0) >= 3:
            avg_win_score = self.winner_dna['entry_score']['sum'] / self.winner_dna['entry_score']['count']
            avg_lose_score = self.loser_dna['entry_score']['sum'] / self.loser_dna['entry_score']['count']
            if avg_win_score > avg_lose_score + 1.5:
                self.adaptive_params['min_score_adjustment'] = int(avg_win_score - 1)

    def get_adaptive_context(self, asset: str, regime: str = 'UNKNOWN') -> Dict[str, Any]:
        """
        Get current adaptive adjustments for trade evaluation.
        Called BEFORE each trade decision.
        """
        with self._lock:
            # Rolling stats
            recent = list(self._recent_trades)
            last_20 = recent[-20:] if len(recent) >= 5 else recent
            wr = sum(1 for t in last_20 if t['won']) / len(last_20) if last_20 else 0.5

            # Asset-specific stats
            asset_trades = list(self._trades_by_asset.get(asset, []))
            asset_wr = sum(1 for t in asset_trades[-10:] if t['won']) / max(len(asset_trades[-10:]), 1) if asset_trades else 0.5

            # Regime stats
            regime_trades = list(self._trades_by_regime.get(regime, []))
            regime_wr = sum(1 for t in regime_trades[-10:] if t['won']) / max(len(regime_trades[-10:]), 1) if regime_trades else 0.5

            # Strategy rankings (best to worst by win rate)
            strategy_rankings = {}
            for strat, sp in self.strategy_performance.items():
                if sp['trades'] >= 3:
                    strategy_rankings[strat] = {
                        'win_rate': sp['wins'] / sp['trades'],
                        'avg_pnl': sp['total_pnl'] / sp['trades'],
                        'trades': sp['trades'],
                    }

            # Winner DNA summary for LLM
            winner_summary = self._get_dna_summary(self.winner_dna, "WINNERS")
            loser_summary = self._get_dna_summary(self.loser_dna, "LOSERS")

            # Confidence calibration map
            calibration = {}
            for bucket, cc in self.confidence_calibration.items():
                if cc['total'] >= 3:
                    calibration[bucket] = cc['wins'] / cc['total']

            return {
                # Multipliers (apply to trade execution)
                'confidence_multiplier': self.adaptive_params['confidence_multiplier'],
                'size_multiplier': self.adaptive_params['size_multiplier'],
                'min_score_adjustment': self.adaptive_params['min_score_adjustment'],
                'strategy_blacklist': list(self.adaptive_params['strategy_blacklist']),
                'regime_warning': self.adaptive_params.get('regime_override'),

                # Rolling stats (for LLM context)
                'rolling_win_rate': round(wr, 3),
                'asset_win_rate': round(asset_wr, 3),
                'regime_win_rate': round(regime_wr, 3),
                'total_trades': len(recent),

                # Strategy rankings (for multi-strategy engine)
                'strategy_rankings': strategy_rankings,

                # Agent weight multipliers
                'agent_weight_mults': {
                    name: aa.get('weight_mult', 1.0)
                    for name, aa in self.agent_accuracy.items()
                },

                # Confidence calibration
                'confidence_calibration': calibration,

                # DNA for LLM prompts
                'winner_dna': winner_summary,
                'loser_dna': loser_summary,

                # Regime profitability
                'regime_profitability': {
                    r: {'win_rate': rp['wins'] / rp['trades'], 'avg_pnl': rp['total_pnl'] / rp['trades']}
                    for r, rp in self.regime_profitability.items()
                    if rp['trades'] >= 3
                },
            }

    def _get_dna_summary(self, dna: Dict, label: str) -> str:
        """Generate human-readable DNA summary for LLM context."""
        if not dna or not dna.get('entry_score', {}).get('count', 0):
            return f"No {label.lower()} data yet."

        parts = []
        for key in ['entry_score', 'llm_confidence', 'risk_score', 'trade_quality', 'hurst', 'duration_min']:
            info = dna.get(key, {})
            if info.get('count', 0) > 0:
                avg = info['sum'] / info['count']
                parts.append(f"{key}={avg:.1f}")

        # Top regime
        regimes = dna.get('regimes', {})
        if regimes:
            top_regime = max(regimes, key=regimes.get)
            parts.append(f"regime={top_regime}({regimes[top_regime]})")

        # Top strategy
        strategies = dna.get('strategies', {})
        if strategies:
            top_strat = max(strategies, key=strategies.get)
            parts.append(f"strategy={top_strat}({strategies[top_strat]})")

        return f"{label}: {', '.join(parts)}"

    def record_evolution_results(self, hall_of_fame: List[Dict], diversity_metrics: Optional[Dict] = None):
        """
        Absorb genetic evolution results into the feedback loop.

        Cross-references evolved parameter sets with live trade outcomes
        and marks hall-of-fame entries as 'validated' if they match
        strategies that have been profitable in live trading.

        Args:
            hall_of_fame: List of dicts from genetic engine (each has genes, fitness, etc.)
            diversity_metrics: Optional diversity metrics from the evolution run
        """
        with self._lock:
            if not hall_of_fame:
                return

            # Cross-reference evolved strategies with live trade outcomes
            validated_count = 0
            for entry in hall_of_fame:
                entry_rule = entry.get('entry_rule', '')
                exit_rule = entry.get('exit_rule', '')
                fitness = entry.get('fitness', 0)

                # Check if any live strategy performance matches the evolved entry/exit rules
                matched_live = False
                for strat_name, sp in self.strategy_performance.items():
                    strat_lower = strat_name.lower()
                    # Match by rule similarity (entry_rule substring in strategy name)
                    if entry_rule and entry_rule.replace('_', '') in strat_lower.replace('_', ''):
                        if sp['trades'] >= 3:
                            live_wr = sp['wins'] / sp['trades']
                            live_pnl = sp['total_pnl'] / sp['trades']
                            # Validated = live performance corroborates evolved fitness
                            if live_wr > 0.4 and live_pnl > 0:
                                entry['validated'] = True
                                entry['live_wr'] = live_wr
                                entry['live_avg_pnl'] = live_pnl
                                validated_count += 1
                                matched_live = True
                                break

                if not matched_live:
                    entry['validated'] = False

            # Store evolution results in adaptive state
            self._evolution_results = {
                'hall_of_fame': hall_of_fame,
                'diversity_metrics': diversity_metrics or {},
                'validated_count': validated_count,
                'total_count': len(hall_of_fame),
                'timestamp': time.time(),
            }

            # Save to disk
            evo_path = os.path.join(PROJECT_ROOT, 'data', 'evolution_feedback.json')
            try:
                os.makedirs(os.path.dirname(evo_path), exist_ok=True)
                with open(evo_path, 'w') as f:
                    json.dump(self._evolution_results, f, indent=2, default=str)
            except Exception as e:
                logger.debug(f"[ADAPTIVE] Evolution feedback save failed: {e}")

            logger.info(
                f"[ADAPTIVE] Evolution results absorbed: "
                f"{len(hall_of_fame)} strategies, {validated_count} validated by live data"
            )

    def get_strategy_weight_adjustments(self) -> Dict[str, float]:
        """
        Return weight multipliers for multi-strategy engine.
        Winning strategies get boosted, losing ones dampened.
        """
        adjustments = {}
        for strat, sp in self.strategy_performance.items():
            if sp['trades'] < 3:
                adjustments[strat] = 1.0
                continue
            wr = sp['wins'] / sp['trades']
            avg_pnl = sp['total_pnl'] / sp['trades']
            # Boost strategies with WR > 50% AND positive PnL
            if wr > 0.5 and avg_pnl > 0:
                adjustments[strat] = min(1.5, 0.8 + wr)
            elif wr < 0.3:
                adjustments[strat] = max(0.3, wr)
            else:
                adjustments[strat] = 1.0
        return adjustments

    def get_llm_learning_context(self) -> str:
        """
        Generate a learning context block for LLM prompts.
        Tells the LLM what's been working and what hasn't.
        """
        recent = list(self._recent_trades)
        if len(recent) < 3:
            return "ADAPTIVE LEARNING: Not enough trades yet for pattern extraction."

        last_20 = recent[-20:]
        wins = [t for t in last_20 if t['won']]
        losses = [t for t in last_20 if not t['won']]
        wr = len(wins) / len(last_20)

        lines = [
            f"ADAPTIVE LEARNING (last {len(last_20)} trades, {wr:.0%} WR):",
        ]

        # What's working
        if wins:
            avg_win_score = sum(t['entry_score'] for t in wins) / len(wins)
            avg_win_conf = sum(t['llm_confidence'] for t in wins) / len(wins)
            win_regimes = [t['regime'] for t in wins]
            top_win_regime = max(set(win_regimes), key=win_regimes.count) if win_regimes else '?'
            lines.append(f"  WINNERS: avg_score={avg_win_score:.1f} avg_conf={avg_win_conf:.2f} top_regime={top_win_regime}")

        # What's failing
        if losses:
            avg_lose_score = sum(t['entry_score'] for t in losses) / len(losses)
            avg_lose_conf = sum(t['llm_confidence'] for t in losses) / len(losses)
            lose_reasons = [t.get('exit_reason', '?') for t in losses]
            lines.append(f"  LOSERS: avg_score={avg_lose_score:.1f} avg_conf={avg_lose_conf:.2f} exits={lose_reasons[:3]}")

        # Strategy performance
        for strat, sp in self.strategy_performance.items():
            if sp['trades'] >= 3:
                strat_wr = sp['wins'] / sp['trades']
                lines.append(f"  {strat}: {strat_wr:.0%} WR ({sp['trades']} trades, {sp['total_pnl']:+.1f}% PnL)")

        # Actionable insight
        if wr < 0.35:
            lines.append("  ACTION: Recent performance POOR — raise conviction bar, reduce size")
        elif wr > 0.60:
            lines.append("  ACTION: Recent performance GOOD — maintain current approach")

        return "\n".join(lines)

    def get_fitness_report(self) -> Dict[str, Any]:
        """Compute multi-metric fitness for recent trades (Sortino, Calmar, expectancy, grade)."""
        try:
            from src.trading.optimizer import MultiMetricFitness
            fitness = MultiMetricFitness()
            recent = list(self._recent_trades)
            if len(recent) < 3:
                return {'grade': 'N/A', 'fitness_score': 0, 'trade_count': 0}
            trade_dicts = [{
                'pnl_usd': t.get('pnl_usd', 0),
                'pnl_pct': t.get('pnl_pct', 0),
                'duration_min': t.get('duration_min', 60),
                'asset': t.get('asset', '?'),
            } for t in recent[-50:]]
            return fitness.compute(trade_dicts)
        except Exception as e:
            logger.debug(f"Fitness report failed: {e}")
            return {'grade': 'N/A', 'fitness_score': 0, 'trade_count': len(list(self._recent_trades))}

    def _save_state(self):
        """Persist adaptive state to disk."""
        try:
            state = {
                'strategy_performance': self.strategy_performance,
                'agent_accuracy': self.agent_accuracy,
                'regime_profitability': self.regime_profitability,
                'confidence_calibration': self.confidence_calibration,
                'winner_dna': self.winner_dna,
                'loser_dna': self.loser_dna,
                'adaptive_params': {
                    k: list(v) if isinstance(v, set) else v
                    for k, v in self.adaptive_params.items()
                },
                'total_trades_processed': len(self._recent_trades),
                'saved_at': datetime.now(tz=timezone.utc).isoformat(),
            }
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            tmp = self._state_file + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp, self._state_file)
        except Exception as e:
            logger.debug(f"[ADAPTIVE] State save failed: {e}")

    def _load_state(self):
        """Load adaptive state from disk."""
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file) as f:
                state = json.load(f)
            self.strategy_performance = state.get('strategy_performance', {})
            self.agent_accuracy = state.get('agent_accuracy', {})
            self.regime_profitability = state.get('regime_profitability', {})
            self.confidence_calibration = state.get('confidence_calibration', {})
            self.winner_dna = state.get('winner_dna', {})
            self.loser_dna = state.get('loser_dna', {})
            saved_params = state.get('adaptive_params', {})
            if 'strategy_blacklist' in saved_params:
                saved_params['strategy_blacklist'] = set(saved_params['strategy_blacklist'])
            self.adaptive_params.update(saved_params)
            total = state.get('total_trades_processed', 0)
            if total > 0:
                logger.info(f"[ADAPTIVE] Loaded state: {total} trades processed, {len(self.strategy_performance)} strategies tracked")
        except Exception as e:
            logger.debug(f"[ADAPTIVE] State load failed: {e}")
