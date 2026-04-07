"""
PHASE 6: Reinforcement Learning — EMA(8) Trend Strategy Optimizer
==================================================================
RL agent that learns WHEN and HOW to apply our proven EMA(8) trend-following
strategy. It does NOT learn a new strategy — it learns to optimize:
  1. Entry timing: Is this EMA new-line worth entering? (filter quality)
  2. Position sizing: How much to risk based on setup quality
  3. Exit patience: How long to ride before taking profit vs waiting for EMA flip
  4. Risk management: Optimal SL buffer distance from EMA line

Proven strategy rules (HARDCODED, never overridden by RL):
  - ENTER on EMA(8) new line (direction change after 3+ bars opposite)
  - RIDE while price on correct side of EMA
  - EXIT when EMA flips direction (profit-only) or SL hit (losers)
  - EMA line-following SL activates after 5 min, requires EMA moved in trade direction
  - Grace period 180s, hard stop -2%

RL learns AROUND these rules — it cannot disable them.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Strategy Constants (from 6-month backtest: 72% WR, PF 1.19) ───
EMA_PERIOD = 8
MIN_REVERSAL_BARS = 2          # Bars of EMA reversal before exit
GRACE_PERIOD_BARS = 4          # Don't check SL for first 4 bars
HARD_STOP_PCT = 2.0            # Maximum loss per trade
EMA_SL_ACTIVATION_BARS = 8    # Bars before EMA line-following SL activates
EMA_SL_BUFFER_ATR = 0.5       # ATR multiplier for EMA SL buffer
PROFIT_ONLY_EMA_EXIT = True    # EMA new line exit ONLY when in profit (100% WR)


@dataclass
class EMATradeState:
    """State representation aligned with EMA(8) strategy."""
    # EMA features
    ema_slope: float              # EMA direction strength (positive = rising)
    ema_slope_bars: int           # How many bars EMA has been trending this direction
    price_ema_distance_atr: float # Price distance from EMA in ATR units
    ema_acceleration: float       # Rate of change of EMA slope

    # Trend quality
    trend_bars_since_flip: int    # Bars since last EMA direction change
    trend_consistency: float      # % of recent bars price stayed on correct side of EMA
    higher_tf_alignment: float    # 1h/4h trend agreement (-1 to +1)

    # Volatility context
    atr_percentile: float         # Current ATR vs 100-bar range (0-1)
    volume_ratio: float           # Current vol vs 20-bar avg
    spread_atr_ratio: float       # Spread relative to ATR (liquidity proxy)

    # Risk context
    recent_win_rate: float        # Win rate of last 20 trades
    daily_pnl_pct: float          # Today's P&L
    open_positions: int           # Number of concurrent positions
    consecutive_losses: int       # Current loss streak

    # Time features
    hour_of_day: int              # 0-23 UTC
    day_of_week: int              # 0-6


@dataclass
class RLDecision:
    """RL output — adjustments AROUND the core strategy."""
    enter_trade: bool             # Should we take this EMA signal?
    quality_score: float          # 0-1, how good this setup looks
    position_size_mult: float     # 0.5-1.5 multiplier on base position size
    sl_buffer_mult: float         # 0.8-1.5 multiplier on SL ATR buffer
    patience_mult: float          # 0.8-2.0 multiplier on min hold time
    reasoning: str


class EMAStrategyRL:
    """
    Reinforcement Learning agent that optimizes EMA(8) strategy execution.

    DOES NOT override strategy rules. Instead learns:
    - Which EMA signals to skip (entry filtering)
    - How to size positions based on setup quality
    - How tight/wide to set SL buffer
    - When to be patient vs take quick profits

    Uses Q-learning with experience replay on discretized state space.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.learning_rate = cfg.get('rl_learning_rate', 0.01)
        self.gamma = cfg.get('rl_gamma', 0.95)
        self.epsilon = cfg.get('rl_epsilon', 0.15)       # Exploration rate
        self.epsilon_min = cfg.get('rl_epsilon_min', 0.05)
        self.epsilon_decay = cfg.get('rl_epsilon_decay', 0.999)

        # State discretization bins
        self.n_ema_slope_bins = 5      # strong_down, down, flat, up, strong_up
        self.n_dist_bins = 5           # far_below, below, at_ema, above, far_above
        self.n_vol_bins = 3            # low, medium, high
        self.n_trend_bins = 3          # new, established, extended
        self.n_wr_bins = 3             # losing, break_even, winning

        # Action space: (enter?, size_mult, sl_mult)
        # Discretized into 8 actions
        self.actions = [
            {'enter': False, 'size': 0.0, 'sl': 1.0, 'patience': 1.0, 'label': 'SKIP'},
            {'enter': True,  'size': 0.5, 'sl': 1.3, 'patience': 0.8, 'label': 'SMALL_TIGHT'},
            {'enter': True,  'size': 0.7, 'sl': 1.0, 'patience': 1.0, 'label': 'MEDIUM_NORMAL'},
            {'enter': True,  'size': 1.0, 'sl': 1.0, 'patience': 1.0, 'label': 'FULL_NORMAL'},
            {'enter': True,  'size': 1.0, 'sl': 0.8, 'patience': 1.5, 'label': 'FULL_WIDE_PATIENT'},
            {'enter': True,  'size': 1.2, 'sl': 1.0, 'patience': 1.2, 'label': 'AGGRESSIVE'},
            {'enter': True,  'size': 0.5, 'sl': 1.5, 'patience': 0.5, 'label': 'CAUTIOUS_QUICK'},
            {'enter': True,  'size': 1.0, 'sl': 1.2, 'patience': 2.0, 'label': 'FULL_VERY_PATIENT'},
        ]
        self.n_actions = len(self.actions)

        # Q-table: state_key -> action_values
        self.q_table: Dict[str, np.ndarray] = {}

        # Experience replay
        self.replay_buffer: deque = deque(maxlen=50000)
        self.batch_size = 32

        # Performance tracking
        self.stats = {
            'total_decisions': 0,
            'entries_taken': 0,
            'entries_skipped': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'action_counts': {a['label']: 0 for a in self.actions},
            'action_pnl': {a['label']: 0.0 for a in self.actions},
        }

        # Model persistence
        self.model_path = cfg.get('rl_model_path', 'models/rl_ema_strategy.json')
        self._load_model()

    def _discretize_state(self, state: EMATradeState) -> str:
        """Convert continuous state to discrete key for Q-table lookup."""
        # EMA slope bin
        if state.ema_slope < -0.3:
            slope_bin = 0  # strong down
        elif state.ema_slope < -0.05:
            slope_bin = 1  # down
        elif state.ema_slope < 0.05:
            slope_bin = 2  # flat
        elif state.ema_slope < 0.3:
            slope_bin = 3  # up
        else:
            slope_bin = 4  # strong up

        # Price-EMA distance bin (in ATR units)
        d = state.price_ema_distance_atr
        if d < -1.0:
            dist_bin = 0   # far below
        elif d < -0.2:
            dist_bin = 1   # below
        elif d < 0.2:
            dist_bin = 2   # at EMA
        elif d < 1.0:
            dist_bin = 3   # above
        else:
            dist_bin = 4   # far above

        # Volatility bin
        if state.atr_percentile < 0.33:
            vol_bin = 0    # low vol
        elif state.atr_percentile < 0.66:
            vol_bin = 1    # medium vol
        else:
            vol_bin = 2    # high vol

        # Trend age bin
        if state.trend_bars_since_flip < 5:
            trend_bin = 0  # new trend
        elif state.trend_bars_since_flip < 20:
            trend_bin = 1  # established
        else:
            trend_bin = 2  # extended

        # Win rate bin
        if state.recent_win_rate < 0.45:
            wr_bin = 0     # losing streak
        elif state.recent_win_rate < 0.60:
            wr_bin = 1     # break even
        else:
            wr_bin = 2     # winning

        # Higher TF alignment (aligned vs not)
        htf_bin = 1 if state.higher_tf_alignment > 0.3 else 0

        # Hour danger (known bad hours from journal)
        hour_bin = 1 if state.hour_of_day in (19, 20, 21, 22, 23, 0, 1, 2) else 0

        return f"{slope_bin}_{dist_bin}_{vol_bin}_{trend_bin}_{wr_bin}_{htf_bin}_{hour_bin}"

    def _get_q_values(self, state_key: str) -> np.ndarray:
        """Get Q-values for a state, initializing if needed."""
        if state_key not in self.q_table:
            # Initialize with strategy-aligned priors:
            # - FULL_NORMAL (action 3) gets slight bonus (default strategy)
            # - SKIP (action 0) gets slight bonus during dangerous hours
            init = np.zeros(self.n_actions)
            init[3] = 0.1  # Slight bias toward FULL_NORMAL (proven strategy)
            self.q_table[state_key] = init
        return self.q_table[state_key]

    def decide(self, state: EMATradeState) -> RLDecision:
        """
        Given current market state at an EMA signal, decide how to execute.

        This is called AFTER the core strategy detects a valid EMA new-line signal.
        RL decides whether to take it and how to size it.
        """
        state_key = self._discretize_state(state)
        q_values = self._get_q_values(state_key)

        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.n_actions)
        else:
            action_idx = int(np.argmax(q_values))

        action = self.actions[action_idx]
        self.stats['total_decisions'] += 1
        self.stats['action_counts'][action['label']] += 1

        if action['enter']:
            self.stats['entries_taken'] += 1
        else:
            self.stats['entries_skipped'] += 1

        # Build reasoning string
        reasoning_parts = [f"Action: {action['label']}"]
        reasoning_parts.append(f"State: slope={state.ema_slope:.2f}, dist={state.price_ema_distance_atr:.2f}ATR")
        reasoning_parts.append(f"Vol: p{state.atr_percentile:.0%}, HTF: {state.higher_tf_alignment:+.1f}")
        reasoning_parts.append(f"Q-values: best={q_values.max():.3f}, this={q_values[action_idx]:.3f}")
        reasoning_parts.append(f"Epsilon: {self.epsilon:.3f}")

        return RLDecision(
            enter_trade=action['enter'],
            quality_score=float(np.clip(q_values[action_idx], 0, 1)),
            position_size_mult=action['size'],
            sl_buffer_mult=action['sl'],
            patience_mult=action['patience'],
            reasoning=' | '.join(reasoning_parts),
        )

    def compute_reward(self, trade_result: Dict[str, float]) -> float:
        """
        Compute reward from a completed trade.

        Reward function aligned with EMA(8) strategy goals:
        - Profitable EMA exits get HIGHEST reward (we want to ride trends)
        - Quick SL exits on bad setups get moderate reward (cutting losses fast is good)
        - Large SL losses get penalty (should have been filtered)
        - Skipped signals that would have lost get reward
        - Skipped signals that would have won get penalty

        Args:
            trade_result: Dict with keys:
                'pnl_pct': float — P&L percentage
                'exit_type': str — 'ema_exit', 'sl', 'ratchet', 'time', 'skip_would_win', 'skip_would_lose'
                'hold_bars': int — how many bars held
                'was_skipped': bool — if RL said don't enter
        """
        pnl = trade_result.get('pnl_pct', 0)
        exit_type = trade_result.get('exit_type', 'unknown')
        hold_bars = trade_result.get('hold_bars', 0)
        was_skipped = trade_result.get('was_skipped', False)

        if was_skipped:
            # Reward for correctly skipping bad signals
            if trade_result.get('would_have_pnl', 0) < -0.3:
                return 1.0   # Great skip — avoided a loss
            elif trade_result.get('would_have_pnl', 0) > 0.5:
                return -0.8  # Bad skip — missed a winner
            else:
                return 0.0   # Neutral skip

        # Actual trade rewards
        if exit_type == 'ema_exit' and pnl > 0:
            # EMA exit in profit — the IDEAL outcome (100% WR in backtest)
            # Bonus for riding longer trends
            ride_bonus = min(hold_bars / 50, 0.5)  # Up to 0.5 bonus for patience
            return 2.0 + pnl * 0.5 + ride_bonus

        elif exit_type == 'ratchet' and pnl > 0:
            # Locked in profit via ratchet — good but not ideal (exited too early)
            return 0.5 + pnl * 0.3

        elif exit_type == 'sl' and pnl > -0.5:
            # Quick small loss — good risk management
            return -0.2

        elif exit_type == 'sl' and pnl <= -0.5:
            # Larger SL loss — should have been filtered or sized smaller
            return -1.0 + pnl * 0.3  # More negative for bigger losses

        elif exit_type == 'hard_stop':
            # Hit -2% hard stop — worst outcome
            return -2.0

        elif exit_type == 'time' and pnl <= 0:
            # Stale loser closed by time — bad entry
            return -0.5

        else:
            # Default: proportional to P&L
            return pnl * 0.5

    def update(self, state_key: str, action_idx: int, reward: float,
               next_state_key: Optional[str] = None):
        """
        Q-learning update: Q(s,a) += lr * (reward + gamma * max(Q(s')) - Q(s,a))
        """
        q_values = self._get_q_values(state_key)
        old_q = q_values[action_idx]

        if next_state_key is not None:
            next_q = self._get_q_values(next_state_key)
            target = reward + self.gamma * np.max(next_q)
        else:
            target = reward  # Terminal state

        # Q-learning update
        q_values[action_idx] += self.learning_rate * (target - old_q)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Store in replay buffer
        self.replay_buffer.append((state_key, action_idx, reward, next_state_key))

    def replay_learn(self):
        """Experience replay — sample random batch and update Q-values."""
        if len(self.replay_buffer) < self.batch_size:
            return

        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        for idx in indices:
            s, a, r, ns = self.replay_buffer[idx]
            self.update(s, a, r, ns)

    def record_trade_result(self, state: EMATradeState, action_idx: int,
                            trade_result: Dict[str, float],
                            next_state: Optional[EMATradeState] = None):
        """
        Record a completed trade and learn from it.
        Called by executor after each trade closes.
        """
        reward = self.compute_reward(trade_result)
        state_key = self._discretize_state(state)
        next_key = self._discretize_state(next_state) if next_state else None

        # Update Q-table
        self.update(state_key, action_idx, reward, next_key)

        # Update stats
        pnl = trade_result.get('pnl_pct', 0)
        action_label = self.actions[action_idx]['label']
        self.stats['action_pnl'][action_label] += pnl

        if not trade_result.get('was_skipped', False):
            if pnl > 0:
                self.stats['wins'] += 1
            else:
                self.stats['losses'] += 1
            self.stats['total_pnl'] += pnl

        # Periodic replay learning
        if self.stats['total_decisions'] % 10 == 0:
            self.replay_learn()

        # Periodic model save
        if self.stats['total_decisions'] % 50 == 0:
            self._save_model()

        logger.info(
            f"[RL] Trade result: {action_label} | PnL: {pnl:+.2f}% | "
            f"Reward: {reward:+.2f} | Epsilon: {self.epsilon:.3f} | "
            f"WR: {self.stats['wins']}/{self.stats['wins']+self.stats['losses']}"
        )

    def get_strategy_insights(self) -> Dict[str, Any]:
        """Get RL-learned insights about the strategy."""
        total = self.stats['wins'] + self.stats['losses']
        wr = self.stats['wins'] / total if total > 0 else 0

        # Find best and worst actions by P&L
        action_performance = {}
        for label in self.stats['action_counts']:
            count = self.stats['action_counts'][label]
            pnl = self.stats['action_pnl'][label]
            action_performance[label] = {
                'count': count,
                'total_pnl': round(pnl, 2),
                'avg_pnl': round(pnl / count, 3) if count > 0 else 0,
            }

        # Find states where SKIP is preferred (RL learned to avoid)
        skip_states = []
        for state_key, q_vals in self.q_table.items():
            if np.argmax(q_vals) == 0:  # SKIP is best action
                skip_states.append(state_key)

        return {
            'total_decisions': self.stats['total_decisions'],
            'win_rate': round(wr, 3),
            'total_pnl': round(self.stats['total_pnl'], 2),
            'entries_taken': self.stats['entries_taken'],
            'entries_skipped': self.stats['entries_skipped'],
            'skip_rate': round(self.stats['entries_skipped'] / max(1, self.stats['total_decisions']), 3),
            'epsilon': round(self.epsilon, 4),
            'q_table_size': len(self.q_table),
            'action_performance': action_performance,
            'learned_skip_states': len(skip_states),
            'replay_buffer_size': len(self.replay_buffer),
        }

    def _save_model(self):
        """Save Q-table and stats to disk."""
        try:
            data = {
                'q_table': {k: v.tolist() for k, v in self.q_table.items()},
                'stats': self.stats,
                'epsilon': self.epsilon,
                'saved_at': datetime.now().isoformat(),
            }
            os.makedirs(os.path.dirname(self.model_path) or '.', exist_ok=True)
            with open(self.model_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"[RL] Model saved: {len(self.q_table)} states, epsilon={self.epsilon:.3f}")
        except Exception as e:
            logger.warning(f"[RL] Failed to save model: {e}")

    def _load_model(self):
        """Load Q-table and stats from disk."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'r') as f:
                    data = json.load(f)
                self.q_table = {k: np.array(v) for k, v in data.get('q_table', {}).items()}
                loaded_stats = data.get('stats', {})
                # Merge loaded stats (preserve structure)
                for key in self.stats:
                    if key in loaded_stats:
                        self.stats[key] = loaded_stats[key]
                self.epsilon = data.get('epsilon', self.epsilon)
                logger.info(
                    f"[RL] Model loaded: {len(self.q_table)} states, "
                    f"epsilon={self.epsilon:.3f}, "
                    f"trades={self.stats['wins']+self.stats['losses']}"
                )
            except Exception as e:
                logger.warning(f"[RL] Failed to load model: {e}")


class AdaptiveAlgorithmLayer:
    """
    Adapts trading parameters based on market regime.
    Works WITH the EMA(8) strategy, not against it.

    Three modes based on market conditions:
    - AGGRESSIVE: Strong trend + low vol + high WR → bigger positions, wider targets
    - CONSERVATIVE: High vol + losing streak + crisis → smaller positions, tighter stops
    - NORMAL: Default EMA(8) parameters from backtest
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.rl_agent = EMAStrategyRL(cfg)
        self.current_mode = 'NORMAL'
        self.mode_history = []
        self.mode_performance = {'AGGRESSIVE': [], 'CONSERVATIVE': [], 'NORMAL': []}

    def select_mode(self, market_metrics: Dict[str, float],
                    recent_trades: Optional[List[Dict]] = None) -> str:
        """
        Select execution mode based on market conditions.
        This doesn't change WHAT we trade (always EMA signals) — only HOW.
        """
        volatility = market_metrics.get('atr_percentile', 0.5)
        trend_strength = abs(market_metrics.get('ema_slope', 0))
        htf_alignment = market_metrics.get('higher_tf_alignment', 0)

        # Recent performance
        recent_wr = 0.5
        consecutive_losses = 0
        if recent_trades:
            wins = sum(1 for t in recent_trades[-20:] if t.get('pnl_pct', 0) > 0)
            total = len(recent_trades[-20:])
            recent_wr = wins / total if total > 0 else 0.5

            # Count consecutive losses
            for t in reversed(recent_trades):
                if t.get('pnl_pct', 0) < 0:
                    consecutive_losses += 1
                else:
                    break

        # Mode selection logic aligned with proven strategy
        if consecutive_losses >= 4 or volatility > 0.8:
            new_mode = 'CONSERVATIVE'
        elif (trend_strength > 0.2 and htf_alignment > 0.5
              and volatility < 0.5 and recent_wr > 0.55):
            new_mode = 'AGGRESSIVE'
        else:
            new_mode = 'NORMAL'

        if new_mode != self.current_mode:
            logger.info(f"[ADAPTIVE] Mode: {self.current_mode} → {new_mode} "
                       f"(vol={volatility:.2f}, trend={trend_strength:.2f}, wr={recent_wr:.1%})")
            self.mode_history.append({
                'from': self.current_mode,
                'to': new_mode,
                'reason': f"vol={volatility:.2f} trend={trend_strength:.2f} wr={recent_wr:.1%}",
                'timestamp': datetime.now().isoformat(),
            })
            self.current_mode = new_mode

        return self.current_mode

    def get_mode_config(self) -> Dict[str, Any]:
        """Get parameter adjustments for current mode."""
        configs = {
            'AGGRESSIVE': {
                'position_size_mult': 1.3,      # 30% bigger positions
                'sl_buffer_mult': 0.9,           # Tighter SL (confident)
                'min_entry_score_adj': -1,       # Accept score 6 instead of 7
                'patience_mult': 1.5,            # Hold longer for bigger trends
                'max_concurrent': 2,
            },
            'CONSERVATIVE': {
                'position_size_mult': 0.5,       # Half position size
                'sl_buffer_mult': 1.3,           # Wider SL buffer
                'min_entry_score_adj': +1,       # Require score 8 instead of 7
                'patience_mult': 0.7,            # Take profits quicker
                'max_concurrent': 1,
            },
            'NORMAL': {
                'position_size_mult': 1.0,       # Default from backtest
                'sl_buffer_mult': 1.0,           # Default ATR buffer
                'min_entry_score_adj': 0,        # Default score 7
                'patience_mult': 1.0,            # Default hold time
                'max_concurrent': 2,
            },
        }
        return configs[self.current_mode]

    def record_trade(self, mode: str, trade_result: Dict[str, float]):
        """Track performance per mode for adaptive selection."""
        if mode in self.mode_performance:
            self.mode_performance[mode].append(trade_result.get('pnl_pct', 0))
            # Keep last 100 per mode
            if len(self.mode_performance[mode]) > 100:
                self.mode_performance[mode] = self.mode_performance[mode][-100:]

    def get_summary(self) -> Dict[str, Any]:
        """Get adaptive layer performance summary."""
        summary = {
            'current_mode': self.current_mode,
            'mode_switches': len(self.mode_history),
        }
        for mode, pnls in self.mode_performance.items():
            if pnls:
                wins = sum(1 for p in pnls if p > 0)
                summary[f'{mode.lower()}_wr'] = round(wins / len(pnls), 3)
                summary[f'{mode.lower()}_avg_pnl'] = round(np.mean(pnls), 3)
                summary[f'{mode.lower()}_trades'] = len(pnls)
        return summary


class SelfModifyingStrategyEngine:
    """
    Genetic algorithm that evolves EMA(8) strategy PARAMETERS.

    Does NOT evolve the strategy itself — the rules are fixed (EMA entry/exit).
    Evolves the tunable parameters:
    - grace_period_bars: How many bars before SL checks (backtest default: 4)
    - ema_sl_activation_bars: Bars before EMA line-following starts (default: 8)
    - ema_sl_buffer_atr: ATR multiplier for SL below EMA (default: 0.5)
    - min_reversal_bars: Bars of EMA reversal before exit (default: 2)
    - breakeven_trigger_pct: Profit % to activate breakeven ratchet (default: 1.0)
    - min_entry_score: Score threshold for entry (default: 7)
    """

    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.population = self._initialize_population()
        self.fitness_history = []
        self.generation = 0
        self.best_ever = None

    def _initialize_population(self) -> List[Dict[str, Any]]:
        """Initialize population with backtest-proven defaults at center."""
        population = []
        for i in range(self.population_size):
            if i == 0:
                # First individual = proven backtest parameters (seed the population)
                params = {
                    'grace_period_bars': 4,
                    'ema_sl_activation_bars': 8,
                    'ema_sl_buffer_atr': 0.5,
                    'min_reversal_bars': 2,
                    'breakeven_trigger_pct': 1.0,
                    'min_entry_score': 7,
                    'ratchet_breakeven_pct': 1.0,
                    'hard_stop_pct': 2.0,
                }
            else:
                # Random variations around proven parameters
                params = {
                    'grace_period_bars': int(np.clip(np.random.normal(4, 1.5), 1, 10)),
                    'ema_sl_activation_bars': int(np.clip(np.random.normal(8, 3), 3, 20)),
                    'ema_sl_buffer_atr': float(np.clip(np.random.normal(0.5, 0.2), 0.1, 1.5)),
                    'min_reversal_bars': int(np.clip(np.random.normal(2, 0.8), 1, 5)),
                    'breakeven_trigger_pct': float(np.clip(np.random.normal(1.0, 0.3), 0.3, 2.5)),
                    'min_entry_score': int(np.clip(np.random.normal(7, 1), 4, 9)),
                    'ratchet_breakeven_pct': float(np.clip(np.random.normal(1.0, 0.3), 0.3, 2.0)),
                    'hard_stop_pct': float(np.clip(np.random.normal(2.0, 0.5), 1.0, 4.0)),
                }

            strategy = {
                'id': i,
                'params': params,
                'fitness': 0.0,
                'metrics': {},
                'age': 0,
            }
            population.append(strategy)
        return population

    def evaluate_population(self, backtest_results: Dict[int, Dict[str, float]]):
        """
        Evaluate fitness from backtest results.

        Fitness prioritizes:
        1. Win rate (40% weight) — most important for our strategy
        2. Profit factor (30% weight) — risk-adjusted returns
        3. Max drawdown penalty (20% weight) — capital preservation
        4. Trade count bonus (10% weight) — avoid over-filtering
        """
        for strategy in self.population:
            sid = strategy['id']
            if sid in backtest_results:
                r = backtest_results[sid]
                wr = r.get('win_rate', 0)
                pf = r.get('profit_factor', 0)
                dd = r.get('max_drawdown', 1.0)
                trades = r.get('total_trades', 0)

                # Fitness function aligned with EMA strategy goals
                fitness = (
                    wr * 0.40 +                              # Win rate
                    min(pf, 3.0) / 3.0 * 0.30 +             # Profit factor (capped at 3)
                    max(0, 1.0 - dd) * 0.20 +                # Drawdown penalty
                    min(trades / 500, 1.0) * 0.10             # Trade count (enough samples)
                )

                # Penalty for extreme parameters (regularization)
                params = strategy['params']
                if params['hard_stop_pct'] > 3.0:
                    fitness *= 0.9
                if params['min_entry_score'] > 8:
                    fitness *= 0.9  # Too strict = misses good trades

                strategy['fitness'] = fitness
                strategy['metrics'] = r

            strategy['age'] += 1

    def selection(self) -> List[Dict[str, Any]]:
        """Tournament selection — keep top 40%."""
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        n_elite = max(2, int(0.4 * len(sorted_pop)))
        return [s.copy() for s in sorted_pop[:n_elite]]

    def crossover(self, p1: Dict, p2: Dict) -> Dict:
        """Uniform crossover between two parents."""
        child_params = {}
        for param in p1['params']:
            if np.random.rand() < 0.5:
                child_params[param] = p1['params'][param]
            else:
                child_params[param] = p2['params'][param]

        return {
            'id': np.random.randint(0, 1_000_000),
            'params': child_params,
            'fitness': 0.0,
            'metrics': {},
            'age': 0,
        }

    def mutate(self, strategy: Dict, rate: float = 0.15):
        """Gaussian mutation on random parameters."""
        params = strategy['params']
        for param in params:
            if np.random.rand() < rate:
                val = params[param]
                if isinstance(val, int):
                    params[param] = int(np.clip(val + np.random.choice([-1, 0, 1]), 1, 20))
                else:
                    noise = np.random.normal(0, abs(val) * 0.15 + 0.01)
                    params[param] = round(float(np.clip(val + noise, 0.1, 10.0)), 2)

    def evolve(self, backtest_results: Dict[int, Dict[str, float]]):
        """Run one generation of evolution."""
        self.evaluate_population(backtest_results)

        best_fitness = max(s['fitness'] for s in self.population)
        self.fitness_history.append(best_fitness)

        best = max(self.population, key=lambda x: x['fitness'])
        if self.best_ever is None or best['fitness'] > self.best_ever['fitness']:
            self.best_ever = best.copy()

        # Selection
        elite = self.selection()

        # Create next generation
        new_pop = [elite[0]]  # Always keep best (elitism)

        while len(new_pop) < self.population_size:
            p1, p2 = np.random.choice(elite, 2, replace=True)
            child = self.crossover(p1, p2)
            self.mutate(child)
            new_pop.append(child)

        self.population = new_pop[:self.population_size]
        self.generation += 1

        logger.info(
            f"[GA] Gen {self.generation}: best_fitness={best_fitness:.4f} | "
            f"Best params: {best['params']} | "
            f"Metrics: WR={best['metrics'].get('win_rate', 0):.1%} "
            f"PF={best['metrics'].get('profit_factor', 0):.2f}"
        )

    def get_best_params(self) -> Dict[str, Any]:
        """Return best strategy parameters found so far."""
        if self.best_ever:
            return self.best_ever['params']
        # Default: proven backtest parameters
        return {
            'grace_period_bars': 4,
            'ema_sl_activation_bars': 8,
            'ema_sl_buffer_atr': 0.5,
            'min_reversal_bars': 2,
            'breakeven_trigger_pct': 1.0,
            'min_entry_score': 7,
            'ratchet_breakeven_pct': 1.0,
            'hard_stop_pct': 2.0,
        }

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Summary of evolutionary progress."""
        return {
            'generation': self.generation,
            'population_size': self.population_size,
            'best_fitness': round(self.fitness_history[-1], 4) if self.fitness_history else 0,
            'best_params': self.get_best_params(),
            'best_metrics': self.best_ever.get('metrics', {}) if self.best_ever else {},
            'fitness_trend': [round(f, 4) for f in self.fitness_history[-10:]],
        }
