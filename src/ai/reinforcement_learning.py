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

    # ── NEW: Robinhood / Spot-specific state features ──
    spread_cost_pct: float = 0.0       # Round-trip spread cost as % (Robinhood ~3.3%)
    expected_move_pct: float = 0.0     # ATR × TP_mult / price as % (how far can it go?)
    move_to_spread_ratio: float = 0.0  # expected_move / spread_cost (>2.0 = profitable)
    is_spot: bool = False              # True = spot (no leverage), False = futures
    confluence_count: int = 0          # Number of independent signals agreeing (Sniper Mode)
    entry_score: int = 0               # Quality score from EMA analysis (0-15)
    timeframe_rank: int = 0            # 0=5m, 1=15m, 2=1h, 3=4h, 4=1d (higher = better for spot)


@dataclass
class RLDecision:
    """RL output — adjustments AROUND the core strategy."""
    enter_trade: bool             # Should we take this EMA signal?
    quality_score: float          # 0-1, how good this setup looks
    position_size_mult: float     # 0.5-1.5 multiplier on base position size
    sl_buffer_mult: float         # 0.8-1.5 multiplier on SL ATR buffer
    patience_mult: float          # 0.8-2.0 multiplier on min hold time
    reasoning: str
    action_idx: int = 0           # Index into actions array (for feedback loop)


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
        self.learning_rate = cfg.get('rl_learning_rate', 0.02)   # Faster learning for new spread-aware data
        self.gamma = cfg.get('rl_gamma', 0.90)                    # Slightly lower discount — spot trades resolve slower
        self.epsilon = cfg.get('rl_epsilon', 0.30)                # HIGH exploration — relearning for Robinhood
        self.epsilon_min = cfg.get('rl_epsilon_min', 0.08)        # Keep some exploration permanently
        self.epsilon_decay = cfg.get('rl_epsilon_decay', 0.997)   # Slower decay — explore longer

        # ── ACTION SPACE: 12 actions (was 8) ──
        # Added WAIT actions for Robinhood — sometimes the best trade is NO trade right now
        # Added SNIPER action — only enter on perfect confluence
        self.actions = [
            # Skip / Wait actions
            {'enter': False, 'size': 0.0, 'sl': 1.0, 'patience': 1.0, 'wait_bars': 0, 'label': 'SKIP'},
            {'enter': False, 'size': 0.0, 'sl': 1.0, 'patience': 1.0, 'wait_bars': 1, 'label': 'WAIT_1BAR'},
            {'enter': False, 'size': 0.0, 'sl': 1.0, 'patience': 1.0, 'wait_bars': 3, 'label': 'WAIT_PULLBACK'},
            # Small / Cautious entries (Robinhood: test the water)
            {'enter': True,  'size': 0.3, 'sl': 1.5, 'patience': 2.0, 'wait_bars': 0, 'label': 'TINY_PATIENT'},
            {'enter': True,  'size': 0.5, 'sl': 1.3, 'patience': 1.5, 'wait_bars': 0, 'label': 'SMALL_PATIENT'},
            # Standard entries
            {'enter': True,  'size': 0.7, 'sl': 1.0, 'patience': 1.0, 'wait_bars': 0, 'label': 'MEDIUM_NORMAL'},
            {'enter': True,  'size': 1.0, 'sl': 1.0, 'patience': 1.0, 'wait_bars': 0, 'label': 'FULL_NORMAL'},
            # Patient entries (hold through spread recovery)
            {'enter': True,  'size': 1.0, 'sl': 0.8, 'patience': 2.0, 'wait_bars': 0, 'label': 'FULL_WIDE_PATIENT'},
            {'enter': True,  'size': 1.0, 'sl': 1.2, 'patience': 3.0, 'wait_bars': 0, 'label': 'FULL_VERY_PATIENT'},
            # Aggressive entries (high confluence only)
            {'enter': True,  'size': 1.3, 'sl': 1.0, 'patience': 1.5, 'wait_bars': 0, 'label': 'AGGRESSIVE'},
            # Sniper entry (max size, max patience — for perfect setups)
            {'enter': True,  'size': 1.5, 'sl': 0.7, 'patience': 3.0, 'wait_bars': 0, 'label': 'SNIPER_MAX'},
            # Quick scalp (only viable when spread is low)
            {'enter': True,  'size': 0.5, 'sl': 1.5, 'patience': 0.5, 'wait_bars': 0, 'label': 'QUICK_SCALP'},
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
            'waits': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'total_net_pnl': 0.0,  # After spread
            'action_counts': {a['label']: 0 for a in self.actions},
            'action_pnl': {a['label']: 0.0 for a in self.actions},
        }

        # Model persistence
        self.model_path = cfg.get('rl_model_path', 'models/rl_ema_strategy.json')
        self._load_model()

    def _discretize_state(self, state: EMATradeState) -> str:
        """
        Convert continuous state to discrete key for Q-table lookup.

        REDESIGNED for Robinhood spot trading:
        - Added spread_quality bin (is this trade worth the spread cost?)
        - Added confluence bin (how many signals agree?)
        - Wider slope/distance bins for 4h timeframe
        - Removed hour_bin (crypto 24/7, hours don't matter as much)
        - Total states: 3 × 3 × 3 × 3 × 2 × 3 × 3 × 2 = 2,916 (manageable)
        """
        # 1. EMA SLOPE (3 bins — simplified from 5, 4h slopes are different scale)
        #    Uses absolute value — direction handled by signal, slope = strength
        abs_slope = abs(state.ema_slope)
        if abs_slope < 0.1:
            slope_bin = 0  # weak momentum
        elif abs_slope < 0.5:
            slope_bin = 1  # moderate momentum
        else:
            slope_bin = 2  # strong momentum

        # 2. PRICE-EMA DISTANCE (3 bins — how extended are we?)
        d = abs(state.price_ema_distance_atr)
        if d < 0.5:
            dist_bin = 0   # close to EMA (good entry)
        elif d < 1.5:
            dist_bin = 1   # moderate extension
        else:
            dist_bin = 2   # overextended (risky entry)

        # 3. SPREAD QUALITY — THE KEY BIN FOR ROBINHOOD (3 bins)
        #    move_to_spread_ratio: expected profit / spread cost
        #    < 1.5 = spread eats most profit = BAD
        #    1.5-3 = marginal = MAYBE
        #    > 3 = plenty of room = GOOD
        msr = state.move_to_spread_ratio
        if msr < 1.5:
            spread_bin = 0  # spread_killer (expected move barely covers spread)
        elif msr < 3.0:
            spread_bin = 1  # spread_marginal (possible but tight)
        else:
            spread_bin = 2  # spread_good (move >> spread cost)

        # 4. CONFLUENCE (3 bins — how many signals agree?)
        conf = state.confluence_count
        if conf < 3:
            confluence_bin = 0  # weak setup
        elif conf < 5:
            confluence_bin = 1  # moderate setup
        else:
            confluence_bin = 2  # strong multi-confluence

        # 5. HTF ALIGNMENT (2 bins)
        htf_bin = 1 if state.higher_tf_alignment > 0.3 else 0

        # 6. VOLATILITY (3 bins)
        if state.atr_percentile < 0.33:
            vol_bin = 0    # low vol (moves may be too small for spread)
        elif state.atr_percentile < 0.66:
            vol_bin = 1    # medium vol
        else:
            vol_bin = 2    # high vol (big moves possible)

        # 7. ENTRY SCORE QUALITY (3 bins)
        score = state.entry_score
        if score < 6:
            score_bin = 0  # weak
        elif score < 9:
            score_bin = 1  # decent
        else:
            score_bin = 2  # excellent

        # 8. RISK CONTEXT (2 bins — are we in a drawdown?)
        if state.consecutive_losses >= 2 or state.daily_pnl_pct < -2.0:
            risk_bin = 1   # elevated risk
        else:
            risk_bin = 0   # normal

        return f"{slope_bin}_{dist_bin}_{spread_bin}_{confluence_bin}_{htf_bin}_{vol_bin}_{score_bin}_{risk_bin}"

    def _get_q_values(self, state_key: str) -> np.ndarray:
        """
        Get Q-values for a state, initializing with SMART PRIORS if new.

        Prior design for Robinhood:
        - When spread_bin=0 (spread killer): bias toward SKIP/WAIT
        - When spread_bin=2 + confluence high: bias toward FULL_PATIENT/SNIPER
        - When score_bin=0 (weak): bias toward SKIP
        - Default: slight bias toward SMALL_PATIENT (conservative start)
        """
        if state_key not in self.q_table:
            init = np.zeros(self.n_actions)
            parts = state_key.split('_')

            # Parse key: slope_dist_spread_confluence_htf_vol_score_risk
            spread_bin = int(parts[2]) if len(parts) > 2 else 1
            confluence_bin = int(parts[3]) if len(parts) > 3 else 1
            score_bin = int(parts[6]) if len(parts) > 6 else 1
            risk_bin = int(parts[7]) if len(parts) > 7 else 0

            if spread_bin == 0:
                # Spread will eat the profit — strongly prefer SKIP/WAIT
                init[0] = 0.5   # SKIP
                init[1] = 0.3   # WAIT_1BAR
                init[2] = 0.2   # WAIT_PULLBACK
                init[11] = -0.5  # QUICK_SCALP penalty (worst action with high spread)
            elif spread_bin == 2 and confluence_bin >= 2:
                # Great setup: big expected move + strong confluence
                init[10] = 0.4  # SNIPER_MAX
                init[8] = 0.3   # FULL_VERY_PATIENT
                init[7] = 0.2   # FULL_WIDE_PATIENT
            elif score_bin == 0:
                # Weak score — prefer skip or tiny
                init[0] = 0.3   # SKIP
                init[3] = 0.1   # TINY_PATIENT
            else:
                # Default: slight bias toward patient entries (Robinhood needs patience)
                init[4] = 0.15  # SMALL_PATIENT
                init[7] = 0.10  # FULL_WIDE_PATIENT

            if risk_bin == 1:
                # In drawdown — extra bias toward skip/small
                init[0] += 0.2
                init[3] += 0.1
                init[9] -= 0.3   # AGGRESSIVE penalty
                init[10] -= 0.3  # SNIPER_MAX penalty in drawdown

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
        elif action.get('wait_bars', 0) > 0:
            self.stats['waits'] += 1
        else:
            self.stats['entries_skipped'] += 1

        # Build reasoning string with Robinhood-specific context
        reasoning_parts = [f"Action: {action['label']}"]
        reasoning_parts.append(f"slope={state.ema_slope:.2f} dist={state.price_ema_distance_atr:.2f}ATR")
        reasoning_parts.append(f"spread_ratio={state.move_to_spread_ratio:.1f}x confl={state.confluence_count}")
        reasoning_parts.append(f"score={state.entry_score} Q={q_values[action_idx]:+.3f} eps={self.epsilon:.3f}")
        if action.get('wait_bars', 0) > 0:
            reasoning_parts.append(f"WAIT {action['wait_bars']} bars")

        return RLDecision(
            enter_trade=action['enter'],
            quality_score=float(np.clip(q_values[action_idx], 0, 1)),
            position_size_mult=action['size'],
            sl_buffer_mult=action['sl'],
            patience_mult=action['patience'],
            reasoning=' | '.join(reasoning_parts),
            action_idx=action_idx,
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

        SPREAD-AWARE: Deducts spread cost from P&L before computing reward.
        Robinhood spread ~1.67% per side = ~3.3% round-trip. A +2% trade is
        actually -1.3% after spread. This teaches the RL agent to only enter
        trades where expected move >> spread cost.

        Args:
            trade_result: Dict with keys:
                'pnl_pct': float — P&L percentage (raw, before spread)
                'exit_type': str — 'ema_exit', 'sl', 'ratchet', 'time', 'skip_would_win', 'skip_would_lose'
                'hold_bars': int — how many bars held
                'was_skipped': bool — if RL said don't enter
                'spread_cost_pct': float — round-trip spread cost as % (e.g., 3.3 for Robinhood)
                'is_spot': bool — True for spot exchanges (no leverage), affects reward scaling
        """
        pnl = trade_result.get('pnl_pct', 0)
        exit_type = trade_result.get('exit_type', 'unknown')
        hold_bars = trade_result.get('hold_bars', 0)
        was_skipped = trade_result.get('was_skipped', False)
        spread_cost_pct = trade_result.get('spread_cost_pct', 0.0)
        is_spot = trade_result.get('is_spot', False)

        # SPREAD-ADJUSTED P&L: This is the REAL profit after paying spread
        # For Robinhood: pnl=2.0% - spread=3.3% = net=-1.3% (LOSS)
        # For Bybit: pnl=2.0% - spread=0.1% = net=1.9% (still a WIN)
        net_pnl = pnl - spread_cost_pct

        if was_skipped:
            # Reward for correctly skipping bad signals
            # Also apply spread to "would have" P&L (skip is even smarter on high-spread exchanges)
            would_have = trade_result.get('would_have_pnl', 0) - spread_cost_pct
            if would_have < -0.3:
                # Skipping bad trade on Robinhood is EXTRA valuable (saved 3.3% spread too)
                spread_bonus = min(spread_cost_pct * 0.3, 0.5) if spread_cost_pct > 0 else 0
                return 0.8 + spread_bonus  # Great skip (was 1.0 — too high, biased toward never trading)
            elif would_have > 2.0:
                return -1.5  # Missed a BIG winner after spread — serious penalty
            elif would_have > 0.5:
                return -0.5  # Missed a moderate winner — mild penalty (was -0.8, too harsh)
            else:
                return 0.1 if spread_cost_pct > 1.0 else 0.0  # Slight reward for skipping on high-spread

        # ── Spot market scaling: bigger moves needed, so scale patience bonus ──
        patience_scale = 1.5 if is_spot else 1.0  # Reward holding longer on spot

        # Actual trade rewards (using NET P&L after spread)
        if exit_type == 'ema_exit' and net_pnl > 0:
            # EMA exit in profit — the IDEAL outcome
            # Bonus for riding longer trends (spot needs more patience for bigger moves)
            ride_bonus = min(hold_bars / 50, 0.5) * patience_scale
            return 2.0 + net_pnl * 0.5 + ride_bonus

        elif exit_type == 'ratchet' and net_pnl > 0:
            # Locked in profit via ratchet — good but not ideal
            return 0.5 + net_pnl * 0.3

        elif exit_type == 'ema_exit' and net_pnl <= 0 and pnl > 0:
            # Profitable before spread but LOSING after spread — spread ate the profit
            # Strong penalty: teaches RL to only enter when move >> spread
            return -1.5  # Worse than a normal SL loss — wasted opportunity

        elif exit_type == 'sl' and net_pnl > -0.5:
            # Quick small loss — good risk management
            return -0.2

        elif exit_type == 'sl' and net_pnl <= -0.5:
            # Larger SL loss — spread makes it even worse
            return -1.0 + net_pnl * 0.3

        elif exit_type == 'hard_stop':
            # Hit hard stop — worst outcome (spread makes it catastrophic on spot)
            spread_penalty = min(spread_cost_pct * 0.2, 0.5)
            return -2.0 - spread_penalty

        elif exit_type == 'time' and net_pnl <= 0:
            # Stale loser closed by time — spread cost wasted
            return -0.5 - min(spread_cost_pct * 0.1, 0.3)

        else:
            # Default: proportional to NET P&L (after spread)
            return net_pnl * 0.5

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

        net_pnl = pnl - trade_result.get('spread_cost_pct', 0.0)
        if not trade_result.get('was_skipped', False):
            if net_pnl > 0:
                self.stats['wins'] += 1
            else:
                self.stats['losses'] += 1
            self.stats['total_pnl'] += pnl
            self.stats['total_net_pnl'] = self.stats.get('total_net_pnl', 0.0) + net_pnl

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

                # Ensure action_counts/action_pnl have all current action labels
                # (handles model version upgrades — old model may have fewer actions)
                for action in self.actions:
                    label = action['label']
                    if label not in self.stats['action_counts']:
                        self.stats['action_counts'][label] = 0
                    if label not in self.stats['action_pnl']:
                        self.stats['action_pnl'][label] = 0.0

                # Handle Q-table dimension mismatch (old model had 8 actions, new has 12)
                for key in list(self.q_table.keys()):
                    if len(self.q_table[key]) != self.n_actions:
                        # Discard old state — dimensions don't match
                        del self.q_table[key]

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
