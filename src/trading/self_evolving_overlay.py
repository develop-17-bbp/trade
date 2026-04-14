"""
Self-Evolving Overlay — Makes ALL Static Parts Adaptive
=========================================================
Sits on top of existing modules and adjusts their parameters
based on trade outcomes. No existing code is modified.

Evolves:
1. Risk thresholds — circuit breaker limits adjust based on recent drawdown
2. Agent weights — agents that predict correctly get more influence
3. LLM context — winning patterns fed back into prompt
4. Indicator parameters — EMA period, RSI threshold auto-tuned from backtest

All state persisted to data/evolution_state.json
"""

import json
import os
import time
import math
import logging
import threading
import re
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVOLUTION_STATE_FILE = os.path.join(PROJECT_ROOT, 'data', 'evolution_state.json')
GENETIC_RESULTS_FILE = os.path.join(PROJECT_ROOT, 'logs', 'genetic_evolution_results.json')
EVOLUTION_FEEDBACK_FILE = os.path.join(PROJECT_ROOT, 'data', 'evolution_feedback.json')
BACKTEST_RESULTS_FILE = os.path.join(PROJECT_ROOT, 'logs', 'strategy_backtest_results.json')
ADAPTIVE_STATE_FILE = os.path.join(PROJECT_ROOT, 'data', 'adaptive_state.json')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_load_json(path: str) -> Optional[Dict]:
    """Load a JSON file, returning None on any failure."""
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.debug(f"[EVOLVE] Could not load {path}: {e}")
    return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _streak(trades: List[Dict], want_win: bool) -> int:
    """Count consecutive wins (want_win=True) or losses from end of list."""
    count = 0
    for t in reversed(trades):
        if t.get('won', False) == want_win:
            count += 1
        else:
            break
    return count


# ===========================================================================
# 1. RiskEvolver
# ===========================================================================

class RiskEvolver:
    """Auto-adjusts risk parameters based on recent performance.

    Produces override dicts that the executor applies ON TOP of the
    existing DynamicRiskManager / RiskLimits defaults.  Nothing in the
    original risk module is patched — the executor just multiplies its
    limits by the factors returned here.
    """

    # Sane bounds so evolution never goes crazy
    MIN_SIZE_MULT = 0.25
    MAX_SIZE_MULT = 1.5
    MIN_SL_MULT = 0.5       # stops can tighten to 50 %
    MAX_SL_MULT = 1.3       # stops can widen to 130 %
    MIN_HEAT_MULT = 0.4
    MAX_HEAT_MULT = 1.2

    def __init__(self):
        self._overrides: Dict[str, float] = {
            'position_size_mult': 1.0,
            'stop_loss_mult': 1.0,
            'max_daily_loss_mult': 1.0,
            'max_drawdown_limit_mult': 1.0,
            'portfolio_heat_mult': 1.0,
            'kill_switch_mult': 1.0,
        }
        self._recent_trades: deque = deque(maxlen=100)

    # ── public API ──

    def record_trade(self, trade: Dict) -> None:
        self._recent_trades.append(trade)

    def compute_adaptive_risk(
        self,
        recent_trades: Optional[List[Dict]] = None,
        current_drawdown: float = 0.0,
    ) -> Dict[str, float]:
        """Return adjusted risk multipliers.

        * Winning streak (3+): slightly increase position size (1.1x)
        * Losing streak (3+): reduce position size (0.5x)
        * Drawdown > 5 %: tighten all stops, reduce max position
        * Drawdown < 1 % and winning: slightly loosen limits
        """
        trades = recent_trades or list(self._recent_trades)
        if not trades:
            return dict(self._overrides)

        last_20 = trades[-20:]
        wins = sum(1 for t in last_20 if t.get('won'))
        wr = wins / len(last_20) if last_20 else 0.5

        win_streak = _streak(last_20, want_win=True)
        lose_streak = _streak(last_20, want_win=False)

        # ── Position sizing ──
        if lose_streak >= 5:
            size_mult = 0.3
        elif lose_streak >= 3:
            size_mult = 0.5
        elif win_streak >= 5:
            size_mult = 1.2
        elif win_streak >= 3:
            size_mult = 1.1
        else:
            size_mult = 0.8 + wr * 0.4   # 0.8 .. 1.2

        # ── Drawdown adjustments ──
        dd = abs(current_drawdown)
        if dd > 0.08:
            # Severe drawdown — aggressive protection
            size_mult *= 0.4
            sl_mult = 0.6        # very tight stops
            heat_mult = 0.4
            dd_limit_mult = 0.8  # lower drawdown ceiling
            kill_mult = 0.8      # earlier kill switch
        elif dd > 0.05:
            size_mult *= 0.6
            sl_mult = 0.75
            heat_mult = 0.6
            dd_limit_mult = 0.9
            kill_mult = 0.9
        elif dd < 0.01 and wr > 0.55:
            # Healthy + winning — slight relaxation
            sl_mult = 1.1
            heat_mult = 1.1
            dd_limit_mult = 1.0
            kill_mult = 1.0
        else:
            sl_mult = 1.0
            heat_mult = 1.0
            dd_limit_mult = 1.0
            kill_mult = 1.0

        # ── Daily loss limit ──
        daily_loss_mult = 1.0
        if lose_streak >= 3:
            daily_loss_mult = 0.7   # tighten daily loss limit
        elif wr > 0.6 and dd < 0.02:
            daily_loss_mult = 1.1

        self._overrides = {
            'position_size_mult': _clamp(size_mult, self.MIN_SIZE_MULT, self.MAX_SIZE_MULT),
            'stop_loss_mult': _clamp(sl_mult, self.MIN_SL_MULT, self.MAX_SL_MULT),
            'max_daily_loss_mult': _clamp(daily_loss_mult, 0.5, 1.3),
            'max_drawdown_limit_mult': _clamp(dd_limit_mult, 0.7, 1.2),
            'portfolio_heat_mult': _clamp(heat_mult, self.MIN_HEAT_MULT, self.MAX_HEAT_MULT),
            'kill_switch_mult': _clamp(kill_mult, 0.7, 1.1),
        }
        return dict(self._overrides)

    def get_current_overrides(self) -> Dict[str, float]:
        return dict(self._overrides)

    def to_dict(self) -> Dict:
        return {
            'overrides': dict(self._overrides),
            'recent_trades_count': len(self._recent_trades),
        }

    def from_dict(self, d: Dict) -> None:
        if d and 'overrides' in d:
            self._overrides.update(d['overrides'])


# ===========================================================================
# 2. AgentEvolver
# ===========================================================================

# Agents known to be structurally unreliable (always mark down)
_ALWAYS_PENALISED = {'polymarket_arb', 'PolymarketArbitrageAgent'}

# Bounds for weight multipliers
_AGENT_WEIGHT_MIN = 0.01
_AGENT_WEIGHT_MAX = 3.0
_MIN_SAMPLES = 5         # need at least N votes before adjusting


class AgentEvolver:
    """Auto-adjusts agent trust weights based on prediction accuracy.

    For each agent we track how many times the agent's directional vote
    matched the realised trade outcome.  Agents that exceed 60 % accuracy
    get boosted; those below 40 % get dampened.  PolymarketAgent (known to
    always fail because the API has been shut down) is pinned to 0.01.
    """

    def __init__(self):
        # {agent_name: {'correct': int, 'total': int}}
        self._history: Dict[str, Dict[str, int]] = {}
        self._weights: Dict[str, float] = {}

    # ── public API ──

    def record_votes(self, agent_votes: Dict[str, Any], won: bool, direction: int) -> None:
        """Record how each agent voted relative to the realised outcome.

        Args:
            agent_votes: {agent_name: vote_dict_or_value}
            won: whether the trade was profitable
            direction: +1 for LONG, -1 for SHORT
        """
        for name, vote in agent_votes.items():
            if name not in self._history:
                self._history[name] = {'correct': 0, 'total': 0}
            h = self._history[name]
            h['total'] += 1

            vote_dir = vote.get('direction', 0) if isinstance(vote, dict) else 0
            # correct = voted same direction AND won, OR voted opposite AND lost
            if (vote_dir == direction and won) or (vote_dir != direction and not won):
                h['correct'] += 1

    def compute_agent_weights(
        self,
        agent_accuracy_history: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, float]:
        """Return {agent_name: weight_multiplier}.

        * >60 % accuracy  =>  weight *= 1.3
        * <40 % accuracy  =>  weight *= 0.7
        * structurally broken agents => 0.01
        """
        history = agent_accuracy_history or self._history
        weights: Dict[str, float] = {}

        for name, h in history.items():
            # Structurally broken agents
            if name in _ALWAYS_PENALISED:
                weights[name] = _AGENT_WEIGHT_MIN
                continue

            total = h.get('total', 0)
            if total < _MIN_SAMPLES:
                weights[name] = 1.0
                continue

            accuracy = h.get('correct', 0) / total

            if accuracy > 0.70:
                mult = 1.5
            elif accuracy > 0.60:
                mult = 1.3
            elif accuracy < 0.30:
                mult = 0.5
            elif accuracy < 0.40:
                mult = 0.7
            else:
                mult = 1.0

            weights[name] = _clamp(mult, _AGENT_WEIGHT_MIN, _AGENT_WEIGHT_MAX)

        self._weights = weights
        return dict(weights)

    def get_current_weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def to_dict(self) -> Dict:
        return {
            'history': {k: dict(v) for k, v in self._history.items()},
            'weights': dict(self._weights),
        }

    def from_dict(self, d: Dict) -> None:
        if not d:
            return
        self._history = {k: dict(v) for k, v in d.get('history', {}).items()}
        self._weights = dict(d.get('weights', {}))


# ===========================================================================
# 3. LLMEvolver
# ===========================================================================

class LLMEvolver:
    """Evolves LLM context based on trade outcomes.

    Instead of rewriting frozen system prompts, we generate an
    ADAPTIVE CONTEXT block that gets injected into the user-message
    portion of the LLM call.  The block contains:
      - Winning-trade patterns (from winner DNA)
      - Losing-trade patterns (from loser DNA)
      - Current regime performance stats
      - Strategy profitability rankings
      - Confidence calibration warning (LLM said 80 % but WR was 45 %)
    """

    def __init__(self):
        self._current_context: str = ''
        self._performance_window: deque = deque(maxlen=50)
        self._confidence_buckets: Dict[str, Dict[str, int]] = {}
        self._strategy_stats: Dict[str, Dict[str, float]] = {}

    # ── public API ──

    def record_trade(self, trade: Dict) -> None:
        self._performance_window.append(trade)
        # confidence calibration
        conf = trade.get('llm_confidence', 0.0)
        bucket = f"{int(conf * 10) / 10:.1f}"
        if bucket not in self._confidence_buckets:
            self._confidence_buckets[bucket] = {'wins': 0, 'total': 0}
        self._confidence_buckets[bucket]['total'] += 1
        if trade.get('won'):
            self._confidence_buckets[bucket]['wins'] += 1
        # strategy stats
        strat = trade.get('strategy', 'unknown')
        if strat not in self._strategy_stats:
            self._strategy_stats[strat] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
        ss = self._strategy_stats[strat]
        if trade.get('won'):
            ss['wins'] += 1
        else:
            ss['losses'] += 1
        ss['total_pnl'] += trade.get('pnl_pct', 0.0)

    def build_evolved_context(
        self,
        winner_dna: Optional[Dict] = None,
        loser_dna: Optional[Dict] = None,
        recent_performance: Optional[List[Dict]] = None,
    ) -> str:
        """Build an adaptive context block for injection into the LLM prompt."""
        lines: List[str] = ['--- ADAPTIVE INTELLIGENCE CONTEXT (auto-evolved) ---']
        perf = recent_performance or list(self._performance_window)

        # 1. Overall stats
        if perf:
            last_n = perf[-20:]
            wins = sum(1 for t in last_n if t.get('won'))
            total = len(last_n)
            wr = wins / total if total else 0
            total_pnl = sum(t.get('pnl_pct', 0) for t in last_n)
            lines.append(f"RECENT PERFORMANCE ({total} trades): WR={wr:.0%} PnL={total_pnl:+.2f}%")
        else:
            lines.append("RECENT PERFORMANCE: No trades yet.")

        # 2. Winner DNA
        lines.append('')
        if winner_dna and winner_dna.get('entry_score', {}).get('count', 0) > 0:
            lines.append("WINNING TRADE PATTERNS (replicate these):")
            for key in ['entry_score', 'llm_confidence', 'risk_score', 'trade_quality', 'hurst', 'duration_min']:
                info = winner_dna.get(key, {})
                if info.get('count', 0) > 0:
                    avg = info['sum'] / info['count']
                    lines.append(f"  avg {key} = {avg:.2f} (n={info['count']})")
            # top regime/strategy for winners
            regimes = winner_dna.get('regimes', {})
            if regimes:
                top = max(regimes, key=regimes.get)
                lines.append(f"  best regime for winners: {top} ({regimes[top]} trades)")
            strats = winner_dna.get('strategies', {})
            if strats:
                top = max(strats, key=strats.get)
                lines.append(f"  best strategy for winners: {top} ({strats[top]} trades)")
        else:
            lines.append("WINNING TRADE PATTERNS: Not enough data yet.")

        # 3. Loser DNA
        lines.append('')
        if loser_dna and loser_dna.get('entry_score', {}).get('count', 0) > 0:
            lines.append("LOSING TRADE PATTERNS (avoid these):")
            for key in ['entry_score', 'llm_confidence', 'risk_score', 'trade_quality', 'hurst', 'duration_min']:
                info = loser_dna.get(key, {})
                if info.get('count', 0) > 0:
                    avg = info['sum'] / info['count']
                    lines.append(f"  avg {key} = {avg:.2f} (n={info['count']})")
            regimes = loser_dna.get('regimes', {})
            if regimes:
                top = max(regimes, key=regimes.get)
                lines.append(f"  worst regime for losers: {top} ({regimes[top]} trades)")
        else:
            lines.append("LOSING TRADE PATTERNS: Not enough data yet.")

        # 4. Strategy profitability rankings
        lines.append('')
        ranked = sorted(
            self._strategy_stats.items(),
            key=lambda kv: kv[1]['total_pnl'],
            reverse=True,
        )
        if ranked:
            lines.append("STRATEGY RANKINGS (by PnL):")
            for strat, ss in ranked[:8]:
                total = ss['wins'] + ss['losses']
                wr_s = ss['wins'] / total if total else 0
                lines.append(
                    f"  {strat}: {wr_s:.0%} WR, {ss['total_pnl']:+.2f}% PnL ({total} trades)"
                )

        # 5. Confidence calibration warning
        lines.append('')
        cal_warnings: List[str] = []
        for bucket, cc in sorted(self._confidence_buckets.items()):
            if cc['total'] >= 3:
                actual_wr = cc['wins'] / cc['total']
                stated_conf = float(bucket)
                gap = stated_conf - actual_wr
                if gap > 0.15:
                    cal_warnings.append(
                        f"  OVERCONFIDENT at {bucket}: LLM says {stated_conf:.0%} but actual WR={actual_wr:.0%} (gap={gap:+.0%})"
                    )
                elif gap < -0.15:
                    cal_warnings.append(
                        f"  UNDERCONFIDENT at {bucket}: LLM says {stated_conf:.0%} but actual WR={actual_wr:.0%} (gap={gap:+.0%})"
                    )
        if cal_warnings:
            lines.append("CONFIDENCE CALIBRATION WARNINGS:")
            lines.extend(cal_warnings)
        else:
            lines.append("CONFIDENCE CALIBRATION: In range (no significant drift).")

        lines.append('--- END ADAPTIVE CONTEXT ---')
        self._current_context = '\n'.join(lines)
        return self._current_context

    def get_current_context(self) -> str:
        return self._current_context

    def to_dict(self) -> Dict:
        return {
            'confidence_buckets': dict(self._confidence_buckets),
            'strategy_stats': dict(self._strategy_stats),
            'current_context': self._current_context,
        }

    def from_dict(self, d: Dict) -> None:
        if not d:
            return
        self._confidence_buckets = d.get('confidence_buckets', {})
        self._strategy_stats = d.get('strategy_stats', {})
        self._current_context = d.get('current_context', '')


# ===========================================================================
# 4. IndicatorEvolver
# ===========================================================================

# Default indicator params (mirrors what indicators.py uses today)
DEFAULT_INDICATOR_PARAMS = {
    'ema_fast': 8,
    'ema_slow': 21,
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2.0,
    'atr_period': 14,
    'stoch_k': 14,
    'stoch_d': 3,
    'lookback_window': 20,
}

# Hard min/max so genetic weirdness can't produce nonsense
_PARAM_BOUNDS = {
    'ema_fast':       (3, 30),
    'ema_slow':       (10, 60),
    'rsi_period':     (5, 30),
    'rsi_oversold':   (15, 40),
    'rsi_overbought': (60, 90),
    'macd_fast':      (5, 20),
    'macd_slow':      (15, 50),
    'macd_signal':    (3, 15),
    'bb_period':      (10, 40),
    'bb_std':         (1.0, 3.5),
    'atr_period':     (5, 30),
    'stoch_k':        (5, 30),
    'stoch_d':        (2, 8),
    'lookback_window': (8, 50),
}


def _parse_strategy_params(strategy_name: str) -> Dict[str, Any]:
    """Extract numeric params embedded in strategy names like U_RSI_21_20_80."""
    parts = strategy_name.split('_')
    params: Dict[str, Any] = {}

    name_lower = strategy_name.lower()

    # RSI variants: U_RSI_<period>_<oversold>_<overbought>
    if 'rsi' in name_lower and len(parts) >= 4:
        try:
            nums = [p for p in parts if p.replace('.', '').isdigit()]
            if len(nums) >= 3:
                params['rsi_period'] = int(nums[0])
                params['rsi_oversold'] = int(nums[1])
                params['rsi_overbought'] = int(nums[2])
            elif len(nums) >= 1:
                params['rsi_period'] = int(nums[0])
        except (ValueError, IndexError):
            pass

    # OBV_EMA variants: U_OBV_EMA_<lookback>_<fast>_<slow>
    if 'obv' in name_lower and 'ema' in name_lower:
        try:
            nums = [p for p in parts if p.replace('.', '').isdigit()]
            if len(nums) >= 3:
                params['lookback_window'] = int(nums[0])
                params['ema_fast'] = int(nums[1])
                params['ema_slow'] = int(nums[2])
        except (ValueError, IndexError):
            pass

    # EMA variants: extract fast/slow
    if 'ema' in name_lower and 'obv' not in name_lower:
        try:
            nums = [int(p) for p in parts if p.isdigit()]
            if len(nums) >= 2:
                params['ema_fast'] = min(nums[0], nums[1])
                params['ema_slow'] = max(nums[0], nums[1])
            elif len(nums) == 1:
                params['ema_fast'] = nums[0]
        except (ValueError, IndexError):
            pass

    # MACD variants
    if 'macd' in name_lower:
        try:
            nums = [int(p) for p in parts if p.isdigit()]
            if len(nums) >= 3:
                params['macd_fast'] = nums[0]
                params['macd_slow'] = nums[1]
                params['macd_signal'] = nums[2]
        except (ValueError, IndexError):
            pass

    return params


class IndicatorEvolver:
    """Auto-tunes indicator parameters from genetic evolution + backtest results.

    Reads the hall-of-fame from genetic evolution and strategy backtest
    rankings.  Winning strategies' embedded parameters (e.g. RSI_21_25_75)
    are extracted, weighted by their composite score, and blended into
    optimal parameter recommendations.
    """

    def __init__(self):
        self._optimal: Dict[str, Any] = dict(DEFAULT_INDICATOR_PARAMS)
        self._source: str = 'defaults'

    # ── public API ──

    def get_optimal_params(
        self,
        genetic_hall_of_fame: Optional[List[Dict]] = None,
        backtest_results: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Compute optimal indicator params from evolution + backtest data.

        Priority:
        1. Genetic hall-of-fame winners (if any exist with positive fitness)
        2. Top-ranked backtest strategies (parsed from strategy names)
        3. Defaults
        """
        candidates: List[Tuple[Dict[str, Any], float]] = []  # (params, weight)

        # --- Genetic hall-of-fame ---
        hof = genetic_hall_of_fame
        if hof is None:
            raw = _safe_load_json(GENETIC_RESULTS_FILE)
            if raw:
                hof = raw.get('hall_of_fame', [])
        if hof:
            for individual in hof:
                fitness = individual.get('fitness', 0)
                if fitness <= 0:
                    continue
                params = individual.get('params', individual.get('genes', {}))
                if params:
                    # Validated entries (cross-referenced with live trades) get 2x weight
                    weight = fitness
                    if individual.get('validated', False):
                        weight *= 2.0
                    candidates.append((params, weight))

        # --- Evolution feedback (validated entries from adaptive feedback loop) ---
        evo_feedback = _safe_load_json(EVOLUTION_FEEDBACK_FILE)
        if evo_feedback:
            fb_hof = evo_feedback.get('hall_of_fame', [])
            for individual in fb_hof:
                fitness = individual.get('fitness', 0)
                if fitness <= 0:
                    continue
                params = individual.get('params', individual.get('genes', {}))
                if params:
                    weight = fitness
                    if individual.get('validated', False):
                        weight *= 3.0  # Validated entries from feedback get 3x weight
                    candidates.append((params, weight))

        # --- Backtest rankings ---
        bt = backtest_results
        if bt is None:
            bt = _safe_load_json(BACKTEST_RESULTS_FILE)
        if bt:
            for asset_key, asset_data in bt.get('assets', {}).items():
                rankings = asset_data if isinstance(asset_data, list) else asset_data.get('rankings', [])
                for entry in rankings:
                    metrics = entry.get('metrics', {})
                    strat_name = entry.get('strategy', '')
                    score = metrics.get('composite_score', 0)
                    total_pnl = metrics.get('total_pnl', 0)
                    win_rate = metrics.get('win_rate', 0)

                    # Only learn from strategies with positive PnL and decent WR
                    if total_pnl <= 0 or win_rate < 0.4:
                        continue

                    parsed = _parse_strategy_params(strat_name)
                    if parsed:
                        weight = score * (1.0 + total_pnl / 10.0)
                        candidates.append((parsed, weight))

        if not candidates:
            self._source = 'defaults'
            return dict(self._optimal)

        # --- Weighted blend ---
        total_weight = sum(w for _, w in candidates)
        if total_weight <= 0:
            self._source = 'defaults'
            return dict(self._optimal)

        blended: Dict[str, float] = {}
        counts: Dict[str, float] = {}
        for params, weight in candidates:
            for k, v in params.items():
                if k in DEFAULT_INDICATOR_PARAMS:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    blended[k] = blended.get(k, 0.0) + fv * weight
                    counts[k] = counts.get(k, 0.0) + weight

        optimal = dict(DEFAULT_INDICATOR_PARAMS)
        for k in blended:
            if counts.get(k, 0) > 0:
                raw_val = blended[k] / counts[k]
                lo, hi = _PARAM_BOUNDS.get(k, (raw_val, raw_val))
                # integer params stay integer
                if isinstance(DEFAULT_INDICATOR_PARAMS.get(k), int):
                    optimal[k] = int(_clamp(round(raw_val), lo, hi))
                else:
                    optimal[k] = round(_clamp(raw_val, lo, hi), 2)

        # Sanity: ema_fast < ema_slow, macd_fast < macd_slow, rsi_oversold < rsi_overbought
        if optimal['ema_fast'] >= optimal['ema_slow']:
            optimal['ema_fast'] = max(3, optimal['ema_slow'] - 5)
        if optimal['macd_fast'] >= optimal['macd_slow']:
            optimal['macd_fast'] = max(5, optimal['macd_slow'] - 5)
        if optimal['rsi_oversold'] >= optimal['rsi_overbought']:
            optimal['rsi_oversold'] = max(15, optimal['rsi_overbought'] - 20)

        self._optimal = optimal
        self._source = f'evolved({len(candidates)} candidates)'
        return dict(self._optimal)

    def get_current_params(self) -> Dict[str, Any]:
        return dict(self._optimal)

    def get_source(self) -> str:
        return self._source

    def to_dict(self) -> Dict:
        return {
            'optimal': dict(self._optimal),
            'source': self._source,
        }

    def from_dict(self, d: Dict) -> None:
        if not d:
            return
        saved = d.get('optimal', {})
        # only load params that are in our schema
        for k in DEFAULT_INDICATOR_PARAMS:
            if k in saved:
                self._optimal[k] = saved[k]
        self._source = d.get('source', 'loaded')


# ===========================================================================
# 5. SelfEvolvingOverlay (master coordinator)
# ===========================================================================

class SelfEvolvingOverlay:
    """Master coordinator — updates ALL evolving parameters after each trade.

    Usage::

        overlay = SelfEvolvingOverlay()

        # After every trade close:
        overlay.update_all(trade_outcome, agent_votes, regime)

        # Before every new trade decision:
        overrides = overlay.get_overrides()
        # overrides['risk']            -> dict of risk multipliers
        # overrides['agent_weights']   -> dict of {agent: weight}
        # overrides['llm_context']     -> string to inject into LLM prompt
        # overrides['indicator_params']-> dict of optimal indicator values
    """

    def __init__(self, state_file: Optional[str] = None):
        self._lock = threading.Lock()
        self._state_file = state_file or EVOLUTION_STATE_FILE

        self.risk_evolver = RiskEvolver()
        self.agent_evolver = AgentEvolver()
        self.llm_evolver = LLMEvolver()
        self.indicator_evolver = IndicatorEvolver()

        self._trade_count = 0
        self._last_update: Optional[str] = None

        # Load persisted state
        self.load_state()

        # Pre-compute indicator params from files on disk
        try:
            self.indicator_evolver.get_optimal_params()
        except Exception as e:
            logger.debug(f"[EVOLVE] Initial indicator param load: {e}")

        logger.info(
            f"[EVOLVE] SelfEvolvingOverlay ACTIVE | "
            f"trades={self._trade_count} | "
            f"indicator_source={self.indicator_evolver.get_source()}"
        )

    # ── Core update (called after every trade close) ──

    def update_all(
        self,
        trade_outcome: Dict[str, Any],
        agent_votes: Optional[Dict[str, Any]] = None,
        regime: str = 'UNKNOWN',
        current_drawdown: float = 0.0,
        winner_dna: Optional[Dict] = None,
        loser_dna: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Called after every trade close — updates all evolvers.

        Args:
            trade_outcome: dict with keys: won, pnl_pct, pnl_usd, asset,
                           strategy, llm_confidence, entry_score, exit_reason, etc.
            agent_votes: {agent_name: vote_dict}
            regime: current market regime string
            current_drawdown: portfolio drawdown as fraction (0.05 = 5 %)
            winner_dna: accumulated winner DNA from AdaptiveFeedbackLoop
            loser_dna: accumulated loser DNA from AdaptiveFeedbackLoop

        Returns:
            The full overrides dict (same as get_overrides).
        """
        with self._lock:
            self._trade_count += 1
            self._last_update = datetime.now(tz=timezone.utc).isoformat()
            won = trade_outcome.get('won', False)
            direction = 1 if trade_outcome.get('direction', 'LONG') == 'LONG' else -1

            # 1. Risk evolution
            self.risk_evolver.record_trade(trade_outcome)
            self.risk_evolver.compute_adaptive_risk(
                current_drawdown=current_drawdown,
            )

            # 2. Agent weight evolution
            if agent_votes:
                self.agent_evolver.record_votes(agent_votes, won, direction)
                self.agent_evolver.compute_agent_weights()

            # 3. LLM context evolution
            self.llm_evolver.record_trade(trade_outcome)
            self.llm_evolver.build_evolved_context(
                winner_dna=winner_dna,
                loser_dna=loser_dna,
            )

            # 4. Indicator evolution (re-read files every 20 trades)
            if self._trade_count % 20 == 0:
                try:
                    self.indicator_evolver.get_optimal_params()
                    logger.info(
                        f"[EVOLVE] Indicator params refreshed: "
                        f"source={self.indicator_evolver.get_source()}"
                    )
                except Exception as e:
                    logger.warning(f"[EVOLVE] Indicator refresh failed: {e}")

            # 5. Persist
            self.save_state()

            overrides = self._build_overrides()

            logger.info(
                f"[EVOLVE] Trade #{self._trade_count} | "
                f"{'WIN' if won else 'LOSS'} {trade_outcome.get('asset','?')} "
                f"{trade_outcome.get('pnl_pct',0):+.2f}% | "
                f"risk_size={overrides['risk'].get('position_size_mult',1):.2f} "
                f"agents_evolved={len(overrides['agent_weights'])} "
                f"indicator_src={self.indicator_evolver.get_source()}"
            )

            return overrides

    # ── Query current overrides ──

    def get_overrides(self) -> Dict[str, Any]:
        """Returns all current parameter overrides for executor to apply."""
        with self._lock:
            return self._build_overrides()

    def _build_overrides(self) -> Dict[str, Any]:
        return {
            'risk': self.risk_evolver.get_current_overrides(),
            'agent_weights': self.agent_evolver.get_current_weights(),
            'llm_context': self.llm_evolver.get_current_context(),
            'indicator_params': self.indicator_evolver.get_current_params(),
            'meta': {
                'trade_count': self._trade_count,
                'last_update': self._last_update,
                'indicator_source': self.indicator_evolver.get_source(),
            },
        }

    # ── Persistence ──

    def save_state(self) -> None:
        """Persist evolution state to disk (atomic write)."""
        try:
            state = {
                'version': 1,
                'trade_count': self._trade_count,
                'last_update': self._last_update,
                'risk': self.risk_evolver.to_dict(),
                'agents': self.agent_evolver.to_dict(),
                'llm': self.llm_evolver.to_dict(),
                'indicators': self.indicator_evolver.to_dict(),
                'saved_at': datetime.now(tz=timezone.utc).isoformat(),
            }
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            tmp = self._state_file + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp, self._state_file)
        except Exception as e:
            logger.warning(f"[EVOLVE] State save failed: {e}")

    def load_state(self) -> None:
        """Load evolution state from disk."""
        state = _safe_load_json(self._state_file)
        if not state:
            # Try loading partial data from the adaptive_state used by feedback loop
            adaptive = _safe_load_json(ADAPTIVE_STATE_FILE)
            if adaptive:
                self._bootstrap_from_adaptive(adaptive)
            return

        try:
            self._trade_count = state.get('trade_count', 0)
            self._last_update = state.get('last_update')
            self.risk_evolver.from_dict(state.get('risk', {}))
            self.agent_evolver.from_dict(state.get('agents', {}))
            self.llm_evolver.from_dict(state.get('llm', {}))
            self.indicator_evolver.from_dict(state.get('indicators', {}))
            logger.info(
                f"[EVOLVE] Loaded state: {self._trade_count} trades, "
                f"last_update={self._last_update}"
            )
        except Exception as e:
            logger.warning(f"[EVOLVE] State load partially failed: {e}")

    def _bootstrap_from_adaptive(self, adaptive: Dict) -> None:
        """Seed evolver state from the existing adaptive_state.json (one-time migration)."""
        try:
            # Agent accuracy -> AgentEvolver
            agent_acc = adaptive.get('agent_accuracy', {})
            if agent_acc:
                history = {}
                for name, data in agent_acc.items():
                    history[name] = {
                        'correct': data.get('correct', 0),
                        'total': data.get('total', 0),
                    }
                self.agent_evolver.from_dict({'history': history, 'weights': {}})
                self.agent_evolver.compute_agent_weights()

            # Confidence calibration -> LLMEvolver
            conf_cal = adaptive.get('confidence_calibration', {})
            if conf_cal:
                self.llm_evolver._confidence_buckets = dict(conf_cal)

            # Strategy performance -> LLMEvolver strategy stats
            strat_perf = adaptive.get('strategy_performance', {})
            if strat_perf:
                for strat, sp in strat_perf.items():
                    self.llm_evolver._strategy_stats[strat] = {
                        'wins': sp.get('wins', 0),
                        'losses': sp.get('losses', 0),
                        'total_pnl': sp.get('total_pnl', 0.0),
                    }

            # Winner/Loser DNA -> LLMEvolver context
            winner_dna = adaptive.get('winner_dna', {})
            loser_dna = adaptive.get('loser_dna', {})
            if winner_dna or loser_dna:
                self.llm_evolver.build_evolved_context(
                    winner_dna=winner_dna,
                    loser_dna=loser_dna,
                )

            total = adaptive.get('total_trades_processed', 0)
            self._trade_count = total
            logger.info(
                f"[EVOLVE] Bootstrapped from adaptive_state.json: "
                f"{total} trades, {len(agent_acc)} agents"
            )
        except Exception as e:
            logger.warning(f"[EVOLVE] Bootstrap from adaptive state failed: {e}")

    # ── Convenience: summary for logging / dashboard ──

    def summary(self) -> str:
        """Human-readable summary of current evolution state."""
        o = self.get_overrides()
        risk = o['risk']
        aw = o['agent_weights']
        ip = o['indicator_params']

        lines = [
            f"=== Self-Evolving Overlay (trade #{self._trade_count}) ===",
            f"Risk overrides: size={risk.get('position_size_mult',1):.2f}x "
            f"SL={risk.get('stop_loss_mult',1):.2f}x "
            f"heat={risk.get('portfolio_heat_mult',1):.2f}x "
            f"kill={risk.get('kill_switch_mult',1):.2f}x",
        ]
        if aw:
            top_agents = sorted(aw.items(), key=lambda kv: kv[1], reverse=True)[:5]
            agent_str = ', '.join(f"{n}={w:.2f}" for n, w in top_agents)
            lines.append(f"Top agent weights: {agent_str}")

        changed = {k: v for k, v in ip.items() if v != DEFAULT_INDICATOR_PARAMS.get(k)}
        if changed:
            param_str = ', '.join(f"{k}={v}" for k, v in changed.items())
            lines.append(f"Evolved indicators: {param_str}")
        else:
            lines.append("Indicators: defaults (no evolution data yet)")

        lines.append(f"Indicator source: {self.indicator_evolver.get_source()}")
        return '\n'.join(lines)
