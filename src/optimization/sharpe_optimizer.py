"""
Sharpe Ratio Optimizer — ACT v8.0
Target: Sharpe 1.5+, Sortino 2.0+, Calmar 1.5+

Three modes:
  RECOVERY   (Sharpe < 0.8): tighten all filters, reduce size
  NORMAL     (0.8 <= Sharpe <= 1.5): standard operation
  MOMENTUM   (Sharpe > 1.5): slight size increase allowed
"""
import time
import math
import logging
from typing import List, Dict, Optional
from collections import deque

logger = logging.getLogger(__name__)


class SharpeOptimizer:
    def __init__(self, config: dict = None):
        cfg = config or {}
        self.target_sharpe = cfg.get('target', 1.5)
        self.recovery_threshold = cfg.get('recovery_mode_threshold', 0.8)
        self.momentum_threshold = cfg.get('momentum_mode_threshold', 1.5)
        self.min_quality_score = cfg.get('min_trade_quality_score', 55)
        self.rolling_window = cfg.get('rolling_window_trades', 20)

        # Rolling trade returns
        self._returns: deque = deque(maxlen=200)
        self._trade_log: List[Dict] = []
        self._current_mode = 'NORMAL'

        # Session-level tracking
        self._session_returns: Dict[str, List[float]] = {}
        self._regime_returns: Dict[str, List[float]] = {}
        self._day_of_week_returns: Dict[int, List[float]] = {}

    def record_trade(self, pnl_pct: float, session: str = 'unknown',
                     regime: str = 'unknown', day_of_week: int = 0,
                     quality_score: int = 50):
        self._returns.append(pnl_pct)
        self._trade_log.append({
            'pnl_pct': pnl_pct, 'session': session, 'regime': regime,
            'day': day_of_week, 'quality': quality_score, 'time': time.time(),
        })
        self._session_returns.setdefault(session, []).append(pnl_pct)
        self._regime_returns.setdefault(regime, []).append(pnl_pct)
        self._day_of_week_returns.setdefault(day_of_week, []).append(pnl_pct)
        self._update_mode()

    def _update_mode(self):
        sharpe = self.get_rolling_sharpe()
        if sharpe is None:
            self._current_mode = 'NORMAL'
        elif sharpe < self.recovery_threshold:
            self._current_mode = 'RECOVERY'
        elif sharpe > self.momentum_threshold:
            self._current_mode = 'MOMENTUM'
        else:
            self._current_mode = 'NORMAL'

    def get_rolling_sharpe(self, window: int = None) -> Optional[float]:
        w = window or self.rolling_window
        returns = list(self._returns)[-w:]
        if len(returns) < 5:
            return None
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = math.sqrt(var) if var > 0 else 0.001
        return round(mean / std * math.sqrt(252), 3)  # annualized

    def get_sortino_ratio(self, window: int = None) -> Optional[float]:
        w = window or self.rolling_window
        returns = list(self._returns)[-w:]
        if len(returns) < 5:
            return None
        mean = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return 5.0  # no downside = excellent
        downside_var = sum(r ** 2 for r in downside) / len(downside)
        downside_std = math.sqrt(downside_var) if downside_var > 0 else 0.001
        return round(mean / downside_std * math.sqrt(252), 3)

    def get_calmar_ratio(self) -> Optional[float]:
        returns = list(self._returns)
        if len(returns) < 10:
            return None
        total_return = sum(returns)
        peak = 0
        max_dd = 0.001
        cumulative = 0
        for r in returns:
            cumulative += r
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return round(total_return / max_dd, 3) if max_dd > 0 else 0

    @property
    def mode(self) -> str:
        return self._current_mode

    def get_filter_adjustments(self) -> dict:
        """Returns filter adjustments based on current Sharpe mode."""
        if self._current_mode == 'RECOVERY':
            return {
                'min_entry_score_add': 2,      # 6 → 8
                'min_confluence_add': 2,        # 5 → 7
                'position_size_mult': 0.6,      # reduce 40%
                'session_filter': 'prime_only',  # UTC 13-21 only
                'reason': f'Sharpe Recovery (rolling={self.get_rolling_sharpe()})',
            }
        elif self._current_mode == 'MOMENTUM':
            return {
                'min_entry_score_add': 0,
                'min_confluence_add': 0,
                'position_size_mult': 1.2,       # slight increase
                'session_filter': 'normal',
                'reason': f'Sharpe Momentum (rolling={self.get_rolling_sharpe()})',
            }
        else:
            return {
                'min_entry_score_add': 0,
                'min_confluence_add': 0,
                'position_size_mult': 1.0,
                'session_filter': 'normal',
                'reason': 'Normal mode',
            }

    def compute_trade_quality_score(self, regime_clarity: float = 0.5,
                                     adx: float = 20, choppiness: float = 50,
                                     agent_consensus: float = 0.5,
                                     llm_confidence: float = 0.5,
                                     macro_risk: int = 50,
                                     usd_bullish: bool = False) -> int:
        """Compute trade quality score 0-100."""
        score = 0
        # Regime clarity (HMM confidence)
        if regime_clarity > 0.7:
            score += 15
        elif regime_clarity > 0.5:
            score += 8
        # Trend strength (ADX)
        if adx > 35:
            score += 20
        elif adx > 25:
            score += 10
        # Low choppiness
        if choppiness < 38.2:
            score += 10
        elif choppiness < 50:
            score += 5
        # Agent consensus
        if agent_consensus > 0.67:  # >8/12 agents
            score += 15
        elif agent_consensus > 0.5:
            score += 8
        # LLM confidence
        if llm_confidence > 0.75:
            score += 10
        elif llm_confidence > 0.6:
            score += 5
        # Low macro risk
        if macro_risk < 40:
            score += 10
        elif macro_risk < 60:
            score += 5
        # USD weakness (crypto bullish)
        if usd_bullish:
            score += 10

        return min(100, score)

    def should_take_trade(self, quality_score: int) -> bool:
        return quality_score >= self.min_quality_score

    def sharpe_adjusted_size(self, base_size: float, quality_score: int) -> float:
        sharpe = self.get_rolling_sharpe() or 1.0
        sharpe_mult = min(1.2, max(0.3, sharpe / 1.5))
        quality_mult = quality_score / 100.0
        adjustments = self.get_filter_adjustments()
        mode_mult = adjustments['position_size_mult']
        return base_size * sharpe_mult * quality_mult * mode_mult

    def get_session_sharpe(self, session: str) -> Optional[float]:
        returns = self._session_returns.get(session, [])
        if len(returns) < 5:
            return None
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = math.sqrt(var) if var > 0 else 0.001
        return round(mean / std * math.sqrt(252), 3)

    def get_regime_sharpe(self, regime: str) -> Optional[float]:
        returns = self._regime_returns.get(regime, [])
        if len(returns) < 5:
            return None
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = math.sqrt(var) if var > 0 else 0.001
        return round(mean / std * math.sqrt(252), 3)

    def get_stats(self) -> dict:
        return {
            'mode': self._current_mode,
            'rolling_sharpe': self.get_rolling_sharpe(),
            'sortino': self.get_sortino_ratio(),
            'calmar': self.get_calmar_ratio(),
            'total_trades': len(self._returns),
            'min_quality_score': self.min_quality_score,
        }
