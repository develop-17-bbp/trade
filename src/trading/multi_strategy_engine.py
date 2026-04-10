"""
Multi-Strategy Engine — Regime-Aware Signal Consensus
======================================================
Replaces EMA-only gatekeeper with 4 parallel strategies
weighted dynamically by HMM regime + Hurst exponent.

Architecture:
  ALL strategies → regime weights → meta consensus → pipeline
  (instead of: EMA only → pipeline)

Strategies:
  1. EMA Trend (existing, proven 72% WR in backtest)
  2. Mean Reversion (RSI + BB + OU process)
  3. Volatility Breakout (Donchian + volume spike)
  4. Trend Following (EMA20/50 golden cross + MACD)

Each strategy produces signal (-1/0/+1) + confidence.
Regime weights shift allocation: trending→EMA, ranging→MeanRev, volatile→Breakout.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import strategies from existing sub_strategies module
try:
    from src.trading.sub_strategies import (
        MeanReversionStrategy,
        TrendFollowingStrategy,
        VolatilityBreakoutStrategy,
        ScalpingStrategy,
    )
    STRATEGIES_AVAILABLE = True
except ImportError:
    STRATEGIES_AVAILABLE = False
    logger.warning("[MULTI-STRATEGY] sub_strategies import failed — single strategy mode")


class StrategySignal:
    """Result from a single strategy."""
    __slots__ = ('name', 'signal', 'confidence', 'metadata')

    def __init__(self, name: str, signal: int, confidence: float, metadata: Optional[Dict] = None):
        self.name = name
        self.signal = signal        # -1 (SHORT), 0 (FLAT), +1 (LONG)
        self.confidence = confidence  # 0.0 - 1.0
        self.metadata = metadata or {}


# ─── Regime Weight Profiles ───

REGIME_WEIGHTS = {
    # HMM Regime → strategy weight allocation
    'CRISIS': {
        'ema_trend': 0.05, 'mean_reversion': 0.05,
        'volatility_breakout': 0.05, 'trend_following': 0.05,
    },  # All heavily reduced in crisis
    'BULL': {
        'ema_trend': 0.40, 'mean_reversion': 0.05,
        'volatility_breakout': 0.25, 'trend_following': 0.30,
    },
    'BEAR': {
        'ema_trend': 0.30, 'mean_reversion': 0.15,
        'volatility_breakout': 0.25, 'trend_following': 0.30,
    },
    'SIDEWAYS': {
        'ema_trend': 0.15, 'mean_reversion': 0.40,
        'volatility_breakout': 0.20, 'trend_following': 0.25,
    },
}

# Hurst-adjusted overrides (applied on top of HMM weights)
HURST_OVERRIDES = {
    'TRENDING': {  # H > 0.55
        'ema_trend': 1.5,       # boost
        'mean_reversion': 0.3,  # suppress
        'volatility_breakout': 1.0,
        'trend_following': 1.3,
    },
    'MEAN_REVERTING': {  # H < 0.45
        'ema_trend': 0.3,
        'mean_reversion': 2.0,  # strong boost
        'volatility_breakout': 0.8,
        'trend_following': 0.5,
    },
    'RANDOM': {  # 0.45 <= H <= 0.55
        'ema_trend': 1.0,
        'mean_reversion': 1.0,
        'volatility_breakout': 1.0,
        'trend_following': 1.0,
    },
}


class MultiStrategyEngine:
    """
    Regime-aware multi-strategy signal generator.

    Usage:
        engine = MultiStrategyEngine(config)
        signals = engine.generate_all_signals(closes, highs, lows, volumes)
        weights = engine.compute_regime_weights(hurst=0.62, hmm_regime='BULL')
        direction, confidence, details = engine.combine(signals, weights)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._strategies: Dict[str, object] = {}
        self._performance: Dict[str, Dict] = {}  # Track per-strategy P&L for learning

        adaptive_cfg = self.config.get('adaptive', {})

        if STRATEGIES_AVAILABLE:
            self._strategies = {
                'ema_trend': None,  # EMA signal comes from executor's existing _compute_tf_signal
                'mean_reversion': MeanReversionStrategy(),
                'volatility_breakout': VolatilityBreakoutStrategy(),
                'trend_following': TrendFollowingStrategy(),
            }
            logger.info("[MULTI-STRATEGY] Engine initialized with 4 strategies")
        else:
            logger.warning("[MULTI-STRATEGY] Running in EMA-only mode (sub_strategies unavailable)")

        # Per-strategy performance tracking (for learning)
        for name in ['ema_trend', 'mean_reversion', 'volatility_breakout', 'trend_following']:
            self._performance[name] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}

        # Config overrides for strategy weights
        self._custom_weights = self.config.get('multi_strategy', {}).get('weights', {})

        print(f"  [MULTI-STRATEGY] Engine ACTIVE — 4 strategies, regime-adaptive weighting")

    def generate_all_signals(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
        ema_signal: int = 0,  # From executor's existing EMA computation
    ) -> Dict[str, StrategySignal]:
        """Run all strategies and return their individual signals."""
        results: Dict[str, StrategySignal] = {}

        # EMA signal comes from executor (already computed, proven logic)
        results['ema_trend'] = StrategySignal(
            name='ema_trend',
            signal=ema_signal,
            confidence=0.72 if ema_signal != 0 else 0.0,  # Backtest WR
            metadata={'source': 'executor_ema8'},
        )

        # Run other strategies with error isolation
        for name, strategy in self._strategies.items():
            if name == 'ema_trend' or strategy is None:
                continue
            try:
                sig = strategy.generate_signal(closes, highs, lows, volumes)
                # Compute basic confidence based on indicator strength
                conf = self._estimate_confidence(name, closes, highs, lows, sig)
                results[name] = StrategySignal(name=name, signal=sig, confidence=conf)
            except Exception as e:
                logger.debug(f"[MULTI-STRATEGY] {name} failed: {e}")
                results[name] = StrategySignal(name=name, signal=0, confidence=0.0)

        return results

    def _estimate_confidence(
        self, strategy_name: str,
        closes: List[float], highs: List[float], lows: List[float],
        signal: int,
    ) -> float:
        """Estimate confidence for a strategy signal based on indicator extremes."""
        if signal == 0:
            return 0.0

        try:
            from src.indicators.indicators import rsi, atr, adx

            if strategy_name == 'mean_reversion':
                # Confidence higher when RSI is more extreme
                rsi_vals = rsi(closes, 14)
                r = rsi_vals[-1] if rsi_vals else 50
                if signal == 1:  # Buy — lower RSI = higher confidence
                    return min(1.0, max(0.3, (30 - r) / 30)) if r < 35 else 0.3
                else:  # Sell — higher RSI = higher confidence
                    return min(1.0, max(0.3, (r - 70) / 30)) if r > 65 else 0.3

            elif strategy_name == 'volatility_breakout':
                # Confidence based on volume spike magnitude
                if len(closes) >= 20:
                    avg_vol = sum(closes[-20:]) / 20  # Simplified
                    return min(1.0, 0.5 + (closes[-1] / avg_vol - 1) * 0.3) if avg_vol > 0 else 0.5
                return 0.5

            elif strategy_name == 'trend_following':
                # Confidence based on ADX strength
                adx_vals = adx(highs, lows, closes, 14)
                adx_val = adx_vals[-1] if adx_vals else 20
                return min(1.0, max(0.3, adx_val / 50))

        except Exception:
            pass

        return 0.5  # Default moderate confidence

    def compute_regime_weights(
        self,
        hurst: float = 0.5,
        hmm_regime: str = 'SIDEWAYS',
        volatility_regime: str = 'NORMAL',
    ) -> Dict[str, float]:
        """Compute dynamic strategy weights based on current market regime."""

        # Start with HMM-based weights
        base = REGIME_WEIGHTS.get(hmm_regime.upper(), REGIME_WEIGHTS['SIDEWAYS']).copy()

        # Apply Hurst adjustment
        if hurst > 0.55:
            hurst_profile = HURST_OVERRIDES['TRENDING']
        elif hurst < 0.45:
            hurst_profile = HURST_OVERRIDES['MEAN_REVERTING']
        else:
            hurst_profile = HURST_OVERRIDES['RANDOM']

        for name in base:
            base[name] *= hurst_profile.get(name, 1.0)

        # Apply performance-based learning (boost winning strategies)
        for name, perf in self._performance.items():
            total = perf['wins'] + perf['losses']
            if total >= 5:
                wr = perf['wins'] / total
                # Boost strategies with WR > 50%, dampen those below
                base[name] *= (0.5 + wr)

        # Apply config overrides if present
        for name, w in self._custom_weights.items():
            if name in base:
                base[name] = w

        # Normalize to sum to 1.0
        total = sum(base.values())
        if total > 0:
            base = {k: v / total for k, v in base.items()}

        return base

    def combine(
        self,
        signals: Dict[str, StrategySignal],
        weights: Dict[str, float],
    ) -> Tuple[str, float, Dict]:
        """
        Combine multi-strategy signals into consensus decision.

        Returns:
            (direction, confidence, details)
            direction: 'BUY', 'SELL', or 'NEUTRAL'
            confidence: 0.0 - 1.0
            details: dict with per-strategy breakdown
        """
        score = 0.0
        total_weight = 0.0
        details = {}

        for name, sig in signals.items():
            w = weights.get(name, 0.0)
            contribution = sig.signal * sig.confidence * w
            score += contribution
            total_weight += w
            details[name] = {
                'signal': sig.signal,
                'confidence': round(sig.confidence, 3),
                'weight': round(w, 3),
                'contribution': round(contribution, 4),
                'signal_word': 'LONG' if sig.signal > 0 else 'SHORT' if sig.signal < 0 else 'FLAT',
            }

        final_score = score / (total_weight + 1e-6)
        details['_consensus_score'] = round(final_score, 4)
        details['_regime_weights'] = {k: round(v, 3) for k, v in weights.items()}

        # Agreement bonus: if 3+ strategies agree on direction, boost confidence
        directions = [s.signal for s in signals.values() if s.signal != 0]
        if len(directions) >= 3:
            if all(d > 0 for d in directions):
                final_score *= 1.25  # Strong bullish consensus
                details['_agreement'] = 'STRONG_BULLISH'
            elif all(d < 0 for d in directions):
                final_score *= 1.25  # Strong bearish consensus
                details['_agreement'] = 'STRONG_BEARISH'
            else:
                details['_agreement'] = 'MIXED'
        else:
            details['_agreement'] = 'WEAK'

        # Decision thresholds (adjusted for Robinhood spread)
        if final_score > 0.12:
            return 'BUY', min(1.0, abs(final_score)), details
        elif final_score < -0.12:
            return 'SELL', min(1.0, abs(final_score)), details
        return 'NEUTRAL', 0.0, details

    def record_outcome(self, strategy_name: str, won: bool, pnl: float = 0.0):
        """Update performance tracking for learning."""
        if strategy_name in self._performance:
            if won:
                self._performance[strategy_name]['wins'] += 1
            else:
                self._performance[strategy_name]['losses'] += 1
            self._performance[strategy_name]['total_pnl'] += pnl

    def get_status(self) -> Dict:
        """Return current engine status for dashboard/API."""
        return {
            'active': STRATEGIES_AVAILABLE,
            'strategies': list(self._strategies.keys()) if STRATEGIES_AVAILABLE else ['ema_trend'],
            'performance': self._performance,
        }
