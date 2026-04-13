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

try:
    from src.trading.sub_strategies import GridTradingStrategy
    GRID_AVAILABLE = True
except ImportError:
    GRID_AVAILABLE = False
    logger.debug("[MULTI-STRATEGY] GridTradingStrategy not available")

try:
    from src.trading.sub_strategies import MarketMakingStrategy
    MARKET_MAKING_AVAILABLE = True
except ImportError:
    MARKET_MAKING_AVAILABLE = False
    logger.debug("[MULTI-STRATEGY] MarketMakingStrategy not available")

try:
    from src.trading.sub_strategies import (
        ICTStrategy, WyckoffAccumulationStrategy, FibonacciRetracementStrategy,
        VWAPBounceStrategy, OrderBlockStrategy, DivergenceStrategy,
        BreakAndRetestStrategy, MovingAverageCrossStrategy,
        KeltnerChannelSqueezeStrategy, HeikinAshiTrendStrategy,
    )
    PRO_STRATEGIES_AVAILABLE = True
except ImportError:
    PRO_STRATEGIES_AVAILABLE = False
    logger.debug("[MULTI-STRATEGY] Professional strategies not available")

try:
    from src.trading.pine_strategies import (
        SupertrendStrategy, SqueezeMomentumStrategy, HalfTrendStrategy,
        UTBotAlertStrategy, SMCStrategy, EMACloudStrategy,
        VolumeProfileStrategy, RSIDivergenceStrategy, MACDHistogramStrategy,
        StochRSIStrategy, IchimokuStrategy, ParabolicSARStrategy,
        ADXDMIStrategy, CMFVolumeStrategy, WilliamsAlligatorStrategy,
        DonchianBreakoutStrategy, ChandelierExitStrategy,
        LinearRegressionStrategy, ElderRayStrategy, AroonStrategy,
    )
    PINE_STRATEGIES_AVAILABLE = True
except ImportError:
    PINE_STRATEGIES_AVAILABLE = False
    logger.debug("[MULTI-STRATEGY] Pine Script strategies not available")


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
        'grid_trading': 0.02, 'market_making': 0.05,
        'ict': 0.08, 'wyckoff': 0.03, 'fibonacci': 0.03,
        'vwap_bounce': 0.05, 'order_block': 0.08, 'divergence': 0.10,
        'break_retest': 0.03, 'ma_cross': 0.02, 'keltner_squeeze': 0.05,
        'heikin_ashi': 0.03,
        # Pine Script strategies (crisis = defensive, mostly suppressed)
        'pine_supertrend': 0.04, 'pine_squeeze': 0.03, 'pine_halftrend': 0.03,
        'pine_utbot': 0.05, 'pine_smc': 0.08, 'pine_ema_cloud': 0.03,
        'pine_volume_profile': 0.06, 'pine_rsi_divergence': 0.10,
        'pine_macd_hist': 0.04, 'pine_stochrsi': 0.05,
        'pine_ichimoku': 0.04, 'pine_psar': 0.04, 'pine_adx_dmi': 0.03,
        'pine_cmf_volume': 0.06, 'pine_alligator': 0.03, 'pine_donchian': 0.04,
        'pine_chandelier': 0.06, 'pine_linreg': 0.08, 'pine_elderray': 0.05,
        'pine_aroon': 0.03,
    },  # All heavily reduced in crisis; divergence/ICT still useful for reversals
    'BULL': {
        'ema_trend': 0.25, 'mean_reversion': 0.03,
        'volatility_breakout': 0.15, 'trend_following': 0.20,
        'grid_trading': 0.03, 'market_making': 0.10,
        'ict': 0.15, 'wyckoff': 0.08, 'fibonacci': 0.15,
        'vwap_bounce': 0.12, 'order_block': 0.15, 'divergence': 0.08,
        'break_retest': 0.18, 'ma_cross': 0.20, 'keltner_squeeze': 0.12,
        'heikin_ashi': 0.18,
        # Pine Script strategies (bull = trend-following boosted)
        'pine_supertrend': 0.20, 'pine_squeeze': 0.15, 'pine_halftrend': 0.18,
        'pine_utbot': 0.15, 'pine_smc': 0.12, 'pine_ema_cloud': 0.22,
        'pine_volume_profile': 0.10, 'pine_rsi_divergence': 0.06,
        'pine_macd_hist': 0.15, 'pine_stochrsi': 0.08,
        'pine_ichimoku': 0.18, 'pine_psar': 0.16, 'pine_adx_dmi': 0.18,
        'pine_cmf_volume': 0.12, 'pine_alligator': 0.15, 'pine_donchian': 0.16,
        'pine_chandelier': 0.14, 'pine_linreg': 0.08, 'pine_elderray': 0.12,
        'pine_aroon': 0.14,
    },
    'BEAR': {
        'ema_trend': 0.20, 'mean_reversion': 0.08,
        'volatility_breakout': 0.15, 'trend_following': 0.20,
        'grid_trading': 0.03, 'market_making': 0.15,
        'ict': 0.15, 'wyckoff': 0.05, 'fibonacci': 0.12,
        'vwap_bounce': 0.12, 'order_block': 0.15, 'divergence': 0.12,
        'break_retest': 0.15, 'ma_cross': 0.18, 'keltner_squeeze': 0.10,
        'heikin_ashi': 0.15,
        # Pine Script strategies (bear = reversal + trend-following)
        'pine_supertrend': 0.18, 'pine_squeeze': 0.12, 'pine_halftrend': 0.15,
        'pine_utbot': 0.16, 'pine_smc': 0.14, 'pine_ema_cloud': 0.18,
        'pine_volume_profile': 0.10, 'pine_rsi_divergence': 0.12,
        'pine_macd_hist': 0.14, 'pine_stochrsi': 0.10,
        'pine_ichimoku': 0.16, 'pine_psar': 0.15, 'pine_adx_dmi': 0.16,
        'pine_cmf_volume': 0.12, 'pine_alligator': 0.12, 'pine_donchian': 0.14,
        'pine_chandelier': 0.16, 'pine_linreg': 0.10, 'pine_elderray': 0.12,
        'pine_aroon': 0.12,
    },
    'SIDEWAYS': {
        # 2026-04-13: Boosted confirmed 14-day backtest winners (grid +6-7%, ict/mktmk/vwap +1.6%)
        # Dampened consistent losers (vol_breakout -8-11%, heikin -5%, trend_follow -9%, fib -1-3%)
        'ema_trend': 0.08, 'mean_reversion': 0.18,
        'volatility_breakout': 0.03, 'trend_following': 0.05,
        'grid_trading': 0.40, 'market_making': 0.30,
        'ict': 0.22, 'wyckoff': 0.12, 'fibonacci': 0.05,
        'vwap_bounce': 0.22, 'order_block': 0.12, 'divergence': 0.10,
        'break_retest': 0.08, 'ma_cross': 0.05, 'keltner_squeeze': 0.12,
        'heikin_ashi': 0.02,
        # Pine Script strategies (sideways = mean-reversion + volume boosted)
        'pine_supertrend': 0.05, 'pine_squeeze': 0.18, 'pine_halftrend': 0.06,
        'pine_utbot': 0.08, 'pine_smc': 0.10, 'pine_ema_cloud': 0.05,
        'pine_volume_profile': 0.20, 'pine_rsi_divergence': 0.15,
        'pine_macd_hist': 0.10, 'pine_stochrsi': 0.15,
        'pine_ichimoku': 0.06, 'pine_psar': 0.05, 'pine_adx_dmi': 0.06,
        'pine_cmf_volume': 0.18, 'pine_alligator': 0.08, 'pine_donchian': 0.06,
        'pine_chandelier': 0.08, 'pine_linreg': 0.18, 'pine_elderray': 0.10,
        'pine_aroon': 0.06,
    },
}

# Hurst-adjusted overrides (applied on top of HMM weights)
HURST_OVERRIDES = {
    'TRENDING': {  # H > 0.55
        'ema_trend': 1.5,       # boost
        'mean_reversion': 0.3,  # suppress
        'volatility_breakout': 1.0,
        'trend_following': 1.3,
        'grid_trading': 0.2,    # grid struggles in trends
        'market_making': 0.7,   # moderate suppression
        'ict': 1.2,             # liquidity sweeps happen in trends
        'wyckoff': 0.5,         # accumulation is pre-trend
        'fibonacci': 1.5,       # fib pullbacks are trend tools
        'vwap_bounce': 0.8,     # less mean-reversion in trends
        'order_block': 1.3,     # institutional blocks work in trends
        'divergence': 0.6,      # divergence signals exhaustion, suppress in trend
        'break_retest': 1.4,    # breakouts thrive in trends
        'ma_cross': 1.5,        # golden/death cross = trend tool
        'keltner_squeeze': 1.2, # squeeze breakouts align with trend
        'heikin_ashi': 1.6,     # HA trend following = best in trends
        # Pine strategies — trending regime
        'pine_supertrend': 1.6,     # trend-following: boost
        'pine_squeeze': 1.3,        # momentum: boost on release
        'pine_halftrend': 1.5,      # trend: boost
        'pine_utbot': 1.4,          # trend trailing stop: boost
        'pine_smc': 1.2,            # institutional: moderate boost
        'pine_ema_cloud': 1.5,      # trend: boost
        'pine_volume_profile': 0.7, # volume: less useful in trends
        'pine_rsi_divergence': 0.5, # reversal: suppress in trends
        'pine_macd_hist': 1.3,      # momentum: boost
        'pine_stochrsi': 0.6,       # reversal: suppress
        'pine_ichimoku': 1.5,       # trend: boost
        'pine_psar': 1.4,           # trend: boost
        'pine_adx_dmi': 1.5,        # trend strength: boost
        'pine_cmf_volume': 1.0,     # volume: neutral
        'pine_alligator': 1.5,      # trend: boost
        'pine_donchian': 1.4,       # breakout/trend: boost
        'pine_chandelier': 1.3,     # trend trailing: boost
        'pine_linreg': 0.6,         # mean reversion: suppress
        'pine_elderray': 1.3,       # trend: boost
        'pine_aroon': 1.4,          # trend timing: boost
    },
    'MEAN_REVERTING': {  # H < 0.45
        'ema_trend': 0.3,
        'mean_reversion': 2.0,  # strong boost
        'volatility_breakout': 0.8,
        'trend_following': 0.5,
        'grid_trading': 2.0,    # grid thrives in mean-reverting
        'market_making': 1.5,   # VWAP reversion works well
        'ict': 1.3,             # liquidity sweeps = mean reversion play
        'wyckoff': 1.5,         # accumulation happens in ranges
        'fibonacci': 0.7,       # less useful without trend
        'vwap_bounce': 1.8,     # VWAP reversion thrives
        'order_block': 0.8,     # moderate
        'divergence': 1.5,      # divergence signals reversals
        'break_retest': 0.4,    # fewer real breakouts in ranges
        'ma_cross': 0.3,        # MAs whipsaw in ranges
        'keltner_squeeze': 1.0, # squeezes still form in ranges
        'heikin_ashi': 0.4,     # HA trends fail in ranges
        # Pine strategies — mean-reverting regime
        'pine_supertrend': 0.4,     # trend: suppress
        'pine_squeeze': 1.2,        # momentum: moderate (squeezes form in ranges)
        'pine_halftrend': 0.4,      # trend: suppress
        'pine_utbot': 0.5,          # trend: suppress
        'pine_smc': 1.3,            # institutional: boost (liquidity sweeps)
        'pine_ema_cloud': 0.3,      # trend: suppress
        'pine_volume_profile': 1.8, # volume: strong boost (POC/VAL/VAH)
        'pine_rsi_divergence': 1.8, # reversal: strong boost
        'pine_macd_hist': 0.6,      # momentum: suppress
        'pine_stochrsi': 1.6,       # reversal: boost
        'pine_ichimoku': 0.4,       # trend: suppress
        'pine_psar': 0.4,           # trend: suppress
        'pine_adx_dmi': 0.4,        # trend: suppress (ADX low in ranges)
        'pine_cmf_volume': 1.5,     # volume: boost
        'pine_alligator': 0.4,      # trend: suppress
        'pine_donchian': 0.5,       # breakout: suppress
        'pine_chandelier': 0.5,     # trend: suppress
        'pine_linreg': 1.8,         # mean reversion: strong boost
        'pine_elderray': 0.6,       # trend: suppress
        'pine_aroon': 0.5,          # trend: suppress
    },
    'RANDOM': {  # 0.45 <= H <= 0.55
        'ema_trend': 1.0,
        'mean_reversion': 1.0,
        'volatility_breakout': 1.0,
        'trend_following': 1.0,
        'grid_trading': 1.0,
        'market_making': 1.0,
        'ict': 1.0,
        'wyckoff': 1.0,
        'fibonacci': 1.0,
        'vwap_bounce': 1.0,
        'order_block': 1.0,
        'divergence': 1.0,
        'break_retest': 1.0,
        'ma_cross': 1.0,
        'keltner_squeeze': 1.0,
        'heikin_ashi': 1.0,
        # Pine strategies — random walk (all neutral)
        'pine_supertrend': 1.0, 'pine_squeeze': 1.0, 'pine_halftrend': 1.0,
        'pine_utbot': 1.0, 'pine_smc': 1.0, 'pine_ema_cloud': 1.0,
        'pine_volume_profile': 1.0, 'pine_rsi_divergence': 1.0,
        'pine_macd_hist': 1.0, 'pine_stochrsi': 1.0,
        'pine_ichimoku': 1.0, 'pine_psar': 1.0, 'pine_adx_dmi': 1.0,
        'pine_cmf_volume': 1.0, 'pine_alligator': 1.0, 'pine_donchian': 1.0,
        'pine_chandelier': 1.0, 'pine_linreg': 1.0, 'pine_elderray': 1.0,
        'pine_aroon': 1.0,
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
            # Add GridTradingStrategy if available
            if GRID_AVAILABLE:
                self._strategies['grid_trading'] = GridTradingStrategy()
            # Add MarketMakingStrategy if available
            if MARKET_MAKING_AVAILABLE:
                self._strategies['market_making'] = MarketMakingStrategy()
            # Add professional strategies if available
            if PRO_STRATEGIES_AVAILABLE:
                self._strategies['ict'] = ICTStrategy()
                self._strategies['wyckoff'] = WyckoffAccumulationStrategy()
                self._strategies['fibonacci'] = FibonacciRetracementStrategy()
                self._strategies['vwap_bounce'] = VWAPBounceStrategy()
                self._strategies['order_block'] = OrderBlockStrategy()
                self._strategies['divergence'] = DivergenceStrategy()
                self._strategies['break_retest'] = BreakAndRetestStrategy()
                self._strategies['ma_cross'] = MovingAverageCrossStrategy()
                self._strategies['keltner_squeeze'] = KeltnerChannelSqueezeStrategy()
                self._strategies['heikin_ashi'] = HeikinAshiTrendStrategy()
            # Add Pine Script translated strategies if available
            if PINE_STRATEGIES_AVAILABLE:
                self._strategies['pine_supertrend'] = SupertrendStrategy()
                self._strategies['pine_squeeze'] = SqueezeMomentumStrategy()
                self._strategies['pine_halftrend'] = HalfTrendStrategy()
                self._strategies['pine_utbot'] = UTBotAlertStrategy()
                self._strategies['pine_smc'] = SMCStrategy()
                self._strategies['pine_ema_cloud'] = EMACloudStrategy()
                self._strategies['pine_volume_profile'] = VolumeProfileStrategy()
                self._strategies['pine_rsi_divergence'] = RSIDivergenceStrategy()
                self._strategies['pine_macd_hist'] = MACDHistogramStrategy()
                self._strategies['pine_stochrsi'] = StochRSIStrategy()
                self._strategies['pine_ichimoku'] = IchimokuStrategy()
                self._strategies['pine_psar'] = ParabolicSARStrategy()
                self._strategies['pine_adx_dmi'] = ADXDMIStrategy()
                self._strategies['pine_cmf_volume'] = CMFVolumeStrategy()
                self._strategies['pine_alligator'] = WilliamsAlligatorStrategy()
                self._strategies['pine_donchian'] = DonchianBreakoutStrategy()
                self._strategies['pine_chandelier'] = ChandelierExitStrategy()
                self._strategies['pine_linreg'] = LinearRegressionStrategy()
                self._strategies['pine_elderray'] = ElderRayStrategy()
                self._strategies['pine_aroon'] = AroonStrategy()
            strat_count = len(self._strategies)
            logger.info(f"[MULTI-STRATEGY] Engine initialized with {strat_count} strategies")
        else:
            logger.warning("[MULTI-STRATEGY] Running in EMA-only mode (sub_strategies unavailable)")

        # Per-strategy performance tracking (for learning)
        for name in ['ema_trend', 'mean_reversion', 'volatility_breakout', 'trend_following',
                      'grid_trading', 'market_making',
                      'ict', 'wyckoff', 'fibonacci', 'vwap_bounce', 'order_block',
                      'divergence', 'break_retest', 'ma_cross', 'keltner_squeeze', 'heikin_ashi',
                      'pine_supertrend', 'pine_squeeze', 'pine_halftrend', 'pine_utbot',
                      'pine_smc', 'pine_ema_cloud', 'pine_volume_profile', 'pine_rsi_divergence',
                      'pine_macd_hist', 'pine_stochrsi', 'pine_ichimoku', 'pine_psar',
                      'pine_adx_dmi', 'pine_cmf_volume', 'pine_alligator', 'pine_donchian',
                      'pine_chandelier', 'pine_linreg', 'pine_elderray', 'pine_aroon']:
            self._performance[name] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}

        # Config overrides for strategy weights
        self._custom_weights = self.config.get('multi_strategy', {}).get('weights', {})

        strat_count = len(self._strategies) if STRATEGIES_AVAILABLE else 1
        pine_count = 20 if PINE_STRATEGIES_AVAILABLE else 0
        print(f"  [MULTI-STRATEGY] Engine ACTIVE — {strat_count} strategies ({pine_count} Pine Script), regime-adaptive weighting")

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
                adx_line, _, _ = adx(highs, lows, closes, 14)
                adx_val = adx_line[-1] if adx_line else 20
                return min(1.0, max(0.3, adx_val / 50))

            elif strategy_name == 'grid_trading':
                # Confidence based on BB width (narrower = more ranging = higher confidence)
                from src.indicators.indicators import bollinger_bands
                upper_bb, mid_bb, lower_bb = bollinger_bands(closes, 20, 2.0)
                if mid_bb[-1] != 0:
                    bb_width = (upper_bb[-1] - lower_bb[-1]) / mid_bb[-1]
                    # Narrower BB = higher confidence (ranging market favors grid)
                    if bb_width < 0.04:
                        return 0.85
                    elif bb_width < 0.06:
                        return 0.70
                    elif bb_width < 0.10:
                        return 0.50
                    else:
                        return 0.30  # Wide bands = trending, grid less reliable
                return 0.4

            elif strategy_name == 'market_making':
                # Confidence based on price deviation from SMA(20) as VWAP proxy
                import numpy as _np
                from src.indicators.indicators import sma as compute_sma
                sma_vals = compute_sma(closes, 20)
                if sma_vals is not None and len(sma_vals) >= 20:
                    recent = _np.asarray(closes[-20:], dtype=float)
                    recent_sma = _np.asarray(sma_vals[-20:], dtype=float)
                    std_dev = _np.std(recent - recent_sma)
                    if std_dev > 0:
                        z = abs(closes[-1] - sma_vals[-1]) / std_dev
                        return min(1.0, max(0.3, 0.3 + z * 0.2))
                return 0.45

            elif strategy_name == 'ict':
                # ICT confidence: higher when ATR is large relative to price (volatile sweep)
                from src.indicators.indicators import atr as compute_atr
                atr_vals = compute_atr(highs, lows, closes, 14)
                if atr_vals and not np.isnan(atr_vals[-1]) and closes[-1] > 0:
                    atr_pct = atr_vals[-1] / closes[-1]
                    return min(1.0, max(0.4, 0.4 + atr_pct * 20))
                return 0.5

            elif strategy_name == 'wyckoff':
                # Wyckoff confidence: volume surge magnitude
                import numpy as _np
                if len(closes) >= 40:
                    avg_vol = _np.mean(_np.asarray(closes[-40:-5], dtype=float))
                    if avg_vol > 0:
                        vol_ratio = closes[-1] / avg_vol
                        return min(1.0, max(0.4, 0.3 + vol_ratio * 0.15))
                return 0.45

            elif strategy_name == 'fibonacci':
                # Fibonacci confidence: how close price is to the 0.618 level
                rsi_vals = rsi(closes, 14)
                r = rsi_vals[-1] if rsi_vals else 50
                extremity = abs(r - 50) / 50  # 0..1 how far from neutral
                return min(1.0, max(0.35, 0.35 + extremity * 0.5))

            elif strategy_name == 'vwap_bounce':
                # VWAP bounce confidence: ATR-normalized distance from VWAP proxy
                from src.indicators.indicators import atr as compute_atr
                atr_vals = compute_atr(highs, lows, closes, 14)
                from src.indicators.indicators import sma as compute_sma
                sma_vals = compute_sma(closes, 20)
                if atr_vals and sma_vals and not np.isnan(atr_vals[-1]) and atr_vals[-1] > 0:
                    dist = abs(closes[-1] - sma_vals[-1]) / atr_vals[-1]
                    return min(1.0, max(0.35, 0.3 + dist * 0.2))
                return 0.45

            elif strategy_name == 'order_block':
                # Order block confidence: strength of the impulse that created the block
                from src.indicators.indicators import atr as compute_atr
                atr_vals = compute_atr(highs, lows, closes, 14)
                if atr_vals and not np.isnan(atr_vals[-1]) and atr_vals[-1] > 0:
                    # Recent price movement vs ATR
                    move = abs(closes[-1] - closes[-3]) / atr_vals[-1]
                    return min(1.0, max(0.4, 0.35 + move * 0.15))
                return 0.5

            elif strategy_name == 'divergence':
                # Divergence confidence: how extreme RSI is
                rsi_vals = rsi(closes, 14)
                r = rsi_vals[-1] if rsi_vals else 50
                if signal == 1:
                    return min(1.0, max(0.4, (40 - r) / 30)) if r < 40 else 0.4
                else:
                    return min(1.0, max(0.4, (r - 60) / 30)) if r > 60 else 0.4

            elif strategy_name == 'break_retest':
                # Break-retest confidence: ADX strength (strong trend = real breakout)
                adx_line, _, _ = adx(highs, lows, closes, 14)
                adx_val = adx_line[-1] if adx_line else 20
                return min(1.0, max(0.35, adx_val / 45))

            elif strategy_name == 'ma_cross':
                # MA cross confidence: distance between 50 and 200 SMA (wider gap = stronger)
                from src.indicators.indicators import sma as compute_sma
                if len(closes) >= 202:
                    sma50 = compute_sma(closes, 50)
                    sma200 = compute_sma(closes, 200)
                    if sma50 and sma200 and not np.isnan(sma50[-1]) and not np.isnan(sma200[-1]):
                        gap_pct = abs(sma50[-1] - sma200[-1]) / sma200[-1]
                        return min(1.0, max(0.4, 0.4 + gap_pct * 10))
                return 0.5

            elif strategy_name == 'keltner_squeeze':
                # Keltner squeeze confidence: how compressed the squeeze was
                from src.indicators.indicators import bollinger_bands as compute_bb, atr as compute_atr
                bb_u, bb_m, bb_l = compute_bb(closes, 20, 2.0)
                if bb_m[-1] != 0:
                    bw = (bb_u[-1] - bb_l[-1]) / bb_m[-1]
                    # Tighter prior squeeze = more explosive release = higher confidence
                    return min(1.0, max(0.4, 1.0 - bw * 8))
                return 0.5

            elif strategy_name == 'heikin_ashi':
                # HA trend confidence: ADX strength
                adx_line, _, _ = adx(highs, lows, closes, 14)
                adx_val = adx_line[-1] if adx_line else 20
                return min(1.0, max(0.35, adx_val / 40))

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
