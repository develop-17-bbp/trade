"""
Collection of Tactical Sub-Strategies for the Adaptive Engine.
=============================================================
These are rule-based kernels that the Meta-Controller can switch between.
"""
from typing import List, Dict, Optional
import numpy as np
from src.indicators.indicators import sma, rsi, macd, bollinger_bands, atr

class SubStrategy:
    def generate_signal(self, prices: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> int:
        raise NotImplementedError

class MeanReversionStrategy(SubStrategy):
    """
    Buys oversold RSI + Lower BB touch. Sells overbought.
    Enhanced with Ornstein-Uhlenbeck process for statistically-grounded signals.
    Best for: LOW Volatility, Sideways markets (Hurst < 0.45).
    """
    def generate_signal(self, prices: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 30: return 0

        # ── Primary: OU Process Signal (statistically-grounded) ──
        if len(prices) >= 100:
            try:
                from src.models.ou_process import OUProcess
                ou = OUProcess(entry_threshold=1.5, exit_threshold=0.5, max_half_life=80)
                result = ou.fit_and_signal(np.asarray(prices, dtype=float), window=min(252, len(prices)))

                # Only use OU signal if series is stationary (ADF test passed)
                if result['ou_is_stationary'] and result['ou_half_life'] < 80:
                    ou_signal = result['ou_signal']
                    if ou_signal != 0:
                        return ou_signal  # OU overrides when conditions are met
            except Exception:
                pass

        # ── Fallback: Classic RSI + Bollinger Bands ──
        rsi_vals = rsi(prices, 14)
        upper, mid, lower = bollinger_bands(prices, 20, 2.0)

        last_price = prices[-1]
        last_rsi = rsi_vals[-1]
        last_lower = lower[-1]
        last_upper = upper[-1]

        if last_rsi < 30 and last_price <= last_lower:
            return 1  # Buy
        elif last_rsi > 70 and last_price >= last_upper:
            return -1  # Sell
        return 0

class TrendFollowingStrategy(SubStrategy):
    """
    EMA Golden Cross + MACD alignment.
    Best for: MEDIUM/HIGH Volatility, Trending markets.
    """
    def generate_signal(self, prices: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 50: return 0
        
        ema_short = sma(prices, 20) # Using SMA for simplicity or implement EMA if needed
        ema_long = sma(prices, 50)
        _, _, macd_hist = macd(prices)
        
        if ema_short[-1] > ema_long[-1] and macd_hist[-1] > 0:
            return 1
        elif ema_short[-1] < ema_long[-1] and macd_hist[-1] < 0:
            return -1
        return 0

class VolatilityBreakoutStrategy(SubStrategy):
    """
    Donchian Channel Breakout + Volume spike.
    Best for: EXTREME Volatility, Momentum bursts.
    """
    def generate_signal(self, prices: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 20: return 0
        
        # Donchian 20 Breakout
        arr_highs = np.asarray(highs)
        arr_lows = np.asarray(lows)
        arr_vols = np.asarray(volumes)

        d_upper = np.max(arr_highs[-21:-1])
        d_lower = np.min(arr_lows[-21:-1])
        avg_vol = np.mean(arr_vols[-20:-1])
        
        if prices[-1] > d_upper and volumes[-1] > avg_vol * 1.5:
            return 1
        elif prices[-1] < d_lower and volumes[-1] > avg_vol * 1.5:
            return -1
        return 0

class ScalpingStrategy(SubStrategy):
    """
    Fast-in, fast-out based on stochastic momentum and EMA filter.
    Best for: High-frequency small-range movements in neutral/choppy regimes.
    """
    def generate_signal(self, prices: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 20: return 0
        
        from src.indicators.indicators import stochastic, ema
        ema_fast = ema(prices, 13)
        k_vals, d_vals = stochastic(highs, lows, prices, k_period=5, d_period=3)
        
        last_price = prices[-1]
        last_ema = ema_fast[-1]
        
        # Guard against NaNs
        if np.isnan(last_ema) or np.isnan(k_vals[-1]): return 0
        
        last_k = k_vals[-1]
        last_d = d_vals[-1]
        
        # Long: Price above EMA and Stoch K crosses above D in oversold zone (< 30)
        if last_price > last_ema and last_k < 30 and last_k > last_d:
            return 1
        # Short: Price below EMA and Stoch K crosses below D in overbought zone (> 70)
        elif last_price < last_ema and last_k > 70 and last_k < last_d:
            return -1
        return 0


class PairsStrategy(SubStrategy):
    """
    Cointegration-based pairs trading.
    Trades the spread between two assets when z-score is extreme.
    Best for: Sideways markets with cointegrated pairs.
    """
    def __init__(self, reference_prices: List[float] = None):
        self.reference_prices = reference_prices or []

    def generate_signal(self, prices: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 100 or len(self.reference_prices) < 100:
            return 0

        try:
            from src.models.cointegration import CointegrationEngine
            engine = CointegrationEngine(entry_z=2.0, exit_z=0.5)
            result = engine.spread_signal(
                np.asarray(prices, dtype=float),
                np.asarray(self.reference_prices, dtype=float)
            )
            return result.get('signal', 0)
        except Exception:
            return 0
