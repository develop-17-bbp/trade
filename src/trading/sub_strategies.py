"""
Collection of Tactical Sub-Strategies for the Adaptive Engine.
=============================================================
These are rule-based kernels that the Meta-Controller can switch between.
"""
from typing import List, Dict, Optional
import numpy as np
from src.indicators.indicators import (
    sma, ema, rsi, macd, bollinger_bands, atr, stochastic, vwap, obv, adx,
    bb_width, roc, williams_r, chaikin_money_flow, mfi, supertrend,
    volume_delta, choppiness_index,
)

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


class ICTStrategy(SubStrategy):
    """
    Inner Circle Trader — Smart Money Concepts (Liquidity Sweeps).
    Detects institutional liquidity grabs: price sweeps past a swing level
    then reverses, trapping retail traders.
    Best for: ALL regimes. Used by institutional traders worldwide.
    """
    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 30:
            return 0
        arr_h = np.asarray(highs, dtype=float)
        arr_l = np.asarray(lows, dtype=float)
        arr_c = np.asarray(prices, dtype=float)

        # Find recent swing high/low over lookback window (excluding last 2 bars)
        lb = 20
        swing_high = np.max(arr_h[-lb - 2:-2])
        swing_low = np.min(arr_l[-lb - 2:-2])

        prev_high = arr_h[-2]
        prev_low = arr_l[-2]
        curr_close = arr_c[-1]
        prev_close = arr_c[-2]

        # Bullish liquidity sweep: previous bar swept below swing low, current bar closes back above
        swept_low = prev_low < swing_low
        reclaimed_low = curr_close > swing_low and curr_close > prev_close

        # Bearish liquidity sweep: previous bar swept above swing high, current bar closes back below
        swept_high = prev_high > swing_high
        rejected_high = curr_close < swing_high and curr_close < prev_close

        if swept_low and reclaimed_low:
            return 1
        if swept_high and rejected_high:
            return -1
        return 0


class WyckoffAccumulationStrategy(SubStrategy):
    """
    Wyckoff Accumulation — Detect "Spring" (false breakdown with volume).
    Identifies selling climax followed by a spring (price breaks support
    then snaps back with volume confirmation).
    Best for: SIDEWAYS to TRENDING transition.
    """
    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 40:
            return 0
        arr_c = np.asarray(prices, dtype=float)
        arr_l = np.asarray(lows, dtype=float)
        arr_h = np.asarray(highs, dtype=float)
        arr_v = np.asarray(volumes, dtype=float)

        # Identify trading range: support/resistance from bars [-40:-5]
        support = np.min(arr_l[-40:-5])
        resistance = np.max(arr_h[-40:-5])
        avg_vol = np.mean(arr_v[-40:-5])

        # Spring detection: bar[-2] broke below support, bar[-1] closes back inside range
        spring_break = arr_l[-2] < support
        spring_close = arr_c[-1] > support and arr_c[-1] > arr_c[-2]
        vol_surge = arr_v[-1] > avg_vol * 1.3

        # Upthrust (distribution): bar[-2] broke above resistance, bar[-1] closes back inside
        upthrust_break = arr_h[-2] > resistance
        upthrust_close = arr_c[-1] < resistance and arr_c[-1] < arr_c[-2]
        upthrust_vol = arr_v[-1] > avg_vol * 1.3

        if spring_break and spring_close and vol_surge:
            return 1
        if upthrust_break and upthrust_close and upthrust_vol:
            return -1
        return 0


class FibonacciRetracementStrategy(SubStrategy):
    """
    Fibonacci Retracement — Buy at 0.618 pullback in uptrend, sell at 0.618 in downtrend.
    Waits for a significant move, then enters when price retraces to a key
    Fibonacci level with RSI confirmation.
    Best for: TRENDING markets after pullback.
    """
    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 50:
            return 0
        arr_c = np.asarray(prices, dtype=float)
        rsi_vals = rsi(prices, 14)

        # Detect significant swing: highest high and lowest low in last 50 bars
        recent_high = float(np.max(arr_c[-50:]))
        recent_low = float(np.min(arr_c[-50:]))
        swing_range = recent_high - recent_low
        if swing_range <= 0:
            return 0

        high_idx = int(np.argmax(arr_c[-50:]))
        low_idx = int(np.argmin(arr_c[-50:]))
        last_price = arr_c[-1]
        last_rsi = rsi_vals[-1] if rsi_vals else 50

        # Uptrend (low came before high): look for pullback to buy
        if low_idx < high_idx:
            fib_618 = recent_high - 0.618 * swing_range
            fib_50 = recent_high - 0.50 * swing_range
            # Price near 0.618 level (within 1% tolerance) and RSI oversold-ish
            if fib_618 * 0.99 <= last_price <= fib_50 * 1.01 and last_rsi < 45:
                return 1

        # Downtrend (high came before low): look for rally to sell
        if high_idx < low_idx:
            fib_618 = recent_low + 0.618 * swing_range
            fib_50 = recent_low + 0.50 * swing_range
            if fib_50 * 0.99 <= last_price <= fib_618 * 1.01 and last_rsi > 55:
                return -1

        return 0


class VWAPBounceStrategy(SubStrategy):
    """
    VWAP Bounce — Institutional benchmark mean-reversion.
    VWAP is the volume-weighted average price that institutional algos target.
    Buy when price dips below VWAP by 1+ ATR and starts reverting.
    Best for: INTRADAY, ALL regimes.
    """
    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 20 or len(volumes) < 20:
            return 0
        vwap_vals = vwap(prices[-20:], volumes[-20:])
        atr_vals = atr(highs, lows, prices, 14)
        if vwap_vals is None or atr_vals is None or len(vwap_vals) == 0 or len(atr_vals) == 0:
            return 0

        last_vwap = vwap_vals[-1]
        last_atr = atr_vals[-1]
        if np.isnan(last_vwap) or np.isnan(last_atr) or last_atr <= 0:
            return 0

        last_price = prices[-1]
        prev_price = prices[-2]
        deviation = last_price - last_vwap

        # Buy: price below VWAP by 1+ ATR and starting to revert up
        if deviation < -last_atr and last_price > prev_price:
            return 1
        # Sell: price above VWAP by 1+ ATR and starting to revert down
        if deviation > last_atr and last_price < prev_price:
            return -1
        return 0


class OrderBlockStrategy(SubStrategy):
    """
    Order Block — Identify institutional order blocks (last opposite candle
    before a strong impulsive move). When price returns to the order block,
    enter in the direction of the original impulse.
    Best for: ALL regimes, institutional flow.
    """
    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 30:
            return 0
        arr_c = np.asarray(prices, dtype=float)
        arr_o = np.roll(arr_c, 1)  # approximate open as previous close
        arr_o[0] = arr_c[0]
        arr_h = np.asarray(highs, dtype=float)
        arr_l = np.asarray(lows, dtype=float)
        atr_vals = atr(highs, lows, prices, 14)
        if atr_vals is None or len(atr_vals) == 0:
            return 0
        last_atr = atr_vals[-1]
        if np.isnan(last_atr) or last_atr <= 0:
            return 0

        # Search for bullish order block: last bearish candle before a strong bullish impulse
        for i in range(-15, -3):
            is_bearish = arr_c[i] < arr_o[i]  # red candle
            impulse_up = arr_c[i + 1] - arr_c[i] > 1.5 * last_atr  # strong green follow
            if is_bearish and impulse_up:
                ob_high = arr_h[i]
                ob_low = arr_l[i]
                # Current price has returned to the order block zone
                if ob_low <= arr_c[-1] <= ob_high and arr_c[-1] > arr_c[-2]:
                    return 1

        # Search for bearish order block: last bullish candle before a strong bearish impulse
        for i in range(-15, -3):
            is_bullish = arr_c[i] > arr_o[i]  # green candle
            impulse_down = arr_c[i] - arr_c[i + 1] > 1.5 * last_atr  # strong red follow
            if is_bullish and impulse_down:
                ob_high = arr_h[i]
                ob_low = arr_l[i]
                if ob_low <= arr_c[-1] <= ob_high and arr_c[-1] < arr_c[-2]:
                    return -1

        return 0


class DivergenceStrategy(SubStrategy):
    """
    RSI Divergence — Detect bullish/bearish divergence between price and RSI.
    BUY when price makes lower low but RSI makes higher low (bullish divergence).
    SELL when price makes higher high but RSI makes lower high (bearish divergence).
    Best for: TREND EXHAUSTION detection.
    """
    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 30:
            return 0
        rsi_vals = rsi(prices, 14)
        if rsi_vals is None or len(rsi_vals) < 30:
            return 0
        arr_c = np.asarray(prices, dtype=float)
        arr_r = np.asarray(rsi_vals, dtype=float)

        # Compare two swing windows: [-20:-10] vs [-10:]
        price_prev_low = np.min(arr_c[-20:-10])
        price_curr_low = np.min(arr_c[-10:])
        rsi_prev_low = np.nanmin(arr_r[-20:-10])
        rsi_curr_low = np.nanmin(arr_r[-10:])

        price_prev_high = np.max(arr_c[-20:-10])
        price_curr_high = np.max(arr_c[-10:])
        rsi_prev_high = np.nanmax(arr_r[-20:-10])
        rsi_curr_high = np.nanmax(arr_r[-10:])

        # Bullish divergence: price lower low, RSI higher low
        if price_curr_low < price_prev_low and rsi_curr_low > rsi_prev_low:
            if arr_r[-1] < 40:  # Only in oversold territory
                return 1

        # Bearish divergence: price higher high, RSI lower high
        if price_curr_high > price_prev_high and rsi_curr_high < rsi_prev_high:
            if arr_r[-1] > 60:  # Only in overbought territory
                return -1

        return 0


class BreakAndRetestStrategy(SubStrategy):
    """
    Break and Retest — Wait for price to break a key level, then retest it
    as new support/resistance before entering.
    BUY: break above resistance, pullback to retest (now support), hold.
    SELL: break below support, rally to retest (now resistance), rejected.
    Best for: BREAKOUT markets.
    """
    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 40:
            return 0
        arr_c = np.asarray(prices, dtype=float)
        arr_h = np.asarray(highs, dtype=float)
        arr_l = np.asarray(lows, dtype=float)

        # Key levels from bars [-40:-10]
        resistance = np.max(arr_h[-40:-10])
        support = np.min(arr_l[-40:-10])

        # Check if price broke above resistance in bars [-10:-3]
        broke_above = np.any(arr_c[-10:-3] > resistance)
        # Retest: price pulled back near resistance (within 0.5%) in bars [-3:-1]
        near_resistance = np.any(np.abs(arr_l[-3:-1] - resistance) / resistance < 0.005)
        # Holding: current close is above resistance
        holding_above = arr_c[-1] > resistance and arr_c[-1] > arr_c[-2]

        if broke_above and near_resistance and holding_above:
            return 1

        # Check if price broke below support
        broke_below = np.any(arr_c[-10:-3] < support)
        near_support = np.any(np.abs(arr_h[-3:-1] - support) / support < 0.005)
        holding_below = arr_c[-1] < support and arr_c[-1] < arr_c[-2]

        if broke_below and near_support and holding_below:
            return -1

        return 0


class MovingAverageCrossStrategy(SubStrategy):
    """
    Golden Cross / Death Cross — 50 SMA crosses 200 SMA.
    Classic institutional strategy used by fund managers worldwide.
    BUY: 50 SMA crosses above 200 SMA + price above both.
    SELL: 50 SMA crosses below 200 SMA.
    Best for: LONG-TERM trends.
    """
    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 202:
            return 0
        sma50 = sma(prices, 50)
        sma200 = sma(prices, 200)

        curr_50, prev_50 = sma50[-1], sma50[-2]
        curr_200, prev_200 = sma200[-1], sma200[-2]
        if any(np.isnan(v) for v in [curr_50, prev_50, curr_200, prev_200]):
            return 0

        last_price = prices[-1]

        # Golden cross: 50 SMA crosses above 200 SMA
        if prev_50 <= prev_200 and curr_50 > curr_200 and last_price > curr_50:
            return 1
        # Death cross: 50 SMA crosses below 200 SMA
        if prev_50 >= prev_200 and curr_50 < curr_200 and last_price < curr_50:
            return -1

        # Sustain signal: already in golden/death cross regime with strong ADX
        adx_line, _, _ = adx(highs, lows, prices, 14)
        last_adx = adx_line[-1] if adx_line else 0
        if not np.isnan(last_adx) and last_adx > 25:
            if curr_50 > curr_200 and last_price > curr_50:
                return 1
            if curr_50 < curr_200 and last_price < curr_50:
                return -1

        return 0


class KeltnerChannelSqueezeStrategy(SubStrategy):
    """
    Keltner Channel Squeeze — Volatility breakout detection.
    When Bollinger Bands squeeze inside Keltner Channel, an explosion is imminent.
    BUY: squeeze releases upward (BB expands + price breaks above Keltner upper).
    SELL: squeeze releases downward.
    Best for: VOLATILITY BREAKOUTS.
    """
    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 30:
            return 0
        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = bollinger_bands(prices, 20, 2.0)
        # Keltner Channel: EMA(20) +/- 1.5 * ATR(10)
        ema_vals = ema(prices, 20)
        atr_vals = atr(highs, lows, prices, 10)
        if atr_vals is None or len(atr_vals) == 0:
            return 0

        last_ema = ema_vals[-1]
        last_atr = atr_vals[-1]
        if np.isnan(last_ema) or np.isnan(last_atr) or last_atr <= 0:
            return 0

        kc_upper = last_ema + 1.5 * last_atr
        kc_lower = last_ema - 1.5 * last_atr

        # Check if BB was squeezed inside KC recently (bars -5 to -2)
        was_squeezed = False
        for i in range(-5, -1):
            if (bb_upper[i] < (ema_vals[i] + 1.5 * atr_vals[i]) and
                    bb_lower[i] > (ema_vals[i] - 1.5 * atr_vals[i])):
                was_squeezed = True
                break

        if not was_squeezed:
            return 0

        # Squeeze release: BB now wider than KC
        bb_expanding = bb_upper[-1] > kc_upper or bb_lower[-1] < kc_lower
        last_price = prices[-1]

        if bb_expanding:
            if last_price > kc_upper:
                return 1
            if last_price < kc_lower:
                return -1
        return 0


class HeikinAshiTrendStrategy(SubStrategy):
    """
    Heikin-Ashi Trend — Smoothed candle trend following.
    HA candles filter noise; 3+ consecutive same-color candles with no
    opposing wick signal strong trend conviction.
    BUY: 3+ green HA candles with no lower wick (strong uptrend).
    SELL: 3+ red HA candles with no upper wick (strong downtrend).
    Best for: TRENDING markets, noise filtering.
    """
    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 10:
            return 0
        arr_c = np.asarray(prices, dtype=float)
        arr_h = np.asarray(highs, dtype=float)
        arr_l = np.asarray(lows, dtype=float)
        # Approximate open as previous close
        arr_o = np.roll(arr_c, 1)
        arr_o[0] = arr_c[0]

        # Compute Heikin-Ashi candles
        n = len(arr_c)
        ha_close = (arr_o + arr_h + arr_l + arr_c) / 4.0
        ha_open = np.empty(n, dtype=float)
        ha_open[0] = (arr_o[0] + arr_c[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high = np.maximum(arr_h, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(arr_l, np.minimum(ha_open, ha_close))

        # Check last 3 candles for strong trend
        green_count = 0
        red_count = 0
        for i in range(-3, 0):
            is_green = ha_close[i] > ha_open[i]
            is_red = ha_close[i] < ha_open[i]
            # No lower wick on green = strong (open == low)
            no_lower_wick = abs(ha_open[i] - ha_low[i]) < (ha_high[i] - ha_low[i]) * 0.05
            # No upper wick on red = strong (open == high)
            no_upper_wick = abs(ha_open[i] - ha_high[i]) < (ha_high[i] - ha_low[i]) * 0.05

            if is_green and no_lower_wick:
                green_count += 1
            elif is_red and no_upper_wick:
                red_count += 1

        if green_count >= 3:
            return 1
        if red_count >= 3:
            return -1
        return 0


class EMACrossoverStrategy(SubStrategy):
    """
    EMA Crossover + Dynamic Trailing Stop-Loss Strategy.
    =====================================================
    Exact implementation from the TradingView chart reference:

    DOWNTREND ENTRY (P1):
      1. Market is in downtrend (EMA descending)
      2. EMA crosses through previous candle (EMA between H and L)
      3. Next candle forms entirely BELOW EMA (high < ema)
      4. → SELL entry (P1)

    UPTREND ENTRY (P1):
      1. Market is in uptrend (EMA ascending)
      2. EMA crosses through previous candle
      3. Next candle forms entirely ABOVE EMA (low > ema)
      4. → BUY entry (P1)

    EXIT (E1): Exact reverse crossover confirmed.

    DYNAMIC STOP-LOSS (L1→L2→L3→L4):
      - L1 = initial structure point (local high for SELL, local low for BUY)
      - As trade profits, push SL to new structure points (L2, L3, L4...)
      - SL only moves in profitable direction, never widens
      - Losses always covered by previously secured profits
      - Exit at last SL level on reversal

    Best for: TRENDING markets on 5m timeframe.
    """
    def __init__(self, ema_period: int = 8, struct_window: int = 5):
        self.ema_period = ema_period
        self.struct_window = struct_window
        # State tracking for active trades
        self._active_direction = 0       # 1=BUY, -1=SELL, 0=flat
        self._entry_price = 0.0
        self._stop_levels: list = []     # [{label, level, idx}]
        self._current_sl = 0.0
        self._peak_favorable = 0.0
        self._entry_idx = 0

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        """Generate EMA crossover signal with full entry/exit/SL logic."""
        from src.indicators.indicators import ema as compute_ema
        n = len(prices)
        if n < self.ema_period + 3:
            return 0

        ema_vals = compute_ema(prices, self.ema_period)
        idx = n - 1  # current bar

        ema_curr = ema_vals[idx]
        ema_prev = ema_vals[idx - 1]
        if np.isnan(ema_curr) or np.isnan(ema_prev):
            return 0

        # If we have an active trade, manage it
        if self._active_direction != 0:
            return self._manage_trade(prices, highs, lows, ema_vals, idx)

        # No active trade → look for entry
        return self._check_entry(prices, highs, lows, ema_vals, idx)

    def _check_entry(self, prices, highs, lows, ema_vals, idx) -> int:
        """Check for EMA crossover entry pattern."""
        ema_curr = ema_vals[idx]
        ema_prev = ema_vals[idx - 1]

        # Previous candle: EMA crosses through it (between high and low)
        prev_cross = (ema_prev <= highs[idx - 1]) and (ema_prev >= lows[idx - 1])
        if not prev_cross:
            return 0

        # ── DOWNTREND ENTRY: current candle entirely below EMA, EMA descending ──
        if highs[idx] < ema_curr and ema_curr < ema_prev:
            self._active_direction = -1
            self._entry_price = prices[idx]
            self._entry_idx = idx
            self._peak_favorable = prices[idx]
            self._stop_levels = []
            # L1: nearest local high (structure resistance)
            l1 = self._find_structure_high(highs, idx)
            self._stop_levels.append({'label': 'L1', 'level': l1, 'idx': idx})
            self._current_sl = l1
            return -1  # SELL

        # ── UPTREND ENTRY: current candle entirely above EMA, EMA ascending ──
        if lows[idx] > ema_curr and ema_curr > ema_prev:
            self._active_direction = 1
            self._entry_price = prices[idx]
            self._entry_idx = idx
            self._peak_favorable = prices[idx]
            self._stop_levels = []
            # L1: nearest local low (structure support)
            l1 = self._find_structure_low(lows, idx)
            self._stop_levels.append({'label': 'L1', 'level': l1, 'idx': idx})
            self._current_sl = l1
            return 1  # BUY

        return 0

    def _manage_trade(self, prices, highs, lows, ema_vals, idx) -> int:
        """Manage active trade: check SL, check exit, update trailing SL."""
        current_price = prices[idx]
        ema_curr = ema_vals[idx]
        ema_prev = ema_vals[idx - 1]

        # 1. Check stop-loss hit
        if self._active_direction == -1:  # SELL trade
            if highs[idx] >= self._current_sl:
                self._close_trade()
                return 1  # Exit SELL = BUY signal
            # Track peak favorable (lowest price)
            self._peak_favorable = min(self._peak_favorable, current_price)
        elif self._active_direction == 1:  # BUY trade
            if lows[idx] <= self._current_sl:
                self._close_trade()
                return -1  # Exit BUY = SELL signal
            self._peak_favorable = max(self._peak_favorable, current_price)

        # 2. Check for EMA reversal exit (E1)
        prev_cross = (ema_prev <= highs[idx - 1]) and (ema_prev >= lows[idx - 1])
        if prev_cross:
            if self._active_direction == -1:
                # Exit SELL: candle above EMA + EMA rising
                if lows[idx] > ema_curr and ema_curr > ema_prev:
                    self._close_trade()
                    return 1  # BUY (exit short)
            elif self._active_direction == 1:
                # Exit BUY: candle below EMA + EMA falling
                if highs[idx] < ema_curr and ema_curr < ema_prev:
                    self._close_trade()
                    return -1  # SELL (exit long)

        # 3. Update trailing stop-loss (L2→L3→L4...)
        self._update_trailing_sl(prices, highs, lows, idx)

        return 0  # HOLD

    def _update_trailing_sl(self, prices, highs, lows, idx):
        """
        Push stop-loss toward profit direction.
        Key rule: SL only moves in profitable direction, never widens.
        Losses covered by previously secured profits (max 30% giveback of peak profit).
        """
        current_price = prices[idx]
        max_giveback_pct = 0.30  # 30% max giveback of peak profit

        if self._active_direction == -1:  # SELL trade
            profit = self._entry_price - current_price
            peak_profit = self._entry_price - self._peak_favorable
            if peak_profit <= 0:
                return
            # Min profit threshold to start trailing
            if profit / self._entry_price < 0.002:
                return
            # Find new structure high (lower than current SL = tighter)
            new_struct = self._find_structure_high(highs, idx, lookback=15)
            if new_struct < self._current_sl and new_struct > current_price:
                next_label = f"L{len(self._stop_levels) + 1}"
                self._stop_levels.append({'label': next_label, 'level': new_struct, 'idx': idx})
                self._current_sl = new_struct
            else:
                # Profit-based trail: max giveback from peak
                max_giveback_abs = peak_profit * max_giveback_pct
                profit_trail_sl = self._peak_favorable + max_giveback_abs
                if profit_trail_sl < self._current_sl:
                    next_label = f"L{len(self._stop_levels) + 1}"
                    self._stop_levels.append({'label': next_label, 'level': profit_trail_sl, 'idx': idx})
                    self._current_sl = profit_trail_sl

        elif self._active_direction == 1:  # BUY trade
            profit = current_price - self._entry_price
            peak_profit = self._peak_favorable - self._entry_price
            if peak_profit <= 0:
                return
            if profit / self._entry_price < 0.002:
                return
            new_struct = self._find_structure_low(lows, idx, lookback=15)
            if new_struct > self._current_sl and new_struct < current_price:
                next_label = f"L{len(self._stop_levels) + 1}"
                self._stop_levels.append({'label': next_label, 'level': new_struct, 'idx': idx})
                self._current_sl = new_struct
            else:
                max_giveback_abs = peak_profit * max_giveback_pct
                profit_trail_sl = self._peak_favorable - max_giveback_abs
                if profit_trail_sl > self._current_sl:
                    next_label = f"L{len(self._stop_levels) + 1}"
                    self._stop_levels.append({'label': next_label, 'level': profit_trail_sl, 'idx': idx})
                    self._current_sl = profit_trail_sl

    def _find_structure_high(self, highs, current_idx, lookback=None) -> float:
        """Find the nearest local high (resistance) for SELL trade SL."""
        lookback = lookback or min(20, current_idx)
        start = max(0, current_idx - lookback)
        window_highs = highs[start:current_idx]
        if not window_highs:
            return highs[current_idx] * 1.005  # fallback: 0.5% above

        # Find local peaks
        arr = np.asarray(window_highs, dtype=float)
        w = min(self.struct_window, len(arr) // 2)
        if w < 1:
            return float(np.max(arr)) * 1.001

        peaks = []
        for i in range(w, len(arr) - w):
            if arr[i] == np.max(arr[max(0, i - w):i + w + 1]):
                peaks.append(arr[i])

        if peaks:
            return float(max(peaks)) * 1.001  # tiny buffer above structure
        return float(np.max(arr)) * 1.001

    def _find_structure_low(self, lows, current_idx, lookback=None) -> float:
        """Find the nearest local low (support) for BUY trade SL."""
        lookback = lookback or min(20, current_idx)
        start = max(0, current_idx - lookback)
        window_lows = lows[start:current_idx]
        if not window_lows:
            return lows[current_idx] * 0.995

        arr = np.asarray(window_lows, dtype=float)
        w = min(self.struct_window, len(arr) // 2)
        if w < 1:
            return float(np.min(arr)) * 0.999

        troughs = []
        for i in range(w, len(arr) - w):
            if arr[i] == np.min(arr[max(0, i - w):i + w + 1]):
                troughs.append(arr[i])

        if troughs:
            return float(min(troughs)) * 0.999
        return float(np.min(arr)) * 0.999

    def _close_trade(self):
        """Reset trade state."""
        self._active_direction = 0
        self._entry_price = 0.0
        self._stop_levels = []
        self._current_sl = 0.0
        self._peak_favorable = 0.0

    def get_stop_loss_progression(self) -> list:
        """Return current SL level history for logging."""
        return list(self._stop_levels)

    def get_current_stop_loss(self) -> float:
        """Return current SL level."""
        return self._current_sl

    def get_trade_state(self) -> Dict:
        """Return full trade state for the executor/journal."""
        return {
            'direction': self._active_direction,
            'entry_price': self._entry_price,
            'current_sl': self._current_sl,
            'peak_favorable': self._peak_favorable,
            'sl_progression': [f"{sl['label']}={sl['level']:.2f}" for sl in self._stop_levels],
            'sl_count': len(self._stop_levels),
        }


class GridTradingStrategy(SubStrategy):
    """
    Grid Trading — auto buy low / sell high in a range.
    Places virtual grid levels above and below current price.
    Best for: SIDEWAYS markets, LOW volatility, Hurst < 0.45.

    Logic:
    - Compute ATR-based grid spacing (e.g., 1x ATR between levels)
    - If price drops to a grid level below → BUY signal
    - If price rises to a grid level above → SELL signal
    - Works best when price oscillates in a range
    """

    def generate_signal(self, prices: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 30 or len(highs) < 30 or len(lows) < 30:
            return 0

        # ATR(14) for grid spacing
        atr_vals = atr(highs, lows, prices, 14)
        if atr_vals is None or len(atr_vals) == 0:
            return 0
        last_atr = atr_vals[-1]
        if np.isnan(last_atr) or last_atr <= 0:
            return 0

        last_price = prices[-1]

        # Grid levels: midpoint is the 20-bar SMA (center of the range)
        sma_vals = sma(prices, 20)
        grid_center = sma_vals[-1] if sma_vals is not None and len(sma_vals) > 0 else last_price
        if np.isnan(grid_center):
            return 0

        # Grid levels: center ± (1x, 2x, 3x) ATR
        lower_grid_1 = grid_center - last_atr
        upper_grid_1 = grid_center + last_atr

        # RSI for confirmation
        rsi_vals = rsi(prices, 14)
        if rsi_vals is None or len(rsi_vals) == 0:
            return 0
        last_rsi = rsi_vals[-1]
        if np.isnan(last_rsi):
            return 0

        # Bollinger Band width — narrow = ranging market = higher confidence for grid
        upper_bb, mid_bb, lower_bb = bollinger_bands(prices, 20, 2.0)
        bb_width = (upper_bb[-1] - lower_bb[-1]) / mid_bb[-1] if mid_bb[-1] != 0 else 999
        is_ranging = bb_width < 0.06  # narrow bands indicate ranging market

        # BUY: price at or below lower grid AND RSI < 40
        if last_price <= lower_grid_1 and last_rsi < 40:
            return 1

        # SELL: price at or above upper grid AND RSI > 60
        if last_price >= upper_grid_1 and last_rsi > 60:
            return -1

        # If market is ranging and price is at 2x ATR grid level, signal even without strict RSI
        lower_grid_2 = grid_center - 2 * last_atr
        upper_grid_2 = grid_center + 2 * last_atr

        if is_ranging:
            if last_price <= lower_grid_2 and last_rsi < 45:
                return 1
            if last_price >= upper_grid_2 and last_rsi > 55:
                return -1

        return 0


class MarketMakingStrategy(SubStrategy):
    """
    Market Making — profit from bid-ask spread and orderbook imbalance.
    Not actual market making (we can't place limit orders on Robinhood),
    but identifies when spread/imbalance creates favorable entry.
    Best for: ALL regimes, especially SIDEWAYS.

    Logic:
    - If price is at VWAP support + volume rising → BUY
    - If price deviates far above VWAP + volume declining → SELL
    - Uses mean-reversion to VWAP as the edge
    """

    def generate_signal(self, prices: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 20 or len(volumes) < 20 or len(highs) < 20 or len(lows) < 20:
            return 0

        # Compute VWAP from last 20 bars
        recent_prices = prices[-20:]
        recent_volumes = volumes[-20:]
        vwap_vals = vwap(recent_prices, recent_volumes)
        if vwap_vals is None or len(vwap_vals) == 0:
            return 0
        last_vwap = vwap_vals[-1]
        if np.isnan(last_vwap) or last_vwap <= 0:
            return 0

        last_price = prices[-1]

        # Price deviation from VWAP as z-score
        recent_prices_arr = np.asarray(recent_prices, dtype=float)
        recent_vwap = np.asarray(vwap_vals, dtype=float)
        deviations = recent_prices_arr - recent_vwap
        std_dev = np.std(deviations)
        if std_dev <= 0 or np.isnan(std_dev):
            return 0

        z_score = (last_price - last_vwap) / std_dev

        # Volume analysis: is volume rising or declining?
        arr_vols = np.asarray(volumes[-20:], dtype=float)
        avg_vol = np.mean(arr_vols)
        recent_avg_vol = np.mean(arr_vols[-5:])  # last 5 bars
        volume_rising = recent_avg_vol > avg_vol
        volume_declining = recent_avg_vol < avg_vol * 0.8

        # BUY: z-score < -1.5 AND volume rising (price below VWAP with volume support)
        if z_score < -1.5 and volume_rising:
            return 1

        # SELL: z-score > 1.5 AND volume declining (price extended above VWAP)
        if z_score > 1.5 and volume_declining:
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
