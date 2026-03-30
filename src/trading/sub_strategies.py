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
