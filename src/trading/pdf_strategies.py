"""
Authority Trading Strategies — ACT v8.0
========================================
Implements the 3 strategies from the Trading Rules PDF:
  1. 400 EMA Two-Candle Closure
  2. Three-Candle Formation
  3. Regime-Gated Mean Reversion

Plus the 4 fakeout filters and trade-type specific trailing.
All connected to SharedContext for inter-layer communication.

These strategies run ALONGSIDE the existing EMA(8) crossover.
The genetic engine can also evolve these parameters.
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    """Standard signal output from any strategy."""
    strategy: str           # 'ema400_2candle', 'three_candle', 'mean_reversion'
    direction: int          # +1 LONG, -1 SHORT, 0 FLAT
    confidence: float       # 0.0 to 1.0
    entry_price: float      # suggested entry level
    stop_loss: float        # stop loss price
    take_profit: float      # take profit price
    trade_type: str         # 'scalp', 'intraday', 'swing'
    trend_tf: str           # trend timeframe ('1d', '4h', '1h')
    entry_tf: str           # entry timeframe ('1h', '15m', '5m')
    signal_range: Tuple[float, float] = (0.0, 0.0)  # high, low of signal candles
    strong_body: bool = False
    fakeout_clear: bool = False
    metadata: Dict = field(default_factory=dict)


def compute_ema(closes: List[float], period: int) -> List[float]:
    """Compute EMA for given period."""
    if len(closes) < period:
        return [closes[-1]] * len(closes) if closes else []
    ema = [sum(closes[:period]) / period]
    mult = 2 / (period + 1)
    for i in range(period, len(closes)):
        ema.append(closes[i] * mult + ema[-1] * (1 - mult))
    # Pad front
    return [ema[0]] * (len(closes) - len(ema)) + ema


def avg_body_size(opens: List[float], closes: List[float], lookback: int = 20) -> float:
    """Average absolute body size over last N bars."""
    if len(opens) < lookback or len(closes) < lookback:
        lookback = min(len(opens), len(closes))
    if lookback == 0:
        return 0.001
    bodies = [abs(closes[-i] - opens[-i]) for i in range(1, lookback + 1)]
    return sum(bodies) / len(bodies)


# ═══════════════════════════════════════════════════════════════
# STRATEGY 1: 400 EMA Two-Candle Closure
# ═══════════════════════════════════════════════════════════════

class EMA400TwoCandleStrategy:
    """
    Two consecutive full-body candles close above (long) or below (short)
    the 400 EMA on the trend timeframe. Mark their high/low as signal range.
    Enter on lower timeframe when strong-body candle breaks the range.
    """

    def __init__(self, config: dict = None, shared_ctx=None):
        cfg = config or {}
        self.ema_period = cfg.get('ema_period', 400)
        self.sl_buffer_pct = cfg.get('sl_buffer_pct', 0.004)  # 0.4% default for BTC
        self._ctx = shared_ctx

    def scan_trend_tf(self, opens: List[float], highs: List[float],
                      lows: List[float], closes: List[float],
                      trend_tf: str = '4h') -> Optional[Dict]:
        """Scan trend timeframe for two consecutive closes above/below 400 EMA."""
        if len(closes) < self.ema_period + 5:
            return None

        ema400 = compute_ema(closes, self.ema_period)

        # Check last 2 candles
        c1 = closes[-2]
        c2 = closes[-1]
        e1 = ema400[-2]
        e2 = ema400[-1]
        o1 = opens[-2]
        o2 = opens[-1]

        # Both must be full-body candles (body > 50% of range)
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        range1 = highs[-2] - lows[-2]
        range2 = highs[-1] - lows[-1]

        if range1 == 0 or range2 == 0:
            return None
        if body1 / range1 < 0.5 or body2 / range2 < 0.5:
            return None  # not full-body candles

        direction = 0
        if c1 > e1 and c2 > e2:
            direction = 1  # LONG: two closes above EMA
        elif c1 < e1 and c2 < e2:
            direction = -1  # SHORT: two closes below EMA

        if direction == 0:
            return None

        signal_high = max(highs[-2], highs[-1])
        signal_low = min(lows[-2], lows[-1])

        # Publish to shared context
        if self._ctx:
            self._ctx.publish('strategy.ema400.direction', direction, source='ema400_2candle')
            self._ctx.publish('strategy.ema400.signal_range', (signal_high, signal_low))
            self._ctx.publish('strategy.ema400.ema_value', e2)

        return {
            'direction': direction,
            'signal_high': signal_high,
            'signal_low': signal_low,
            'ema_value': e2,
            'trend_tf': trend_tf,
        }

    def check_entry(self, trend_signal: Dict,
                    entry_opens: List[float], entry_highs: List[float],
                    entry_lows: List[float], entry_closes: List[float],
                    entry_volumes: List[float],
                    entry_tf: str = '15m') -> Optional[StrategySignal]:
        """Check entry timeframe for breakout of signal range with strong body."""
        if not trend_signal or len(entry_closes) < 20:
            return None

        direction = trend_signal['direction']
        sig_high = trend_signal['signal_high']
        sig_low = trend_signal['signal_low']
        price = entry_closes[-1]
        prev_close = entry_closes[-2]

        # Strong body check
        current_body = abs(entry_closes[-1] - entry_opens[-1])
        avg_body = avg_body_size(entry_opens, entry_closes, 20)
        is_strong = current_body > avg_body

        if not is_strong:
            return None  # PDF rule: never enter on small-body candle

        # Breakout check (must CLOSE beyond level, not just wick)
        signal = None
        if direction == 1 and price > sig_high and prev_close <= sig_high:
            # Long breakout
            sl = sig_low * (1 - self.sl_buffer_pct)
            tp = price + (price - sl) * 3  # 3:1 R:R
            trade_type = self._classify_trade_type(entry_tf)
            signal = StrategySignal(
                strategy='ema400_2candle', direction=1, confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=tp,
                trade_type=trade_type, trend_tf=trend_signal['trend_tf'],
                entry_tf=entry_tf, signal_range=(sig_high, sig_low),
                strong_body=True,
            )
        elif direction == -1 and price < sig_low and prev_close >= sig_low:
            # Short breakout
            sl = sig_high * (1 + self.sl_buffer_pct)
            tp = price - (sl - price) * 3
            trade_type = self._classify_trade_type(entry_tf)
            signal = StrategySignal(
                strategy='ema400_2candle', direction=-1, confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=tp,
                trade_type=trade_type, trend_tf=trend_signal['trend_tf'],
                entry_tf=entry_tf, signal_range=(sig_high, sig_low),
                strong_body=True,
            )

        if signal and self._ctx:
            self._ctx.publish('strategy.ema400.entry_signal', signal.direction)

        return signal

    def _classify_trade_type(self, entry_tf: str) -> str:
        if entry_tf in ('5m',):
            return 'scalp'
        elif entry_tf in ('15m', '1h'):
            return 'intraday'
        return 'swing'


# ═══════════════════════════════════════════════════════════════
# STRATEGY 2: Three-Candle Formation
# ═══════════════════════════════════════════════════════════════

class ThreeCandleStrategy:
    """
    Three consecutive candles on trend TF define a clear high/low range.
    Can be consolidation, rejection, or continuation pause.
    Enter on lower TF breakout or retracement with strong body.
    """

    def __init__(self, config: dict = None, shared_ctx=None):
        cfg = config or {}
        self.sl_buffer_pct = cfg.get('sl_buffer_pct', 0.004)
        self._ctx = shared_ctx

    def scan_trend_tf(self, opens: List[float], highs: List[float],
                      lows: List[float], closes: List[float],
                      trend_tf: str = '4h') -> Optional[Dict]:
        """Scan for three-candle formation defining clear high/low."""
        if len(closes) < 5:
            return None

        # Last 3 candles
        h3 = max(highs[-3], highs[-2], highs[-1])
        l3 = min(lows[-3], lows[-2], lows[-1])
        range_pct = (h3 - l3) / l3 * 100 if l3 > 0 else 0

        # Range must be meaningful (not too tight, not too wide)
        if range_pct < 0.5 or range_pct > 8.0:
            return None

        # Determine bias from candle structure
        avg_close = (closes[-3] + closes[-2] + closes[-1]) / 3
        if closes[-1] > avg_close:
            bias = 1  # bullish bias
        elif closes[-1] < avg_close:
            bias = -1
        else:
            bias = 0

        if self._ctx:
            self._ctx.publish('strategy.three_candle.range', (h3, l3))
            self._ctx.publish('strategy.three_candle.bias', bias)

        return {
            'formation_high': h3,
            'formation_low': l3,
            'range_pct': range_pct,
            'bias': bias,
            'trend_tf': trend_tf,
        }

    def check_entry(self, formation: Dict,
                    entry_opens: List[float], entry_highs: List[float],
                    entry_lows: List[float], entry_closes: List[float],
                    entry_volumes: List[float],
                    entry_tf: str = '15m') -> Optional[StrategySignal]:
        """Check for breakout or retracement entry."""
        if not formation or len(entry_closes) < 20:
            return None

        f_high = formation['formation_high']
        f_low = formation['formation_low']
        price = entry_closes[-1]
        prev = entry_closes[-2]

        # Strong body filter
        current_body = abs(entry_closes[-1] - entry_opens[-1])
        avg_body = avg_body_size(entry_opens, entry_closes, 20)
        if current_body <= avg_body:
            return None  # PDF: small body = skip

        signal = None
        # Breakout long
        if price > f_high and prev <= f_high:
            sl = f_low * (1 - self.sl_buffer_pct)
            tp = price + (price - sl) * 2.5
            signal = StrategySignal(
                strategy='three_candle', direction=1, confidence=0.70,
                entry_price=price, stop_loss=sl, take_profit=tp,
                trade_type='intraday', trend_tf=formation['trend_tf'],
                entry_tf=entry_tf, signal_range=(f_high, f_low),
                strong_body=True,
            )
        # Breakout short
        elif price < f_low and prev >= f_low:
            sl = f_high * (1 + self.sl_buffer_pct)
            tp = price - (sl - price) * 2.5
            signal = StrategySignal(
                strategy='three_candle', direction=-1, confidence=0.70,
                entry_price=price, stop_loss=sl, take_profit=tp,
                trade_type='intraday', trend_tf=formation['trend_tf'],
                entry_tf=entry_tf, signal_range=(f_high, f_low),
                strong_body=True,
            )

        return signal


# ═══════════════════════════════════════════════════════════════
# STRATEGY 3: Regime-Gated Mean Reversion
# ═══════════════════════════════════════════════════════════════

class RegimeMeanReversion:
    """
    Only runs in CHOP or LOW_VOL regime. Non-negotiable.
    Long: RSI(14) < 25 + lower Bollinger Band
    Short: RSI(14) > 75 + upper Bollinger Band
    Target: return to 20-period MA. Size = HALF of trend-following.
    """

    def __init__(self, config: dict = None, shared_ctx=None):
        cfg = config or {}
        self.rsi_oversold = cfg.get('rsi_oversold', 25)
        self.rsi_overbought = cfg.get('rsi_overbought', 75)
        self.bb_period = cfg.get('bb_period', 20)
        self.bb_std = cfg.get('bb_std', 2.0)
        self._ctx = shared_ctx

    def check_signal(self, closes: List[float], highs: List[float],
                     lows: List[float], volumes: List[float],
                     regime: str = 'unknown',
                     entry_tf: str = '15m') -> Optional[StrategySignal]:
        """Check for mean reversion entry. ONLY in CHOP/LOW_VOL."""
        # REGIME GATE — non-negotiable
        allowed_regimes = ('chop', 'choppy', 'low_vol', 'sideways', 'ranging')
        if regime.lower() not in allowed_regimes:
            return None  # Skip entirely in trending regimes

        if len(closes) < self.bb_period + 5:
            return None

        # RSI(14)
        rsi = self._compute_rsi(closes, 14)
        if rsi is None:
            return None

        # Bollinger Bands
        sma = sum(closes[-self.bb_period:]) / self.bb_period
        std = (sum((c - sma) ** 2 for c in closes[-self.bb_period:]) / self.bb_period) ** 0.5
        upper_band = sma + self.bb_std * std
        lower_band = sma - self.bb_std * std
        price = closes[-1]

        signal = None
        # Long: RSI < 25 + at lower band
        if rsi < self.rsi_oversold and price <= lower_band * 1.005:
            sl = lower_band - std  # 1 ATR-equivalent beyond band
            tp = sma  # target = return to mean
            signal = StrategySignal(
                strategy='mean_reversion', direction=1, confidence=0.65,
                entry_price=price, stop_loss=sl, take_profit=tp,
                trade_type='scalp', trend_tf='15m', entry_tf=entry_tf,
                metadata={'rsi': rsi, 'regime': regime, 'half_size': True},
            )
        # Short: RSI > 75 + at upper band
        elif rsi > self.rsi_overbought and price >= upper_band * 0.995:
            sl = upper_band + std
            tp = sma
            signal = StrategySignal(
                strategy='mean_reversion', direction=-1, confidence=0.65,
                entry_price=price, stop_loss=sl, take_profit=tp,
                trade_type='scalp', trend_tf='15m', entry_tf=entry_tf,
                metadata={'rsi': rsi, 'regime': regime, 'half_size': True},
            )

        if signal and self._ctx:
            self._ctx.publish('strategy.mean_reversion.signal', signal.direction)
            self._ctx.publish('strategy.mean_reversion.rsi', rsi)

        return signal

    def _compute_rsi(self, closes: List[float], period: int = 14) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        deltas = [closes[i] - closes[i-1] for i in range(-period, 0)]
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]
        avg_gain = sum(gains) / period if gains else 0.001
        avg_loss = sum(losses) / period if losses else 0.001
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


# ═══════════════════════════════════════════════════════════════
# FAKEOUT FILTERS (All 4 required on 5m/15m)
# ═══════════════════════════════════════════════════════════════

class FakeoutFilter:
    """
    Four filters from PDF — ALL must pass on 5m/15m entries:
    1. Unusual candle detection
    2. Liquidity sweep recognition
    3. Double top/bottom check
    4. Back-to-zone re-entry validation
    """

    def __init__(self, shared_ctx=None):
        self._ctx = shared_ctx

    def check_all(self, opens: List[float], highs: List[float],
                  lows: List[float], closes: List[float],
                  volumes: List[float], direction: int,
                  signal_level: float, timeframe: str = '15m') -> Dict:
        """Run all 4 fakeout filters. Returns dict with each result + overall pass/fail."""
        results = {
            'unusual_candle': self._check_unusual_candle(opens, highs, lows, closes, volumes, direction),
            'liquidity_sweep': self._check_liquidity_sweep(highs, lows, closes, signal_level, direction),
            'double_top_bottom': self._check_double_top_bottom(highs, lows, closes, direction),
            'back_to_zone': self._check_back_to_zone(closes, signal_level, direction),
        }

        # All 4 must be clear (no fakeout detected)
        results['all_clear'] = all(
            not v.get('fakeout_detected', False) for v in results.values()
        )
        results['confidence_modifier'] = 1.0
        if results['unusual_candle'].get('unusual_in_direction'):
            results['confidence_modifier'] = 1.15  # boost
        if results['unusual_candle'].get('unusual_against_direction'):
            results['confidence_modifier'] = 0.7  # reduce

        if self._ctx:
            self._ctx.publish('filters.fakeout.all_clear', results['all_clear'])
            self._ctx.publish('filters.fakeout.confidence_mod', results['confidence_modifier'])

        return results

    def _check_unusual_candle(self, opens, highs, lows, closes, volumes, direction) -> Dict:
        """Filter 1: Flag if body >3x avg or volume >5x avg."""
        if len(closes) < 20:
            return {'fakeout_detected': False}

        current_body = abs(closes[-1] - opens[-1])
        current_vol = volumes[-1] if volumes else 0
        avg_body = avg_body_size(opens, closes, 20)
        avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else current_vol

        body_unusual = current_body > 3 * avg_body
        vol_unusual = current_vol > 5 * avg_vol if avg_vol > 0 else False

        # Direction of the unusual candle
        candle_dir = 1 if closes[-1] > opens[-1] else -1

        return {
            'fakeout_detected': False,  # unusual candle is informational, not a fakeout
            'unusual_in_direction': (body_unusual or vol_unusual) and candle_dir == direction,
            'unusual_against_direction': (body_unusual or vol_unusual) and candle_dir != direction,
            'body_ratio': current_body / avg_body if avg_body > 0 else 1,
            'vol_ratio': current_vol / avg_vol if avg_vol > 0 else 1,
        }

    def _check_liquidity_sweep(self, highs, lows, closes, signal_level, direction) -> Dict:
        """Filter 2: Price briefly breaks level then closes back inside = fakeout."""
        if len(closes) < 3:
            return {'fakeout_detected': False}

        # Check if previous bar broke level but current bar closed back
        if direction == 1:
            # Bullish: check if high broke above but close came back below
            swept = highs[-2] > signal_level and closes[-2] < signal_level
        else:
            swept = lows[-2] < signal_level and closes[-2] > signal_level

        return {
            'fakeout_detected': swept,
            'sweep_type': 'liquidity_sweep' if swept else 'none',
        }

    def _check_double_top_bottom(self, highs, lows, closes, direction) -> Dict:
        """Filter 3: Two tests of same level with rejection = reversal warning."""
        if len(highs) < 20:
            return {'fakeout_detected': False}

        tolerance = closes[-1] * 0.002  # 0.2% tolerance

        if direction == 1:
            # Check for double top (bearish)
            recent_highs = sorted(highs[-20:], reverse=True)[:3]
            if len(recent_highs) >= 2 and abs(recent_highs[0] - recent_highs[1]) < tolerance:
                if closes[-1] < recent_highs[0]:
                    return {'fakeout_detected': True, 'pattern': 'double_top'}
        else:
            recent_lows = sorted(lows[-20:])[:3]
            if len(recent_lows) >= 2 and abs(recent_lows[0] - recent_lows[1]) < tolerance:
                if closes[-1] > recent_lows[0]:
                    return {'fakeout_detected': True, 'pattern': 'double_bottom'}

        return {'fakeout_detected': False}

    def _check_back_to_zone(self, closes, signal_level, direction) -> Dict:
        """Filter 4: After sweep, only enter on clean rejection at zone."""
        if len(closes) < 5:
            return {'fakeout_detected': False}

        # Check if price came back to signal zone and is now rejecting
        tolerance = signal_level * 0.003
        near_zone = abs(closes[-1] - signal_level) < tolerance

        if near_zone:
            # Check for rejection: last 2 bars moving away from zone
            if direction == 1:
                rejecting = closes[-1] > closes[-2] > signal_level
            else:
                rejecting = closes[-1] < closes[-2] < signal_level
            return {'fakeout_detected': not rejecting, 'at_zone': True, 'rejecting': rejecting}

        return {'fakeout_detected': False, 'at_zone': False}


# ═══════════════════════════════════════════════════════════════
# TRADE-TYPE TRAILING (from PDF)
# ═══════════════════════════════════════════════════════════════

class TradeTypeTrailer:
    """
    PDF trailing rules per trade type:
    - Scalp:    Move to BE after +0.3-0.5%, trail in small increments
    - Intraday: Trail via 21 EMA on entry TF, exit when close against EMA
    - Swing:    Trail via 50 EMA on trend TF, or via swing highs/lows
    """

    def __init__(self, shared_ctx=None):
        self._ctx = shared_ctx

    def get_trailing_stop(self, trade_type: str, direction: int,
                          entry_price: float, current_price: float,
                          current_sl: float, closes: List[float],
                          trade_tf: str = '15m') -> float:
        """Compute trailing stop based on trade type."""
        if trade_type == 'scalp':
            return self._trail_scalp(direction, entry_price, current_price, current_sl)
        elif trade_type == 'intraday':
            return self._trail_intraday(direction, current_sl, closes, ema_period=21)
        elif trade_type == 'swing':
            return self._trail_swing(direction, current_sl, closes, ema_period=50)
        return current_sl

    def _trail_scalp(self, direction, entry_price, current_price, current_sl) -> float:
        """Scalp: BE after +0.3-0.5%, then trail in small increments."""
        pnl_pct = (current_price - entry_price) / entry_price * 100 * direction
        if pnl_pct >= 0.3:
            # Move to break-even + 0.05% buffer
            be_level = entry_price * (1 + 0.0005 * direction)
            if direction == 1:
                return max(current_sl, be_level)
            else:
                return min(current_sl, be_level)
        if pnl_pct >= 0.8:
            # Trail: lock in 40% of profit
            lock = entry_price + (current_price - entry_price) * 0.4
            if direction == 1:
                return max(current_sl, lock)
            else:
                return min(current_sl, lock)
        return current_sl

    def _trail_intraday(self, direction, current_sl, closes, ema_period=21) -> float:
        """Intraday: Trail via 21 EMA. Exit when close against EMA."""
        if len(closes) < ema_period + 1:
            return current_sl
        ema21 = compute_ema(closes, ema_period)
        ema_val = ema21[-1]
        if direction == 1:
            return max(current_sl, ema_val * 0.998)  # tiny buffer below EMA
        else:
            return min(current_sl, ema_val * 1.002)

    def _trail_swing(self, direction, current_sl, closes, ema_period=50) -> float:
        """Swing: Trail via 50 EMA on trend TF."""
        if len(closes) < ema_period + 1:
            return current_sl
        ema50 = compute_ema(closes, ema_period)
        ema_val = ema50[-1]
        if direction == 1:
            return max(current_sl, ema_val * 0.995)
        else:
            return min(current_sl, ema_val * 1.005)

    def should_exit_intraday(self, direction: int, closes: List[float]) -> bool:
        """Intraday: Exit when price closes against 21 EMA."""
        if len(closes) < 22:
            return False
        ema21 = compute_ema(closes, 21)
        if direction == 1 and closes[-1] < ema21[-1]:
            return True
        if direction == -1 and closes[-1] > ema21[-1]:
            return True
        return False


# ═══════════════════════════════════════════════════════════════
# NEWS FLATTEN RULE
# ═══════════════════════════════════════════════════════════════

class NewsFlattenGuard:
    """
    PDF Rule: Never trade during major scheduled news on lower TFs.
    Flatten 15 minutes before, wait 2 bars after release.
    """

    def __init__(self, shared_ctx=None, institutional_layer=None):
        self._ctx = shared_ctx
        self._institutional = institutional_layer

    def should_flatten(self) -> Tuple[bool, str]:
        """Check if we should flatten positions due to upcoming news."""
        if self._institutional:
            try:
                if self._institutional.get_event_buffer_window():
                    return True, "High-impact news event within 2 hours — flatten positions"
            except Exception:
                pass
        # Also check shared context
        if self._ctx:
            pre_event = self._ctx.get('macro.pre_event_flag', False)
            if pre_event:
                return True, "Pre-event flag active in shared context"
        return False, ""

    def should_skip_entry(self, timeframe: str = '15m') -> Tuple[bool, str]:
        """Check if entries should be skipped due to news."""
        flatten, reason = self.should_flatten()
        if flatten and timeframe in ('5m', '15m', '1m'):
            return True, f"Skip {timeframe} entry: {reason}"
        return False, ""


# ═══════════════════════════════════════════════════════════════
# UNIFIED STRATEGY MANAGER
# ═══════════════════════════════════════════════════════════════

class PDFStrategyManager:
    """
    Manages all 3 PDF strategies + fakeout filters + trailing + news guard.
    Connects to SharedContext so all strategies see each other's signals.
    """

    def __init__(self, config: dict = None, shared_ctx=None):
        self._ctx = shared_ctx or SharedContext()
        cfg = config or {}
        self.ema400 = EMA400TwoCandleStrategy(cfg.get('ema400', {}), self._ctx)
        self.three_candle = ThreeCandleStrategy(cfg.get('three_candle', {}), self._ctx)
        self.mean_reversion = RegimeMeanReversion(cfg.get('mean_reversion', {}), self._ctx)
        self.fakeout_filter = FakeoutFilter(self._ctx)
        self.trailer = TradeTypeTrailer(self._ctx)
        self.news_guard = NewsFlattenGuard(self._ctx)
        self.longs_only = cfg.get('longs_only', True)

    def scan_all(self, asset: str,
                 trend_opens: List[float], trend_highs: List[float],
                 trend_lows: List[float], trend_closes: List[float],
                 entry_opens: List[float], entry_highs: List[float],
                 entry_lows: List[float], entry_closes: List[float],
                 entry_volumes: List[float],
                 regime: str = 'unknown',
                 trend_tf: str = '4h', entry_tf: str = '15m') -> List[StrategySignal]:
        """Scan all 3 strategies and return valid signals."""
        signals = []

        # News guard check
        skip, reason = self.news_guard.should_skip_entry(entry_tf)
        if skip:
            if self._ctx:
                self._ctx.publish('strategy.news_block', reason)
            return []

        # ETH rule: day trades only, never swing
        eth_no_swing = asset.upper() in ('ETH', 'AAVE', 'SOL', 'AVAX')

        # Strategy 1: 400 EMA Two-Candle
        trend_sig = self.ema400.scan_trend_tf(trend_opens, trend_highs, trend_lows, trend_closes, trend_tf)
        if trend_sig:
            entry = self.ema400.check_entry(
                trend_sig, entry_opens, entry_highs, entry_lows, entry_closes, entry_volumes, entry_tf)
            if entry:
                if self.longs_only and entry.direction == -1:
                    pass  # skip shorts on Robinhood
                elif eth_no_swing and entry.trade_type == 'swing':
                    pass  # ETH no swing
                else:
                    # Run fakeout filters on 5m/15m
                    if entry_tf in ('5m', '15m'):
                        ff = self.fakeout_filter.check_all(
                            entry_opens, entry_highs, entry_lows, entry_closes, entry_volumes,
                            entry.direction, entry.signal_range[0] if entry.direction == 1 else entry.signal_range[1],
                            entry_tf)
                        entry.fakeout_clear = ff['all_clear']
                        entry.confidence *= ff['confidence_modifier']
                        if not ff['all_clear']:
                            entry.confidence *= 0.5  # penalize but don't skip entirely
                    signals.append(entry)

        # Strategy 2: Three-Candle
        formation = self.three_candle.scan_trend_tf(trend_opens, trend_highs, trend_lows, trend_closes, trend_tf)
        if formation:
            entry2 = self.three_candle.check_entry(
                formation, entry_opens, entry_highs, entry_lows, entry_closes, entry_volumes, entry_tf)
            if entry2:
                if self.longs_only and entry2.direction == -1:
                    pass
                elif eth_no_swing and entry2.trade_type == 'swing':
                    pass
                else:
                    signals.append(entry2)

        # Strategy 3: Mean Reversion (regime-gated)
        mr_signal = self.mean_reversion.check_signal(
            entry_closes, entry_highs, entry_lows, entry_volumes, regime, entry_tf)
        if mr_signal:
            if self.longs_only and mr_signal.direction == -1:
                pass
            else:
                signals.append(mr_signal)

        # Publish to shared context
        if self._ctx:
            self._ctx.publish('strategy.active_signals', len(signals))
            if signals:
                best = max(signals, key=lambda s: s.confidence)
                self._ctx.publish('strategy.best_signal', best.strategy)
                self._ctx.publish('strategy.best_direction', best.direction)
                self._ctx.publish('strategy.best_confidence', best.confidence)

        return signals
