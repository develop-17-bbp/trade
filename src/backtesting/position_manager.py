"""
Backtest Position Manager — Trailing SL Ratchet System
=======================================================
Replicates executor.py _manage_position() logic.
Handles: hard stop, time exit, grace period, 11-level ratchet, SL checks.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from src.indicators.indicators import atr, ema


@dataclass
class Position:
    direction: str  # LONG or SHORT
    entry_price: float
    qty: float
    sl: float
    entry_bar: int
    entry_ts: int  # ms
    confidence: float = 0.5
    entry_score: int = 0
    peak_price: float = 0.0
    sl_levels: List[str] = field(default_factory=lambda: ['L1'])
    is_reversal: bool = False
    ratchet_scale: float = 1.0
    timeframe: str = '5m'

    def __post_init__(self):
        if self.peak_price == 0:
            self.peak_price = self.entry_price


@dataclass
class TradeRecord:
    direction: str
    entry_price: float
    exit_price: float
    entry_bar: int
    exit_bar: int
    entry_ts: int
    exit_ts: int
    qty: float
    pnl_pct: float
    pnl_usd: float
    duration_bars: int
    duration_min: float
    exit_reason: str
    max_sl_level: str
    entry_score: int
    sl_levels_hit: List[str] = field(default_factory=list)


# TF SL bounds (from executor)
TF_SL_MIN_PCT = {'1m': 0.002, '5m': 0.005, '15m': 0.008, '1h': 0.015, '4h': 0.03}
TF_SL_MAX_PCT = {'1m': 0.008, '5m': 0.02, '15m': 0.035, '1h': 0.06, '4h': 0.10}
TF_RATCHET_SCALE = {'1m': 0.5, '5m': 1.0, '15m': 1.5, '1h': 2.5, '4h': 5.0}
TF_SECONDS = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400}


class BacktestPositionManager:
    """Manages positions with full trailing SL ratchet system."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.hard_stop_pct = config.get('hard_stop_pct', -1.8)  # Configurable hard stop
        self.max_hold_minutes = 360  # 6hr for losers
        self.max_hold_winners = max(config.get('max_hold_minutes', 720), 720)  # 12hr winners
        self.atr_sl_mult = 3.0  # Emergency ATR backstop (3x — wide)
        self.ema_period = config.get('ema_period', 8)
        self.grace_bars = 12  # 12 bars grace (60m on 5m, 180m on 15m)
        self.min_ema_exit_pct = 0.15  # Min profit to take EMA exit (10yr: tiny wins avg +0.09% drag PF)

    def open_position(self, direction: str, price: float, ohlcv: dict,
                      bar_index: int, bar_ts: int, qty: float,
                      entry_score: int = 0, timeframe: str = '5m',
                      is_reversal: bool = False) -> Position:
        """Open a new position with EMA-based SL (ATR emergency fallback)."""
        highs = ohlcv['highs']
        lows = ohlcv['lows']
        closes = ohlcv['closes']

        atr_vals = atr(highs, lows, closes, 14)
        current_atr = atr_vals[-1] if atr_vals else price * 0.01
        ema_vals = ema(closes, self.ema_period)

        tf_min_pct = TF_SL_MIN_PCT.get(timeframe, 0.005)
        tf_max_pct = TF_SL_MAX_PCT.get(timeframe, 0.02)

        # Emergency ATR backstop (wide)
        sl_distance_emergency = current_atr * self.atr_sl_mult
        sl_distance_emergency = max(sl_distance_emergency, price * tf_min_pct)
        sl_distance_emergency = min(sl_distance_emergency, price * tf_max_pct * 1.5)

        # INITIAL SL: Set to hard stop distance (-2%).
        # The EMA new-line exit is the PRIMARY exit mechanism.
        # L1 SL should NOT independently kill trades — it's just a safety net.
        # As the trade moves into profit, the EMA line-following SL will tighten
        # to track just below/above the EMA line.
        hard_stop_dist = price * abs(self.hard_stop_pct) / 100.0  # Configurable hard stop

        if direction == 'LONG':
            sl_price = price - hard_stop_dist
        else:
            sl_price = price + hard_stop_dist

        rs = TF_RATCHET_SCALE.get(timeframe, 1.0)

        return Position(
            direction=direction,
            entry_price=price,
            qty=qty,
            sl=sl_price,
            entry_bar=bar_index,
            entry_ts=bar_ts,
            entry_score=entry_score,
            peak_price=price,
            is_reversal=is_reversal,
            ratchet_scale=rs,
            timeframe=timeframe,
        )

    def update_position(self, pos: Position, bar_index: int, bar_ts: int,
                        price: float, ohlcv: dict) -> tuple:
        """Update position and check exits.

        Returns:
            (updated_position_or_None, trade_record_or_None)
            If position closed: (None, TradeRecord)
            If still open: (Position, None)
        """
        entry = pos.entry_price
        direction = pos.direction
        sl = pos.sl

        closes = ohlcv['closes']
        highs = ohlcv['highs']
        lows = ohlcv['lows']

        atr_vals = atr(highs, lows, closes, 14)
        current_atr = atr_vals[-1] if atr_vals else price * 0.01

        # Update peak
        if direction == 'LONG':
            if price > pos.peak_price:
                pos.peak_price = price
            pnl_pct = ((price - entry) / entry) * 100.0
        else:
            if price < pos.peak_price:
                pos.peak_price = price
            pnl_pct = ((entry - price) / entry) * 100.0

        peak = pos.peak_price
        duration_bars = bar_index - pos.entry_bar
        tf_seconds = TF_SECONDS.get(pos.timeframe, 300)
        duration_min = (duration_bars * tf_seconds) / 60.0

        # Helper to create trade record
        def _close(reason: str) -> tuple:
            pnl_usd = pnl_pct / 100.0 * entry * pos.qty
            return None, TradeRecord(
                direction=direction,
                entry_price=entry,
                exit_price=price,
                entry_bar=pos.entry_bar,
                exit_bar=bar_index,
                entry_ts=pos.entry_ts,
                exit_ts=bar_ts,
                qty=pos.qty,
                pnl_pct=pnl_pct,
                pnl_usd=pnl_usd,
                duration_bars=duration_bars,
                duration_min=duration_min,
                exit_reason=reason,
                max_sl_level=pos.sl_levels[-1],
                entry_score=pos.entry_score,
                sl_levels_hit=list(pos.sl_levels),
            )

        # 1. HARD STOP — non-negotiable max loss
        if duration_bars < self.grace_bars:
            # Grace period: wider emergency stop only
            if pnl_pct <= -3.0:
                return _close(f"Emergency stop {pnl_pct:+.1f}%")
            return pos, None
        else:
            if pnl_pct <= self.hard_stop_pct:
                return _close(f"Hard stop {pnl_pct:+.1f}%")

        # 2. TIME EXIT — only for very stale losers (12hr+)
        if pnl_pct <= -0.5 and duration_min >= 720:
            return _close(f"Time exit ({duration_min:.0f}min, P&L={pnl_pct:+.2f}%)")

        # 4. SL CHECK on confirmed close
        confirmed_close = closes[-2] if len(closes) >= 2 else price
        sl_hit = False
        if direction == 'LONG' and confirmed_close <= sl:
            sl_hit = True
        elif direction == 'SHORT' and confirmed_close >= sl:
            sl_hit = True

        if sl_hit:
            return _close(f"SL {pos.sl_levels[-1]} hit (close ${confirmed_close:,.2f})")

        # 5. TRAILING SL RATCHET (widened — EMA line is primary SL now)
        new_sl = self._compute_ratchet_sl(pos, pnl_pct, peak, entry, price,
                                           current_atr, direction, duration_min, lows, highs)

        # 5b. EMA LINE-FOLLOWING SL — activates after EMA has built trend (8+ bars)
        # Don't tighten SL to EMA immediately — the EMA hasn't moved enough yet.
        # Wait until the EMA line has had time to trend in our direction.
        ema_vals = ema(closes, self.ema_period) if len(closes) >= self.ema_period + 2 else []
        ema_follow_min_bars = 8  # ~40 min on 5m TF — EMA needs this many bars to show trend
        if len(ema_vals) >= 3 and duration_bars >= ema_follow_min_bars:
            ema_now = ema_vals[-2]  # Confirmed bar's EMA
            ema_buffer = current_atr * 0.5

            if direction == 'LONG':
                ema_sl = ema_now - ema_buffer
                # Only tighten if EMA has moved in our favor (EMA > entry = trend confirmed)
                if ema_sl > new_sl and ema_sl < price and ema_now > entry:
                    new_sl = ema_sl
            else:
                ema_sl = ema_now + ema_buffer
                # Only tighten if EMA has moved in our favor (EMA < entry = trend confirmed)
                if ema_sl < new_sl and ema_sl > price and ema_now < entry:
                    new_sl = ema_sl

        if new_sl != sl:
            pos.sl = new_sl
            # Track SL level advancement
            level_label = self._get_level_label(pnl_pct, pos.ratchet_scale, pos.is_reversal)
            if level_label and level_label != pos.sl_levels[-1]:
                pos.sl_levels.append(level_label)

        # 6. EMA NEW LINE EXIT — exit when EMA forms opposite line
        # This is the CORE exit: entered on new EMA line, exit when opposite line forms
        if len(ema_vals) >= 5:
            confirmed_ema = ema_vals[-2]
            confirmed_close_val = closes[-2] if len(closes) >= 2 else price

            reversal_bars = 0
            if direction == 'LONG':
                for ri in range(2, min(8, len(ema_vals))):
                    if ema_vals[-ri] < ema_vals[-ri - 1]:
                        reversal_bars += 1
                    else:
                        break
            else:
                for ri in range(2, min(8, len(ema_vals))):
                    if ema_vals[-ri] > ema_vals[-ri - 1]:
                        reversal_bars += 1
                    else:
                        break

            # EMA new line exit: ONLY when in MEANINGFUL profit
            # 10yr analysis: 5,383 tiny EMA wins (<0.3%) avg only +0.09% = dead weight
            # Require min_ema_exit_pct (0.15%) to exit — let smaller moves ride longer
            # When losing, let the L1 SL / hard stop handle the exit
            min_reversal = 2  # Quick: 2 bars reversal when in profit
            if pnl_pct >= self.min_ema_exit_pct:
                if direction == 'LONG' and reversal_bars >= min_reversal and confirmed_close_val < confirmed_ema:
                    return _close(f"EMA new down line ({reversal_bars} bars, P&L={pnl_pct:+.2f}%)")
                elif direction == 'SHORT' and reversal_bars >= min_reversal and confirmed_close_val > confirmed_ema:
                    return _close(f"EMA new up line ({reversal_bars} bars, P&L={pnl_pct:+.2f}%)")

        return pos, None

    def _compute_ratchet_sl(self, pos: Position, pnl_pct: float,
                             peak: float, entry: float, price: float,
                             current_atr: float, direction: str,
                             duration_min: float,
                             lows: list, highs: list) -> float:
        """Compute new SL using the ratchet system."""
        sl = pos.sl
        new_sl = sl
        rs = pos.ratchet_scale
        is_reversal = pos.is_reversal

        # Min age for breakeven (in minutes)
        min_age_be = 5.0 if is_reversal else 8.0

        # WIDENED ratchet: EMA line-following is the PRIMARY SL now.
        # Ratchet only kicks in at higher profit levels to lock gains.
        if is_reversal:
            ratchet_levels = [
                (1.0*rs,  0.0,  2.0, "BREAKEVEN"),
                (1.5*rs,  0.15, 1.8, "LOCK-15%"),
                (2.0*rs,  0.25, 1.5, "LOCK-25%"),
                (3.0*rs,  0.35, 1.3, "LOCK-35%"),
                (5.0*rs,  0.50, 1.0, "LOCK-50%"),
                (7.0*rs,  0.60, 0.8, "LOCK-60%"),
                (10.0*rs, 0.70, 0.6, "LOCK-70%"),
            ]
        else:
            ratchet_levels = [
                (1.0*rs,  0.0,  2.0, "BREAKEVEN"),
                (1.5*rs,  0.10, 1.8, "LOCK-10%"),
                (2.0*rs,  0.20, 1.5, "LOCK-20%"),
                (3.0*rs,  0.30, 1.3, "LOCK-30%"),
                (4.0*rs,  0.40, 1.2, "LOCK-40%"),
                (5.0*rs,  0.50, 1.0, "LOCK-50%"),
                (7.0*rs,  0.55, 0.9, "LOCK-55%"),
                (10.0*rs, 0.60, 0.8, "LOCK-60%"),
                (12.0*rs, 0.65, 0.7, "LOCK-65%"),
                (15.0*rs, 0.70, 0.6, "LOCK-70%"),
            ]

        if direction == 'LONG':
            for min_pnl, protect, atr_m, label in reversed(ratchet_levels):
                if pnl_pct >= min_pnl:
                    if protect == 0.0:
                        if duration_min >= min_age_be and sl < entry:
                            new_sl = entry
                    else:
                        profit_range = peak - entry
                        floor_sl = entry + (profit_range * protect)
                        atr_trail_sl = peak - (current_atr * atr_m)
                        best_sl = max(floor_sl, atr_trail_sl)
                        if best_sl > new_sl and best_sl < price:
                            new_sl = best_sl
                    break

            # Swing low tightening at 1.5%+
            if pnl_pct >= 1.5:
                lookback = min(15, len(lows))
                if lookback >= 3:
                    recent_lows = lows[-lookback:]
                    for i in range(1, len(recent_lows) - 1):
                        if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
                            if recent_lows[i] > entry and recent_lows[i] > new_sl and recent_lows[i] < price:
                                new_sl = recent_lows[i]

        else:  # SHORT
            for min_pnl, protect, atr_m, label in reversed(ratchet_levels):
                if pnl_pct >= min_pnl:
                    if protect == 0.0:
                        if duration_min >= min_age_be and sl > entry:
                            new_sl = entry
                    else:
                        profit_range = entry - peak
                        floor_sl = entry - (profit_range * protect)
                        atr_trail_sl = peak + (current_atr * atr_m)
                        best_sl = min(floor_sl, atr_trail_sl)
                        if best_sl < new_sl and best_sl > price:
                            new_sl = best_sl
                    break

            # Swing high tightening at 1.5%+
            if pnl_pct >= 1.5:
                lookback = min(15, len(highs))
                if lookback >= 3:
                    recent_highs = highs[-lookback:]
                    for i in range(1, len(recent_highs) - 1):
                        if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
                            if recent_highs[i] < entry and recent_highs[i] < new_sl and recent_highs[i] > price:
                                new_sl = recent_highs[i]

        return new_sl

    def _get_level_label(self, pnl_pct: float, rs: float, is_reversal: bool) -> str:
        """Get the current ratchet level label."""
        if is_reversal:
            levels = [(1.0*rs, "L2"), (1.5*rs, "L3"), (2.0*rs, "L4"),
                      (3.0*rs, "L5"), (5.0*rs, "L6"), (7.0*rs, "L7"), (10.0*rs, "L8")]
        else:
            levels = [(1.0*rs, "L2"), (1.5*rs, "L3"), (2.0*rs, "L4"),
                      (3.0*rs, "L5"), (4.0*rs, "L6"), (5.0*rs, "L7"),
                      (7.0*rs, "L8"), (10.0*rs, "L9"), (12.0*rs, "L10"),
                      (15.0*rs, "L11")]

        label = "L1"
        for threshold, lvl in levels:
            if pnl_pct >= threshold:
                label = lvl
        return label
