"""
Pine Script v5 Exporter — ACT v8.0
Generates TradingView strategy code from ACT's current live parameters.
Auto-updates when hyperopt finds new optimal parameters.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PineScriptExporter:
    def __init__(self, config: dict = None):
        self.config = config or {}

    def generate_strategy(self, asset: str = "BTCUSD", timeframe: str = "5",
                          ema_period: int = 8, atr_stop_mult: float = 3.0,
                          atr_tp_mult: float = 9.0, min_entry_score: int = 6,
                          spread_pct: float = 1.69) -> str:
        """Generate Pine Script v5 strategy mirroring ACT's live logic."""
        script = f'''//@version=5
strategy("ACT v8.0 — EMA Crossover + ATR Risk", overlay=true,
         default_qty_type=strategy.percent_of_equity, default_qty_value=2,
         commission_type=strategy.commission.percent, commission_value={spread_pct/2},
         slippage=2, initial_capital=16000, currency=currency.USD)

// ═══════════════════════════════════════════════════════════════
// ACT TRADING SYSTEM — Pine Script Mirror
// Asset: {asset} | Timeframe: {timeframe}m
// Generated from live parameters
// ═══════════════════════════════════════════════════════════════

// ── Inputs ──
ema_len     = input.int({ema_period}, "EMA Period", minval=3, maxval=50)
atr_len     = input.int(14, "ATR Period")
atr_sl_mult = input.float({atr_stop_mult}, "ATR Stop Loss Multiplier", step=0.5)
atr_tp_mult = input.float({atr_tp_mult}, "ATR Take Profit Multiplier", step=0.5)
min_score   = input.int({min_entry_score}, "Min Entry Score (0-15)", minval=0, maxval=15)
rsi_len     = input.int(14, "RSI Period")
adx_len     = input.int(14, "ADX Period")
session_filter = input.bool(true, "Session Filter (UTC 07:00-21:00)")

// ── Indicators ──
ema_val = ta.ema(close, ema_len)
atr_val = ta.atr(atr_len)
rsi_val = ta.rsi(close, rsi_len)

// ADX calculation
[diplus, diminus, adx_val] = ta.dmi(adx_len, adx_len)

// EMA slope
ema_slope = (ema_val - ema_val[3]) / ema_val[3] * 100
ema_rising = ema_slope > 0.05
ema_falling = ema_slope < -0.05

// Volume confirmation
vol_avg = ta.sma(volume, 20)
vol_surge = volume > vol_avg * 1.5

// ── Entry Score (0-15) ──
score = 0
// EMA slope strength (0-3)
score := score + (math.abs(ema_slope) > 0.5 ? 3 : math.abs(ema_slope) > 0.2 ? 2 : math.abs(ema_slope) > 0.05 ? 1 : 0)
// Consecutive EMA direction (0-3)
consec = 0
for i = 1 to 5
    if ema_val[i] < ema_val[i-1]
        consec := consec + 1
score := score + math.min(consec, 3)
// Price vs EMA (0-2)
sep = math.abs(close - ema_val) / ema_val * 100
score := score + (sep > 2 ? 0 : sep > 0.5 ? 2 : 1)
// Momentum (0-2)
score := score + (close > close[1] and close[1] > close[2] ? 2 : close > close[1] ? 1 : 0)
// Volume (0-1)
score := score + (vol_surge ? 1 : 0)
// RSI confirmation (0-2)
score := score + (rsi_val > 50 and rsi_val < 70 ? 2 : rsi_val > 40 ? 1 : 0)
// ADX trend strength (0-2)
score := score + (adx_val > 30 ? 2 : adx_val > 20 ? 1 : 0)

// ── Session Filter ──
utc_hour = hour(time, "UTC")
in_session = not session_filter or (utc_hour >= 7 and utc_hour < 21)

// ── Entry Conditions (LONG ONLY — Robinhood) ──
long_signal = ta.crossover(close, ema_val) and ema_rising
long_confirm = score >= min_score and rsi_val > 40 and rsi_val < 80 and in_session
entry_long = long_signal and long_confirm

// ── SL / TP ──
sl_price = close - atr_val * atr_sl_mult
tp_price = close + atr_val * atr_tp_mult

// ── Execute ──
if entry_long and strategy.position_size == 0
    strategy.entry("LONG", strategy.long)
    strategy.exit("Exit", "LONG", stop=sl_price, limit=tp_price)

// ── Trailing Stop (L1→L4 progression) ──
if strategy.position_size > 0
    entry_price = strategy.position_avg_price
    profit_pct = (close - entry_price) / entry_price * 100
    // L2: move SL to -1.5% when profit > 1.5%
    if profit_pct > 1.5
        new_sl = entry_price * 0.985
        strategy.exit("Exit_L2", "LONG", stop=new_sl, limit=tp_price)
    // L3: move SL to breakeven when profit > 3%
    if profit_pct > 3.0
        strategy.exit("Exit_L3", "LONG", stop=entry_price * 1.001, limit=tp_price)
    // L4: lock in 50% of profit when profit > 5%
    if profit_pct > 5.0
        lock_sl = entry_price * (1 + profit_pct * 0.5 / 100)
        strategy.exit("Exit_L4", "LONG", stop=lock_sl, limit=tp_price)

// ── Plots ──
plot(ema_val, "EMA", color=ema_rising ? color.green : color.red, linewidth=2)
plotshape(entry_long, "Buy Signal", shape.triangleup, location.belowbar, color.green, size=size.small)
bgcolor(score >= min_score ? color.new(color.green, 95) : na)

// ── Info Table ──
var table info = table.new(position.top_right, 2, 6, border_width=1)
if barstate.islast
    table.cell(info, 0, 0, "Score", bgcolor=color.gray)
    table.cell(info, 1, 0, str.tostring(score) + "/15", bgcolor=score >= min_score ? color.green : color.red)
    table.cell(info, 0, 1, "RSI", bgcolor=color.gray)
    table.cell(info, 1, 1, str.tostring(rsi_val, "#.0"))
    table.cell(info, 0, 2, "ADX", bgcolor=color.gray)
    table.cell(info, 1, 2, str.tostring(adx_val, "#.0"))
    table.cell(info, 0, 3, "ATR", bgcolor=color.gray)
    table.cell(info, 1, 3, str.tostring(atr_val, "#.0"))
    table.cell(info, 0, 4, "EMA Slope", bgcolor=color.gray)
    table.cell(info, 1, 4, str.tostring(ema_slope, "#.##") + "%")
    table.cell(info, 0, 5, "Mode", bgcolor=color.gray)
    table.cell(info, 1, 5, in_session ? "ACTIVE" : "BLOCKED")

// ── Alert Conditions ──
alertcondition(entry_long, "ACT Long Entry", "ACT v8.0: LONG entry signal on {{ticker}}")
alertcondition(ta.crossunder(close, ema_val), "ACT EMA Cross Down", "ACT v8.0: EMA cross DOWN on {{ticker}}")
'''
        return script

    def export_to_file(self, path: str = "pinescript/ACT_strategy.pine", **kwargs):
        """Save Pine Script to file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        script = self.generate_strategy(**kwargs)
        with open(path, 'w') as f:
            f.write(script)
        logger.info(f"[PINE] Exported strategy to {path}")
        return path

    def generate_rsi_strategy(self, asset: str = "BTCUSD") -> str:
        return f'''//@version=5
strategy("ACT RSI Mean Reversion", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=2)
rsi_val = ta.rsi(close, 14)
long = rsi_val < 30
exitl = rsi_val > 70
if long
    strategy.entry("RSI_LONG", strategy.long)
if exitl
    strategy.close("RSI_LONG")
alertcondition(long, "RSI Oversold", "RSI < 30 on {{{{ticker}}}}")
'''

    def generate_macd_strategy(self, asset: str = "BTCUSD") -> str:
        return f'''//@version=5
strategy("ACT MACD Momentum", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=2)
[macd_line, signal_line, hist] = ta.macd(close, 12, 26, 9)
long = ta.crossover(macd_line, signal_line) and hist > 0
exitl = ta.crossunder(macd_line, signal_line)
if long
    strategy.entry("MACD_LONG", strategy.long)
if exitl
    strategy.close("MACD_LONG")
'''
