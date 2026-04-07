"""
Backtest Signal Generator — EMA(8) New Line Detection
======================================================
Pure-function extraction of executor.py signal logic.
No side effects, no exchange calls, no LLM.
"""

from typing import Dict, List, Tuple, Optional
from src.indicators.indicators import ema, atr, rsi, macd, bollinger_bands, stochastic, obv, adx


def compute_tf_signal(ohlcv: dict, ema_period: int = 8) -> dict:
    """Compute EMA(8) new line signal for a single timeframe.

    Exact replica of executor._compute_tf_signal().
    Uses confirmed candles only ([-2] index).
    """
    closes = ohlcv['closes']
    highs = ohlcv['highs']
    lows = ohlcv['lows']
    volumes = ohlcv.get('volumes', [0] * len(closes))

    if len(closes) < 20:
        return {'signal': 'NEUTRAL', 'reason': 'not enough data'}

    ema_vals = ema(closes, ema_period)
    atr_vals = atr(highs, lows, closes, 14)

    current_ema = ema_vals[-2] if len(ema_vals) >= 2 else closes[-1]
    prev_ema = ema_vals[-3] if len(ema_vals) >= 3 else current_ema
    current_atr = atr_vals[-1] if atr_vals and len(atr_vals) > 0 else closes[-1] * 0.01
    price = closes[-2]  # Last confirmed close

    ema_direction = "RISING" if current_ema > prev_ema else "FALLING"
    slope_pct = ((current_ema - prev_ema) / prev_ema * 100) if prev_ema > 0 else 0
    ema_separation = abs(price - current_ema) / current_ema * 100 if current_ema > 0 else 0
    atr_pct = (current_atr / price * 100) if price > 0 else 0

    # New line detection
    new_line_bars = 0
    prior_trend_bars = 0

    if len(ema_vals) >= 5:
        for i in range(2, min(30, len(ema_vals))):
            if ema_direction == "RISING" and ema_vals[-i] > ema_vals[-i - 1]:
                new_line_bars += 1
            elif ema_direction == "FALLING" and ema_vals[-i] < ema_vals[-i - 1]:
                new_line_bars += 1
            else:
                break

        inflection_idx = 2 + new_line_bars
        if inflection_idx < len(ema_vals) - 1:
            for i in range(inflection_idx, min(inflection_idx + 30, len(ema_vals) - 1)):
                if ema_direction == "RISING":
                    if ema_vals[-i] < ema_vals[-i - 1]:
                        prior_trend_bars += 1
                    else:
                        break
                else:
                    if ema_vals[-i] > ema_vals[-i - 1]:
                        prior_trend_bars += 1
                    else:
                        break

    is_new_line = prior_trend_bars >= 3 and new_line_bars >= 1
    is_fresh_entry = is_new_line and new_line_bars <= 5

    # EMA crossover check
    ema_crossed = False
    for i in range(2, min(5, len(highs))):
        h = highs[-i]
        l = lows[-i]
        e = ema_vals[-i] if i <= len(ema_vals) else 0
        if l <= e <= h:
            ema_crossed = True
            break

    # Price momentum
    price_falling = False
    price_rising = False
    if len(closes) >= 5:
        c1, c2, c3 = closes[-2], closes[-3], closes[-4]
        if c1 < c2 and c1 < c3:
            price_falling = True
        elif c1 > c2 and c1 > c3:
            price_rising = True

    # Signal determination
    signal = "NEUTRAL"

    if is_fresh_entry and ema_direction == "RISING" and price > current_ema:
        if not price_falling:
            signal = "BUY"
    elif is_fresh_entry and ema_direction == "FALLING" and price < current_ema:
        if not price_rising:
            signal = "SELL"
    elif is_new_line and ema_crossed:
        if ema_direction == "RISING" and price > current_ema and not price_falling:
            signal = "BUY"
        elif ema_direction == "FALLING" and price < current_ema and not price_rising:
            signal = "SELL"
    elif new_line_bars >= 5 and ema_separation < 3.0:
        if ema_direction == "RISING" and price > current_ema and ema_crossed and not price_falling:
            signal = "BUY"
        elif ema_direction == "FALLING" and price < current_ema and ema_crossed and not price_rising:
            signal = "SELL"

    # Volume trend
    vol_trend = "FLAT"
    if len(volumes) >= 10:
        recent_vol = sum(volumes[-5:]) / 5
        prev_vol = sum(volumes[-10:-5]) / 5
        if prev_vol > 0:
            if recent_vol > prev_vol * 1.2:
                vol_trend = "RISING"
            elif recent_vol < prev_vol * 0.7:
                vol_trend = "DECLINING"

    return {
        'signal': signal,
        'ema_direction': ema_direction,
        'current_ema': current_ema,
        'current_atr': current_atr,
        'atr_pct': round(atr_pct, 4),
        'ema_slope_pct': round(slope_pct, 4),
        'ema_separation_pct': round(ema_separation, 2),
        'trend_bars': new_line_bars,
        'prior_trend_bars': prior_trend_bars,
        'is_new_line': is_new_line,
        'is_fresh_entry': is_fresh_entry,
        'vol_trend': vol_trend,
        'price': price,
        'ema_vals': ema_vals,
        'atr_vals': atr_vals,
    }


def compute_indicator_context(ohlcv: dict) -> dict:
    """Compute all indicator values for entry scoring.

    Returns dict with: rsi, adx, macd_hist, macd_cross, choppiness,
    money_flow, obv_trend, bb_width, stoch_k, stoch_d
    """
    closes = ohlcv['closes']
    highs = ohlcv['highs']
    lows = ohlcv['lows']
    volumes = ohlcv.get('volumes', [0] * len(closes))

    ctx = {}

    try:
        rsi_vals = rsi(closes, 14)
        ctx['rsi'] = rsi_vals[-1] if rsi_vals else 50
    except Exception:
        ctx['rsi'] = 50

    try:
        adx_vals = adx(highs, lows, closes, 14)
        if isinstance(adx_vals, list) and adx_vals:
            ctx['adx'] = float(adx_vals[-1]) if adx_vals[-1] is not None else 20
        elif isinstance(adx_vals, (int, float)):
            ctx['adx'] = float(adx_vals)
        else:
            ctx['adx'] = 20
    except Exception:
        ctx['adx'] = 20

    try:
        macd_line, signal_line, hist = macd(closes, 12, 26, 9)
        ctx['macd_hist'] = hist[-1] if hist else 0
        if len(hist) >= 2:
            if hist[-1] > 0 and hist[-2] <= 0:
                ctx['macd_cross'] = 'BULLISH'
            elif hist[-1] < 0 and hist[-2] >= 0:
                ctx['macd_cross'] = 'BEARISH'
            else:
                ctx['macd_cross'] = 'NONE'
        else:
            ctx['macd_cross'] = 'NONE'
    except Exception:
        ctx['macd_hist'] = 0
        ctx['macd_cross'] = 'NONE'

    try:
        obv_vals = obv(closes, volumes)
        if len(obv_vals) >= 10:
            obv_sma = sum(obv_vals[-10:]) / 10
            if obv_vals[-1] > obv_sma * 1.02:
                ctx['obv_trend'] = 'RISING'
            elif obv_vals[-1] < obv_sma * 0.98:
                ctx['obv_trend'] = 'FALLING'
            else:
                ctx['obv_trend'] = 'FLAT'
        else:
            ctx['obv_trend'] = 'FLAT'
    except Exception:
        ctx['obv_trend'] = 'FLAT'

    # Choppiness Index (simplified)
    try:
        if len(closes) >= 14:
            atr_vals = atr(highs, lows, closes, 1)
            atr_sum = sum(atr_vals[-14:])
            high_14 = max(highs[-14:])
            low_14 = min(lows[-14:])
            hl_range = high_14 - low_14
            if hl_range > 0 and atr_sum > 0:
                import math
                ctx['choppiness'] = 100 * math.log10(atr_sum / hl_range) / math.log10(14)
            else:
                ctx['choppiness'] = 50
        else:
            ctx['choppiness'] = 50
    except Exception:
        ctx['choppiness'] = 50

    # Money flow (simplified)
    try:
        if len(closes) >= 14 and len(volumes) >= 14:
            pos_flow = 0
            neg_flow = 0
            for i in range(-14, 0):
                tp = (highs[i] + lows[i] + closes[i]) / 3
                tp_prev = (highs[i-1] + lows[i-1] + closes[i-1]) / 3
                mf = tp * volumes[i]
                if tp > tp_prev:
                    pos_flow += mf
                else:
                    neg_flow += mf
            if neg_flow > 0:
                mfi = 100 - (100 / (1 + pos_flow / neg_flow))
            else:
                mfi = 100
            ctx['money_flow'] = 'INFLOW' if mfi > 60 else ('OUTFLOW' if mfi < 40 else 'NEUTRAL')
        else:
            ctx['money_flow'] = 'NEUTRAL'
    except Exception:
        ctx['money_flow'] = 'NEUTRAL'

    # ATR values for volatility filter in entry scoring
    try:
        atr_vals_14 = atr(highs, lows, closes, 14)
        ctx['atr_vals'] = atr_vals_14 if atr_vals_14 else []
    except Exception:
        ctx['atr_vals'] = []

    return ctx


def compute_entry_score(signal: str, ohlcv: dict, ema_vals: list,
                        ema_direction: str, price: float,
                        indicator_context: dict, asset: str = None,
                        config: dict = None) -> Tuple[int, List[str]]:
    """Compute entry quality score (0-20+).

    Exact replica of executor's 9-component scoring system.
    Returns (score, reasons_list).
    """
    closes = ohlcv['closes']
    entry_score = 0
    score_reasons = []

    if len(ema_vals) >= 5:
        # 1. EMA slope strength (0-3)
        ema_slope = abs(ema_vals[-1] - ema_vals[-3]) / ema_vals[-3] * 100 if ema_vals[-3] > 0 else 0
        if ema_slope > 0.3:
            entry_score += 3
            score_reasons.append(f"steep_slope={ema_slope:.2f}%")
        elif ema_slope > 0.1:
            entry_score += 2
            score_reasons.append(f"good_slope={ema_slope:.2f}%")
        elif ema_slope > 0.03:
            entry_score += 1
            score_reasons.append(f"mild_slope={ema_slope:.2f}%")

        # 2. Consecutive EMA direction (0-3)
        consec = 0
        for i in range(len(ema_vals)-2, max(0, len(ema_vals)-12), -1):
            if i > 0:
                if ema_direction == "RISING" and ema_vals[i] > ema_vals[i-1]:
                    consec += 1
                elif ema_direction == "FALLING" and ema_vals[i] < ema_vals[i-1]:
                    consec += 1
                else:
                    break
        if consec >= 5:
            entry_score += 3
            score_reasons.append(f"trend_{consec}bars")
        elif consec >= 3:
            entry_score += 2
            score_reasons.append(f"trend_{consec}bars")
        elif consec >= 2:
            entry_score += 1
            score_reasons.append(f"trend_{consec}bars")

        # 3. Price vs EMA separation (0-2)
        separation = abs(price - ema_vals[-1]) / ema_vals[-1] * 100 if ema_vals[-1] > 0 else 0
        if separation > 0.5:
            entry_score += 2
            score_reasons.append(f"sep={separation:.2f}%")
        elif separation > 0.2:
            entry_score += 1
            score_reasons.append(f"sep={separation:.2f}%")

        # 4. Candle momentum (0-2)
        if len(closes) >= 4:
            if signal == "BUY" and closes[-1] > closes[-2] > closes[-3]:
                entry_score += 2
                score_reasons.append("3_green")
            elif signal == "BUY" and closes[-1] > closes[-2]:
                entry_score += 1
                score_reasons.append("2_green")
            elif signal == "SELL" and closes[-1] < closes[-2] < closes[-3]:
                entry_score += 2
                score_reasons.append("3_red")
            elif signal == "SELL" and closes[-1] < closes[-2]:
                entry_score += 1
                score_reasons.append("2_red")

    # 5. RSI confirmation (0-2)
    _rsi = indicator_context.get('rsi', 50)
    if signal == "BUY" and 55 < _rsi < 75:
        entry_score += 2
        score_reasons.append(f"rsi_bull={_rsi:.0f}")
    elif signal == "SELL" and 25 < _rsi < 45:
        entry_score += 2
        score_reasons.append(f"rsi_bear={_rsi:.0f}")
    elif (signal == "BUY" and _rsi < 30) or (signal == "SELL" and _rsi > 70):
        entry_score -= 1
        score_reasons.append(f"rsi_against={_rsi:.0f}")

    # 6. ADX trend strength (0-2)
    _adx = indicator_context.get('adx', 20)
    if _adx > 30:
        entry_score += 2
        score_reasons.append(f"adx_strong={_adx:.0f}")
    elif _adx > 25:
        entry_score += 1
        score_reasons.append(f"adx_trend={_adx:.0f}")
    elif _adx < 15:
        entry_score -= 1
        score_reasons.append(f"adx_weak={_adx:.0f}")

    # 7. MACD alignment (0-2)
    _macd_cross = indicator_context.get('macd_cross', 'NONE')
    _macd_hist = indicator_context.get('macd_hist', 0)
    if signal == "BUY" and (_macd_cross == 'BULLISH' or _macd_hist > 0):
        entry_score += 2
        score_reasons.append("macd_bull")
    elif signal == "SELL" and (_macd_cross == 'BEARISH' or _macd_hist < 0):
        entry_score += 2
        score_reasons.append("macd_bear")
    elif (signal == "BUY" and _macd_hist < 0) or (signal == "SELL" and _macd_hist > 0):
        entry_score -= 1
        score_reasons.append("macd_against")

    # 8. Choppiness + money flow (0-2)
    _chop = indicator_context.get('choppiness', 50)
    _mflow = indicator_context.get('money_flow', '')
    if _chop < 50:
        entry_score += 1
        score_reasons.append(f"chop_trending={_chop:.0f}")
    elif _chop > 65:
        entry_score -= 1
        score_reasons.append(f"chop_ranging={_chop:.0f}")
    if (signal == "BUY" and _mflow == 'INFLOW') or (signal == "SELL" and _mflow == 'OUTFLOW'):
        entry_score += 1
        score_reasons.append(f"flow_{_mflow}")

    # 9. OBV confirmation (0-1)
    _obv = indicator_context.get('obv_trend', '')
    if (signal == "BUY" and _obv == 'RISING') or (signal == "SELL" and _obv == 'FALLING'):
        entry_score += 1
        score_reasons.append(f"obv_{_obv}")

    # 10. Trendline breakout confirmation (−2 to +2)
    tl_breakout = indicator_context.get('trendline_breakout', 0)
    tl_strength = indicator_context.get('trendline_strength', 0)
    tl_score_adj = indicator_context.get('trendline_score_adj', 0)
    if tl_breakout != 0:
        # Breakout aligned with signal direction?
        if (signal == "BUY" and tl_breakout > 0) or (signal == "SELL" and tl_breakout < 0):
            entry_score += min(2, max(1, tl_score_adj))
            score_reasons.append(f"trendline_break_aligned(str={tl_strength:.1f})")
        elif (signal == "BUY" and tl_breakout < 0) or (signal == "SELL" and tl_breakout > 0):
            entry_score -= 1
            score_reasons.append(f"trendline_break_against")
    elif tl_score_adj < 0:
        # Near trendline resistance/support (no breakout yet)
        entry_score += tl_score_adj  # Negative = penalty
        score_reasons.append(f"near_trendline(adj={tl_score_adj})")

    # 11. Horizontal S/R levels — penalize entries near opposing levels
    # Only apply to assets in sr_assets (default: ETH only — hurts BTC)
    sr_assets = (config or {}).get('adaptive', {}).get('sr_assets', ['ETH']) if config else ['ETH']
    if asset is None or asset in sr_assets:
        try:
            from src.indicators.trendlines import get_sr_score_adjustment
            sr_ctx = get_sr_score_adjustment(
                ohlcv['highs'], ohlcv['lows'], ohlcv['closes'],
                signal, lookback=100
            )
            sr_adj = sr_ctx.get('sr_score_adj', 0)
            if sr_adj != 0:
                entry_score += sr_adj
                score_reasons.append(sr_ctx.get('sr_details', f'sr_adj={sr_adj}'))
        except Exception:
            pass

    # 12. Volatility filter — BLOCK entries during extreme volatility spikes
    # Instead of adjusting score (which shifts distribution), use as hard gate
    # Only block at extreme levels (ATR > 2x) to avoid entering during crashes/pumps
    atr_vals = indicator_context.get('atr_vals', [])
    if len(atr_vals) >= 20:
        current_atr = atr_vals[-1]
        avg_atr = sum(atr_vals[-20:]) / 20
        if avg_atr > 0:
            atr_ratio = current_atr / avg_atr
            if atr_ratio > 2.0:
                entry_score -= 99  # Hard block — extreme volatility
                score_reasons.append(f"VOL_BLOCK({atr_ratio:.1f}x)")

    return entry_score, score_reasons
