"""
Pine Script Strategy Translations — Top 20 TradingView Strategies
=================================================================
Python implementations of the most popular TradingView Pine Script
indicators/strategies. Each class is a SubStrategy compatible with
the Multi-Strategy Engine.

Source indicators used from src.indicators.indicators:
  sma, ema, rsi, macd, bollinger_bands, atr, stochastic, vwap, obv, adx,
  bb_width, roc, williams_r, chaikin_money_flow, mfi, supertrend,
  volume_delta, choppiness_index, parabolic_sar
"""

from typing import List
import math
import numpy as np
from src.indicators.indicators import (
    sma, ema, rsi, macd, bollinger_bands, atr, stochastic, vwap, obv, adx,
    bb_width, roc, williams_r, chaikin_money_flow, mfi, supertrend,
    volume_delta, choppiness_index, parabolic_sar,
)
from src.trading.sub_strategies import SubStrategy


# ---------------------------------------------------------------------------
# 1. SupertrendStrategy
# ---------------------------------------------------------------------------
class SupertrendStrategy(SubStrategy):
    """Pine: Supertrend by KivancOzbilgic — ATR-based trend following.
    Long when price closes above supertrend line, short when below.
    Most popular indicator on TradingView with 500k+ users."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 20:
            return 0
        st_line, st_dir = supertrend(highs, lows, prices, period=10, multiplier=3.0)
        # Signal on direction change
        if st_dir[-1] == 1 and st_dir[-2] == -1:
            return 1   # Flip to bullish
        elif st_dir[-1] == -1 and st_dir[-2] == 1:
            return -1  # Flip to bearish
        # Continuation signal with momentum confirmation
        if st_dir[-1] == 1 and prices[-1] > st_line[-1] * 1.002:
            return 1
        elif st_dir[-1] == -1 and prices[-1] < st_line[-1] * 0.998:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 2. SqueezeMonetumStrategy
# ---------------------------------------------------------------------------
class SqueezeMomentumStrategy(SubStrategy):
    """Pine: Squeeze Momentum by LazyBear — BB inside Keltner Channel = squeeze.
    When squeeze fires, momentum histogram direction gives entry.
    300k+ users on TradingView."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 30:
            return 0
        arr = np.asarray(prices, dtype=float)
        # Bollinger Bands
        bb_u, bb_m, bb_l = bollinger_bands(prices, 20, 2.0)
        # Keltner Channel (EMA20 +/- 1.5*ATR)
        ema20 = ema(prices, 20)
        atr_vals = atr(highs, lows, prices, 20)
        kc_upper = [e + 1.5 * a for e, a in zip(ema20, atr_vals)]
        kc_lower = [e - 1.5 * a for e, a in zip(ema20, atr_vals)]
        # Squeeze = BB inside KC
        squeeze_on = bb_l[-1] > kc_lower[-1] and bb_u[-1] < kc_upper[-1]
        squeeze_prev = bb_l[-2] > kc_lower[-2] and bb_u[-2] < kc_upper[-2]
        # Momentum = linear regression of (close - avg(highest_high, lowest_low, sma20))
        n = min(20, len(arr))
        segment = arr[-n:]
        hh = np.max(np.asarray(highs[-n:], dtype=float))
        ll = np.min(np.asarray(lows[-n:], dtype=float))
        sma20 = np.mean(segment)
        mom = segment[-1] - np.mean([hh, ll, sma20])
        mom_prev = segment[-2] - np.mean([hh, ll, sma20]) if n > 1 else 0
        # Squeeze release + momentum direction
        if squeeze_prev and not squeeze_on:
            return 1 if mom > 0 else -1
        # Active momentum change
        if mom > 0 and mom > mom_prev:
            return 1
        elif mom < 0 and mom < mom_prev:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 3. HalfTrendStrategy
# ---------------------------------------------------------------------------
class HalfTrendStrategy(SubStrategy):
    """Pine: HalfTrend by everget — Smoothed trend detection with ATR envelope.
    Uses amplitude (ATR-based) to filter noise; trend flips on close
    breaking the opposite channel boundary."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 30:
            return 0
        amplitude = 2
        atr_vals = atr(highs, lows, prices, 14)
        arr_h = np.asarray(highs, dtype=float)
        arr_l = np.asarray(lows, dtype=float)
        arr_c = np.asarray(prices, dtype=float)
        n = len(prices)
        # Compute highest-high and lowest-low over amplitude period
        trend = 0
        half_trend = arr_c[-1]
        for i in range(max(amplitude, 1), n):
            hh = np.max(arr_h[max(0, i - amplitude):i + 1])
            ll = np.min(arr_l[max(0, i - amplitude):i + 1])
            dev = atr_vals[i] * amplitude * 0.5 if not math.isnan(atr_vals[i]) else 0
            mid = (hh + ll) / 2.0
            if arr_c[i] > mid + dev:
                trend = 1
                half_trend = ll
            elif arr_c[i] < mid - dev:
                trend = -1
                half_trend = hh
        # Previous bar trend
        prev_trend = 0
        if n > amplitude + 2:
            i = n - 2
            hh = np.max(arr_h[max(0, i - amplitude):i + 1])
            ll = np.min(arr_l[max(0, i - amplitude):i + 1])
            dev = atr_vals[i] * amplitude * 0.5 if not math.isnan(atr_vals[i]) else 0
            mid = (hh + ll) / 2.0
            if arr_c[i] > mid + dev:
                prev_trend = 1
            elif arr_c[i] < mid - dev:
                prev_trend = -1
        if trend == 1 and prev_trend != 1:
            return 1
        elif trend == -1 and prev_trend != -1:
            return -1
        return trend if trend != 0 else 0


# ---------------------------------------------------------------------------
# 4. UTBotAlertStrategy
# ---------------------------------------------------------------------------
class UTBotAlertStrategy(SubStrategy):
    """Pine: UT Bot Alert by QuantNomad — ATR trailing stop with key level detection.
    Generates buy when price crosses above trailing stop, sell when below."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 20:
            return 0
        key_value = 1.0  # sensitivity
        atr_period = 10
        atr_vals = atr(highs, lows, prices, atr_period)
        arr_c = np.asarray(prices, dtype=float)
        n = len(prices)
        # Compute trailing stop
        trail = np.zeros(n)
        trail[0] = arr_c[0]
        for i in range(1, n):
            loss = atr_vals[i] * key_value if not math.isnan(atr_vals[i]) else 0
            if arr_c[i] > trail[i - 1] and arr_c[i - 1] > trail[i - 1]:
                trail[i] = max(trail[i - 1], arr_c[i] - loss)
            elif arr_c[i] < trail[i - 1] and arr_c[i - 1] < trail[i - 1]:
                trail[i] = min(trail[i - 1], arr_c[i] + loss)
            elif arr_c[i] > trail[i - 1]:
                trail[i] = arr_c[i] - loss
            else:
                trail[i] = arr_c[i] + loss
        # Cross detection
        above_now = arr_c[-1] > trail[-1]
        above_prev = arr_c[-2] > trail[-2]
        if above_now and not above_prev:
            return 1
        elif not above_now and above_prev:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 5. SMCStrategy (Smart Money Concepts)
# ---------------------------------------------------------------------------
class SMCStrategy(SubStrategy):
    """Pine: Smart Money Concepts by LuxAlgo — Order blocks, FVG, break of structure.
    Detects institutional order flow: BOS + order block retest = entry."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 40:
            return 0
        arr_h = np.asarray(highs, dtype=float)
        arr_l = np.asarray(lows, dtype=float)
        arr_c = np.asarray(prices, dtype=float)
        lb = 20
        # Break of structure: new higher-high or lower-low
        recent_hh = np.max(arr_h[-lb - 2:-2])
        recent_ll = np.min(arr_l[-lb - 2:-2])
        bos_bull = arr_h[-1] > recent_hh  # Bullish BOS
        bos_bear = arr_l[-1] < recent_ll  # Bearish BOS
        # Fair Value Gap detection (3-candle pattern with gap)
        fvg_bull = arr_l[-1] > arr_h[-3]  # Gap up
        fvg_bear = arr_h[-1] < arr_l[-3]  # Gap down
        # Order block: last down candle before bullish BOS
        ob_bull = arr_c[-3] < arr_c[-4] and bos_bull
        ob_bear = arr_c[-3] > arr_c[-4] and bos_bear
        # Volume confirmation
        avg_vol = np.mean(np.asarray(volumes[-lb:], dtype=float))
        vol_spike = volumes[-1] > avg_vol * 1.3
        if bos_bull and (ob_bull or fvg_bull) and vol_spike:
            return 1
        elif bos_bear and (ob_bear or fvg_bear) and vol_spike:
            return -1
        # Weaker signal: BOS alone
        if bos_bull and vol_spike:
            return 1
        elif bos_bear and vol_spike:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 6. EMACloudStrategy
# ---------------------------------------------------------------------------
class EMACloudStrategy(SubStrategy):
    """Pine: EMA Cloud by Ripster47 — EMA 20/50 cloud with price position.
    Bullish when price above cloud and cloud is green (20>50)."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 55:
            return 0
        ema20 = ema(prices, 20)
        ema50 = ema(prices, 50)
        cloud_bull = ema20[-1] > ema50[-1]
        cloud_prev = ema20[-2] > ema50[-2]
        price_above = prices[-1] > max(ema20[-1], ema50[-1])
        price_below = prices[-1] < min(ema20[-1], ema50[-1])
        # Cloud flip
        if cloud_bull and not cloud_prev and price_above:
            return 1
        elif not cloud_bull and cloud_prev and price_below:
            return -1
        # Continuation: price above green cloud or below red cloud
        if cloud_bull and price_above:
            return 1
        elif not cloud_bull and price_below:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 7. VolumeProfileStrategy
# ---------------------------------------------------------------------------
class VolumeProfileStrategy(SubStrategy):
    """Pine: Volume Profile by LuxAlgo — POC, VAH, VAL from volume distribution.
    Buy when price bounces off VAL with volume, sell at VAH."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 30:
            return 0
        lb = 30
        arr_c = np.asarray(prices[-lb:], dtype=float)
        arr_v = np.asarray(volumes[-lb:], dtype=float)
        # Build simple volume profile using price bins
        price_min, price_max = np.min(arr_c), np.max(arr_c)
        if price_max == price_min:
            return 0
        n_bins = 20
        bins = np.linspace(price_min, price_max, n_bins + 1)
        vol_hist = np.zeros(n_bins)
        for i in range(len(arr_c)):
            idx = min(int((arr_c[i] - price_min) / (price_max - price_min) * n_bins), n_bins - 1)
            vol_hist[idx] += arr_v[i]
        poc_idx = np.argmax(vol_hist)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2.0
        # Value area: 70% of volume around POC
        total_vol = np.sum(vol_hist)
        va_vol = 0
        lo_idx, hi_idx = poc_idx, poc_idx
        va_vol += vol_hist[poc_idx]
        while va_vol < total_vol * 0.7 and (lo_idx > 0 or hi_idx < n_bins - 1):
            below = vol_hist[lo_idx - 1] if lo_idx > 0 else 0
            above = vol_hist[hi_idx + 1] if hi_idx < n_bins - 1 else 0
            if below >= above and lo_idx > 0:
                lo_idx -= 1
                va_vol += vol_hist[lo_idx]
            elif hi_idx < n_bins - 1:
                hi_idx += 1
                va_vol += vol_hist[hi_idx]
            else:
                break
        val = (bins[lo_idx] + bins[lo_idx + 1]) / 2.0  # Value Area Low
        vah = (bins[hi_idx] + bins[hi_idx + 1]) / 2.0  # Value Area High
        last = prices[-1]
        avg_vol = np.mean(arr_v)
        # Buy at VAL support, sell at VAH resistance
        if last <= val * 1.005 and volumes[-1] > avg_vol * 1.2:
            return 1
        elif last >= vah * 0.995 and volumes[-1] > avg_vol * 1.2:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 8. RSIDivergenceStrategy
# ---------------------------------------------------------------------------
class RSIDivergenceStrategy(SubStrategy):
    """Pine: RSI Divergence by ChrisMoody — Bullish/bearish divergence detection.
    Price makes lower low but RSI makes higher low = bullish divergence."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 30:
            return 0
        rsi_vals = rsi(prices, 14)
        arr_c = np.asarray(prices, dtype=float)
        arr_rsi = np.asarray(rsi_vals, dtype=float)
        # Find two recent swing lows/highs (simplified pivot detection)
        lb = 20
        seg_c = arr_c[-lb:]
        seg_r = arr_rsi[-lb:]
        # Bullish divergence: price lower low, RSI higher low
        price_lo1 = np.min(seg_c[:lb // 2])
        price_lo2 = np.min(seg_c[lb // 2:])
        rsi_at_lo1 = seg_r[np.argmin(seg_c[:lb // 2])]
        rsi_at_lo2 = seg_r[lb // 2 + np.argmin(seg_c[lb // 2:])]
        if price_lo2 < price_lo1 and rsi_at_lo2 > rsi_at_lo1 and arr_rsi[-1] < 40:
            return 1  # Bullish divergence
        # Bearish divergence: price higher high, RSI lower high
        price_hi1 = np.max(seg_c[:lb // 2])
        price_hi2 = np.max(seg_c[lb // 2:])
        rsi_at_hi1 = seg_r[np.argmax(seg_c[:lb // 2])]
        rsi_at_hi2 = seg_r[lb // 2 + np.argmax(seg_c[lb // 2:])]
        if price_hi2 > price_hi1 and rsi_at_hi2 < rsi_at_hi1 and arr_rsi[-1] > 60:
            return -1  # Bearish divergence
        return 0


# ---------------------------------------------------------------------------
# 9. MACDHistogramStrategy
# ---------------------------------------------------------------------------
class MACDHistogramStrategy(SubStrategy):
    """Pine: MACD Histogram by MACD_Trader — Histogram color change detection.
    Green->red = sell, red->green = buy. Simple but effective."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 35:
            return 0
        _, _, hist = macd(prices, 12, 26, 9)
        if math.isnan(hist[-1]) or math.isnan(hist[-2]):
            return 0
        # Histogram color change (increasing = green, decreasing = red)
        curr_rising = hist[-1] > hist[-2]
        prev_rising = hist[-2] > hist[-3] if len(hist) > 2 and not math.isnan(hist[-3]) else False
        # Color flip
        if curr_rising and not prev_rising:
            return 1   # Red to green
        elif not curr_rising and prev_rising:
            return -1  # Green to red
        # Zero-line cross for stronger signal
        if hist[-1] > 0 and hist[-2] <= 0:
            return 1
        elif hist[-1] < 0 and hist[-2] >= 0:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 10. StochRSIStrategy
# ---------------------------------------------------------------------------
class StochRSIStrategy(SubStrategy):
    """Pine: Stochastic RSI by TradingView — RSI of RSI with K/D crossover.
    Buy when K crosses above D in oversold, sell when K crosses below D in overbought."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 30:
            return 0
        rsi_vals = rsi(prices, 14)
        # StochRSI = stochastic applied to RSI values
        valid_rsi = [r for r in rsi_vals if not math.isnan(r)]
        if len(valid_rsi) < 14:
            return 0
        period = 14
        stoch_rsi = []
        for i in range(len(valid_rsi)):
            if i < period - 1:
                stoch_rsi.append(float('nan'))
            else:
                window = valid_rsi[i - period + 1:i + 1]
                lo = min(window)
                hi = max(window)
                stoch_rsi.append((valid_rsi[i] - lo) / (hi - lo) * 100 if hi != lo else 50)
        # K = SMA(StochRSI, 3), D = SMA(K, 3)
        k_line = sma(stoch_rsi, 3)
        d_line = sma(k_line, 3)
        if len(k_line) < 2 or math.isnan(k_line[-1]) or math.isnan(d_line[-1]):
            return 0
        k_cross_up = k_line[-1] > d_line[-1] and k_line[-2] <= d_line[-2]
        k_cross_dn = k_line[-1] < d_line[-1] and k_line[-2] >= d_line[-2]
        if k_cross_up and k_line[-1] < 30:
            return 1
        elif k_cross_dn and k_line[-1] > 70:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 11. IchimokuStrategy
# ---------------------------------------------------------------------------
class IchimokuStrategy(SubStrategy):
    """Pine: Ichimoku Cloud by TradingView — Tenkan/Kijun cross + cloud position.
    Full system: TK cross, price vs cloud, Chikou span confirmation."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 56:
            return 0
        arr_h = np.asarray(highs, dtype=float)
        arr_l = np.asarray(lows, dtype=float)

        def donchian_mid(h, l, period, idx):
            s = max(0, idx - period + 1)
            return (np.max(h[s:idx + 1]) + np.min(l[s:idx + 1])) / 2.0

        n = len(prices)
        # Tenkan-sen (9), Kijun-sen (26)
        tenkan = donchian_mid(arr_h, arr_l, 9, n - 1)
        kijun = donchian_mid(arr_h, arr_l, 26, n - 1)
        tenkan_prev = donchian_mid(arr_h, arr_l, 9, n - 2)
        kijun_prev = donchian_mid(arr_h, arr_l, 26, n - 2)
        # Senkou Span A/B (displaced 26 ahead, so use 26 bars ago for current cloud)
        if n > 52:
            ssa_now = (donchian_mid(arr_h, arr_l, 9, n - 27) +
                       donchian_mid(arr_h, arr_l, 26, n - 27)) / 2.0
            ssb_now = donchian_mid(arr_h, arr_l, 52, n - 27)
        else:
            ssa_now = (tenkan + kijun) / 2.0
            ssb_now = donchian_mid(arr_h, arr_l, 52, n - 1)
        cloud_top = max(ssa_now, ssb_now)
        cloud_bot = min(ssa_now, ssb_now)
        # TK Cross
        tk_cross_bull = tenkan > kijun and tenkan_prev <= kijun_prev
        tk_cross_bear = tenkan < kijun and tenkan_prev >= kijun_prev
        # Price vs cloud
        above_cloud = prices[-1] > cloud_top
        below_cloud = prices[-1] < cloud_bot
        if tk_cross_bull and above_cloud:
            return 1
        elif tk_cross_bear and below_cloud:
            return -1
        if above_cloud and tenkan > kijun:
            return 1
        elif below_cloud and tenkan < kijun:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 12. ParabolicSARStrategy
# ---------------------------------------------------------------------------
class ParabolicSARStrategy(SubStrategy):
    """Pine: Parabolic SAR by TradingView — SAR flip = trend reversal.
    Buy when price crosses above SAR, sell when below."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 10:
            return 0
        sar = parabolic_sar(highs, lows, prices, step=0.02, max_step=0.2)
        above_now = prices[-1] > sar[-1]
        above_prev = prices[-2] > sar[-2]
        if above_now and not above_prev:
            return 1   # SAR flip bullish
        elif not above_now and above_prev:
            return -1  # SAR flip bearish
        # Continuation with distance check
        atr_vals = atr(highs, lows, prices, 14)
        if not math.isnan(atr_vals[-1]) and atr_vals[-1] > 0:
            dist = abs(prices[-1] - sar[-1]) / atr_vals[-1]
            if above_now and dist > 1.5:
                return 1
            elif not above_now and dist > 1.5:
                return -1
        return 0


# ---------------------------------------------------------------------------
# 13. ADXDMIStrategy
# ---------------------------------------------------------------------------
class ADXDMIStrategy(SubStrategy):
    """Pine: ADX and DI by TradingView — DI+ cross DI- with ADX>25 filter.
    Classic trend strength system: strong trend + directional move."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 30:
            return 0
        adx_line, plus_di, minus_di = adx(highs, lows, prices, 14)
        if math.isnan(adx_line[-1]):
            return 0
        strong_trend = adx_line[-1] > 25
        di_cross_bull = plus_di[-1] > minus_di[-1] and plus_di[-2] <= minus_di[-2]
        di_cross_bear = plus_di[-1] < minus_di[-1] and plus_di[-2] >= minus_di[-2]
        if di_cross_bull and strong_trend:
            return 1
        elif di_cross_bear and strong_trend:
            return -1
        # Sustained trend
        if strong_trend and plus_di[-1] > minus_di[-1] and adx_line[-1] > adx_line[-2]:
            return 1
        elif strong_trend and minus_di[-1] > plus_di[-1] and adx_line[-1] > adx_line[-2]:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 14. CMFVolumeStrategy
# ---------------------------------------------------------------------------
class CMFVolumeStrategy(SubStrategy):
    """Pine: CMF + Volume by LazyBear — Chaikin Money Flow with volume surge.
    CMF > 0.1 with volume spike = accumulation, CMF < -0.1 = distribution."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 25:
            return 0
        cmf_vals = chaikin_money_flow(highs, lows, prices, volumes, 20)
        if not cmf_vals or math.isnan(cmf_vals[-1]):
            return 0
        avg_vol = np.mean(np.asarray(volumes[-20:], dtype=float))
        vol_surge = volumes[-1] > avg_vol * 1.3
        # Strong accumulation
        if cmf_vals[-1] > 0.10 and vol_surge:
            return 1
        elif cmf_vals[-1] < -0.10 and vol_surge:
            return -1
        # CMF zero-line cross
        if cmf_vals[-1] > 0.05 and cmf_vals[-2] <= 0.05:
            return 1
        elif cmf_vals[-1] < -0.05 and cmf_vals[-2] >= -0.05:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 15. WilliamsAlligatorStrategy
# ---------------------------------------------------------------------------
class WilliamsAlligatorStrategy(SubStrategy):
    """Pine: Williams Alligator by TradingView — 3 smoothed MAs (jaw/teeth/lips).
    Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3). Spread = trending."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 25:
            return 0
        # Median price
        mp = [(h + l) / 2.0 for h, l in zip(highs, lows)]
        # SMMA (Smoothed MA) approximated via SMA
        jaw = sma(mp, 13)    # Displaced 8 bars
        teeth = sma(mp, 8)   # Displaced 5 bars
        lips = sma(mp, 5)    # Displaced 3 bars
        # Use displaced values (shift back)
        j = jaw[-9] if len(jaw) > 9 and not math.isnan(jaw[-9]) else jaw[-1]
        t = teeth[-6] if len(teeth) > 6 and not math.isnan(teeth[-6]) else teeth[-1]
        l_ = lips[-4] if len(lips) > 4 and not math.isnan(lips[-4]) else lips[-1]
        if math.isnan(j) or math.isnan(t) or math.isnan(l_):
            return 0
        # Bullish: lips > teeth > jaw and price above all
        if l_ > t > j and prices[-1] > l_:
            return 1
        elif l_ < t < j and prices[-1] < l_:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 16. DonchianBreakoutStrategy
# ---------------------------------------------------------------------------
class DonchianBreakoutStrategy(SubStrategy):
    """Pine: Donchian Channel Breakout by TradingView — Turtle Trading system.
    Buy on 20-bar high breakout with volume, sell on 20-bar low breakdown."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 25:
            return 0
        period = 20
        arr_h = np.asarray(highs, dtype=float)
        arr_l = np.asarray(lows, dtype=float)
        arr_v = np.asarray(volumes, dtype=float)
        # Donchian channel (excluding current bar)
        dc_upper = np.max(arr_h[-period - 1:-1])
        dc_lower = np.min(arr_l[-period - 1:-1])
        dc_mid = (dc_upper + dc_lower) / 2.0
        avg_vol = np.mean(arr_v[-period:])
        vol_ok = volumes[-1] > avg_vol * 1.2
        # Breakout
        if prices[-1] > dc_upper and vol_ok:
            return 1
        elif prices[-1] < dc_lower and vol_ok:
            return -1
        # Price above/below midline as continuation
        if prices[-1] > dc_mid and prices[-2] <= dc_mid:
            return 1
        elif prices[-1] < dc_mid and prices[-2] >= dc_mid:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 17. ChandelierExitStrategy
# ---------------------------------------------------------------------------
class ChandelierExitStrategy(SubStrategy):
    """Pine: Chandelier Exit by everget — ATR-based trailing stop from highest high.
    Long stop = highest_high(22) - 3*ATR(22). Flip on stop hit."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 25:
            return 0
        period = 22
        mult = 3.0
        arr_h = np.asarray(highs, dtype=float)
        arr_l = np.asarray(lows, dtype=float)
        atr_vals = atr(highs, lows, prices, period)
        if math.isnan(atr_vals[-1]):
            return 0
        # Long chandelier exit
        hh = np.max(arr_h[-period:])
        long_stop = hh - mult * atr_vals[-1]
        # Short chandelier exit
        ll = np.min(arr_l[-period:])
        short_stop = ll + mult * atr_vals[-1]
        # Previous bar stops
        hh_prev = np.max(arr_h[-period - 1:-1])
        atr_prev = atr_vals[-2] if not math.isnan(atr_vals[-2]) else atr_vals[-1]
        long_stop_prev = hh_prev - mult * atr_prev
        ll_prev = np.min(arr_l[-period - 1:-1])
        short_stop_prev = ll_prev + mult * atr_prev
        # Direction
        above_long = prices[-1] > long_stop
        above_long_prev = prices[-2] > long_stop_prev
        below_short = prices[-1] < short_stop
        below_short_prev = prices[-2] < short_stop_prev
        if above_long and not above_long_prev:
            return 1
        elif below_short and not below_short_prev:
            return -1
        if above_long and prices[-1] > short_stop:
            return 1
        elif below_short and prices[-1] < long_stop:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 18. LinearRegressionStrategy
# ---------------------------------------------------------------------------
class LinearRegressionStrategy(SubStrategy):
    """Pine: Linear Regression Channel by TradingView — Mean reversion from
    regression line. Buy at -2 std deviation, sell at +2 std deviation."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 25:
            return 0
        period = 20
        arr = np.asarray(prices[-period:], dtype=float)
        x = np.arange(period, dtype=float)
        # Linear regression: y = mx + b
        m, b = np.polyfit(x, arr, 1)
        reg_line = m * x + b
        residuals = arr - reg_line
        std_dev = np.std(residuals)
        if std_dev == 0:
            return 0
        # Current z-score from regression
        current_dev = (arr[-1] - reg_line[-1]) / std_dev
        prev_dev = (arr[-2] - reg_line[-2]) / std_dev if len(arr) > 1 else 0
        # Mean reversion at extremes
        if current_dev < -2.0 and current_dev > prev_dev:
            return 1   # Oversold bounce
        elif current_dev > 2.0 and current_dev < prev_dev:
            return -1  # Overbought rejection
        # Trend direction from slope
        if current_dev < -1.0 and m > 0:
            return 1   # Pullback in uptrend
        elif current_dev > 1.0 and m < 0:
            return -1  # Rally in downtrend
        return 0


# ---------------------------------------------------------------------------
# 19. ElderRayStrategy
# ---------------------------------------------------------------------------
class ElderRayStrategy(SubStrategy):
    """Pine: Elder Ray Index by Alexander Elder — Bull/Bear power with EMA filter.
    Bull Power = High - EMA(13), Bear Power = Low - EMA(13).
    Buy when Bear Power < 0 but rising + EMA rising."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 20:
            return 0
        ema13 = ema(prices, 13)
        if math.isnan(ema13[-1]) or math.isnan(ema13[-2]):
            return 0
        bull_power = highs[-1] - ema13[-1]
        bear_power = lows[-1] - ema13[-1]
        bull_prev = highs[-2] - ema13[-2]
        bear_prev = lows[-2] - ema13[-2]
        ema_rising = ema13[-1] > ema13[-2]
        ema_falling = ema13[-1] < ema13[-2]
        # Buy: EMA rising, bear power < 0 but increasing
        if ema_rising and bear_power < 0 and bear_power > bear_prev:
            return 1
        # Sell: EMA falling, bull power > 0 but decreasing
        elif ema_falling and bull_power > 0 and bull_power < bull_prev:
            return -1
        return 0


# ---------------------------------------------------------------------------
# 20. AroonStrategy
# ---------------------------------------------------------------------------
class AroonStrategy(SubStrategy):
    """Pine: Aroon Indicator by TradingView — Time-based trend strength.
    Aroon Up = 100*(period - bars_since_high)/period.
    Cross + threshold (70/30) for entry."""

    def generate_signal(self, prices: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]) -> int:
        if len(prices) < 27:
            return 0
        period = 25
        arr_h = np.asarray(highs, dtype=float)
        arr_l = np.asarray(lows, dtype=float)
        # Bars since highest high and lowest low
        window_h = arr_h[-period - 1:]
        window_l = arr_l[-period - 1:]
        bars_since_high = period - np.argmax(window_h)
        bars_since_low = period - np.argmin(window_l)
        aroon_up = 100.0 * (period - bars_since_high) / period
        aroon_dn = 100.0 * (period - bars_since_low) / period
        # Previous bar
        window_h_p = arr_h[-period - 2:-1]
        window_l_p = arr_l[-period - 2:-1]
        bsh_p = period - np.argmax(window_h_p)
        bsl_p = period - np.argmin(window_l_p)
        aroon_up_p = 100.0 * (period - bsh_p) / period
        aroon_dn_p = 100.0 * (period - bsl_p) / period
        # Cross + threshold
        cross_up = aroon_up > aroon_dn and aroon_up_p <= aroon_dn_p
        cross_dn = aroon_dn > aroon_up and aroon_dn_p <= aroon_up_p
        if cross_up and aroon_up > 70:
            return 1
        elif cross_dn and aroon_dn > 70:
            return -1
        # Strong trend continuation
        if aroon_up > 80 and aroon_dn < 30:
            return 1
        elif aroon_dn > 80 and aroon_up < 30:
            return -1
        return 0
