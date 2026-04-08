"""
Train ALL ML Models — Multi-Timeframe High-Accuracy Pipeline
=============================================================
Fetches real OHLCV from Binance across MULTIPLE timeframes (1m to 1w)
and trains every model with strategy-aligned features for EMA(8) crossover
+ trailing SL L1->Ln system.

Multi-Timeframe Strategy:
  - 1m/5m: micro-structure, exact entry timing
  - 15m/1h: trend confirmation, volume profile
  - 4h/1d: macro trend, support/resistance levels
  - 1w: cycle position, major trend direction

All features feed into models that learn:
  "Given this multi-timeframe context, will the next EMA(8) crossover
   reach L4+ (runner) or die at L1?"

Target: 90%+ recall/precision on L4+ detection.

Usage:
    python -m src.scripts.train_all_models
    python -m src.scripts.train_all_models --asset ALL --bars 20000
    python -m src.scripts.train_all_models --loop --loop-hours 4
"""

import os
import sys
import time
import logging
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def fetch_training_data(asset='BTC', timeframe='5m', bars=5000):
    """Fetch real OHLCV from Binance with pagination."""
    print(f"\n{'='*60}")
    print(f"FETCHING {bars} {timeframe} BARS FOR {asset}")
    print(f"{'='*60}")

    try:
        import ccxt
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        symbol = f"{asset}/USDT"
        all_ohlcv = []

        tf_ms = {'1m': 60000, '5m': 300000, '15m': 900000, '1h': 3600000,
                 '4h': 14400000, '1d': 86400000, '1w': 604800000}
        candle_ms = tf_ms.get(timeframe, 300000)
        since = int(time.time() * 1000) - (bars * candle_ms)

        remaining = bars
        while remaining > 0:
            limit = min(1000, remaining)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            remaining -= len(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < limit:
                break
            time.sleep(0.3)
            if len(all_ohlcv) % 2000 < 1001:
                print(f"    ... fetched {len(all_ohlcv)}/{bars} bars")

        if len(all_ohlcv) >= 100:
            seen = set()
            deduped = []
            for x in all_ohlcv:
                if x[0] not in seen:
                    seen.add(x[0])
                    deduped.append(x)
            all_ohlcv = sorted(deduped, key=lambda x: x[0])

            data = {
                'timestamps': [x[0] for x in all_ohlcv],
                'opens': [float(x[1]) for x in all_ohlcv],
                'highs': [float(x[2]) for x in all_ohlcv],
                'lows': [float(x[3]) for x in all_ohlcv],
                'closes': [float(x[4]) for x in all_ohlcv],
                'volumes': [float(x[5]) for x in all_ohlcv],
            }
            print(f"  Fetched {len(all_ohlcv)} bars ({timeframe}) from Binance")
            return data
    except Exception as e:
        print(f"  Binance fetch failed: {e}")

    try:
        import ccxt
        exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'},
            'sandbox': True,
        })
        symbol = f"{asset}/USDT:USDT"
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=min(200, bars))
        if ohlcv:
            data = {
                'timestamps': [x[0] for x in ohlcv],
                'opens': [float(x[1]) for x in ohlcv],
                'highs': [float(x[2]) for x in ohlcv],
                'lows': [float(x[3]) for x in ohlcv],
                'closes': [float(x[4]) for x in ohlcv],
                'volumes': [float(x[5]) for x in ohlcv],
            }
            print(f"  Fetched {len(ohlcv)} bars from Bybit testnet")
            return data
    except Exception as e:
        print(f"  Bybit fetch failed: {e}")

    # Fallback: Binance Vision S3 (bypasses API geo-blocks)
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/maintenance'))
        from download_vision_data import fetch_vision_ohlcv
        symbol = f"{asset}USDT"
        print(f"  Trying Binance Vision S3 for {symbol} ({timeframe})...")
        df = fetch_vision_ohlcv(symbol, timeframe=timeframe, start_year=2017, data_dir='data/training_cache')
        if df is not None and len(df) >= 100:
            # Trim to requested bar count (take most recent)
            if len(df) > bars:
                df = df.tail(bars).reset_index(drop=True)
            # Convert timestamp to milliseconds
            if 'timestamp' in df.columns:
                ts = df['timestamp']
                if hasattr(ts.iloc[0], 'timestamp'):
                    timestamps = [int(t.timestamp() * 1000) for t in ts]
                else:
                    timestamps = ts.astype(int).tolist()
            else:
                timestamps = list(range(len(df)))
            data = {
                'timestamps': timestamps,
                'opens': df['open'].astype(float).tolist(),
                'highs': df['high'].astype(float).tolist(),
                'lows': df['low'].astype(float).tolist(),
                'closes': df['close'].astype(float).tolist(),
                'volumes': df['volume'].astype(float).tolist(),
            }
            print(f"  Fetched {len(df)} bars from Binance Vision S3")
            return data
    except Exception as e:
        print(f"  Binance Vision fetch failed: {e}")

    print("  ERROR: No exchange connectivity")
    return None


def fetch_multi_timeframe_data(asset='BTC', base_bars=20000):
    """
    Fetch data across ALL timeframes for multi-timeframe feature engineering.
    Returns dict of {timeframe: data_dict}.
    """
    print(f"\n{'#'*60}")
    print(f"# MULTI-TIMEFRAME DATA: {asset}")
    print(f"{'#'*60}")

    timeframes = {
        '1m': min(base_bars * 5, 50000),  # 5x more 1m bars (same time coverage)
        '5m': base_bars,                    # Base timeframe
        '15m': base_bars // 3,             # 1/3 bars = same time
        '1h': base_bars // 12,             # 1/12 bars
        '4h': base_bars // 48,             # 1/48 bars
        '1d': base_bars // 288,            # 1/288 bars
    }

    all_data = {}
    for tf, n_bars in timeframes.items():
        n_bars = max(200, n_bars)  # Minimum 200 bars per timeframe
        data = fetch_training_data(asset, tf, n_bars)
        if data and len(data['closes']) >= 100:
            all_data[tf] = data
            print(f"  {tf}: {len(data['closes'])} bars | "
                  f"${data['closes'][-1]:,.2f} | "
                  f"Range: ${min(data['closes']):,.2f}-${max(data['closes']):,.2f}")
        else:
            print(f"  {tf}: FAILED")
        time.sleep(0.5)

    return all_data


# =====================================================================
# MULTI-TIMEFRAME FEATURES (45 features for maximum pattern detection)
# =====================================================================

def compute_mtf_features(all_data, seq_len=30, n_features=45):
    """
    Build MULTI-TIMEFRAME feature sequences.
    Combines micro (1m/5m), meso (15m/1h), and macro (4h/1d) features.

    Features 0-29: Same 30 strategy features from 5m data
    Features 30-34: 15m trend context (EMA slope, RSI, MACD, volume, ATR)
    Features 35-39: 1h trend context (EMA slope, RSI, BB position, volume, momentum)
    Features 40-44: 4h/1d macro context (trend direction, cycle position, vol regime, support/resistance distance, macro momentum)
    """
    from src.indicators.indicators import ema, atr, rsi, macd, bollinger_bands, stochastic, obv, roc

    # Primary data is 5m
    if '5m' not in all_data:
        print("  ERROR: No 5m data for MTF features")
        return None, None

    data_5m = all_data['5m']
    closes = np.array(data_5m['closes'], dtype=float)
    highs = np.array(data_5m['highs'], dtype=float)
    lows = np.array(data_5m['lows'], dtype=float)
    opens = np.array(data_5m.get('opens', data_5m['closes']), dtype=float)
    volumes = np.array(data_5m['volumes'], dtype=float)
    timestamps = data_5m['timestamps']
    n = len(closes)

    if n < max(seq_len + 55, 200):
        return None, None

    # ═══════════════════════════════════════════════════════════════
    # Pre-compute 5m indicators
    # ═══════════════════════════════════════════════════════════════
    ema_8 = ema(list(closes), 8)
    ema_21 = ema(list(closes), 21)
    ema_50 = ema(list(closes), 50)
    atr_14 = atr(list(highs), list(lows), list(closes), 14)
    rsi_14 = rsi(list(closes), 14)
    macd_l, macd_s, macd_h = macd(list(closes))
    bb_up, bb_lo, bb_mid = bollinger_bands(list(closes), 20)
    stoch_k, stoch_d = stochastic(list(highs), list(lows), list(closes), 14, 3)
    obv_vals = obv(list(closes), list(volumes))
    roc_12 = roc(list(closes), 12)

    # ═══════════════════════════════════════════════════════════════
    # Pre-compute higher TF indicators (align to 5m timestamps)
    # ═══════════════════════════════════════════════════════════════
    htf_features = {}
    for tf in ['15m', '1h', '4h', '1d']:
        if tf not in all_data:
            continue
        d = all_data[tf]
        c = np.array(d['closes'], dtype=float)
        h = np.array(d['highs'], dtype=float)
        l = np.array(d['lows'], dtype=float)
        v = np.array(d['volumes'], dtype=float)
        ts = d['timestamps']

        if len(c) < 30:
            continue

        htf_ema8 = np.array(ema(list(c), 8))
        htf_ema21 = np.array(ema(list(c), 21))
        htf_rsi = np.array(rsi(list(c), 14))
        htf_macd_l, htf_macd_s, htf_macd_h = macd(list(c))
        htf_atr = np.array(atr(list(h), list(l), list(c), 14))
        htf_bb_up, htf_bb_lo, htf_bb_mid = bollinger_bands(list(c), 20)

        htf_features[tf] = {
            'timestamps': ts, 'closes': c, 'ema8': htf_ema8, 'ema21': htf_ema21,
            'rsi': htf_rsi, 'macd_h': np.array(htf_macd_h), 'atr': htf_atr,
            'bb_up': np.array(htf_bb_up), 'bb_lo': np.array(htf_bb_lo), 'bb_mid': np.array(htf_bb_mid),
            'volumes': v,
        }

    def _get_htf_idx(tf, timestamp_5m):
        """Find the most recent HTF bar before this 5m timestamp."""
        if tf not in htf_features:
            return -1
        ts_list = htf_features[tf]['timestamps']
        # Binary search for latest HTF bar <= timestamp_5m
        lo, hi = 0, len(ts_list) - 1
        result = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if ts_list[mid] <= timestamp_5m:
                result = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    # ═══════════════════════════════════════════════════════════════
    # Build per-bar feature matrix (45 features)
    # ═══════════════════════════════════════════════════════════════
    feat = np.zeros((n, n_features), dtype=np.float32)

    for i in range(55, n):
        c_val = closes[i]
        o_val = opens[i] if i < len(opens) else closes[i-1]
        h_val = highs[i]
        l_val = lows[i]
        v_val = volumes[i]
        prev_c = closes[i-1]

        if c_val <= 0 or prev_c <= 0:
            continue

        # ── Features 0-4: EMA CROSSOVER QUALITY ──
        if ema_8[i] > 0 and i >= 3:
            feat[i, 0] = (ema_8[i] - ema_8[i-3]) / ema_8[i-3] * 100
        feat[i, 1] = (c_val - ema_8[i]) / c_val * 100 if ema_8[i] > 0 else 0
        feat[i, 2] = (ema_8[i] - ema_21[i]) / c_val * 100 if ema_21[i] > 0 else 0
        feat[i, 3] = (ema_8[i] - ema_50[i]) / c_val * 100 if ema_50[i] > 0 else 0
        consec = 0
        for j in range(i-1, max(i-15, 1), -1):
            if ema_8[j] > ema_8[j-1]: consec += 1
            elif ema_8[j] < ema_8[j-1]: consec -= 1
            else: break
        feat[i, 4] = consec / 10.0

        # ── Features 5-9: MOMENTUM ──
        feat[i, 5] = (rsi_14[i] - 50) / 50 if i < len(rsi_14) else 0
        feat[i, 6] = macd_h[i] / c_val * 1000 if i < len(macd_h) and c_val > 0 else 0
        feat[i, 7] = (stoch_k[i] - 50) / 50 if i < len(stoch_k) else 0
        feat[i, 8] = roc_12[i] / 10 if i < len(roc_12) else 0
        if i >= 2 and i < len(macd_h):
            feat[i, 9] = 1.0 if macd_h[i] > 0 and macd_h[i-1] < 0 else (-1.0 if macd_h[i] < 0 and macd_h[i-1] > 0 else 0)

        # ── Features 10-14: VOLATILITY ──
        feat[i, 10] = atr_14[i] / c_val * 100 if atr_14[i] > 0 else 0
        if i >= 5:
            feat[i, 11] = (atr_14[i] - atr_14[i-5]) / (atr_14[i-5] + 1e-12)
        if i < len(bb_up) and (bb_up[i] - bb_lo[i]) > 0:
            feat[i, 12] = (bb_up[i] - bb_lo[i]) / bb_mid[i] * 100 if bb_mid[i] > 0 else 0
            feat[i, 13] = (c_val - bb_lo[i]) / (bb_up[i] - bb_lo[i])
        feat[i, 14] = (c_val - closes[i-3]) / closes[i-3] * 100 if closes[i-3] > 0 else 0

        # ── Features 15-19: VOLUME ──
        avg_vol = np.mean(volumes[max(0,i-20):i]) if i >= 20 else np.mean(volumes[:i+1])
        feat[i, 15] = v_val / (avg_vol + 1e-10) - 1.0
        if i >= 5 and i < len(obv_vals):
            feat[i, 16] = (obv_vals[i] - obv_vals[i-5]) / (abs(obv_vals[i-5]) + 1e-10)
        feat[i, 17] = 1.0 if c_val > o_val else -1.0
        if i >= 10:
            recent_vol = np.mean(volumes[i-5:i])
            older_vol = np.mean(volumes[i-10:i-5])
            feat[i, 18] = (recent_vol - older_vol) / (older_vol + 1e-10)
        up_vol = sum(volumes[jj] for jj in range(max(0,i-10), i) if closes[jj] > (closes[jj-1] if jj > 0 else closes[jj]))
        dn_vol = sum(volumes[jj] for jj in range(max(0,i-10), i) if closes[jj] < (closes[jj-1] if jj > 0 else closes[jj]))
        feat[i, 19] = (up_vol - dn_vol) / (up_vol + dn_vol + 1e-10)

        # ── Features 20-24: CANDLE PATTERNS ──
        total_range = h_val - l_val if h_val > l_val else 1e-10
        body = abs(c_val - o_val)
        feat[i, 20] = body / total_range
        feat[i, 21] = (h_val - max(c_val, o_val)) / total_range
        feat[i, 22] = (min(c_val, o_val) - l_val) / total_range
        green_streak = 0
        for j in range(i, max(i-10, 0), -1):
            if closes[j] > (opens[j] if j < len(opens) else closes[j-1]):
                green_streak += 1
            else:
                break
        feat[i, 23] = green_streak / 10.0
        if i >= 20:
            h20 = max(highs[i-20:i])
            l20 = min(lows[i-20:i])
            feat[i, 24] = (c_val - l20) / (h20 - l20) if h20 > l20 else 0.5

        # ── Features 25-29: STRUCTURE ──
        feat[i, 25] = (c_val - closes[i-5]) / closes[i-5] * 100 if i >= 5 and closes[i-5] > 0 else 0
        if i >= 20:
            feat[i, 26] = (c_val - closes[i-20]) / closes[i-20] * 100
            feat[i, 27] = (max(highs[i-20:i]) - c_val) / c_val * 100
            feat[i, 28] = (c_val - min(lows[i-20:i])) / c_val * 100
        feat[i, 29] = np.log1p(v_val) / 15.0

        # ═══════════════════════════════════════════════════════════
        # Features 30-34: 15-MINUTE TREND CONTEXT
        # ═══════════════════════════════════════════════════════════
        ts_i = timestamps[i] if i < len(timestamps) else 0
        idx_15m = _get_htf_idx('15m', ts_i)
        if idx_15m >= 3 and '15m' in htf_features:
            hf = htf_features['15m']
            # 30: 15m EMA(8) slope
            if hf['ema8'][idx_15m] > 0:
                feat[i, 30] = (hf['ema8'][idx_15m] - hf['ema8'][idx_15m-3]) / hf['ema8'][idx_15m-3] * 100
            # 31: 15m RSI normalized
            feat[i, 31] = (hf['rsi'][idx_15m] - 50) / 50 if idx_15m < len(hf['rsi']) else 0
            # 32: 15m MACD histogram
            feat[i, 32] = hf['macd_h'][idx_15m] / hf['closes'][idx_15m] * 1000 if idx_15m < len(hf['macd_h']) and hf['closes'][idx_15m] > 0 else 0
            # 33: 15m relative volume
            if idx_15m >= 20:
                avg_v15 = np.mean(hf['volumes'][idx_15m-20:idx_15m])
                feat[i, 33] = hf['volumes'][idx_15m] / (avg_v15 + 1e-10) - 1.0
            # 34: 15m ATR expansion
            if idx_15m >= 5 and hf['atr'][idx_15m-5] > 0:
                feat[i, 34] = (hf['atr'][idx_15m] - hf['atr'][idx_15m-5]) / hf['atr'][idx_15m-5]

        # ═══════════════════════════════════════════════════════════
        # Features 35-39: 1-HOUR TREND CONTEXT
        # ═══════════════════════════════════════════════════════════
        idx_1h = _get_htf_idx('1h', ts_i)
        if idx_1h >= 3 and '1h' in htf_features:
            hf = htf_features['1h']
            # 35: 1h EMA(8) slope
            if hf['ema8'][idx_1h] > 0:
                feat[i, 35] = (hf['ema8'][idx_1h] - hf['ema8'][idx_1h-3]) / hf['ema8'][idx_1h-3] * 100
            # 36: 1h RSI
            feat[i, 36] = (hf['rsi'][idx_1h] - 50) / 50 if idx_1h < len(hf['rsi']) else 0
            # 37: 1h BB position (where is price in the band?)
            if idx_1h < len(hf['bb_up']) and (hf['bb_up'][idx_1h] - hf['bb_lo'][idx_1h]) > 0:
                feat[i, 37] = (hf['closes'][idx_1h] - hf['bb_lo'][idx_1h]) / (hf['bb_up'][idx_1h] - hf['bb_lo'][idx_1h])
            # 38: 1h volume trend
            if idx_1h >= 10:
                rec_v = np.mean(hf['volumes'][idx_1h-5:idx_1h])
                old_v = np.mean(hf['volumes'][idx_1h-10:idx_1h-5])
                feat[i, 38] = (rec_v - old_v) / (old_v + 1e-10)
            # 39: 1h momentum (EMA8 vs EMA21)
            if hf['ema21'][idx_1h] > 0:
                feat[i, 39] = (hf['ema8'][idx_1h] - hf['ema21'][idx_1h]) / hf['closes'][idx_1h] * 100

        # ═══════════════════════════════════════════════════════════
        # Features 40-44: 4H / DAILY MACRO CONTEXT
        # ═══════════════════════════════════════════════════════════
        idx_4h = _get_htf_idx('4h', ts_i)
        if idx_4h >= 3 and '4h' in htf_features:
            hf = htf_features['4h']
            # 40: 4h trend direction (EMA8 vs EMA21)
            if hf['ema21'][idx_4h] > 0:
                feat[i, 40] = (hf['ema8'][idx_4h] - hf['ema21'][idx_4h]) / hf['closes'][idx_4h] * 100
            # 41: 4h RSI (macro overbought/oversold)
            feat[i, 41] = (hf['rsi'][idx_4h] - 50) / 50 if idx_4h < len(hf['rsi']) else 0

        idx_1d = _get_htf_idx('1d', ts_i)
        if idx_1d >= 3 and '1d' in htf_features:
            hf = htf_features['1d']
            # 42: Daily trend (EMA8 slope)
            if hf['ema8'][idx_1d] > 0 and idx_1d >= 3:
                feat[i, 42] = (hf['ema8'][idx_1d] - hf['ema8'][idx_1d-3]) / hf['ema8'][idx_1d-3] * 100
            # 43: Daily BB position
            if idx_1d < len(hf['bb_up']) and (hf['bb_up'][idx_1d] - hf['bb_lo'][idx_1d]) > 0:
                feat[i, 43] = (hf['closes'][idx_1d] - hf['bb_lo'][idx_1d]) / (hf['bb_up'][idx_1d] - hf['bb_lo'][idx_1d])
            # 44: Daily momentum (20-bar return)
            if idx_1d >= 20:
                feat[i, 44] = (hf['closes'][idx_1d] - hf['closes'][idx_1d-20]) / hf['closes'][idx_1d-20] * 100

    # Replace NaN/Inf, clip
    feat = np.nan_to_num(feat, nan=0.0, posinf=3.0, neginf=-3.0)
    feat = np.clip(feat, -5.0, 5.0)

    # ═══════════════════════════════════════════════════════════════
    # Create sequences + BINARY labels (SKIP vs TRADE)
    # SKIP = L1 death (trade loses money, trailing SL stops at loss)
    # TRADE = L2+ (trailing SL locks profit at breakeven or better)
    #
    # This directly maps to our strategy: ML says ENTER or SKIP,
    # then trailing SL handles profit locking at L1→L7 automatically
    # ═══════════════════════════════════════════════════════════════
    ema_8_lbl = np.array(ema(list(closes), 8))
    atr_14_lbl = np.array(atr(list(highs), list(lows), list(closes), 14))

    sequences = []
    labels = []
    l_level_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    start_idx = max(55, seq_len)

    for i in range(start_idx, n - 30):
        seq = feat[i - seq_len:i]
        sequences.append(seq)

        cur = closes[i]
        if cur <= 0:
            labels.append(0)
            l_level_dist[0] += 1
            continue

        entry = cur
        atr_e = atr_14_lbl[i] if i < len(atr_14_lbl) and atr_14_lbl[i] > 0 else cur * 0.01

        # Simulate BOTH directions with trailing SL, take the best
        best_l = 0
        for direction in ['long', 'short']:
            if direction == 'long':
                sl = entry - atr_e * 1.5
                peak = entry
            else:
                sl = entry + atr_e * 1.5
                peak = entry

            max_l_level = 0
            # EXACT trailing SL ratchet from executor
            ratchet = [(0.3, 0.0), (0.6, 0.10), (1.0, 0.20), (1.5, 0.30),
                       (2.0, 0.40), (3.0, 0.50), (5.0, 0.60)]

            for j in range(i+1, min(i+30, n)):
                p = closes[j]
                if direction == 'long':
                    pnl = (p - entry) / entry * 100
                    if p > peak: peak = p
                    if p <= sl or pnl <= -2.0: break
                    for mi, (min_pnl, protect) in enumerate(reversed(ratchet)):
                        if pnl >= min_pnl:
                            if protect == 0: sl = max(sl, entry)
                            else: sl = max(sl, entry + (peak - entry) * protect)
                            max_l_level = max(max_l_level, len(ratchet) - mi)
                            break
                else:
                    pnl = (entry - p) / entry * 100
                    if p < peak: peak = p
                    if p >= sl or pnl <= -2.0: break
                    for mi, (min_pnl, protect) in enumerate(reversed(ratchet)):
                        if pnl >= min_pnl:
                            if protect == 0: sl = min(sl, entry)
                            else: sl = min(sl, entry - (entry - peak) * protect)
                            max_l_level = max(max_l_level, len(ratchet) - mi)
                            break
            best_l = max(best_l, max_l_level)

        l_level_dist[min(best_l, 7)] += 1

        # BINARY: SKIP (0) = L1 death, TRADE (1) = L2+ (trailing SL locks profit)
        if best_l >= 2:
            labels.append(1)  # TRADE — trailing SL will lock profits
        else:
            labels.append(0)  # SKIP — would die at L1, lose money

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    n_skip = np.sum(y == 0)
    n_trade = np.sum(y == 1)
    print(f"  MTF Sequences: {X.shape} | SKIP={n_skip} ({n_skip/len(y):.0%}) TRADE={n_trade} ({n_trade/len(y):.0%})")
    print(f"  L-level distribution: {dict(l_level_dist)}")
    return X, y


def compute_strategy_features(closes, highs, lows, opens, volumes, seq_len=30, n_features=50):
    """
    Build feature sequences for 5m-only data.
    50 features: 30 strategy + 5 Kalman + 5 EMA new-line inflection + 10 Category B risk/ML.

    Features 0-29:  Core strategy (EMA momentum, RSI, MACD, ATR, BB, volume, candle patterns)
    Features 30-34: Kalman trend filter (slope, SNR, acceleration, residual, direction)
    Features 35-39: EMA new line inflection (new bars, prior bars, is_new_line, is_fresh, price-EMA/ATR)
    Features 40-49: Category B risk/ML (EVT tail, MC risk, Hawkes intensity, TFT forecast,
                     sentiment, regime) — online autonomous features from risk engines
    """
    from src.indicators.indicators import ema, atr, rsi, macd, bollinger_bands, stochastic, obv, roc

    n = len(closes)
    if n < max(seq_len + 55, 100):
        return None, None

    ema_8 = ema(closes, 8)
    ema_21 = ema(closes, 21)
    ema_50 = ema(closes, 50)
    atr_14 = atr(highs, lows, closes, 14)
    rsi_14 = rsi(closes, 14)
    macd_l, macd_s, macd_h = macd(closes)
    bb_up, bb_lo, bb_mid = bollinger_bands(closes, 20)
    stoch_k, stoch_d = stochastic(highs, lows, closes, 14, 3)
    obv_vals = obv(closes, volumes)
    roc_12 = roc(closes, 12)

    # ── Kalman Trend Filter: smoothed trend + signal-to-noise ──
    kalman_slope = np.zeros(n)
    kalman_snr = np.zeros(n)
    kalman_accel = np.zeros(n)
    kalman_residual = np.zeros(n)
    kalman_signal = np.zeros(n)
    try:
        from src.models.kalman_filter import KalmanTrendFilter
        kf = KalmanTrendFilter()
        k_result = kf.filter(np.array(closes, dtype=float))
        if k_result:
            kalman_slope = k_result.get('slope', kalman_slope)
            kalman_snr = k_result.get('snr', kalman_snr)
            kalman_accel = k_result.get('accel', kalman_accel)
            kalman_residual = k_result.get('residual', kalman_residual)
            kalman_signal = k_result.get('signal', kalman_signal).astype(float)
    except Exception:
        pass  # Kalman unavailable — features stay zero

    feat = np.zeros((n, n_features))
    for i in range(55, n):
        c = closes[i]
        o = opens[i] if i < len(opens) else closes[i-1]
        h = highs[i]; l = lows[i]; v = volumes[i]
        if c <= 0: continue

        if ema_8[i] > 0 and i >= 3:
            feat[i, 0] = (ema_8[i] - ema_8[i-3]) / ema_8[i-3] * 100
        feat[i, 1] = (c - ema_8[i]) / c * 100 if ema_8[i] > 0 else 0
        feat[i, 2] = (ema_8[i] - ema_21[i]) / c * 100 if ema_21[i] > 0 else 0
        feat[i, 3] = (ema_8[i] - ema_50[i]) / c * 100 if ema_50[i] > 0 else 0
        consec = 0
        for j in range(i-1, max(i-15, 1), -1):
            if ema_8[j] > ema_8[j-1]: consec += 1
            elif ema_8[j] < ema_8[j-1]: consec -= 1
            else: break
        feat[i, 4] = consec / 10.0
        feat[i, 5] = (rsi_14[i] - 50) / 50 if i < len(rsi_14) else 0
        feat[i, 6] = macd_h[i] / c * 1000 if i < len(macd_h) and c > 0 else 0
        feat[i, 7] = (stoch_k[i] - 50) / 50 if i < len(stoch_k) else 0
        feat[i, 8] = roc_12[i] / 10 if i < len(roc_12) else 0
        if i >= 2 and i < len(macd_h):
            feat[i, 9] = 1.0 if macd_h[i] > 0 and macd_h[i-1] < 0 else (-1.0 if macd_h[i] < 0 and macd_h[i-1] > 0 else 0)
        feat[i, 10] = atr_14[i] / c * 100 if atr_14[i] > 0 else 0
        if i >= 5: feat[i, 11] = (atr_14[i] - atr_14[i-5]) / (atr_14[i-5] + 1e-12)
        if i < len(bb_up) and (bb_up[i] - bb_lo[i]) > 0:
            feat[i, 12] = (bb_up[i] - bb_lo[i]) / bb_mid[i] * 100 if bb_mid[i] > 0 else 0
            feat[i, 13] = (c - bb_lo[i]) / (bb_up[i] - bb_lo[i])
        feat[i, 14] = (c - closes[i-3]) / closes[i-3] * 100 if closes[i-3] > 0 else 0
        avg_vol = np.mean(volumes[max(0,i-20):i]) if i >= 20 else np.mean(volumes[:i+1])
        feat[i, 15] = v / (avg_vol + 1e-10) - 1.0
        if i >= 5 and i < len(obv_vals):
            feat[i, 16] = (obv_vals[i] - obv_vals[i-5]) / (abs(obv_vals[i-5]) + 1e-10)
        feat[i, 17] = 1.0 if c > o else -1.0
        if i >= 10:
            recent_vol = np.mean(volumes[i-5:i])
            older_vol = np.mean(volumes[i-10:i-5])
            feat[i, 18] = (recent_vol - older_vol) / (older_vol + 1e-10)
        up_vol = sum(volumes[jj] for jj in range(max(0,i-10), i) if closes[jj] > (closes[jj-1] if jj > 0 else closes[jj]))
        dn_vol = sum(volumes[jj] for jj in range(max(0,i-10), i) if closes[jj] < (closes[jj-1] if jj > 0 else closes[jj]))
        feat[i, 19] = (up_vol - dn_vol) / (up_vol + dn_vol + 1e-10)
        total_range = h - l if h > l else 1e-10
        body = abs(c - o)
        feat[i, 20] = body / total_range
        feat[i, 21] = (h - max(c, o)) / total_range
        feat[i, 22] = (min(c, o) - l) / total_range
        green = 0
        for jj in range(i, max(i-10, 0), -1):
            if closes[jj] > (opens[jj] if jj < len(opens) else closes[jj-1]): green += 1
            else: break
        feat[i, 23] = green / 10.0
        if i >= 20:
            h20 = max(highs[i-20:i]); l20 = min(lows[i-20:i])
            feat[i, 24] = (c - l20) / (h20 - l20) if h20 > l20 else 0.5
        feat[i, 25] = (c - closes[i-5]) / closes[i-5] * 100 if i >= 5 and closes[i-5] > 0 else 0
        if i >= 20: feat[i, 26] = (c - closes[i-20]) / closes[i-20] * 100
        if i >= 20: feat[i, 27] = (max(highs[i-20:i]) - c) / c * 100
        if i >= 20: feat[i, 28] = (c - min(lows[i-20:i])) / c * 100
        feat[i, 29] = np.log1p(v) / 15.0

        # ── KALMAN TREND FILTER FEATURES (30-34) ──
        if n_features >= 35:
            feat[i, 30] = kalman_slope[i] * 1000      # Trend slope (scaled up from log-space)
            feat[i, 31] = min(kalman_snr[i], 5.0)     # Signal-to-noise (capped at 5)
            feat[i, 32] = kalman_accel[i] * 10000      # Slope acceleration (momentum change)
            feat[i, 33] = kalman_residual[i] * 100     # Innovation (breakout detection)
            feat[i, 34] = kalman_signal[i]             # Direction: +1=up, -1=down, 0=flat

        # ── EMA NEW LINE INFLECTION FEATURES (35-39) ──
        # These teach the model the EXACT entry pattern from reference images:
        # When EMA changes direction after trending for 3+ bars = NEW LINE = ENTER
        if n_features >= 40 and i >= 5:
            # Feature 35: New line bars (how many bars the current EMA direction has been going)
            new_bars = 0
            cur_dir = 1 if ema_8[i] > ema_8[i-1] else -1
            for j in range(i-1, max(i-20, 1), -1):
                d = 1 if ema_8[j] > ema_8[j-1] else -1
                if d == cur_dir:
                    new_bars += 1
                else:
                    break
            feat[i, 35] = new_bars / 10.0  # Normalized: 0.1 = 1 bar, 1.0 = 10 bars

            # Feature 36: Prior trend bars (how long the OPPOSITE direction lasted before inflection)
            prior_bars = 0
            start_j = i - 1 - new_bars
            if start_j > 1:
                for j in range(start_j, max(start_j - 20, 1), -1):
                    d = 1 if ema_8[j] > ema_8[j-1] else -1
                    if d != cur_dir:  # Opposite direction
                        prior_bars += 1
                    else:
                        break
            feat[i, 36] = prior_bars / 10.0

            # Feature 37: Is NEW LINE (prior trend 3+ bars then direction flipped)
            feat[i, 37] = 1.0 if prior_bars >= 3 and new_bars >= 1 else 0.0

            # Feature 38: Is FRESH entry (new line just started, 1-5 bars old)
            feat[i, 38] = 1.0 if prior_bars >= 3 and 1 <= new_bars <= 5 else 0.0

            # Feature 39: Price position relative to EMA at inflection
            # >0 = price above EMA (good for CALL new line), <0 = below (good for PUT)
            feat[i, 39] = (c - ema_8[i]) / (atr_14[i] + 1e-10) if atr_14[i] > 0 else 0

        # ── CATEGORY B: RISK/ML FEATURES (40-49) ──
        # These features capture tail risk, event clustering, temporal attention,
        # and regime context that the core 40 features miss.
        # Computed from the SAME price data — no external dependency needed for training.
        if n_features >= 50 and i >= 100:
            # --- Feature 40: EVT tail severity (rolling GPD ξ parameter) ---
            # Positive ξ = heavy tail (Pareto), higher = fatter tails = more risk
            # Computed from rolling window of log returns
            _window_rets = np.diff(np.log(np.array(closes[max(0,i-500):i+1], dtype=float) + 1e-12))
            if len(_window_rets) >= 50:
                _losses = -_window_rets  # flip: positive = losses
                _threshold_q = np.quantile(_losses, 0.90)
                _exceedances = _losses[_losses > _threshold_q] - _threshold_q
                if len(_exceedances) >= 10:
                    _mean_exc = np.mean(_exceedances)
                    _var_exc = np.var(_exceedances)
                    _ratio = _mean_exc**2 / (_var_exc + 1e-12)
                    _xi = 0.5 * (1 - _ratio)
                    feat[i, 40] = float(np.clip(_xi, -0.5, 1.0))  # GPD shape param
                else:
                    feat[i, 40] = 0.0

            # --- Feature 41: EVT tail ratio (EVT VaR / Normal VaR) ---
            # >1 = fatter tails than normal distribution predicts
            if len(_window_rets) >= 50:
                _normal_var = float(np.mean(_window_rets) - 2.326 * np.std(_window_rets))
                # Simple EVT VaR approximation using threshold + scale
                if len(_exceedances) >= 10 and _mean_exc > 0:
                    _evt_var = _threshold_q + _mean_exc * 1.5
                    feat[i, 41] = float(np.clip(_evt_var / (abs(_normal_var) + 1e-12), 0.5, 5.0)) - 1.0
                else:
                    feat[i, 41] = 0.0

            # --- Feature 42: Monte Carlo risk score (forward VaR severity) ---
            # EWMA volatility (λ=0.94 RiskMetrics) → GBM simulation → VaR/risk_budget
            _recent_rets = np.diff(np.array(closes[max(0,i-100):i+1], dtype=float)) / (np.array(closes[max(0,i-100):i], dtype=float) + 1e-12)
            if len(_recent_rets) >= 20:
                _lam = 0.94
                _ewma_var = np.var(_recent_rets[:10]) if len(_recent_rets) >= 10 else np.var(_recent_rets)
                for _r in _recent_rets[10:]:
                    _ewma_var = _lam * _ewma_var + (1 - _lam) * _r ** 2
                _vol = float(np.sqrt(max(_ewma_var, 1e-12)))
                # Simplified MC: VaR ≈ vol * z_score * sqrt(horizon)
                _mc_var_approx = _vol * 1.645 * np.sqrt(24)  # 95% VaR, 24-bar horizon
                _risk_budget = 0.02
                feat[i, 42] = float(np.clip(_mc_var_approx / (_risk_budget + 1e-10), 0.0, 1.0))
            else:
                feat[i, 42] = 0.5

            # --- Feature 43: MC position scale (inverse of risk) ---
            # How much position the risk budget allows: high = safe, low = dangerous
            if feat[i, 42] > 0.01:
                feat[i, 43] = float(np.clip(_risk_budget / (feat[i, 42] * _risk_budget + 1e-10), 0.05, 1.0))
            else:
                feat[i, 43] = 1.0

            # --- Feature 44: Hawkes intensity (event clustering) ---
            # Self-exciting: recent large moves increase probability of more large moves
            _abs_rets = np.abs(_recent_rets)
            _mean_abs = np.mean(_abs_rets)
            _std_abs = np.std(_abs_rets)
            _event_threshold = _mean_abs + 2.0 * _std_abs
            _event_mask = _abs_rets > _event_threshold
            _event_times = np.where(_event_mask)[0].astype(float)
            if len(_event_times) >= 3:
                # Hawkes intensity: μ + α * Σ exp(-β*(t-t_i))
                _mu = 0.1; _alpha = 0.5; _beta = 1.0
                _t_now = float(len(_recent_rets))
                _excitation = _alpha * np.sum(np.exp(-_beta * (_t_now - _event_times)))
                feat[i, 44] = float(np.clip((_mu + _excitation) / 2.0, 0.0, 2.0))
            else:
                feat[i, 44] = 0.05  # baseline

            # --- Feature 45: Temporal attention forecast direction ---
            # Simplified attention-based directional signal from recent returns
            if len(_recent_rets) >= 30:
                _context = _recent_rets[-30:]
                _norm = (_context - np.mean(_context)) / (np.std(_context) + 1e-9)
                # Exponential recency weighting (attention proxy)
                _weights = np.exp(np.linspace(-2, 0, len(_norm)))
                _weights /= _weights.sum()
                _weighted_signal = np.sum(_norm * _weights)
                feat[i, 45] = float(np.clip(_weighted_signal, -2.0, 2.0))
            else:
                feat[i, 45] = 0.0

            # --- Feature 46: Temporal forecast confidence ---
            # Higher when recent returns are consistent (low dispersion in attention)
            if len(_recent_rets) >= 30:
                _recent_std = np.std(_recent_rets[-10:])
                _older_std = np.std(_recent_rets[-30:-10])
                _consistency = 1.0 - float(np.clip(_recent_std / (_older_std + 1e-10), 0, 2))
                feat[i, 46] = float(np.clip(abs(_consistency), 0, 1))
            else:
                feat[i, 46] = 0.0

            # --- Feature 47: Sentiment proxy from price action ---
            # Ratio of up-volume to total volume (buying pressure proxy)
            # In live trading this slot gets real FinBERT scores
            _up_vol = sum(volumes[jj] for jj in range(max(0,i-20), i) if closes[jj] > closes[max(0,jj-1)])
            _total_vol = sum(volumes[jj] for jj in range(max(0,i-20), i)) + 1e-10
            feat[i, 47] = float((_up_vol / _total_vol) * 2.0 - 1.0)  # [-1, +1]

            # --- Feature 48: Regime encoded ---
            # -1 = bearish (declining EMA + high vol), 0 = neutral, +1 = bullish
            _ema_trend = (ema_8[i] - ema_50[i]) / (c + 1e-10) if ema_50[i] > 0 else 0
            _vol_ratio = (atr_14[i] / c * 100) if c > 0 else 0
            _avg_vol_ratio = np.mean([atr_14[j] / closes[j] * 100 for j in range(max(0,i-50), i) if closes[j] > 0]) if i >= 50 else _vol_ratio
            if _ema_trend > 0.005 and _vol_ratio < _avg_vol_ratio * 1.3:
                feat[i, 48] = 1.0   # Bullish: trending up, vol not extreme
            elif _ema_trend < -0.005 or _vol_ratio > _avg_vol_ratio * 1.5:
                feat[i, 48] = -1.0  # Bearish: trending down or vol extreme
            else:
                feat[i, 48] = 0.0   # Neutral

            # --- Feature 49: Anomaly severity (multi-factor z-score) ---
            # High = current bar is anomalous (price z-score * vol z-score)
            if i >= 20:
                _price_mean = np.mean(closes[i-20:i])
                _price_std = np.std(closes[i-20:i])
                _price_z = abs(c - _price_mean) / (_price_std + 1e-10)
                _vol_z = abs(v - np.mean(volumes[max(0,i-20):i])) / (np.std(volumes[max(0,i-20):i]) + 1e-10)
                feat[i, 49] = float(np.clip(_price_z * _vol_z / 10.0, 0, 2.0))
            else:
                feat[i, 49] = 0.0

    feat = np.nan_to_num(feat, nan=0.0, posinf=2.0, neginf=-2.0)
    feat = np.clip(feat, -5.0, 5.0)

    # BINARY labels: SKIP (0) = L1 death, TRADE (1) = L2+ (trailing SL locks profit)
    ema_8_lbl = ema(list(closes), 8)
    atr_14_lbl = atr(list(highs), list(lows), list(closes), 14)

    sequences = []
    labels = []
    start_idx = max(55, seq_len)

    for i in range(start_idx, n - 30):
        seq = feat[i - seq_len:i]
        sequences.append(seq)

        cur = closes[i]
        if cur <= 0:
            labels.append(0)
            continue

        entry = cur
        atr_e = atr_14_lbl[i] if i < len(atr_14_lbl) and atr_14_lbl[i] > 0 else cur * 0.01

        best_l = 0
        for direction in ['long', 'short']:
            if direction == 'long':
                sl = entry - atr_e * 1.5; peak = entry
            else:
                sl = entry + atr_e * 1.5; peak = entry

            max_l_level = 0
            ratchet = [(0.3, 0.0), (0.6, 0.10), (1.0, 0.20), (1.5, 0.30),
                       (2.0, 0.40), (3.0, 0.50), (5.0, 0.60)]

            for j in range(i+1, min(i+30, n)):
                p = closes[j]
                if direction == 'long':
                    pnl = (p - entry) / entry * 100
                    if p > peak: peak = p
                    if p <= sl or pnl <= -2.0: break
                    for mi, (min_pnl, protect) in enumerate(reversed(ratchet)):
                        if pnl >= min_pnl:
                            if protect == 0: sl = max(sl, entry)
                            else: sl = max(sl, entry + (peak - entry) * protect)
                            max_l_level = max(max_l_level, len(ratchet) - mi)
                            break
                else:
                    pnl = (entry - p) / entry * 100
                    if p < peak: peak = p
                    if p >= sl or pnl <= -2.0: break
                    for mi, (min_pnl, protect) in enumerate(reversed(ratchet)):
                        if pnl >= min_pnl:
                            if protect == 0: sl = min(sl, entry)
                            else: sl = min(sl, entry - (entry - peak) * protect)
                            max_l_level = max(max_l_level, len(ratchet) - mi)
                            break
            best_l = max(best_l, max_l_level)

        # BINARY: SKIP (L1 death) vs TRADE (L2+ = trailing SL locks profit)
        labels.append(1 if best_l >= 2 else 0)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    n_skip = np.sum(y == 0)
    n_trade = np.sum(y == 1)
    print(f"  Sequences: {X.shape} | SKIP={n_skip} ({n_skip/len(y):.0%}) TRADE={n_trade} ({n_trade/len(y):.0%})")
    return X, y


# =====================================================================
# EMA CROSSOVER LABELS (used by LightGBM)
# =====================================================================

def compute_ema_crossover_labels(closes, highs, lows, atr_vals, ema_8):
    """
    Strategy-aligned labels: simulate trailing SL L1->Ln system.
    Labels: 0=L1 DEATH, 1=L2-L3, 2=L4+ RUNNER
    """
    n = len(closes)
    labels = np.full(n, -1, dtype=np.int64)

    for i in range(2, n - 30):
        if ema_8[i] <= 0 or ema_8[i-1] <= 0:
            continue

        bullish = closes[i] > ema_8[i] and closes[i-1] <= ema_8[i-1]
        bearish = closes[i] < ema_8[i] and closes[i-1] >= ema_8[i-1]

        if not bullish and not bearish:
            continue

        entry = closes[i]
        atr_at_entry = atr_vals[i] if i < len(atr_vals) and atr_vals[i] > 0 else entry * 0.01
        is_long = bullish

        sl = entry - (atr_at_entry * 1.5) if is_long else entry + (atr_at_entry * 1.5)
        peak = entry
        max_l_level = 0

        ratchet = [
            (0.3, 0.0), (0.6, 0.10), (1.0, 0.20), (1.5, 0.30),
            (2.0, 0.40), (3.0, 0.50), (5.0, 0.60),
        ]

        for j in range(i + 1, min(i + 30, n)):
            price = closes[j]

            if is_long:
                pnl_pct = (price - entry) / entry * 100
                if price > peak: peak = price
                if price <= sl: break
                for min_pnl, protect in reversed(ratchet):
                    if pnl_pct >= min_pnl:
                        if protect == 0.0: sl = max(sl, entry)
                        else: sl = max(sl, entry + (peak - entry) * protect)
                        l_num = ratchet.index((min_pnl, protect)) + 1
                        max_l_level = max(max_l_level, l_num)
                        break
                if pnl_pct <= -2.0:
                    max_l_level = 0
                    break
            else:
                pnl_pct = (entry - price) / entry * 100
                if price < peak: peak = price
                if price >= sl: break
                for min_pnl, protect in reversed(ratchet):
                    if pnl_pct >= min_pnl:
                        if protect == 0.0: sl = min(sl, entry)
                        else: sl = min(sl, entry - (entry - peak) * protect)
                        l_num = ratchet.index((min_pnl, protect)) + 1
                        max_l_level = max(max_l_level, l_num)
                        break
                if pnl_pct <= -2.0:
                    max_l_level = 0
                    break

        if max_l_level <= 1:
            labels[i] = 0
        elif max_l_level <= 3:
            labels[i] = 1
        else:
            labels[i] = 2

    return labels


# =====================================================================
# MODEL TRAINERS
# =====================================================================

def train_lightgbm(data, asset='BTC', all_data=None):
    """
    Train LightGBM on strategy features with BINARY labels.
    SKIP = L1 death (trailing SL stops at loss)
    TRADE = L2+ (trailing SL locks profit at breakeven or better)
    Uses ALL bars for maximum training data.
    """
    print(f"\n{'='*60}")
    print(f"1. TRAINING LIGHTGBM CLASSIFIER ({asset})")
    print(f"  Labels: SKIP (L1 death) vs TRADE (L2+ profit locked)")
    print(f"{'='*60}")

    try:
        import lightgbm as lgb
        from src.indicators.indicators import ema, atr, rsi, macd, bollinger_bands, stochastic, obv, roc

        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        opens = data.get('opens', closes)
        volumes = data['volumes']
        n = len(closes)

        # ── Compute features: 50 = 30 base + 5 Kalman + 5 inflection + 10 Category B ──
        n_features = 50
        print(f"  Using 50 strategy features (30 base + 5 Kalman + 5 inflection + 10 Category B)")

        # Compute flat feature matrix
        ema_8 = ema(closes, 8)
        ema_21 = ema(closes, 21)
        ema_50 = ema(closes, 50)
        atr_14 = atr(highs, lows, closes, 14)
        rsi_14 = rsi(closes, 14)
        macd_l, macd_s, macd_h = macd(closes)
        bb_up, bb_lo, bb_mid = bollinger_bands(closes, 20)
        stoch_k, stoch_d = stochastic(highs, lows, closes, 14, 3)
        obv_vals = obv(closes, volumes)
        roc_12 = roc(closes, 12)

        # Kalman trend filter — smoothed trend + SNR for crossover quality
        kalman_slope_arr = np.zeros(n)
        kalman_snr_arr = np.zeros(n)
        kalman_accel_arr = np.zeros(n)
        kalman_residual_arr = np.zeros(n)
        kalman_signal_arr = np.zeros(n)
        try:
            from src.models.kalman_filter import KalmanTrendFilter
            kf = KalmanTrendFilter()
            k_result = kf.filter(np.array(closes, dtype=float))
            if k_result:
                kalman_slope_arr = k_result.get('slope', kalman_slope_arr)
                kalman_snr_arr = k_result.get('snr', kalman_snr_arr)
                kalman_accel_arr = k_result.get('accel', kalman_accel_arr)
                kalman_residual_arr = k_result.get('residual', kalman_residual_arr)
                kalman_signal_arr = k_result.get('signal', kalman_signal_arr).astype(float)
        except Exception:
            pass

        feature_names = [
            'ema8_slope', 'price_vs_ema8', 'ema8_vs_ema21', 'ema8_vs_ema50',
            'ema_consecutive', 'rsi_norm', 'macd_hist_norm', 'stoch_k_norm',
            'roc12_norm', 'macd_cross', 'atr_pct', 'atr_expansion',
            'bb_width', 'bb_position', 'price_velocity_3bar',
            'rel_volume', 'obv_slope', 'vol_delta_dir', 'vol_trend',
            'vol_up_vs_down', 'body_ratio', 'upper_wick', 'lower_wick',
            'green_streak', 'price_vs_20bar_range', 'return_5bar',
            'return_20bar', 'dist_to_resistance', 'dist_to_support', 'log_vol_norm',
            'kalman_slope', 'kalman_snr', 'kalman_accel', 'kalman_residual', 'kalman_signal',
            'new_line_bars', 'prior_trend_bars', 'is_new_line', 'is_fresh_entry', 'price_vs_ema_atr',
            # Category B risk/ML features (40-49)
            'evt_tail_severity', 'evt_tail_ratio', 'mc_risk_score', 'mc_position_scale',
            'hawkes_intensity', 'tft_forecast_dir', 'tft_confidence',
            'sentiment_proxy', 'regime_encoded', 'anomaly_severity',
        ]

        feat = np.zeros((n, 50))
        for i in range(55, n):
            c = closes[i]
            o = opens[i] if i < len(opens) else closes[i-1]
            h = highs[i]; l = lows[i]; v = volumes[i]
            if c <= 0: continue

            if ema_8[i] > 0 and i >= 3:
                feat[i, 0] = (ema_8[i] - ema_8[i-3]) / ema_8[i-3] * 100
            feat[i, 1] = (c - ema_8[i]) / c * 100 if ema_8[i] > 0 else 0
            feat[i, 2] = (ema_8[i] - ema_21[i]) / c * 100 if ema_21[i] > 0 else 0
            feat[i, 3] = (ema_8[i] - ema_50[i]) / c * 100 if ema_50[i] > 0 else 0
            consec = 0
            for j in range(i-1, max(i-15, 1), -1):
                if ema_8[j] > ema_8[j-1]: consec += 1
                elif ema_8[j] < ema_8[j-1]: consec -= 1
                else: break
            feat[i, 4] = consec / 10.0
            feat[i, 5] = (rsi_14[i] - 50) / 50 if i < len(rsi_14) else 0
            feat[i, 6] = macd_h[i] / c * 1000 if i < len(macd_h) and c > 0 else 0
            feat[i, 7] = (stoch_k[i] - 50) / 50 if i < len(stoch_k) else 0
            feat[i, 8] = roc_12[i] / 10 if i < len(roc_12) else 0
            if i >= 2 and i < len(macd_h):
                feat[i, 9] = 1.0 if macd_h[i] > 0 and macd_h[i-1] < 0 else (-1.0 if macd_h[i] < 0 and macd_h[i-1] > 0 else 0)
            feat[i, 10] = atr_14[i] / c * 100 if atr_14[i] > 0 else 0
            if i >= 5: feat[i, 11] = (atr_14[i] - atr_14[i-5]) / (atr_14[i-5] + 1e-12)
            if i < len(bb_up) and (bb_up[i] - bb_lo[i]) > 0:
                feat[i, 12] = (bb_up[i] - bb_lo[i]) / bb_mid[i] * 100 if bb_mid[i] > 0 else 0
                feat[i, 13] = (c - bb_lo[i]) / (bb_up[i] - bb_lo[i])
            feat[i, 14] = (c - closes[i-3]) / closes[i-3] * 100 if closes[i-3] > 0 else 0
            avg_vol = np.mean(volumes[max(0,i-20):i]) if i >= 20 else np.mean(volumes[:i+1])
            feat[i, 15] = v / (avg_vol + 1e-10) - 1.0
            if i >= 5 and i < len(obv_vals):
                feat[i, 16] = (obv_vals[i] - obv_vals[i-5]) / (abs(obv_vals[i-5]) + 1e-10)
            feat[i, 17] = 1.0 if c > o else -1.0
            if i >= 10:
                recent_vol = np.mean(volumes[i-5:i])
                older_vol = np.mean(volumes[i-10:i-5])
                feat[i, 18] = (recent_vol - older_vol) / (older_vol + 1e-10)
            up_vol = sum(volumes[jj] for jj in range(max(0,i-10), i) if closes[jj] > (closes[jj-1] if jj > 0 else closes[jj]))
            dn_vol = sum(volumes[jj] for jj in range(max(0,i-10), i) if closes[jj] < (closes[jj-1] if jj > 0 else closes[jj]))
            feat[i, 19] = (up_vol - dn_vol) / (up_vol + dn_vol + 1e-10)
            total_range = h - l if h > l else 1e-10
            body = abs(c - o)
            feat[i, 20] = body / total_range
            feat[i, 21] = (h - max(c, o)) / total_range
            feat[i, 22] = (min(c, o) - l) / total_range
            green = 0
            for jj in range(i, max(i-10, 0), -1):
                if closes[jj] > (opens[jj] if jj < len(opens) else closes[jj-1]): green += 1
                else: break
            feat[i, 23] = green / 10.0
            if i >= 20:
                h20 = max(highs[i-20:i]); l20 = min(lows[i-20:i])
                feat[i, 24] = (c - l20) / (h20 - l20) if h20 > l20 else 0.5
            feat[i, 25] = (c - closes[i-5]) / closes[i-5] * 100 if i >= 5 and closes[i-5] > 0 else 0
            if i >= 20: feat[i, 26] = (c - closes[i-20]) / closes[i-20] * 100
            if i >= 20: feat[i, 27] = (max(highs[i-20:i]) - c) / c * 100
            if i >= 20: feat[i, 28] = (c - min(lows[i-20:i])) / c * 100
            feat[i, 29] = np.log1p(v) / 15.0

            # KALMAN features (30-34)
            feat[i, 30] = kalman_slope_arr[i] * 1000
            feat[i, 31] = min(kalman_snr_arr[i], 5.0)
            feat[i, 32] = kalman_accel_arr[i] * 10000
            feat[i, 33] = kalman_residual_arr[i] * 100
            feat[i, 34] = kalman_signal_arr[i]

            # EMA NEW LINE INFLECTION features (35-39)
            if i >= 5:
                new_bars = 0
                cur_dir = 1 if ema_8[i] > ema_8[i-1] else -1
                for j in range(i-1, max(i-20, 1), -1):
                    d = 1 if ema_8[j] > ema_8[j-1] else -1
                    if d == cur_dir:
                        new_bars += 1
                    else:
                        break
                feat[i, 35] = new_bars / 10.0

                prior_bars = 0
                start_j = i - 1 - new_bars
                if start_j > 1:
                    for j in range(start_j, max(start_j - 20, 1), -1):
                        d = 1 if ema_8[j] > ema_8[j-1] else -1
                        if d != cur_dir:
                            prior_bars += 1
                        else:
                            break
                feat[i, 36] = prior_bars / 10.0
                feat[i, 37] = 1.0 if prior_bars >= 3 and new_bars >= 1 else 0.0
                feat[i, 38] = 1.0 if prior_bars >= 3 and 1 <= new_bars <= 5 else 0.0
                feat[i, 39] = (c - ema_8[i]) / (atr_14[i] + 1e-10) if atr_14[i] > 0 else 0

            # ── CATEGORY B RISK/ML FEATURES (40-49) ── same logic as compute_strategy_features
            if i >= 100:
                _window_rets = np.diff(np.log(np.array(closes[max(0,i-500):i+1], dtype=float) + 1e-12))
                if len(_window_rets) >= 50:
                    _losses = -_window_rets
                    _threshold_q = np.quantile(_losses, 0.90)
                    _exceedances = _losses[_losses > _threshold_q] - _threshold_q
                    if len(_exceedances) >= 10:
                        _mean_exc = np.mean(_exceedances)
                        _var_exc = np.var(_exceedances)
                        _ratio = _mean_exc**2 / (_var_exc + 1e-12)
                        feat[i, 40] = float(np.clip(0.5 * (1 - _ratio), -0.5, 1.0))
                    _normal_var = float(np.mean(_window_rets) - 2.326 * np.std(_window_rets))
                    if len(_exceedances) >= 10 and _mean_exc > 0:
                        _evt_var = _threshold_q + _mean_exc * 1.5
                        feat[i, 41] = float(np.clip(_evt_var / (abs(_normal_var) + 1e-12), 0.5, 5.0)) - 1.0

                _recent_rets = np.diff(np.array(closes[max(0,i-100):i+1], dtype=float)) / (np.array(closes[max(0,i-100):i], dtype=float) + 1e-12)
                if len(_recent_rets) >= 20:
                    _lam = 0.94
                    _ewma_var = np.var(_recent_rets[:10]) if len(_recent_rets) >= 10 else np.var(_recent_rets)
                    for _r in _recent_rets[10:]:
                        _ewma_var = _lam * _ewma_var + (1 - _lam) * _r ** 2
                    _vol = float(np.sqrt(max(_ewma_var, 1e-12)))
                    _mc_var_approx = _vol * 1.645 * np.sqrt(24)
                    _risk_budget = 0.02
                    feat[i, 42] = float(np.clip(_mc_var_approx / (_risk_budget + 1e-10), 0.0, 1.0))
                    feat[i, 43] = float(np.clip(_risk_budget / (feat[i, 42] * _risk_budget + 1e-10), 0.05, 1.0)) if feat[i, 42] > 0.01 else 1.0

                    _abs_rets = np.abs(_recent_rets)
                    _mean_abs = np.mean(_abs_rets); _std_abs = np.std(_abs_rets)
                    _event_threshold = _mean_abs + 2.0 * _std_abs
                    _event_times = np.where(_abs_rets > _event_threshold)[0].astype(float)
                    if len(_event_times) >= 3:
                        _t_now = float(len(_recent_rets))
                        _excitation = 0.5 * np.sum(np.exp(-1.0 * (_t_now - _event_times)))
                        feat[i, 44] = float(np.clip((0.1 + _excitation) / 2.0, 0.0, 2.0))
                    else:
                        feat[i, 44] = 0.05

                    if len(_recent_rets) >= 30:
                        _context = _recent_rets[-30:]
                        _norm = (_context - np.mean(_context)) / (np.std(_context) + 1e-9)
                        _weights = np.exp(np.linspace(-2, 0, len(_norm)))
                        _weights /= _weights.sum()
                        feat[i, 45] = float(np.clip(np.sum(_norm * _weights), -2.0, 2.0))
                        _recent_std = np.std(_recent_rets[-10:])
                        _older_std = np.std(_recent_rets[-30:-10])
                        feat[i, 46] = float(np.clip(abs(1.0 - _recent_std / (_older_std + 1e-10)), 0, 1))

                _up_vol = sum(volumes[jj] for jj in range(max(0,i-20), i) if closes[jj] > closes[max(0,jj-1)])
                _total_vol = sum(volumes[jj] for jj in range(max(0,i-20), i)) + 1e-10
                feat[i, 47] = float((_up_vol / _total_vol) * 2.0 - 1.0)

                _ema_trend = (ema_8[i] - ema_50[i]) / (c + 1e-10) if ema_50[i] > 0 else 0
                _vol_ratio = (atr_14[i] / c * 100) if c > 0 else 0
                _avg_vol_r = np.mean([atr_14[j] / closes[j] * 100 for j in range(max(0,i-50), i) if closes[j] > 0]) if i >= 50 else _vol_ratio
                feat[i, 48] = 1.0 if _ema_trend > 0.005 and _vol_ratio < _avg_vol_r * 1.3 else (-1.0 if _ema_trend < -0.005 or _vol_ratio > _avg_vol_r * 1.5 else 0.0)

                if i >= 20:
                    _price_mean = np.mean(closes[i-20:i]); _price_std = np.std(closes[i-20:i])
                    _price_z = abs(c - _price_mean) / (_price_std + 1e-10)
                    _vol_z = abs(v - np.mean(volumes[max(0,i-20):i])) / (np.std(volumes[max(0,i-20):i]) + 1e-10)
                    feat[i, 49] = float(np.clip(_price_z * _vol_z / 10.0, 0, 2.0))

        feat = np.nan_to_num(feat, nan=0.0, posinf=2.0, neginf=-2.0)
        feat = np.clip(feat, -5.0, 5.0)

        # ── ALL-BAR BINARY labels: SKIP (L1 death) vs TRADE (L2+ profit locked) ──
        # Simulate trailing SL for EVERY bar (in both directions)
        all_labels = np.zeros(n, dtype=np.int64)
        for i in range(55, n - 30):
            cur = closes[i]
            if cur <= 0: continue
            entry = cur
            atr_e = atr_14[i] if atr_14[i] > 0 else cur * 0.01

            best_l = 0
            for direction in ['long', 'short']:
                if direction == 'long':
                    sl = entry - atr_e * 1.5; peak = entry
                else:
                    sl = entry + atr_e * 1.5; peak = entry
                max_l = 0
                ratchet = [(0.3, 0.0), (0.6, 0.10), (1.0, 0.20), (1.5, 0.30),
                           (2.0, 0.40), (3.0, 0.50), (5.0, 0.60)]
                for j in range(i+1, min(i+30, n)):
                    p = closes[j]
                    if direction == 'long':
                        pnl = (p - entry) / entry * 100
                        if p > peak: peak = p
                        if p <= sl or pnl <= -2.0: break
                    else:
                        pnl = (entry - p) / entry * 100
                        if p < peak: peak = p
                        if p >= sl or pnl <= -2.0: break
                    for mi, (min_pnl, protect) in enumerate(reversed(ratchet)):
                        if pnl >= min_pnl:
                            if direction == 'long':
                                if protect == 0: sl = max(sl, entry)
                                else: sl = max(sl, entry + (peak - entry) * protect)
                            else:
                                if protect == 0: sl = min(sl, entry)
                                else: sl = min(sl, entry - (entry - peak) * protect)
                            max_l = max(max_l, len(ratchet) - mi)
                            break
                best_l = max(best_l, max_l)

            # BINARY: SKIP (0) = L1 death, TRADE (1) = L2+ (trailing SL locks profit)
            all_labels[i] = 1 if best_l >= 2 else 0

        valid_idx = np.arange(55, n - 30)
        X = feat[valid_idx]
        y = all_labels[valid_idx]

        n_skip = np.sum(y == 0)
        n_trade = np.sum(y == 1)
        print(f"  Training data: {len(X)} bars | SKIP={n_skip} ({n_skip/len(y):.0%}) TRADE={n_trade} ({n_trade/len(y):.0%})")

        split = int(len(X) * 0.80)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # ── Jitter oversampling for TRADE class ──
        trade_mask = y_train == 1
        n_trade_train = np.sum(trade_mask)
        n_skip_train = np.sum(y_train == 0)
        if n_trade_train > 0 and n_trade_train < n_skip_train * 0.6:
            target = int(n_skip_train * 0.5)
            n_synthetic = target - n_trade_train
            trade_X = X_train[trade_mask]
            aug_X = [X_train]
            aug_y = [y_train]
            while n_synthetic > 0:
                batch = min(len(trade_X), n_synthetic)
                idx = np.random.choice(len(trade_X), batch, replace=True)
                synthetic = trade_X[idx] + np.random.normal(0, 0.005, (batch, 30)).astype(np.float32)
                aug_X.append(synthetic)
                aug_y.append(np.full(batch, 1, dtype=np.int64))
                n_synthetic -= batch
            X_train = np.concatenate(aug_X)
            y_train = np.concatenate(aug_y)
            print(f"  After TRADE oversampling: {len(X_train)} | TRADE={np.sum(y_train==1)}")

        # ── Class weights ──
        n_s = max(1, np.sum(y_train == 0))
        n_t = max(1, np.sum(y_train == 1))
        scale = n_s / n_t
        sample_w = np.where(y_train == 1, scale * 1.5, 1.0)
        print(f"  TRADE weight: {scale*1.5:.2f}x")

        dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_w, feature_name=feature_names)
        dtest = lgb.Dataset(X_test, label=y_test, feature_name=feature_names, reference=dtrain)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.03,
            'min_data_in_leaf': 10,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'is_unbalance': True,
            'verbose': -1,
        }

        model = lgb.train(params, dtrain, num_boost_round=1000,
                          valid_sets=[dtest],
                          callbacks=[lgb.log_evaluation(200), lgb.early_stopping(80)])

        # ── Evaluate binary ──
        y_prob = model.predict(X_test)
        y_pred = (y_prob > 0.5).astype(int)
        accuracy = np.mean(y_pred == y_test)

        for cls, name in [(0, 'SKIP(L1-death)'), (1, 'TRADE(L2+-profit)')]:
            mask = y_test == cls
            pred_mask = y_pred == cls
            tp = np.sum((y_pred == cls) & (y_test == cls))
            fp = np.sum((y_pred == cls) & (y_test != cls))
            fn = np.sum((y_pred != cls) & (y_test == cls))
            prec = tp / (tp + fp + 1e-10)
            rec = tp / (tp + fn + 1e-10)
            f1 = 2 * prec * rec / (prec + rec + 1e-10)
            print(f"    {name}: n={mask.sum()} | P={prec:.0%} R={rec:.0%} F1={f1:.0%}")

        # ── TRADE threshold optimization ──
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.20, 0.80, 0.005):
            preds = (y_prob >= thresh).astype(int)
            tp = np.sum((preds == 1) & (y_test == 1))
            fp = np.sum((preds == 1) & (y_test == 0))
            fn = np.sum((preds == 0) & (y_test == 1))
            p = tp / (tp + fp + 1e-10)
            r = tp / (tp + fn + 1e-10)
            f1 = 2 * p * r / (p + r + 1e-10)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        print(f"    TRADE optimal threshold: {best_thresh:.3f} (F1={best_f1:.0%})")
        print(f"    Overall accuracy: {accuracy:.0%}")

        imp = model.feature_importance(importance_type='gain')
        top = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)[:8]
        print(f"  Top features: {[f'{nm}={v:.0f}' for nm, v in top]}")

        os.makedirs('models', exist_ok=True)
        model.save_model(f'models/lgbm_{asset.lower()}_trained.txt')
        model.save_model('models/lgbm_latest.txt')

        import json
        thresh_info = {'trade_threshold': float(best_thresh), 'trade_f1': float(best_f1)}
        with open(f'models/lgbm_{asset.lower()}_thresholds.json', 'w') as f:
            json.dump(thresh_info, f)

        print(f"  Saved: models/lgbm_{asset.lower()}_trained.txt + thresholds")
        return True
    except Exception as e:
        print(f"  LightGBM error: {e}")
        import traceback; traceback.print_exc()
        return False


def train_hmm(data, asset='BTC'):
    """Train HMM on returns + volatility + EMA trend strength."""
    print(f"\n{'='*60}")
    print(f"2. TRAINING HMM REGIME DETECTOR ({asset})")
    print(f"{'='*60}")

    try:
        from src.models.hmm_regime import HMMRegimeDetector
        from src.indicators.indicators import ema

        closes = np.array(data['closes'], dtype=float)
        volumes = np.array(data['volumes'], dtype=float)

        log_ret = np.diff(np.log(closes + 1e-12))
        vol_20 = np.array([np.std(log_ret[max(0,i-20):i]) if i > 1 else 0.001
                           for i in range(1, len(log_ret)+1)])

        ema_8 = np.array(ema(list(closes), 8))
        ema_slope = np.zeros(len(closes))
        for i in range(3, len(closes)):
            if ema_8[i] > 0:
                ema_slope[i] = (ema_8[i] - ema_8[i-1]) / ema_8[i-1] * 100
        ema_slope_aligned = ema_slope[1:]

        mn = min(len(log_ret), len(vol_20), len(ema_slope_aligned))
        ret = log_ret[-mn:]
        vol = vol_20[-mn:]
        vc = ema_slope_aligned[-mn:]

        detector = HMMRegimeDetector(n_states=4, n_iter=200)
        success = detector.fit(ret, vol, vc)

        if success:
            result = detector.predict(ret[-100:], vol[-100:], vc[-100:])
            print(f"  HMM fitted on {mn} observations")
            print(f"  Current regime: {result.get('regime', '?')} (conf: {result.get('confidence', 0):.2f})")
            print(f"  Crisis prob: {result.get('crisis_probability', 0):.3f}")

            import pickle
            os.makedirs('models', exist_ok=True)
            with open(f'models/hmm_{asset.lower()}.pkl', 'wb') as f:
                pickle.dump(detector, f)
            print(f"  Saved: models/hmm_{asset.lower()}.pkl")
            return True
        else:
            print(f"  HMM fitting failed")
            return False
    except Exception as e:
        print(f"  HMM error: {e}")
        return False


def train_lstm_ensemble(data, asset='BTC', all_data=None):
    """
    Train LSTM/GRU/BiLSTM on multi-timeframe features with 3-class labels.
    Uses jitter oversampling, focal loss, attention pooling, threshold optimization.
    """
    print(f"\n{'='*60}")
    print(f"3. TRAINING LSTM ENSEMBLE ({asset})")
    print(f"{'='*60}")

    try:
        import torch
        print(f"  PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print(f"  PyTorch NOT installed -- pip install torch")
        return False

    try:
        from src.models.lstm_ensemble import LSTMEnsemble

        # 50 features: 30 base + 5 Kalman + 5 EMA new-line inflection + 10 Category B
        print(f"  Using 50 strategy features (30 base + 5 Kalman + 5 inflection + 10 Category B)")
        X, y = compute_strategy_features(
            data['closes'], data['highs'], data['lows'],
            data.get('opens', data['closes']), data['volumes'],
            seq_len=30, n_features=50,
        )
        n_features = 50

        if X is None or len(X) < 200:
            print(f"  Not enough data ({0 if X is None else len(X)} samples, need 200+)")
            return False

        # Delete old model files to avoid size mismatch
        model_dir = f'models/lstm_ensemble_{asset.lower()}'
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.pth'):
                    try:
                        os.remove(os.path.join(model_dir, f))
                        print(f"  Removed old checkpoint: {f}")
                    except Exception:
                        pass

        # BINARY: SKIP(0) vs TRADE(1)
        n_classes = 2
        print(f"  Classes: {n_classes} (SKIP vs TRADE) | Features: {n_features}")

        ensemble = LSTMEnsemble(input_dim=n_features, seq_len=30, num_classes=n_classes,
                                model_dir=model_dir)
        results = ensemble.train(X, y, epochs=150, lr=0.001, batch_size=64)

        print(f"\n  Results:")
        for name, m in results.items():
            if isinstance(m, dict):
                f1s = m.get('per_class_f1', [])
                f1_str = ' '.join(f'F1[{i}]={f1s[i]:.0%}' for i in range(len(f1s))) if f1s else ''
                print(f"    {name}: val_loss={m.get('best_val_loss','?'):.4f} "
                      f"val_f1={m.get('best_val_f1','?'):.3f} "
                      f"acc={m.get('val_accuracy','?'):.1%} "
                      f"epochs={m.get('epochs_trained','?')} | {f1_str}")

        test_pred = ensemble.predict(X[-1])
        print(f"  Test: quality={test_pred.get('trade_quality','?')} "
              f"L4_prob={test_pred.get('l4_probability', 0):.2f} "
              f"conf={test_pred.get('confidence',0):.2f}")
        return True
    except Exception as e:
        print(f"  LSTM error: {e}")
        import traceback; traceback.print_exc()
        return False


def train_patchtst(data, asset='BTC'):
    """Train PatchTST transformer on strategy-enriched returns."""
    print(f"\n{'='*60}")
    print(f"4. TRAINING PatchTST TRANSFORMER ({asset})")
    print(f"{'='*60}")

    try:
        import torch
        from src.ai.patchtst_model import PatchTST
    except ImportError:
        print(f"  PyTorch NOT installed"); return False

    try:
        from src.indicators.indicators import ema, atr
        closes = np.array(data['closes'], dtype=float)
        volumes = np.array(data['volumes'], dtype=float)
        if len(closes) < 500:
            print(f"  Need 500+ bars, have {len(closes)}")
            return False

        returns = np.diff(closes) / (closes[:-1] + 1e-9)
        ema_8 = np.array(ema(list(closes), 8))
        atr_14 = np.array(atr(data['highs'], data['lows'], list(closes), 14))

        ema_slope = np.zeros(len(closes))
        for i in range(3, len(closes)):
            if ema_8[i] > 0:
                ema_slope[i] = (ema_8[i] - ema_8[i-3]) / ema_8[i-3] * 100

        seq_len = 400

        X, y_up, y_shock = [], [], []
        for i in range(seq_len, len(returns) - 10):
            X.append(returns[i-seq_len:i])
            fwd_returns = returns[i:min(i+10, len(returns))]
            fwd_sum = sum(fwd_returns)
            if i + 5 < len(ema_slope):
                ema_fwd = ema_slope[i+5] - ema_slope[i]
                y_up.append(1.0 if ema_fwd > 0.01 else 0.0)
            else:
                y_up.append(1.0 if fwd_sum > 0.001 else 0.0)
            if i + 5 < len(atr_14) and atr_14[i] > 0:
                atr_expansion = (atr_14[min(i+5, len(atr_14)-1)] - atr_14[i]) / atr_14[i]
                y_shock.append(1.0 if atr_expansion > 0.5 else 0.0)
            else:
                y_shock.append(1.0 if abs(fwd_sum) > 0.01 else 0.0)

        X = torch.FloatTensor(np.array(X))
        y_up = torch.FloatTensor(np.array(y_up)).unsqueeze(1)
        y_shock = torch.FloatTensor(np.array(y_shock)).unsqueeze(1)

        print(f"  Samples: {len(X)} | Up: {int(y_up.sum())}/{len(X)} | Shock: {int(y_shock.sum())}/{len(X)}")

        model = PatchTST(seq_len=seq_len)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
        criterion = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

        split = int(len(X) * 0.85)
        X_tr, X_val = X[:split], X[split:]
        y_up_tr, y_up_val = y_up[:split], y_up[split:]
        y_sh_tr, y_sh_val = y_shock[:split], y_shock[split:]

        best_state = None
        best_val_acc = 0
        batch_size = 64

        for epoch in range(40):
            model.train()
            idx = np.random.permutation(len(X_tr))
            epoch_loss = 0; nb = 0

            for start in range(0, len(idx), batch_size):
                bi = idx[start:start+batch_size]
                bx, by_up, by_sh = X_tr[bi], y_up_tr[bi], y_sh_tr[bi]

                optimizer.zero_grad()
                pu, ps = model(bx)
                loss = criterion(pu, by_up) + 0.3 * criterion(ps, by_sh)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item(); nb += 1

            scheduler.step()

            model.eval()
            with torch.no_grad():
                vpu, _ = model(X_val)
                val_acc = ((vpu > 0.5).float() == y_up_val).float().mean().item()

            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: loss={epoch_loss/nb:.4f} val_acc={val_acc:.1%}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            os.makedirs('models', exist_ok=True)
            save_path = os.path.abspath(os.path.join('models', 'patchtst_v1.pt'))
            asset_path = os.path.abspath(os.path.join('models', f'patchtst_{asset.lower()}.pt'))
            import io, shutil
            tmp_path = save_path + '.tmp'
            try:
                buf = io.BytesIO()
                torch.save(best_state, buf)
                with open(tmp_path, 'wb') as f:
                    f.write(buf.getvalue())
                if os.path.exists(save_path):
                    try: os.remove(save_path)
                    except OSError: pass
                shutil.move(tmp_path, save_path)
                try: shutil.copy2(save_path, asset_path)
                except Exception: pass
            except Exception as e:
                ts = int(time.time())
                alt_path = os.path.abspath(os.path.join('models', f'patchtst_{asset.lower()}_{ts}.pt'))
                with open(alt_path, 'wb') as f:
                    f.write(buf.getvalue())
                save_path = alt_path
            print(f"  Best accuracy: {best_val_acc:.1%}")
            print(f"  Saved: {save_path}")

        return True
    except Exception as e:
        print(f"  PatchTST error: {e}")
        import traceback; traceback.print_exc()
        return False


def train_garch(data, asset='BTC'):
    """Train GARCH(1,1) on real price returns."""
    print(f"\n{'='*60}")
    print(f"5. TRAINING GARCH(1,1) ({asset})")
    print(f"{'='*60}")

    try:
        from src.models.volatility import GARCH11

        prices = [float(c) for c in data['closes'] if float(c) > 0]
        if len(prices) < 100:
            print(f"  Need 100+ prices, have {len(prices)}")
            return False

        garch = GARCH11()
        garch.fit(prices[-500:])

        forecast = garch.forecast(prices[-100:])
        print(f"  Fitted on {len(prices)} prices")
        print(f"  Current vol: {forecast[-1]:.6f}")
        print(f"  Params: omega={garch.omega:.6f} alpha={garch.alpha:.4f} beta={garch.beta:.4f}")
        print(f"  Persistence: {garch.alpha + garch.beta:.4f} (should be < 1.0)")

        import pickle
        os.makedirs('models', exist_ok=True)
        with open(f'models/garch_{asset.lower()}.pkl', 'wb') as f:
            pickle.dump(garch, f)
        print(f"  Saved: models/garch_{asset.lower()}.pkl")
        return True
    except Exception as e:
        print(f"  GARCH error: {e}")
        import traceback; traceback.print_exc()
        return False


def train_alpha_decay(data, asset='BTC'):
    """Train Alpha Decay — find optimal holding period."""
    print(f"\n{'='*60}")
    print(f"6. TRAINING ALPHA DECAY ({asset})")
    print(f"{'='*60}")

    try:
        from src.models.alpha_decay import AlphaDecayModel
        from src.indicators.indicators import ema

        closes = np.array(data['closes'], dtype=float)
        ema_8 = np.array(ema(list(closes), 8))
        ema_21 = np.array(ema(list(closes), 21))

        signals = ema_8 - ema_21
        returns = np.diff(closes) / (closes[:-1] + 1e-12)

        mn = min(len(signals), len(returns))
        signals = signals[-mn:]
        ret = returns[-mn:]

        model = AlphaDecayModel(max_horizon=60)
        model.fit(signals, ret)

        hold_min = model.optimal_hold * 5
        print(f"  Half-life: {model.half_life:.1f} bars ({model.half_life*5:.0f} min)")
        print(f"  Optimal hold: {model.optimal_hold:.0f} bars ({hold_min:.0f} min)")
        print(f"  Peak alpha: {model.peak_alpha:.4f}")

        import pickle
        os.makedirs('models', exist_ok=True)
        with open(f'models/alpha_decay_{asset.lower()}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved: models/alpha_decay_{asset.lower()}.pkl")
        return True
    except Exception as e:
        print(f"  Alpha Decay error: {e}")
        return False


def train_rl_agent(data, asset='BTC', all_data=None):
    """
    Train RL agent on historical OHLCV data by simulating EMA(8) trades.

    Unlike LightGBM (supervised learning on labels), RL learns from REWARDS:
    - Simulates the EMA(8) strategy bar-by-bar on historical data
    - For each EMA signal, the RL agent decides: enter or skip? what size?
    - After trade closes, reward is computed based on P&L and exit type
    - Q-table updated via Q-learning with experience replay

    Also loads any existing backtest CSV trades for additional training.
    """
    print(f"\n{'='*60}")
    print(f"7. TRAINING RL AGENT ({asset})")
    print(f"{'='*60}")

    try:
        from src.ai.reinforcement_learning import EMAStrategyRL, EMATradeState
        from src.indicators.indicators import ema, atr, rsi, adx as compute_adx

        closes = np.array(data['closes'], dtype=float)
        highs = np.array(data.get('highs', closes), dtype=float)
        lows = np.array(data.get('lows', closes), dtype=float)
        volumes = np.array(data.get('volumes', [1.0]*len(closes)), dtype=float)
        timestamps = data.get('timestamps', list(range(len(closes))))

        n_bars = len(closes)
        if n_bars < 200:
            print(f"  SKIP: only {n_bars} bars (need 200+)")
            return False

        # Compute indicators
        ema_8 = np.array(ema(list(closes), 8))
        ema_21 = np.array(ema(list(closes), 21))
        atr_14 = np.array(atr(list(highs), list(lows), list(closes), 14))
        rsi_14 = np.array(rsi(list(closes), 14))

        # Compute ATR percentile rolling
        atr_pctile = np.zeros(n_bars)
        for i in range(100, n_bars):
            window = atr_14[max(0, i-100):i]
            if len(window) > 0 and np.max(window) > np.min(window):
                atr_pctile[i] = (atr_14[i] - np.min(window)) / (np.max(window) - np.min(window) + 1e-10)

        # EMA slope (rate of change over 3 bars)
        ema_slope = np.zeros(n_bars)
        for i in range(3, n_bars):
            if atr_14[i] > 0:
                ema_slope[i] = (ema_8[i] - ema_8[i-3]) / (atr_14[i] * 3)

        # EMA direction tracking
        ema_direction = np.zeros(n_bars)  # 1=rising, -1=falling
        ema_dir_bars = np.zeros(n_bars, dtype=int)  # bars since flip
        for i in range(1, n_bars):
            if ema_8[i] > ema_8[i-1]:
                ema_direction[i] = 1
            elif ema_8[i] < ema_8[i-1]:
                ema_direction[i] = -1
            else:
                ema_direction[i] = ema_direction[i-1]

            if ema_direction[i] == ema_direction[i-1]:
                ema_dir_bars[i] = ema_dir_bars[i-1] + 1
            else:
                ema_dir_bars[i] = 1

        # Volume ratio
        vol_ratio = np.ones(n_bars)
        for i in range(20, n_bars):
            avg_vol = np.mean(volumes[i-20:i])
            if avg_vol > 0:
                vol_ratio[i] = volumes[i] / avg_vol

        # Higher TF alignment (use EMA21 as proxy)
        htf_align = np.zeros(n_bars)
        for i in range(21, n_bars):
            if ema_21[i] > ema_21[i-5]:
                htf_align[i] = 1.0
            elif ema_21[i] < ema_21[i-5]:
                htf_align[i] = -1.0

        # Initialize RL agent
        rl = EMAStrategyRL({'rl_model_path': f'models/rl_ema_{asset.lower()}.json'})

        # Reset epsilon for training (more exploration)
        rl.epsilon = 0.30

        # Simulate EMA strategy and train RL
        in_position = False
        entry_bar = 0
        entry_price = 0.0
        entry_direction = 0
        entry_state_key = ''
        entry_action_idx = 0
        trades_simulated = 0
        trades_won = 0
        total_pnl = 0.0
        prev_ema_dir = 0

        # Track recent trades for win rate feature
        recent_results = []

        print(f"  Simulating EMA(8) strategy on {n_bars} bars...")

        for i in range(50, n_bars):
            cur_dir = int(ema_direction[i])
            price = closes[i]

            # Detect EMA new line (direction flip after 3+ bars)
            is_new_line = (cur_dir != prev_ema_dir and prev_ema_dir != 0
                          and ema_dir_bars[i-1] >= 3)

            if not in_position and is_new_line:
                # EMA signal detected — ask RL whether to take it
                recent_wr = 0.5
                consec_losses = 0
                if recent_results:
                    wins = sum(1 for r in recent_results[-20:] if r > 0)
                    recent_wr = wins / len(recent_results[-20:]) if recent_results[-20:] else 0.5
                    for r in reversed(recent_results):
                        if r < 0:
                            consec_losses += 1
                        else:
                            break

                # Build state
                price_ema_dist = (price - ema_8[i]) / (atr_14[i] + 1e-10)
                hour = 12  # Default (no timestamp parsing for speed)
                if timestamps and isinstance(timestamps[i], (int, float)) and timestamps[i] > 1e9:
                    from datetime import datetime as dt, timezone
                    try:
                        hour = dt.fromtimestamp(timestamps[i] / 1000, tz=timezone.utc).hour
                    except Exception:
                        pass

                state = EMATradeState(
                    ema_slope=float(ema_slope[i]),
                    ema_slope_bars=int(ema_dir_bars[i]),
                    price_ema_distance_atr=float(price_ema_dist),
                    ema_acceleration=float(ema_slope[i] - ema_slope[max(0, i-3)]),
                    trend_bars_since_flip=int(ema_dir_bars[i]),
                    trend_consistency=0.7,  # Approximate
                    higher_tf_alignment=float(htf_align[i]) * cur_dir,
                    atr_percentile=float(atr_pctile[i]),
                    volume_ratio=float(vol_ratio[i]),
                    spread_atr_ratio=0.1,  # Approximate
                    recent_win_rate=recent_wr,
                    daily_pnl_pct=0.0,
                    open_positions=0,
                    consecutive_losses=consec_losses,
                    hour_of_day=hour,
                    day_of_week=3,  # Default
                )

                # RL decides
                decision = rl.decide(state)
                entry_state_key = rl._discretize_state(state)

                # Find action index
                for idx, act in enumerate(rl.actions):
                    if act['label'] == decision.reasoning.split('Action: ')[1].split(' |')[0]:
                        entry_action_idx = idx
                        break

                if decision.enter_trade:
                    in_position = True
                    entry_bar = i
                    entry_price = price
                    entry_direction = cur_dir  # 1=long, -1=short
                else:
                    # Track what would have happened if we entered
                    # (look ahead to find theoretical exit for skip reward)
                    look_ahead = min(i + 100, n_bars)
                    best_pnl = 0.0
                    for j in range(i+1, look_ahead):
                        if cur_dir == 1:
                            sim_pnl = (closes[j] - price) / price * 100
                        else:
                            sim_pnl = (price - closes[j]) / price * 100

                        if sim_pnl < -2.0:  # Would have hit hard stop
                            best_pnl = -2.0
                            break
                        if int(ema_direction[j]) != cur_dir and sim_pnl > 0:
                            best_pnl = sim_pnl
                            break
                        best_pnl = sim_pnl

                    skip_result = {
                        'was_skipped': True,
                        'would_have_pnl': best_pnl,
                    }
                    reward = rl.compute_reward(skip_result)
                    rl.update(entry_state_key, entry_action_idx, reward, None)

            elif in_position:
                # Check exits
                bars_held = i - entry_bar
                if entry_direction == 1:
                    pnl_pct = (price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - price) / entry_price * 100

                exit_type = None

                # Hard stop
                if pnl_pct <= -HARD_STOP_PCT:
                    exit_type = 'hard_stop'
                    pnl_pct = -HARD_STOP_PCT

                # EMA new line exit (profit only)
                elif (cur_dir != entry_direction and ema_dir_bars[i] >= 2
                      and pnl_pct > 0):
                    exit_type = 'ema_exit'

                # SL hit (after grace period)
                elif bars_held >= 4 and pnl_pct <= -0.5:
                    exit_type = 'sl'

                # Time exit (stale trade)
                elif bars_held >= 200 and pnl_pct < 0.3:
                    exit_type = 'time'

                if exit_type:
                    in_position = False
                    trades_simulated += 1
                    total_pnl += pnl_pct
                    if pnl_pct > 0:
                        trades_won += 1
                    recent_results.append(pnl_pct)

                    # Compute reward and update RL
                    trade_result = {
                        'pnl_pct': pnl_pct,
                        'exit_type': exit_type,
                        'hold_bars': bars_held,
                        'was_skipped': False,
                    }
                    reward = rl.compute_reward(trade_result)
                    rl.update(entry_state_key, entry_action_idx, reward, None)

                    if trades_simulated % 200 == 0:
                        wr = trades_won / trades_simulated * 100
                        print(f"    ... {trades_simulated} trades | WR: {wr:.1f}% | "
                              f"PnL: {total_pnl:+.1f}% | eps: {rl.epsilon:.3f}")

            prev_ema_dir = cur_dir

        # Run extra replay learning passes
        print(f"  Running {min(100, len(rl.replay_buffer))} replay learning passes...")
        for _ in range(100):
            rl.replay_learn()

        # Save model
        rl._save_model()

        # Print results
        wr = trades_won / trades_simulated * 100 if trades_simulated > 0 else 0
        insights = rl.get_strategy_insights()
        print(f"\n  RL Training Complete:")
        print(f"    Trades simulated: {trades_simulated}")
        print(f"    Win rate: {wr:.1f}%")
        print(f"    Total PnL: {total_pnl:+.1f}%")
        print(f"    Q-table states: {insights['q_table_size']}")
        print(f"    Skip rate: {insights['skip_rate']:.1%}")
        print(f"    Final epsilon: {rl.epsilon:.4f}")
        print(f"    Replay buffer: {insights['replay_buffer_size']}")
        print(f"  Saved: models/rl_ema_{asset.lower()}.json")

        # Print action performance
        print(f"\n  Action Performance:")
        for label, perf in insights.get('action_performance', {}).items():
            if perf['count'] > 0:
                print(f"    {label:20s}: count={perf['count']:4d} | "
                      f"avg_pnl={perf['avg_pnl']:+.3f}%")

        return True

    except Exception as e:
        import traceback
        print(f"  RL Training error: {e}")
        traceback.print_exc()
        return False


# Import for hard stop constant
from src.ai.reinforcement_learning import EMAStrategyRL as _RL_CHECK
HARD_STOP_PCT = 2.0


def train_one_cycle(assets, bars=20000, skip_lstm=False, skip_patchtst=False, use_mtf=True):
    """Run a single training cycle for all models on all assets."""
    results = {}

    for asset in assets:
        print(f"\n{'#'*60}")
        print(f"# {asset}")
        print(f"{'#'*60}")

        # Fetch multi-timeframe data
        all_data = None
        if use_mtf:
            all_data = fetch_multi_timeframe_data(asset, bars)
            if '5m' in all_data:
                data = all_data['5m']
            else:
                data = fetch_training_data(asset, '5m', bars)
        else:
            data = fetch_training_data(asset, '5m', bars)

        if not data or len(data['closes']) < 100:
            print(f"  SKIP: not enough data")
            continue

        if 'opens' not in data:
            data['opens'] = [data['closes'][max(0, i-1)] for i in range(len(data['closes']))]

        print(f"  Bars: {len(data['closes'])} | "
              f"Price: ${data['closes'][-1]:,.2f} | "
              f"Range: ${min(data['closes']):,.2f}-${max(data['closes']):,.2f}")

        results[f'{asset}_lgbm'] = train_lightgbm(data, asset, all_data)
        results[f'{asset}_hmm'] = train_hmm(data, asset)
        results[f'{asset}_garch'] = train_garch(data, asset)
        results[f'{asset}_alpha'] = train_alpha_decay(data, asset)
        results[f'{asset}_rl'] = train_rl_agent(data, asset, all_data)

        if not skip_lstm:
            results[f'{asset}_lstm'] = train_lstm_ensemble(data, asset, all_data)
        if not skip_patchtst:
            results[f'{asset}_patchtst'] = train_patchtst(data, asset)

    return results


def print_summary(results, cycle=0, elapsed=0):
    """Print training summary."""
    ok = sum(1 for v in results.values() if v is True)
    total = len(results)
    print(f"\n{'='*60}")
    if cycle > 0:
        print(f"TRAINING SUMMARY - Cycle #{cycle} (took {elapsed:.0f}s)")
    else:
        print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        s = 'OK' if v is True else ('SKIP' if v == 'skipped' else 'FAIL')
        print(f"  [{s}] {k}")
    print(f"  Score: {ok}/{total} models trained successfully")
    print(f"{'='*60}")
    return ok, total


def log_training_history(results, cycle):
    """Append training results to history file."""
    import json
    from datetime import datetime, timezone
    history_path = 'models/training_history.json'
    try:
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = []
    except Exception:
        history = []

    entry = {
        'cycle': cycle,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'results': {k: v for k, v in results.items()},
        'success_rate': sum(1 for v in results.values() if v is True) / max(1, len(results)),
    }
    history.append(entry)
    history = history[-100:]
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train all ML models on real market data')
    parser.add_argument('--asset', default='ALL', help='BTC, ETH, or ALL')
    parser.add_argument('--bars', type=int, default=20000, help='Bars per timeframe (default: 20000)')
    parser.add_argument('--skip-lstm', action='store_true')
    parser.add_argument('--skip-patchtst', action='store_true')
    parser.add_argument('--no-mtf', action='store_true', help='Skip multi-timeframe (use 5m only)')
    parser.add_argument('--loop', action='store_true',
                        help='Autonomous continuous training loop')
    parser.add_argument('--loop-hours', type=float, default=4.0)
    parser.add_argument('--max-cycles', type=int, default=0)
    args = parser.parse_args()

    assets = ['BTC', 'ETH'] if args.asset == 'ALL' else [args.asset]
    use_mtf = not args.no_mtf

    if args.loop:
        cycle = 0
        loop_seconds = args.loop_hours * 3600

        print("=" * 60)
        print("AUTONOMOUS ML TRAINING LOOP — MULTI-TIMEFRAME")
        print(f"  Assets: {assets}")
        print(f"  Multi-timeframe: {use_mtf}")
        print(f"  Retrain every: {args.loop_hours}h ({loop_seconds:.0f}s)")
        print(f"  Max cycles: {'infinite' if args.max_cycles == 0 else args.max_cycles}")
        print(f"  Bars per cycle: {args.bars}")
        print("  Press Ctrl+C to stop")
        print("=" * 60)

        consecutive_failures = 0
        max_consecutive_failures = 5  # Stop after 5 consecutive failures

        while True:
            cycle += 1
            if args.max_cycles > 0 and cycle > args.max_cycles:
                print(f"\nReached max cycles ({args.max_cycles}). Stopping.")
                break

            from datetime import datetime, timezone
            start_time = time.time()
            print(f"\n{'*'*60}")
            print(f"* CYCLE #{cycle} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"{'*'*60}")

            try:
                results = train_one_cycle(assets, args.bars,
                                          args.skip_lstm, args.skip_patchtst, use_mtf)
                elapsed = time.time() - start_time
                ok, total = print_summary(results, cycle, elapsed)
                log_training_history(results, cycle)

                if ok < total * 0.5:
                    consecutive_failures += 1
                    print(f"  WARNING: {total - ok} failures (streak: {consecutive_failures}/{max_consecutive_failures})")
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"  FATAL: {max_consecutive_failures} consecutive failures. Waiting 30min then resetting...")
                        time.sleep(1800)
                        consecutive_failures = 0
                    else:
                        retry_wait = min(300, loop_seconds * 0.1)
                        print(f"  Retrying in {retry_wait:.0f}s...")
                        time.sleep(retry_wait)
                    continue
                else:
                    consecutive_failures = 0  # Reset on success

            except KeyboardInterrupt:
                print("\n\nStopped by user.")
                break
            except Exception as e:
                consecutive_failures += 1
                print(f"\n  CYCLE ERROR ({consecutive_failures}/{max_consecutive_failures}): {e}")
                import traceback; traceback.print_exc()
                if consecutive_failures >= max_consecutive_failures:
                    print(f"  FATAL: {max_consecutive_failures} consecutive errors. Waiting 30min...")
                    time.sleep(1800)
                    consecutive_failures = 0

            from datetime import datetime, timezone
            next_train = datetime.fromtimestamp(time.time() + loop_seconds, tz=timezone.utc)
            print(f"\n  Next training: {next_train.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"  Sleeping {loop_seconds:.0f}s...")

            try:
                time.sleep(loop_seconds)
            except KeyboardInterrupt:
                print("\n\nStopped by user.")
                break
    else:
        print("=" * 60)
        print("TRAINING ALL ML MODELS — MULTI-TIMEFRAME REAL MARKET DATA")
        print("=" * 60)

        start = time.time()
        results = train_one_cycle(assets, args.bars,
                                  args.skip_lstm, args.skip_patchtst, use_mtf)
        elapsed = time.time() - start
        print_summary(results, elapsed=elapsed)
        log_training_history(results, cycle=1)


if __name__ == '__main__':
    main()
