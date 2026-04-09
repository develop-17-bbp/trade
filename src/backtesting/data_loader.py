"""
Backtest Data Loader — Multi-Timeframe Historical OHLCV
========================================================
Fetches and caches historical data for backtesting.
Uses Binance mainnet for realistic candle data.
"""

import os
import time
import json
import ccxt
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class BacktestData:
    """Container for multi-timeframe OHLCV data."""
    asset: str
    primary_tf: str
    start_ts: int  # ms
    end_ts: int    # ms
    timeframes: Dict[str, dict] = field(default_factory=dict)  # tf -> {opens, highs, lows, closes, volumes, timestamps}

    @property
    def primary(self) -> dict:
        return self.timeframes.get(self.primary_tf, {})

    @property
    def bar_count(self) -> int:
        return len(self.primary.get('closes', []))


def fetch_backtest_data(
    asset: str = 'BTC',
    days: int = 30,
    primary_tf: str = '5m',
    start_date: str = None,
    end_date: str = None,
    exchange_id: str = 'binance',
    cache_dir: str = None,
) -> BacktestData:
    """Fetch multi-timeframe OHLCV data for backtesting.

    Args:
        asset: BTC or ETH
        days: number of days of history
        primary_tf: main simulation timeframe
        start_date: explicit start (YYYY-MM-DD), overrides days
        end_date: explicit end (YYYY-MM-DD)
        exchange_id: CCXT exchange for data
        cache_dir: directory for cached data

    Returns:
        BacktestData with all timeframes loaded
    """
    # Setup cache
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), 'data', 'backtest_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Date range
    if end_date:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_dt = datetime.utcnow()

    if start_date:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start_dt = end_dt - timedelta(days=days)

    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    symbol = f"{asset}/USDT"

    # Exchange initialized lazily — only if cache miss requires API fetch
    exchange = None

    # Timeframes to fetch — primary + context
    tf_list = [primary_tf]
    # Skip 1m context — not used by backtest engine, adds 2.7M bars and causes OOM
    context_tfs = {'5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h'}
    for tf in context_tfs.values():
        if tf not in tf_list:
            tf_list.append(tf)

    data = BacktestData(
        asset=asset,
        primary_tf=primary_tf,
        start_ts=since_ms,
        end_ts=until_ms,
    )

    for tf in tf_list:
        cache_key = f"{asset}_{tf}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}"
        cache_path = os.path.join(cache_dir, f"{cache_key}.json")

        # Check exact cache match
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                if len(cached.get('closes', [])) > 0:
                    data.timeframes[tf] = cached
                    print(f"  [CACHE] {asset} {tf}: {len(cached['closes'])} bars loaded from cache")
                    continue
            except Exception:
                pass

        # Check ANY cached file that COVERS the requested date range
        # e.g., BTC_5m_20160409_20260407.json covers a request for 20170101-20260401
        import glob as _glob
        _found_superset = False
        _pattern = os.path.join(cache_dir, f"{asset}_{tf}_*.json")
        _candidates = sorted(_glob.glob(_pattern), key=os.path.getsize, reverse=True)
        for _cpath in _candidates:
            _csize = os.path.getsize(_cpath)
            if _csize < 500:  # Skip LFS pointers
                continue
            try:
                with open(_cpath, 'r') as f:
                    _cdata = json.load(f)
                if not isinstance(_cdata, dict) or 'closes' not in _cdata or len(_cdata['closes']) < 100:
                    continue
                _cts = _cdata.get('timestamps', [])
                if not _cts:
                    continue
                # Check if cached data covers our requested range
                _c_start = _cts[0]
                _c_end = _cts[-1]
                if _c_start <= since_ms and _c_end >= until_ms - 86400000:  # 1 day tolerance
                    # Trim to requested range
                    _trimmed = {}
                    _indices = [i for i, t in enumerate(_cts) if since_ms <= t <= until_ms]
                    if len(_indices) >= 100:
                        _s, _e = _indices[0], _indices[-1] + 1
                        for _k, _v in _cdata.items():
                            _trimmed[_k] = _v[_s:_e]
                        data.timeframes[tf] = _trimmed
                        print(f"  [CACHE] {asset} {tf}: {len(_trimmed['closes']):,} bars from {os.path.basename(_cpath)} (superset)")
                        _found_superset = True
                        break
            except Exception:
                continue
        if _found_superset:
            continue

        # ── PARQUET FALLBACK: search training_cache for parquet files ──
        # Useful when Binance API is blocked (e.g. India server HTTP 451)
        _found_parquet = False
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _parquet_dirs = [
            os.path.join(_project_root, 'data', 'training_cache'),
            os.path.join(_project_root, 'data', 'backtest_cache'),
            os.path.join(_project_root, 'data'),
        ]
        _tf_map = {'5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d', '1m': '1m'}
        _pq_tf = _tf_map.get(tf, tf)
        for _pq_dir in _parquet_dirs:
            for _pq_name in [f"{asset}USDT-{_pq_tf}.parquet", f"{asset}-{_pq_tf}.parquet",
                             f"{asset}USDT_{_pq_tf}.parquet"]:
                _pq_path = os.path.join(_pq_dir, _pq_name)
                if os.path.exists(_pq_path):
                    try:
                        import pandas as pd
                        _fsize_mb = os.path.getsize(_pq_path) / (1024 * 1024)
                        print(f"  [PARQUET] Found {_pq_name} ({_fsize_mb:.1f} MB), loading...", flush=True)
                        df = pd.read_parquet(_pq_path)
                        if len(df) < 100:
                            continue
                        # Detect timestamp column
                        ts_col = None
                        for col in ['timestamp', 'open_time', 'time', 'date']:
                            if col in df.columns:
                                ts_col = col
                                break
                        if ts_col is None:
                            ts_col = df.columns[0]
                        # Convert to milliseconds as numpy array (vectorized, handles 900K+ rows)
                        if pd.api.types.is_datetime64_any_dtype(df[ts_col]):
                            ts_arr = df[ts_col].values.astype('int64') // 10**6
                        else:
                            ts_arr = df[ts_col].values.astype(np.float64)
                            if ts_arr[0] < 1e12:  # seconds
                                ts_arr = (ts_arr * 1000).astype(np.int64)
                            else:
                                ts_arr = ts_arr.astype(np.int64)
                        # Map OHLCV columns
                        _col_map = {}
                        for target, candidates in [
                            ('opens', ['open', 'Open']), ('highs', ['high', 'High']),
                            ('lows', ['low', 'Low']), ('closes', ['close', 'Close']),
                            ('volumes', ['volume', 'Volume']),
                        ]:
                            for c in candidates:
                                if c in df.columns:
                                    _col_map[target] = c
                                    break
                        if len(_col_map) < 5:
                            # Fallback: assume columns by position (ts, o, h, l, c, v)
                            cols = df.columns.tolist()
                            if len(cols) >= 6:
                                _col_map = {'opens': cols[1], 'highs': cols[2], 'lows': cols[3],
                                            'closes': cols[4], 'volumes': cols[5]}
                        if len(_col_map) >= 5:
                            # Vectorized date range filter (fast for 900K+ rows)
                            _mask = (ts_arr >= since_ms) & (ts_arr <= until_ms)
                            _count = int(_mask.sum())
                            print(f"  [PARQUET] {_pq_name}: {len(df):,} total rows, {_count:,} in date range", flush=True)
                            if _count >= 100:
                                _df_slice = df.loc[_mask].reset_index(drop=True)
                                _ts_slice = ts_arr[_mask]
                                ohlcv = {
                                    'timestamps': _ts_slice.tolist(),
                                    'opens': _df_slice[_col_map['opens']].tolist(),
                                    'highs': _df_slice[_col_map['highs']].tolist(),
                                    'lows': _df_slice[_col_map['lows']].tolist(),
                                    'closes': _df_slice[_col_map['closes']].tolist(),
                                    'volumes': _df_slice[_col_map['volumes']].tolist(),
                                }
                                data.timeframes[tf] = ohlcv
                                print(f"  [PARQUET] {asset} {tf}: {_count:,} bars loaded", flush=True)
                                # Cache as JSON for faster subsequent loads
                                try:
                                    with open(cache_path, 'w') as f:
                                        json.dump(ohlcv, f)
                                    print(f"  [PARQUET] Cached to {os.path.basename(cache_path)}", flush=True)
                                except Exception:
                                    pass
                                _found_parquet = True
                                break
                    except Exception as e:
                        import traceback
                        print(f"  [PARQUET] Error loading {_pq_path}: {e}", flush=True)
                        traceback.print_exc()
                        continue
            if _found_parquet:
                break
        if _found_parquet:
            continue

        # Fetch from exchange
        print(f"  [FETCH] {asset} {tf}: fetching from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}...")

        # Lazy init exchange only when actually needed
        if exchange is None:
            exchange = ccxt.binance({'enableRateLimit': True})

        all_candles = []
        fetch_since = since_ms
        batch_size = 1000
        exchange._bt_retry = 0

        while fetch_since < until_ms:
            try:
                candles = exchange.fetch_ohlcv(symbol, tf, since=fetch_since, limit=batch_size)
                if not candles or len(candles) == 0:
                    break

                # Filter to date range
                candles = [c for c in candles if c[0] <= until_ms]
                if not candles:
                    break
                all_candles.extend(candles)

                # Move forward
                last_ts = candles[-1][0]
                if last_ts <= fetch_since:
                    break
                fetch_since = last_ts + 1

                if len(all_candles) % 5000 < batch_size:
                    print(f"    ... fetched {len(all_candles)} bars")

                time.sleep(exchange.rateLimit / 1000)

            except Exception as e:
                import traceback
                print(f"    Error: {e}")
                traceback.print_exc()
                time.sleep(2)
                retry_count = getattr(exchange, '_bt_retry', 0) + 1
                exchange._bt_retry = retry_count
                if retry_count > 3:
                    print(f"    GIVING UP on {asset} {tf} after 3 retries")
                    break

        if not all_candles:
            print(f"  [WARN] No data for {asset} {tf}")
            continue

        # Deduplicate by timestamp
        seen = set()
        unique = []
        for c in all_candles:
            if c[0] not in seen:
                seen.add(c[0])
                unique.append(c)
        unique.sort(key=lambda x: x[0])

        ohlcv = {
            'timestamps': [c[0] for c in unique],
            'opens': [c[1] for c in unique],
            'highs': [c[2] for c in unique],
            'lows': [c[3] for c in unique],
            'closes': [c[4] for c in unique],
            'volumes': [c[5] for c in unique],
        }

        data.timeframes[tf] = ohlcv
        print(f"  [FETCH] {asset} {tf}: {len(unique)} bars fetched")

        # Cache
        try:
            with open(cache_path, 'w') as f:
                json.dump(ohlcv, f)
        except Exception:
            pass

    return data


def get_context_at_bar(data: BacktestData, tf: str, bar_timestamp: int,
                       lookback: int = 100) -> Optional[dict]:
    """Get OHLCV slice for a given timeframe at a specific timestamp.

    Returns lookback bars ending at or before bar_timestamp.
    Prevents lookahead bias by strictly filtering by timestamp.
    """
    tf_data = data.timeframes.get(tf)
    if not tf_data or not tf_data.get('timestamps'):
        return None

    timestamps = tf_data['timestamps']

    # Find last bar at or before our timestamp
    end_idx = 0
    for i, ts in enumerate(timestamps):
        if ts <= bar_timestamp:
            end_idx = i
        else:
            break

    start_idx = max(0, end_idx - lookback + 1)

    if end_idx - start_idx < 10:
        return None

    return {
        'opens': tf_data['opens'][start_idx:end_idx + 1],
        'highs': tf_data['highs'][start_idx:end_idx + 1],
        'lows': tf_data['lows'][start_idx:end_idx + 1],
        'closes': tf_data['closes'][start_idx:end_idx + 1],
        'volumes': tf_data['volumes'][start_idx:end_idx + 1],
        'timestamps': timestamps[start_idx:end_idx + 1],
    }
