#!/usr/bin/env python3
"""
Binance Vision Historical Data Downloader
==========================================
Downloads OHLCV kline data directly from Binance's public S3 repository
(https://data.binance.vision/) — bypasses API region blocks and rate limits.

Supports:
  - Futures (UM) and Spot klines
  - Multiple timeframes: 1m, 5m, 15m, 1h, 4h, 1d
  - Monthly ZIP archives (2020-present)
  - Saves as .parquet for fast loading

Usage:
    python download_vision_data.py
    python download_vision_data.py --symbols BTCUSDT ETHUSDT --timeframe 1h
    python download_vision_data.py --symbols SOLUSDT --timeframe 4h --start-year 2022
    python download_vision_data.py --train  # Download AND train LightGBM models

Data source: https://data.binance.vision/?prefix=data/futures/um/monthly/klines/
"""

import os
import sys
import io
import argparse
import zipfile
import logging
import pandas as pd
import requests
from datetime import datetime
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

VISION_BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
SPOT_BASE = "https://data.binance.vision/data/spot/monthly/klines"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "AAVEUSDT"]
DEFAULT_TIMEFRAME = "1h"
DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = 2026
OUTPUT_DIR = "data"

# Binance kline CSV columns (no header in files)
KLINE_COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
    'taker_buy_quote', 'ignore'
]

# Model output mapping
SYMBOL_TO_MODEL = {
    'BTCUSDT': 'models/lgbm_btc.txt',
    'ETHUSDT': 'models/lgbm_eth.txt',
    'AAVEUSDT': 'models/lgbm_aave.txt',
    'SOLUSDT': 'models/lgbm_sol.txt',
    'BNBUSDT': 'models/lgbm_bnb.txt',
    'ADAUSDT': 'models/lgbm_ada.txt',
    'DOGEUSDT': 'models/lgbm_doge.txt',
    'XRPUSDT': 'models/lgbm_xrp.txt',
    'AVAXUSDT': 'models/lgbm_avax.txt',
    'DOTUSDT': 'models/lgbm_dot.txt',
    'LINKUSDT': 'models/lgbm_link.txt',
    'MATICUSDT': 'models/lgbm_matic.txt',
}


# ============================================================================
# DOWNLOAD ENGINE
# ============================================================================

def generate_month_list(start_year: int, end_year: int) -> List[Tuple[int, int]]:
    """Generate (year, month) pairs from start_year-01 to current month."""
    months = []
    now = datetime.utcnow()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == now.year and month > now.month:
                break
            months.append((year, month))
    return months


def download_monthly_zip(symbol: str, timeframe: str, year: int, month: int,
                         use_futures: bool = True) -> Optional[pd.DataFrame]:
    """
    Download a single monthly kline ZIP from Binance Vision.
    Returns DataFrame or None if not available.
    """
    base = VISION_BASE if use_futures else SPOT_BASE
    filename = f"{symbol}-{timeframe}-{year}-{month:02d}.zip"
    url = f"{base}/{symbol}/{timeframe}/{filename}"

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            logger.debug(f"  HTTP {resp.status_code} for {filename}")
            return None

        # Extract CSV from ZIP in memory
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as csv_file:
                # Try reading with header first — some files have headers
                df = pd.read_csv(csv_file, header=None)

        # Check if first row is a header (string values)
        first_val = str(df.iloc[0, 0])
        if first_val.startswith('open') or not first_val.replace('.', '').replace('-', '').isdigit():
            df = df.iloc[1:].reset_index(drop=True)

        # Handle varying column counts (some files have 12 cols, some have fewer)
        if len(df.columns) >= 12:
            df.columns = KLINE_COLUMNS[:len(df.columns)]
        elif len(df.columns) >= 6:
            df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume'] + \
                         [f'extra_{i}' for i in range(len(df.columns) - 6)]
        else:
            logger.warning(f"  Unexpected column count in {filename}: {len(df.columns)}")
            return None

        # Normalize to standard OHLCV — convert open_time to int/float first
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
        result = pd.DataFrame({
            'timestamp': pd.to_datetime(df['open_time'], unit='ms'),
            'open': pd.to_numeric(df['open'], errors='coerce'),
            'high': pd.to_numeric(df['high'], errors='coerce'),
            'low': pd.to_numeric(df['low'], errors='coerce'),
            'close': pd.to_numeric(df['close'], errors='coerce'),
            'volume': pd.to_numeric(df['volume'], errors='coerce'),
        })
        result = result.dropna()

        return result

    except zipfile.BadZipFile:
        logger.debug(f"  Bad ZIP: {filename}")
        return None
    except Exception as e:
        logger.debug(f"  Error downloading {filename}: {e}")
        return None


def download_symbol(symbol: str, timeframe: str = "1h",
                    start_year: int = 2020, end_year: int = 2026,
                    output_dir: str = "data", force: bool = False) -> Optional[str]:
    """
    Download all available monthly data for a symbol.
    Saves as .parquet file. Returns output path or None.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{symbol}-{timeframe}.parquet")

    # Check if already downloaded (skip with force=True)
    if os.path.exists(output_path) and not force:
        existing = pd.read_parquet(output_path)
        logger.info(f"  {symbol}: Already have {len(existing)} rows in {output_path}")
        return output_path
    elif os.path.exists(output_path) and force:
        logger.info(f"  {symbol}: Force re-download — removing old {output_path}")
        os.remove(output_path)

    months = generate_month_list(start_year, end_year)
    all_dfs = []
    downloaded = 0
    skipped = 0

    logger.info(f"  Downloading {symbol} {timeframe} ({start_year}-{end_year})...")

    for year, month in months:
        # Try futures first, fall back to spot
        df = download_monthly_zip(symbol, timeframe, year, month, use_futures=True)
        if df is None:
            df = download_monthly_zip(symbol, timeframe, year, month, use_futures=False)

        if df is not None and not df.empty:
            all_dfs.append(df)
            downloaded += 1
        else:
            skipped += 1

    if not all_dfs:
        logger.warning(f"  {symbol}: No data downloaded")
        return None

    # Combine and deduplicate
    combined = pd.concat(all_dfs, ignore_index=True)

    # Safety: strip any lingering header rows that slipped through per-month detection
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')
    if 'timestamp' in combined.columns:
        combined['timestamp'] = pd.to_datetime(combined['timestamp'], errors='coerce')
    combined = combined.dropna(subset=['timestamp', 'open', 'close'])

    combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    # Save as parquet
    combined.to_parquet(output_path, index=False, engine='pyarrow')

    logger.info(f"  {symbol}: {len(combined)} rows saved to {output_path} "
                f"({downloaded} months downloaded, {skipped} unavailable)")

    return output_path


def download_all(symbols: List[str] = None, timeframe: str = "1h",
                 start_year: int = 2020, output_dir: str = "data") -> dict:
    """Download data for all specified symbols."""
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    logger.info(f"Downloading {len(symbols)} symbols ({timeframe}) from Binance Vision...")
    results = {}

    for symbol in symbols:
        path = download_symbol(symbol, timeframe, start_year, DEFAULT_END_YEAR, output_dir)
        if path:
            df = pd.read_parquet(path)
            # Ensure timestamp is datetime (some parquets have raw int64 ms)
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            try:
                start_dt = df['timestamp'].iloc[0]
                end_dt = df['timestamp'].iloc[-1]
                start_str = str(start_dt.date()) if hasattr(start_dt, 'date') else str(start_dt)
                end_str = str(end_dt.date()) if hasattr(end_dt, 'date') else str(end_dt)
            except Exception:
                start_str, end_str = '?', '?'
            results[symbol] = {
                'path': path,
                'rows': len(df),
                'start': start_str,
                'end': end_str,
            }
            logger.info(f"  {symbol}: {len(df)} rows ({start_str} to {end_str})")

    return results


# ============================================================================
# SHARED UTILITY: Used by all training scripts as primary data source
# ============================================================================

def fetch_vision_ohlcv(symbol: str, timeframe: str = '1h',
                       start_year: int = 2020, data_dir: str = 'data') -> pd.DataFrame:
    """
    Primary data source for ALL training scripts.
    Downloads from Binance Vision (S3) — works even when Binance API returns 451.

    1. Checks for existing parquet file
    2. Downloads from Vision if missing
    3. Returns DataFrame with columns: [timestamp, open, high, low, close, volume]

    Usage (from any training script):
        from download_vision_data import fetch_vision_ohlcv
        df = fetch_vision_ohlcv('BTC/USDT')  # or 'BTCUSDT'
    """
    # Normalize symbol: 'BTC/USDT' -> 'BTCUSDT'
    clean_symbol = symbol.replace('/', '')

    # Check for existing parquet
    parquet_path = os.path.join(data_dir, f"{clean_symbol}-{timeframe}.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        # Ensure timestamp is datetime (some parquets saved raw int64 ms)
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            # Re-save with proper datetime so this fix is permanent
            try:
                df.to_parquet(parquet_path, index=False, engine='pyarrow')
                logger.info(f"Fixed int64 timestamps in {parquet_path}")
            except Exception:
                pass
        logger.info(f"Loaded {len(df)} bars from {parquet_path}")
        return df

    # Check for existing CSV (legacy naming)
    csv_candidates = [
        os.path.join(data_dir, f"{clean_symbol}-{timeframe}.csv"),
        os.path.join(data_dir, f"{clean_symbol.replace('USDT','')}_USDT_{timeframe}.csv"),
        os.path.join('data', f"{clean_symbol.replace('USDT','')}_USDT_{timeframe}.csv"),
    ]
    for csv_path in csv_candidates:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Normalize columns
            col_map = {}
            for c in df.columns:
                low = c.lower().strip()
                if low in ('timestamp', 'time', 'date', 'open_time', 'datetime'):
                    col_map[c] = 'timestamp'
                elif low == 'open': col_map[c] = 'open'
                elif low == 'high': col_map[c] = 'high'
                elif low == 'low': col_map[c] = 'low'
                elif low == 'close': col_map[c] = 'close'
                elif low in ('volume', 'vol'): col_map[c] = 'volume'
            df = df.rename(columns=col_map)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.info(f"Loaded {len(df)} bars from {csv_path}")
            return df

    # Download from Binance Vision
    path = download_symbol(clean_symbol, timeframe, start_year, DEFAULT_END_YEAR, data_dir)
    if path:
        df = pd.read_parquet(path)
        logger.info(f"Downloaded {len(df)} bars for {clean_symbol} via Binance Vision")
        return df

    logger.error(f"Could not fetch data for {clean_symbol}/{timeframe}")
    return pd.DataFrame()


# ============================================================================
# TRAINING INTEGRATION
# ============================================================================

def train_from_parquet(symbol: str, parquet_path: str, model_out: str) -> bool:
    """Train LightGBM model from downloaded parquet data."""
    try:
        from src.scripts.train_lgbm import build_dataset, train_model

        logger.info(f"  Training {symbol} from {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        features, labels = build_dataset(df)
        if not features:
            logger.warning(f"  No features extracted for {symbol}")
            return False

        os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
        train_model(features, labels, model_out)
        logger.info(f"  Model saved: {model_out}")
        return True

    except Exception as e:
        logger.error(f"  Training failed for {symbol}: {e}")
        return False


def download_and_train(symbols: List[str] = None, timeframe: str = "1h",
                       start_year: int = 2020, output_dir: str = "data") -> dict:
    """Download data and train models for all symbols."""
    downloads = download_all(symbols, timeframe, start_year, output_dir)

    trained = {}
    for symbol, info in downloads.items():
        model_out = SYMBOL_TO_MODEL.get(symbol,
                    f"models/lgbm_{symbol.replace('USDT','').lower()}.txt")
        success = train_from_parquet(symbol, info['path'], model_out)
        trained[symbol] = {
            **info,
            'model_path': model_out,
            'trained': success,
        }

    return trained


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download historical klines from Binance Vision")
    parser.add_argument('--symbols', nargs='+', default=DEFAULT_SYMBOLS,
                        help=f'Symbols to download (default: {DEFAULT_SYMBOLS})')
    parser.add_argument('--timeframe', default=DEFAULT_TIMEFRAME,
                        help=f'Timeframe (default: {DEFAULT_TIMEFRAME})')
    parser.add_argument('--start-year', type=int, default=DEFAULT_START_YEAR,
                        help=f'Start year (default: {DEFAULT_START_YEAR})')
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--train', action='store_true',
                        help='Also train LightGBM models after downloading')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if parquet files exist')
    args = parser.parse_args()

    # Pass force flag to download functions
    if hasattr(args, 'force') and args.force:
        # Monkey-patch to pass force through
        original_download = download_symbol
        def forced_download(symbol, timeframe="1h", start_year=2020, end_year=2026, output_dir="data", force=True):
            return original_download(symbol, timeframe, start_year, end_year, output_dir, force=True)
        globals()['download_symbol'] = forced_download

    if args.train:
        results = download_and_train(args.symbols, args.timeframe, args.start_year, args.output_dir)
    else:
        results = download_all(args.symbols, args.timeframe, args.start_year, args.output_dir)

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for sym, info in results.items():
        rows = info.get('rows', 0)
        trained_str = " [TRAINED]" if info.get('trained') else ""
        print(f"  {sym}: {rows:,} rows ({info.get('start', '?')} to {info.get('end', '?')}){trained_str}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
