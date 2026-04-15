#!/usr/bin/env python3
"""
Refresh 1d OHLCV parquets from Kraken (live).
=============================================
The existing continuous adaptation loop (src/scripts/continuous_adapt.py)
only refreshes 1h and 4h data. The 1d parquets were originally backfilled
from Binance Vision (monthly archives) and then went stale because nothing
was keeping them current — which broke the chart's 1d view.

This script fetches recent 1d bars from Kraken via ccxt and *merges* them
into the existing parquet files, so the historical bars going back years
are preserved.

Usage:
    python scripts/maintenance/refresh_1d_data.py
    python scripts/maintenance/refresh_1d_data.py --symbols BTC ETH SOL
"""

import os
import sys
import time
import argparse
import logging
from typing import List

import ccxt
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('refresh_1d_data')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


def refresh_symbol_1d(asset: str, data_dir: str, limit: int = 720) -> bool:
    """Fetch recent 1d bars for `asset` from Kraken and merge into existing parquet."""
    parquet_path = os.path.join(data_dir, f'{asset}USDT-1d.parquet')

    # Load existing (if any) so we can preserve history
    if os.path.exists(parquet_path):
        existing = pd.read_parquet(parquet_path)
        # Normalize timestamp to datetime64[ns] so the merge is consistent
        if not str(existing['timestamp'].dtype).startswith('datetime'):
            ts = existing['timestamp']
            if ts.max() > 1e12:  # int ms
                existing['timestamp'] = pd.to_datetime(ts, unit='ms')
            else:
                existing['timestamp'] = pd.to_datetime(ts)
        logger.info(f"  {asset}: existing parquet has {len(existing)} bars "
                    f"(last {existing['timestamp'].max().date()})")
    else:
        existing = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        logger.info(f"  {asset}: no existing parquet, creating from scratch")

    exchange = ccxt.kraken({'enableRateLimit': True})
    symbol = f'{asset}/USD'
    try:
        bars = exchange.fetch_ohlcv(symbol, '1d', limit=limit)
    except Exception as e:
        logger.error(f"  {asset}: Kraken fetch failed: {e}")
        return False

    if not bars:
        logger.warning(f"  {asset}: Kraken returned no bars")
        return False

    fresh = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    fresh['timestamp'] = pd.to_datetime(fresh['timestamp'], unit='ms')

    # Merge: keep the newest copy of any duplicated timestamp (Kraken is the source of truth
    # for any overlapping range — close prices may have been provisional in the older file).
    combined = (
        pd.concat([existing, fresh], ignore_index=True)
        .drop_duplicates(subset='timestamp', keep='last')
        .sort_values('timestamp')
        .reset_index(drop=True)
    )

    combined.to_parquet(parquet_path, index=False)
    last_row = combined.iloc[-1]
    logger.info(f"  {asset}: wrote {len(combined)} bars "
                f"(last {last_row['timestamp'].date()} close ${float(last_row['close']):,.2f})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Refresh 1d OHLCV parquets from Kraken")
    parser.add_argument('--symbols', nargs='+', default=['BTC', 'ETH'],
                        help='Asset symbols to refresh (default: BTC ETH)')
    parser.add_argument('--limit', type=int, default=720,
                        help='Max bars to fetch per symbol (default: 720 = ~2 years)')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                        help=f'Data directory containing the parquet files (default: {DEFAULT_DATA_DIR})')
    args = parser.parse_args()

    print("=" * 60)
    print("  REFRESHING 1d OHLCV PARQUETS FROM KRAKEN")
    print("=" * 60)

    print(f"  data dir: {args.data_dir}")
    failures: List[str] = []
    for i, asset in enumerate(args.symbols):
        if i > 0:
            time.sleep(2)  # be polite to Kraken's rate limit
        ok = refresh_symbol_1d(asset, data_dir=args.data_dir, limit=args.limit)
        if not ok:
            failures.append(asset)

    print("=" * 60)
    if failures:
        print(f"  FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("  All symbols refreshed.")


if __name__ == '__main__':
    main()
