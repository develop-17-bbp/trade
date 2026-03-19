#!/usr/bin/env python3
"""Check what was done on March 10, 2026"""

import json
from datetime import datetime
from collections import defaultdict

# Load trading journal
with open('logs/trading_journal.json', 'r') as f:
    trades = json.load(f)

# Filter for March 10
march_10_trades = [t for t in trades if t['timestamp'].startswith('2026-03-10')]

print("=" * 70)
print("MARCH 10, 2026 - PROJECT ACTIVITY SUMMARY")
print("=" * 70)

print(f"\nTrades on March 10, 2026: {len(march_10_trades)}")

if march_10_trades:
    print(f"\nFirst trade timestamp: {march_10_trades[0]['timestamp']}")
    print(f"Last trade timestamp:  {march_10_trades[-1]['timestamp']}")
    
    # Count by asset
    by_asset = defaultdict(int)
    by_side = defaultdict(int)
    
    for t in march_10_trades:
        by_asset[t['asset']] += 1
        by_side[t['side']] += 1
    
    print(f"\nAssets traded on March 10:")
    for asset in sorted(by_asset.keys()):
        print(f"  {asset}: {by_asset[asset]} trades")
    
    print(f"\nTrade directions:")
    print(f"  BUY: {by_side['buy']}")
    print(f"  SELL: {by_side['sell']}")

# Check all dates
dates = {}
for t in trades:
    date = t['timestamp'][:10]
    dates[date] = dates.get(date, 0) + 1

print(f"\n\nTrades by date (all records):")
print("-" * 40)
for date in sorted(dates.keys()):
    print(f"{date}: {dates[date]:3} trades")

print("\n" + "=" * 70)
