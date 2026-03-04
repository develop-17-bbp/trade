#!/usr/bin/env python
"""Simple test runner for trading system."""

import sys
import traceback

print("=" * 60)
print("Testing import of TradingExecutor...")
print("=" * 60)

try:
    from src.trading.executor import TradingExecutor
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\nCreating executor with paper config...")
    config = {'mode': 'paper', 'assets': ['BTC', 'ETH']}
    executor = TradingExecutor(config)
    print("✓ Executor created")
except Exception as e:
    print(f"✗ Failed to create executor: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All basic checks passed!")
print("=" * 60)
