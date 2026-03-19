#!/usr/bin/env python
"""Test training with proper output capture."""
import subprocess
import sys

result = subprocess.run([
    sys.executable, '-m', 'src.scripts.train_lgbm',
    '--symbol', 'BTC/USDT',
    '--timeframe', '1h',
    '--since', '2024-09-01',
    '--until', '2026-03-10',
    '--model-out', 'models/lgbm_model_test.txt'
], capture_output=True, text=True, timeout=300)

print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("\nSTDERR:")
    print(result.stderr)
print(f"\nReturn code: {result.returncode}")
