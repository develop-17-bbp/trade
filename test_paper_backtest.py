#!/usr/bin/env python
"""Quick paper mode test with CSV data."""

import sys
import pandas as pd
sys.path.insert(0, '.')

print("[TEST] Loading AAVE data from CSV...")
df = pd.read_csv('data/AAVE_USDT_1h.csv')
closes = df['close'].tolist()
highs = df['high'].tolist()
lows = df['low'].tolist()
volumes = df['volume'].tolist()
print(f"[OK] Loaded {len(closes):,} bars")

print("\n[TEST] Initializing LightGBM classifier...")
from src.models.lightgbm_classifier import LightGBMClassifier
clf = LightGBMClassifier()
print(f"[OK] Classifier ready, {len(clf.FEATURE_NAMES)} features")

print("\n[TEST] Extracting features (this may take 30-60 seconds)...")
features = clf.extract_features(closes, highs, lows, volumes)
print(f"[OK] Extracted {len(features)} feature vectors")

print("\n[TEST] Making predictions...")
predictions = clf.predict(features)
print(f"[OK] Got {len(predictions)} predictions")

# Count signals
long_signals = sum(1 for p, c in predictions if p == 1)
flat_signals = sum(1 for p, c in predictions if p == 0)
short_signals = sum(1 for p, c in predictions if p == -1)

print(f"\n[SIGNAL DISTRIBUTION]")
print(f"  Long (+1):  {long_signals:5d} ({100*long_signals/len(predictions):5.1f}%)")
print(f"  Flat   (0): {flat_signals:5d} ({100*flat_signals/len(predictions):5.1f}%)")
print(f"  Short (-1): {short_signals:5d} ({100*short_signals/len(predictions):5.1f}%)")

print("\n[TEST] Running backtest...")
from src.trading.backtest import run_backtest, BacktestConfig

config = BacktestConfig(
    fee_pct=0.0,
    slippage_pct=0.375,
    risk_per_trade_pct=1.0,
    max_position_pct=2.0,
    use_stops=True,
    atr_stop_mult=2.0,
    atr_tp_mult=3.0,
)

signals = [int(p) for p, c in predictions]
result = run_backtest(
    prices=closes,
    signals=signals,
    config=config,
    highs=highs,
    lows=lows
)

print(f"\n[BACKTEST RESULTS]")
print(f"  Total Return: {result.total_return_pct:+.2f}%")
print(f"  Total Trades: {result.total_trades}")
print(f"  Win Rate: {100*result.win_rate:.1f}%")
print(f"  Profit Factor: {result.profit_factor:.2f}")
print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
print(f"  Max Drawdown: {result.max_drawdown_pct:.2f}%")

if result.total_return_pct >= -2.0:
    print(f"\n[GO] Signal quality acceptable (return {result.total_return_pct:+.2f}% >= -2.0%). Ready for orders.")
else:
    print(f"\n[VETO] Returns catastrophic ({result.total_return_pct:+.2f}% < -2.0%). Skipping orders for risk prevention.")

print("\n[SUCCESS] Test completed!")
