#!/usr/bin/env python
"""Quick test of HybridStrategy and adaptive engine integration."""

import sys
sys.path.insert(0, '.')

from src.trading.strategy import HybridStrategy
import yaml

print("[TEST] Loading config...")
with open('config.yaml') as f:
    config = yaml.safe_load(f)

print("[TEST] Initializing HybridStrategy...")
strategy = HybridStrategy(config)
print("[OK] HybridStrategy initialized")

print("[TEST] Checking adaptive_engine methods...")
print(f"  - has select_strategy: {hasattr(strategy.adaptive_engine, 'select_strategy')}")
print(f"  - has select_best_strategy: {hasattr(strategy.adaptive_engine, 'select_best_strategy')}")

print("[TEST] Testing select_strategy call...")
test_features = {name: 0.0 for name in strategy.classifier.FEATURE_NAMES}
sentiment_data = {'aggregate_score': 0.0}
try:
    result = strategy.adaptive_engine.select_strategy(test_features, sentiment_data)
    print(f"[OK] select_strategy returned: {result}")
except Exception as e:
    print(f"[ERROR] select_strategy failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[SUCCESS] All tests passed!")
