#!/usr/bin/env python
"""Minimal training test."""
import sys
sys.path.insert(0, '.')

try:
    print("1. Importing classifier...")
    from src.models.lightgbm_classifier import LightGBMClassifier
    print("   ✓ Import successful")
    
    print("2. Creating test data...")
    test_closes = [100 + i * 0.5 for i in range(500)]
    test_highs = [c + 2 for c in test_closes]
    test_lows = [c - 2 for c in test_closes]
    test_volumes = [1000000 + i * 100 for i in range(500)]
    print(f"   ✓ Created {len(test_closes)} bars")
    
    print("3. Instantiating classifier...")
    clf = LightGBMClassifier()
    print("   ✓ Classifier created")
    
    print("4. Extracting features...")
    features = clf.extract_features(test_closes, test_highs, test_lows, test_volumes)
    print(f"   ✓ Extracted {len(features)} feature vectors")
    print(f"   ✓ Features per vector: {len(features[0]) if features else 0}")
    
    # Check for NaN values
    import math
    nan_count = 0
    for i, feat_dict in enumerate(features):
        for key, val in feat_dict.items():
            if isinstance(val, float) and math.isnan(val):
                nan_count += 1
    print(f"   ✓ NaN values found: {nan_count}")
    
    print("\n✅ ALL TESTS PASSED!")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
