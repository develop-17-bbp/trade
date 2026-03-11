#!/usr/bin/env python
import ccxt
import pandas as pd
from src.models.lightgbm_classifier import LightGBMClassifier

# Fetch small sample
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])

# Test feature extraction
clf = LightGBMClassifier()
try:
    features = clf.extract_features(df['close'].tolist(), df['high'].tolist(), df['low'].tolist(), df['volume'].tolist())
    print(f'✓ Extracted {len(features)} feature vectors')
    if features:
        print(f'✓ Features per vector: {len(features[0])}')
        print(f'✓ Sample feature keys: {list(features[0].keys())[:5]}')
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
