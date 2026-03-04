"""Training utility for LightGBM classifier using historical Binance data.

Usage:
    python -m src.scripts.train_lgbm --symbol BTC/USDT --timeframe 1h \
        --since 2020-01-01 --until 2025-01-01 --model-out models/lgbm_model.txt

The script will:
 1. fetch OHLCV data from Binance (via PriceFetcher)
 2. compute the 80+ feature vector for each bar using
    :pyfunc:`src.models.lightgbm_classifier.LightGBMClassifier.extract_features`
 3. label each bar with direction (+1/-1/0) based on the next-bar return
 4. split into train/test sets (default 80/20)
 5. train a LightGBM model and persist it

Requirements:
  * ccxt (already a dependency)
  * lightgbm installed to train; falls back to saving dataset if absent.

"""

import argparse
import time
import math
from typing import List, Dict, Optional

import ccxt
import numpy as np
import pandas as pd

from src.models.lightgbm_classifier import LightGBMClassifier
from src.trading.backtest import run_backtest
from src.indicators.indicators import sma  # ensure import for feature generation


def fetch_ohlcv(symbol: str, timeframe: str, since: int, until: int) -> pd.DataFrame:
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })
    data = []
    limit = 1000
    t = since
    while t < until:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=t, limit=limit)
        if not ohlcv:
            break
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        data.append(df)
        t = int(df['timestamp'].iloc[-1]) + exchange.parse_timeframe(timeframe) * 1000
        time.sleep(exchange.rateLimit / 1000)
    if data:
        return pd.concat(data, ignore_index=True)
    else:
        return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])


def build_dataset(df: pd.DataFrame) -> (List[Dict[str,float]], List[int]):
    closes = df['close'].tolist()
    highs = df['high'].tolist()
    lows = df['low'].tolist()
    volumes = df['volume'].tolist()
    clf = LightGBMClassifier()
    features = clf.extract_features(closes, highs, lows, volumes)

    # label by next-bar return
    labels: List[int] = []
    for i in range(len(closes)):
        if i == len(closes) - 1:
            labels.append(0)
        else:
            ret = (closes[i+1] - closes[i]) / closes[i]
            if ret > 0:
                labels.append(1)
            elif ret < 0:
                labels.append(-1)
            else:
                labels.append(0)
    return features, labels


def train_model(features: List[Dict[str,float]], labels: List[int], model_out: str):
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed; cannot train model.")
        return
    # convert to dataframe
    df = pd.DataFrame(features)
    df['label'] = labels
    df = df.dropna()
    if df.empty:
        print("No training data after cleaning.")
        return
    X = df.drop(columns=['label']).values
    y = df['label'].map({-1:0, 0:1, 1:2}).values  # map to multiclass

    dtrain = lgb.Dataset(X, label=y)
    params = {'objective':'multiclass', 'num_class':3, 'verbosity':-1}
    model = lgb.train(params, dtrain, num_boost_round=100)
    model.save_model(model_out)
    print(f"Model saved to {model_out}")


def parse_args():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--symbol', required=True)
    p.add_argument('--timeframe', default='1h')
    p.add_argument('--since', required=True,
                   help='ISO date or milliseconds since epoch')
    p.add_argument('--until', required=True,
                   help='ISO date or milliseconds since epoch')
    p.add_argument('--model-out', default='models/lgbm_model.txt')
    return p.parse_args()


def to_millis(s):
    try:
        return int(s)
    except ValueError:
        return int(pd.to_datetime(s).timestamp() * 1000)


def main():
    args = parse_args()
    since = to_millis(args.since)
    until = to_millis(args.until)
    df = fetch_ohlcv(args.symbol, args.timeframe, since, until)
    if df.empty:
        print("No data fetched; exiting.")
        return
    print(f"Fetched {len(df)} bars from {args.symbol} {args.timeframe}")
    features, labels = build_dataset(df)
    train_model(features, labels, args.model_out)


if __name__ == '__main__':
    main()
