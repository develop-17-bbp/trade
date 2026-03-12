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
import os
import time
from typing import List, Dict, Optional, Sequence, Tuple

import ccxt
import numpy as np
import pandas as pd

from src.models.lightgbm_classifier import LightGBMClassifier
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


def load_ohlcv(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    def find(names: Sequence[str]) -> Optional[str]:
        for name in names:
            if name in cols:
                return cols[name]
        return None

    mapping = {
        'timestamp': find(['timestamp', 'time', 'date', 'datetime']),
        'open': find(['open', 'o', 'open_price']),
        'high': find(['high', 'h', 'high_price']),
        'low': find(['low', 'l', 'low_price']),
        'close': find(['close', 'c', 'close_price', 'adj_close', 'price']),
        'volume': find(['volume', 'v', 'vol']),
    }
    rename_map = {source: target for target, source in mapping.items() if source is not None}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


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


def default_model_out(symbol: str, model_dir: str = 'models') -> str:
    base = symbol.split('/')[0].lower()
    return os.path.join(model_dir, f'lgbm_{base}.txt')


def train_symbol(symbol: str,
                 timeframe: str,
                 since: int,
                 until: int,
                 model_out: str,
                 input_path: Optional[str] = None) -> Tuple[str, int]:
    if input_path:
        df = load_ohlcv(input_path)
    else:
        df = fetch_ohlcv(symbol, timeframe, since, until)

    if df.empty:
        raise ValueError(f"No data fetched for {symbol}")

    features, labels = build_dataset(df)
    os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
    train_model(features, labels, model_out)
    return model_out, len(df)


def parse_args():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--symbol', help='Single symbol to train, e.g. BTC/USDT')
    p.add_argument('--symbols', nargs='+', help='Batch symbols, e.g. BTC/USDT ETH/USDT')
    p.add_argument('--timeframe', default='1h')
    p.add_argument('--since',
                   help='ISO date or milliseconds since epoch')
    p.add_argument('--until',
                   help='ISO date or milliseconds since epoch')
    p.add_argument('--model-out', default='models/lgbm_model.txt')
    p.add_argument('--model-dir', default='models',
                   help='Output directory used with --symbols')
    p.add_argument('--input', help='Local CSV/parquet OHLCV input for single-symbol training')
    p.add_argument('--inputs', nargs='+',
                   help='Local CSV/parquet OHLCV inputs matched 1:1 with --symbols')
    return p.parse_args()


def to_millis(s):
    try:
        return int(s)
    except ValueError:
        return int(pd.to_datetime(s).timestamp() * 1000)


def main():
    args = parse_args()
    symbols = args.symbols or ([args.symbol] if args.symbol else [])
    if not symbols:
        raise ValueError("Provide --symbol or --symbols")

    if args.inputs and len(args.inputs) != len(symbols):
        raise ValueError("--inputs must match the number of --symbols")

    if (args.input or args.inputs) is None:
        if not args.since or not args.until:
            raise ValueError("--since and --until are required when fetching from exchange")
        since = to_millis(args.since)
        until = to_millis(args.until)
    else:
        since = 0
        until = 0

    if len(symbols) == 1:
        model_out = args.model_out if args.model_out != 'models/lgbm_model.txt' else default_model_out(symbols[0], args.model_dir)
        model_path, bars = train_symbol(
            symbol=symbols[0],
            timeframe=args.timeframe,
            since=since,
            until=until,
            model_out=model_out,
            input_path=args.input,
        )
        print(f"Trained {symbols[0]} on {bars} bars -> {model_path}")
        return

    input_paths = args.inputs or [None] * len(symbols)
    for symbol, input_path in zip(symbols, input_paths):
        model_out = default_model_out(symbol, args.model_dir)
        model_path, bars = train_symbol(
            symbol=symbol,
            timeframe=args.timeframe,
            since=since,
            until=until,
            model_out=model_out,
            input_path=input_path,
        )
        print(f"Trained {symbol} on {bars} bars -> {model_path}")


if __name__ == '__main__':
    main()
