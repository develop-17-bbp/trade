"""Train LightGBM from a local OHLCV CSV/Parquet file.

Usage:
    python -m src.scripts.train_lgbm_local --input C:/path/to/your.csv --model-out models/lgbm_local.txt

The file must contain columns: timestamp, open, high, low, close, volume
"""
import argparse
import os
import pandas as pd
from typing import List, Dict

from src.scripts import train_lgbm


def parse_args():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input', required=True, help='CSV or Parquet file with OHLCV')
    p.add_argument('--model-out', default='models/lgbm_local.txt')
    return p.parse_args()


def load_input(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    args = parse_args()
    df = load_input(args.input)

    # tolerant column mapping: accept common variants (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    def find(names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    mapping = {
        'timestamp': find(['timestamp','time','date','datetime']),
        'open': find(['open','o','open_price']),
        'high': find(['high','h','high_price']),
        'low': find(['low','l','low_price']),
        'close': find(['close','c','close_price','adj_close','price']),
        'volume': find(['volume','v','vol']),
    }

    missing = [k for k,v in mapping.items() if v is None]
    if missing:
        raise ValueError(f"Input missing required columns (or unrecognized names): {missing}. Found columns: {list(df.columns)[:10]}")

    # rename to canonical names
    df = df.rename(columns={mapping[k]: k for k in mapping})

    print(f"Loaded {len(df)} rows from {args.input}")
    features, labels = train_lgbm.build_dataset(df)
    print(f"Built dataset: {len(features)} examples")

    # try training
    # attempt to train. If LightGBM unavailable or training did not produce
    # a model file, save the prepared dataset for offline training.
    train_lgbm.train_model(features, labels, args.model_out)
    if not os.path.exists(args.model_out):
        print("Model file not found after train attempt; saving dataset for offline training.")
        os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)
        df_features = pd.DataFrame(features)
        df_features['label'] = labels
        out_path = os.path.splitext(args.model_out)[0] + '_dataset.parquet'
        df_features.to_parquet(out_path)
        print(f"Saved prepared dataset to {out_path}")


if __name__ == '__main__':
    main()
