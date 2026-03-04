"""Run backtest using a saved LightGBM model and produce performance metrics.

Usage:
    python -m src.scripts.run_backtest_model --input "C:/path/to/data.csv" --model models/lgbm_aave.txt --out logs/backtest_report.txt
"""
import argparse
import os
import pandas as pd
import lightgbm as lgb

from src.models.lightgbm_classifier import LightGBMClassifier
from src.trading.backtest import run_backtest, format_backtest_report, monte_carlo_simulation, format_monte_carlo_report, walk_forward_validation, format_walk_forward_report


def parse_args():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input', help='CSV or parquet OHLCV file')
    p.add_argument('--symbol', help='symbol to fetch from exchange (e.g. AAVE/USDT)')
    p.add_argument('--timeframe', default='1h', help='timeframe when fetching')
    p.add_argument('--since', help='ISO date or milliseconds to start fetching')
    p.add_argument('--until', help='ISO date or milliseconds to stop fetching')
    p.add_argument('--model', required=True, help='LightGBM model file')
    p.add_argument('--out', default='logs/backtest_report.txt')
    p.add_argument('--n-windows', type=int, default=12, help='Number of walk-forward windows')
    p.add_argument('--train-ratio', type=float, default=0.7, help='Train ratio per window')
    return p.parse_args()


def load_ohlcv(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    df = pd.read_csv(path)
    # tolerant column mapping (case-insensitive common variants)
    cols = {c.lower(): c for c in df.columns}
    def find(names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    mapping = {
        'timestamp': find(['timestamp', 'time', 'date', 'datetime']),
        'open': find(['open', 'o', 'open_price']),
        'high': find(['high', 'h', 'high_price']),
        'low': find(['low', 'l', 'low_price']),
        'close': find(['close', 'c', 'close_price', 'adj_close', 'price']),
        'volume': find(['volume', 'v', 'vol']),
    }

    # rename available columns to canonical names
    rename_map = {mapping[k]: k for k in mapping if mapping[k] is not None}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def main():
    args = parse_args()
    # obtain OHLCV data either by loading file or fetching symbol
    try:
        if args.input:
            df = load_ohlcv(args.input)
        elif args.symbol:
        from src.scripts.train_lgbm import fetch_ohlcv
        import pandas as _pd

        def to_millis(s):
            try:
                return int(s)
            except Exception:
                return int(_pd.to_datetime(s).timestamp() * 1000)

        if not args.since or not args.until:
            raise ValueError("--since and --until required when using --symbol")
        since = to_millis(args.since)
        until = to_millis(args.until)
        print(f"Fetching {args.symbol} {args.timeframe} from {args.since} to {args.until}")
        df = fetch_ohlcv(args.symbol, args.timeframe, since, until)
        os.makedirs('data', exist_ok=True)
        df.to_csv(f"data/{args.symbol.replace('/','_')}_{args.timeframe}.csv", index=False)
    else:
        raise ValueError("Either --input or --symbol must be provided")
    print(f"Loaded dataframe with {len(df)} rows")
    closes = df['close'].tolist()
    highs = df['high'].tolist() if 'high' in df.columns else closes
    lows = df['low'].tolist() if 'low' in df.columns else closes
    volumes = df['volume'].tolist() if 'volume' in df.columns else [1.0] * len(closes)

    clf = LightGBMClassifier()
    # load LightGBM booster
    booster = lgb.Booster(model_file=args.model)
    clf._lgb_model = booster
    clf._fitted = True

    features = clf.extract_features(closes, highs, lows, volumes)
    preds = clf.predict(features)
    signals = [int(c) for c, _ in preds]

    # run backtest
    bt = run_backtest(prices=closes, signals=signals, highs=highs, lows=lows, features=features)

    report = format_backtest_report(bt)

    # monte carlo
    mc = monte_carlo_simulation(bt.trades, n_simulations=1000, initial_capital=100_000.0)
    mc_report = format_monte_carlo_report(mc)

    # walk-forward (lightweight)
    wf = walk_forward_validation(closes, signals, n_windows=args.n_windows, train_ratio=args.train_ratio)
    wf_report = format_walk_forward_report(wf)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(report)
        f.write('\n')
        f.write(mc_report)
        f.write('\n')
        f.write(wf_report)

    print(report)
    print(mc_report)
    print(wf_report)
    print(f"Saved backtest report to {args.out}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Backtest runner failed: {e}")
        raise


if __name__ == '__main__':
    main()
