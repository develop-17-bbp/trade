import pandas as pd
import lightgbm as lgb
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.models.lightgbm_classifier import LightGBMClassifier
from src.trading.backtest import run_backtest, format_backtest_report, monte_carlo_simulation, format_monte_carlo_report, walk_forward_validation, format_walk_forward_report

# load data
path = 'data/AAVE_USDT_1h.csv'
df = pd.read_csv(path)
print(f'Loaded {len(df)} rows from {path}')
closes = df['close'].tolist()
highs = df['high'].tolist() if 'high' in df.columns else closes
lows = df['low'].tolist() if 'low' in df.columns else closes
volumes = df['volume'].tolist() if 'volume' in df.columns else [1.0]*len(closes)

# load model
model_path = 'models/lgbm_aave.txt'
clf = LightGBMClassifier()
clf._lgb_model = lgb.Booster(model_file=model_path)
clf._fitted = True

features = clf.extract_features(closes, highs, lows, volumes)
preds = clf.predict(features)
signals = [int(c) for c, _ in preds]

bt = run_backtest(prices=closes, signals=signals, highs=highs, lows=lows, features=features)
report = format_backtest_report(bt)
print(report)

mc = monte_carlo_simulation(bt.trades)
mc_report = format_monte_carlo_report(mc)
print(mc_report)

wf = walk_forward_validation(closes, signals, n_windows=12)
wf_report = format_walk_forward_report(wf)
print(wf_report)

# persist all reports
os.makedirs('logs', exist_ok=True)
with open('logs/backtest_full_direct.txt','w',encoding='utf-8') as f:
    f.write(report + '\n')
    f.write(mc_report + '\n')
    f.write(wf_report + '\n')
print('Reports saved to logs/backtest_full_direct.txt')
