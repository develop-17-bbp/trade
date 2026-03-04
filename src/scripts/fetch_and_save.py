from src.scripts.train_lgbm import fetch_ohlcv
import pandas as pd
import os

# adjust dates as needed
since = int(pd.to_datetime('2019-01-01').timestamp()*1000)
until = int(pd.to_datetime('2026-03-01').timestamp()*1000)
df = fetch_ohlcv('AAVE/USDT', '1h', since, until)
print('fetched', len(df), 'rows')
os.makedirs('data', exist_ok=True)
df.to_csv('data/aave_1h_full.csv', index=False)
print('saved to data/aave_1h_full.csv')
