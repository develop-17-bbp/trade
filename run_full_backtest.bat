@echo off
python -m src.scripts.run_backtest_model --symbol AAVE/USDT --timeframe 1h --since 2019-01-01 --until 2026-03-04 --model models/lgbm_aave.txt --out logs/backtest_full.txt
