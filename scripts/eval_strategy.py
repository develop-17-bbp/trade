"""Quick strategy evaluator for the autonomous iteration loop.

Reads engine.py's current defaults, runs the 2y BTC 15m backtest once,
prints metrics + contract pass/fail. Exit 0 if both contracts (Sharpe>1.5,
DD<15%) hold, else 1.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtesting.engine import BacktestEngine

df = pd.read_parquet("data/BTCUSDT-15m.parquet").sort_values("timestamp").reset_index(drop=True)
eng = BacktestEngine()
r = eng.run(
    opens=df["open"].astype(float).tolist(),
    highs=df["high"].astype(float).tolist(),
    lows=df["low"].astype(float).tolist(),
    closes=df["close"].astype(float).tolist(),
    volumes=df["volume"].astype(float).tolist(),
)
sharpe = r.sharpe_ratio
dd = r.max_drawdown_pct
sharpe_pass = sharpe > 1.5
dd_pass = dd < 15.0
both = sharpe_pass and dd_pass
print(
    f"params: ema={eng.ema_period} atr_p={eng.atr_period} "
    f"atr_mult={eng.atr_stop_mult} min_score={eng.min_score} "
    f"max_hold={eng.max_hold_bars} overext={eng.overextension_pct}"
)
print(
    f"result: trades={r.total_trades} WR={r.win_rate:.1%} pnl={r.total_pnl_pct:+.1f}% "
    f"Sharpe={sharpe:.2f} DD={dd:.2f}% PF={r.profit_factor:.2f}"
)
print(f"contracts: sharpe>1.5={'PASS' if sharpe_pass else 'FAIL'}  dd<15%={'PASS' if dd_pass else 'FAIL'}")
sys.exit(0 if both else 1)
