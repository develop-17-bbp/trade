"""Sweep BacktestEngine parameter space to find configs that satisfy the
contract (Sharpe > 1.5, DD < 15%) on the 2-year BTC 15m baseline.

Each row is one hypothesis: a (param-set, result) pair. The autonomous
iteration loop's job is to enumerate hypotheses, score them, and surface
the winner. Output is a markdown table the operator can read.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtesting.engine import BacktestEngine

DATA = Path(__file__).resolve().parents[1] / "data" / "BTCUSDT-15m.parquet"

df = pd.read_parquet(DATA).sort_values("timestamp").reset_index(drop=True)
opens = df["open"].astype(float).tolist()
highs = df["high"].astype(float).tolist()
lows = df["low"].astype(float).tolist()
closes = df["close"].astype(float).tolist()
volumes = df["volume"].astype(float).tolist()


def evaluate(label: str, **kwargs) -> dict:
    eng = BacktestEngine(**kwargs)
    r = eng.run(opens=opens, highs=highs, lows=lows, closes=closes, volumes=volumes)
    return {
        "id": label,
        "params": kwargs or {"<defaults>": "—"},
        "trades": r.total_trades,
        "wr": r.win_rate,
        "pnl": r.total_pnl_pct,
        "sharpe": r.sharpe_ratio,
        "dd": r.max_drawdown_pct,
        "pf": r.profit_factor,
        "pass_sharpe": r.sharpe_ratio > 1.5,
        "pass_dd": r.max_drawdown_pct < 15.0,
    }


# Baseline first
results = [evaluate("H0_baseline")]

# H1-H6: single-knob sweeps
results.append(evaluate("H1_atr_wider",       atr_stop_mult=2.5))
results.append(evaluate("H2_atr_tight",       atr_stop_mult=1.0))
results.append(evaluate("H3_score_strict",    min_score=5))
results.append(evaluate("H4_ema_slow",        ema_period=21))
results.append(evaluate("H5_ema_fast",        ema_period=5))
results.append(evaluate("H6_atr_period_long", atr_period=28))
results.append(evaluate("H7_max_hold_short",  max_hold_bars=24))
results.append(evaluate("H8_max_hold_long",   max_hold_bars=144))
results.append(evaluate("H9_overext_strict",  overextension_pct=5.0))

# H10-H15: combos with promising single-knob ideas
results.append(evaluate("H10_score5_ema21",   min_score=5, ema_period=21))
results.append(evaluate("H11_score5_hold24",  min_score=5, max_hold_bars=24))
results.append(evaluate("H12_score7_ema21",   min_score=7, ema_period=21))
results.append(evaluate("H13_combo_strict",   min_score=5, ema_period=21, max_hold_bars=24))
results.append(evaluate("H14_combo_strict_overext", min_score=5, ema_period=21, max_hold_bars=24, overextension_pct=5.0))
results.append(evaluate("H15_score3_ema21_atr20", ema_period=21, atr_period=20))

# H16-H19: more aggressive combos
results.append(evaluate("H16_score9",         min_score=9))
results.append(evaluate("H17_score5_atr_tight", min_score=5, atr_stop_mult=1.0))
results.append(evaluate("H18_ema34_score5",   ema_period=34, min_score=5))
results.append(evaluate("H19_super_strict",   ema_period=34, min_score=7, max_hold_bars=24, overextension_pct=5.0))

# Print markdown table
print("| ID | params | trades | WR | pnl% | Sharpe | DD% | PF | Sharpe>1.5 | DD<15% |")
print("|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|")
for r in results:
    pstr = ", ".join(f"{k}={v}" for k, v in r["params"].items() if not k.startswith("<"))
    if not pstr:
        pstr = "<defaults>"
    sh = "PASS" if r["pass_sharpe"] else "FAIL"
    dd = "PASS" if r["pass_dd"] else "FAIL"
    print(
        f"| {r['id']} | {pstr} | {r['trades']} | {r['wr']:.1%} | "
        f"{r['pnl']:+.1f} | {r['sharpe']:.2f} | {r['dd']:.2f} | {r['pf']:.2f} | "
        f"{sh} | {dd} |"
    )

# Highlight winners (both contracts pass)
winners = [r for r in results if r["pass_sharpe"] and r["pass_dd"]]
print()
if winners:
    print(f"WINNERS ({len(winners)}): both contracts satisfied")
    for w in sorted(winners, key=lambda r: -r["sharpe"]):
        pstr = ", ".join(f"{k}={v}" for k, v in w["params"].items() if not k.startswith("<"))
        print(f"  {w['id']:30s}  Sharpe={w['sharpe']:.2f}  DD={w['dd']:.2f}%  ({pstr})")
else:
    print("NO WINNERS — no tested config satisfied both Sharpe>1.5 AND DD<15%")
    # Best DD with Sharpe still passing
    sharpe_pass = [r for r in results if r["pass_sharpe"]]
    if sharpe_pass:
        best = min(sharpe_pass, key=lambda r: r["dd"])
        pstr = ", ".join(f"{k}={v}" for k, v in best["params"].items() if not k.startswith("<"))
        print(f"  Best DD with Sharpe>1.5:  {best['id']}  Sharpe={best['sharpe']:.2f}  DD={best['dd']:.2f}%  ({pstr})")
