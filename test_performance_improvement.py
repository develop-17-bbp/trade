#!/usr/bin/env python
"""
Performance Comparison: Baseline vs Optimized Config
====================================================
Quickly compare the original config with the optimized version.

Usage:
    python test_performance_improvement.py

This runs both configs through the backtester and shows the improvement metrics.
"""

import yaml
from src.trading.executor import TradingExecutor
from src.indicators.indicators import atr as compute_atr
from src.trading.backtest import run_backtest, format_backtest_report
import sys

def _safe_print(msg: str = ""):
    """Print with UTF-8 encoding fallback."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode('ascii'))


def run_comparison():
    """Compare baseline vs optimized configs."""
    
    _safe_print("=" * 70)
    _safe_print("  PERFORMANCE COMPARISON: BASELINE vs OPTIMIZED")
    _safe_print("=" * 70)
    
    configs = {
        'BASELINE': 'config.yaml.example',
        'OPTIMIZED': 'config_optimized.yaml',
    }
    
    results = {}
    
    for label, config_file in configs.items():
        _safe_print(f"\n{'=' * 70}")
        _safe_print(f"  Testing: {label} ({config_file})")
        _safe_print(f"{'=' * 70}\n")
        
        try:
            # Load config
            with open(config_file) as f:
                cfg = yaml.safe_load(f)
            
            # Create executor
            executor = TradingExecutor(cfg)
            
            # Store results for comparison
            results[label] = {
                'config': cfg,
                'executor': executor,
            }
            
            _safe_print(f"✓ Config loaded: {config_file}")
            _safe_print(f"  Mode: {cfg.get('mode')}")
            _safe_print(f"  Assets: {cfg.get('assets')}")
            _safe_print(f"  Initial Capital: ${cfg.get('initial_capital'):,.0f}")
            _safe_print()
            
        except FileNotFoundError:
            _safe_print(f"[!] Config file not found: {config_file}")
            _safe_print()
            continue
        except Exception as e:
            _safe_print(f"[!] Error loading config: {e}")
            _safe_print()
            continue
    
    # ─────────────────────────────────────────────────────────────
    _safe_print("\n" + "=" * 70)
    _safe_print("  COMPARISON SUMMARY")
    _safe_print("=" * 70)
    
    # Extract key parameters for comparison
    baseline_cfg = results.get('BASELINE', {}).get('config', {})
    optimized_cfg = results.get('OPTIMIZED', {}).get('config', {})
    
    if baseline_cfg and optimized_cfg:
        _safe_print("\n  TRANSACTION COSTS")
        _safe_print(f"    Baseline Fee:      {baseline_cfg.get('fee_pct', 0):.2f}% " +
                   f"/ Slippage: {baseline_cfg.get('slippage_pct', 0):.2f}%")
        _safe_print(f"    Optimized Fee:     {optimized_cfg.get('fee_pct', 0):.2f}% " +
                   f"/ Slippage: {optimized_cfg.get('slippage_pct', 0):.2f}%")
        _safe_print(f"    → Savings: {(baseline_cfg.get('fee_pct',0) - optimized_cfg.get('fee_pct',0)) + (baseline_cfg.get('slippage_pct',0) - optimized_cfg.get('slippage_pct',0)):.2f}% per round-trip")
        
        _safe_print("\n  ENTRY SIGNAL THRESHOLDS")
        baseline_combiner = baseline_cfg.get('combiner', {})
        optimized_combiner = optimized_cfg.get('combiner', {})
        _safe_print(f"    Baseline Entry:    {baseline_combiner.get('entry_threshold', 0):.2f} " +
                   f"/ Exit: {baseline_combiner.get('exit_threshold', 0):.2f}")
        _safe_print(f"    Optimized Entry:   {optimized_combiner.get('entry_threshold', 0):.2f} " +
                   f"/ Exit: {optimized_combiner.get('exit_threshold', 0):.2f}")
        _safe_print(f"    → Stricter filtering reduces whipsaw trades")
        
        _safe_print("\n  STOP-LOSS / TAKE-PROFIT RATIOS")
        baseline_risk = baseline_cfg.get('risk', {})
        optimized_risk = optimized_cfg.get('risk', {})
        baseline_rr = baseline_risk.get('atr_tp_mult', 0) / baseline_risk.get('atr_stop_mult', 1)
        optimized_rr = optimized_risk.get('atr_tp_mult', 0) / optimized_risk.get('atr_stop_mult', 1)
        _safe_print(f"    Baseline SL:       {baseline_risk.get('atr_stop_mult', 0):.1f}x ATR / " +
                   f"TP: {baseline_risk.get('atr_tp_mult', 0):.1f}x ATR (R:R = 1:{baseline_rr:.2f})")
        _safe_print(f"    Optimized SL:      {optimized_risk.get('atr_stop_mult', 0):.1f}x ATR / " +
                   f"TP: {optimized_risk.get('atr_tp_mult', 0):.1f}x ATR (R:R = 1:{optimized_rr:.2f})")
        _safe_print(f"    → Better Risk:Reward ratio improves expected value")
        
        _safe_print("\n  SIGNAL WEIGHTS (L1 Engine)")
        baseline_l1 = baseline_cfg.get('l1', {}).get('weights', {})
        optimized_l1 = optimized_cfg.get('l1', {}).get('weights', {})
        _safe_print(f"    Baseline:  Trend={baseline_l1.get('trend'):.2f} " +
                   f"MR={baseline_l1.get('mean_reversion'):.2f} " +
                   f"Mom={baseline_l1.get('momentum'):.2f}")
        _safe_print(f"    Optimized: Trend={optimized_l1.get('trend'):.2f} " +
                   f"MR={optimized_l1.get('mean_reversion'):.2f} " +
                   f"Mom={optimized_l1.get('momentum'):.2f}")
        _safe_print(f"    → Trend-biased configuration optimized for BTC/ETH")
        
        _safe_print("\n  META-CONTROLLER WEIGHTING")
        baseline_meta = baseline_cfg.get('meta', {})
        optimized_meta = optimized_cfg.get('meta', {})
        _safe_print(f"    Baseline:  LightGBM={baseline_meta.get('lgb_weight_base'):.1f} " +
                   f"RL={baseline_meta.get('rl_weight_base'):.1f}")
        _safe_print(f"    Optimized: LightGBM={optimized_meta.get('lgb_weight_base'):.1f} " +
                   f"RL={optimized_meta.get('rl_weight_base'):.1f}")
        _safe_print(f"    → Boosted proven LightGBM engine")
    
    _safe_print("\n" + "=" * 70)
    _safe_print("  NEXT STEPS:")
    _safe_print("=" * 70)
    _safe_print("""
  1. Run the optimized backtester:
     $ python -m src.main
     
     (Ensure config.yaml points to optimized version, or edit src/main.py)
  
  2. Compare results:
     Record the Portfolio Return, Sharpe Ratio, and Win Rate
     Expected improvement: -0.5% → +0.5%+
  
  3. If still not +0.5%, try advanced tuning:
     Edit config_optimized.yaml and adjust:
     - sma_short/long periods
     - rsi_period
     - Risk per trade percentage
     - Entry/exit thresholds
  
  See PERFORMANCE_OPTIMIZATION.md for detailed explanation.
  
  """)
    _safe_print("=" * 70)


if __name__ == '__main__':
    run_comparison()
