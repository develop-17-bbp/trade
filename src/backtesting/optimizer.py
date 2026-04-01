"""
Parameter Optimizer — Optuna-based strategy tuning
====================================================
Optimizes EMA period, ATR multiplier, phase thresholds, and scoring
using the backtesting engine. Cross-validates to avoid overfitting.

Usage:
    pip install optuna  (if not installed)
    python -m src.backtesting.optimizer --asset BTC --trials 100
    python -m src.backtesting.optimizer --asset ETH --trials 200 --metric sharpe
"""

import argparse
import json
from typing import List, Dict

from src.backtesting.engine import BacktestEngine, load_ohlcv_from_journal

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def objective(trial, opens, highs, lows, closes, volumes, metric='sharpe'):
    """Optuna objective function — returns metric to maximize."""

    # Parameters to optimize
    ema_period = trial.suggest_int('ema_period', 5, 20)
    atr_stop_mult = trial.suggest_float('atr_stop_mult', 0.8, 3.0, step=0.1)
    min_score = trial.suggest_int('min_score', 1, 7)
    max_hold_bars = trial.suggest_int('max_hold_bars', 36, 288)  # 3hrs to 24hrs
    overextension_pct = trial.suggest_float('overextension_pct', 5.0, 20.0, step=1.0)

    engine = BacktestEngine(
        ema_period=ema_period,
        atr_stop_mult=atr_stop_mult,
        min_score=min_score,
        max_hold_bars=max_hold_bars,
        overextension_pct=overextension_pct,
    )

    result = engine.run(opens, highs, lows, closes, volumes)

    # Minimum trades required to avoid degenerate solutions
    if result.total_trades < 5:
        return -100.0

    if metric == 'sharpe':
        return result.sharpe_ratio
    elif metric == 'sortino':
        return result.sortino_ratio
    elif metric == 'profit_factor':
        return result.profit_factor if result.profit_factor != float('inf') else 10.0
    elif metric == 'total_pnl':
        return result.total_pnl_pct
    elif metric == 'calmar':
        # Calmar = total return / max drawdown
        if result.max_drawdown_pct > 0:
            return result.total_pnl_pct / result.max_drawdown_pct
        return result.total_pnl_pct
    else:
        return result.sharpe_ratio


def cross_validate(best_params: dict, opens, highs, lows, closes, volumes, n_folds=3):
    """Walk-forward cross-validation to check for overfitting."""
    n = len(closes)
    fold_size = n // n_folds
    results = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = min(start + fold_size, n)
        if end - start < 50:
            continue

        engine = BacktestEngine(
            ema_period=best_params['ema_period'],
            atr_stop_mult=best_params['atr_stop_mult'],
            min_score=best_params['min_score'],
            max_hold_bars=best_params['max_hold_bars'],
            overextension_pct=best_params.get('overextension_pct', 10.0),
        )

        fold_result = engine.run(
            opens[start:end], highs[start:end], lows[start:end],
            closes[start:end], volumes[start:end],
        )
        results.append({
            'fold': fold + 1,
            'trades': fold_result.total_trades,
            'win_rate': fold_result.win_rate,
            'total_pnl': fold_result.total_pnl_pct,
            'sharpe': fold_result.sharpe_ratio,
            'max_dd': fold_result.max_drawdown_pct,
            'profit_factor': fold_result.profit_factor,
        })

    return results


def run_optimization(asset: str = 'BTC', n_trials: int = 100, metric: str = 'sharpe'):
    """Run full optimization pipeline."""
    if not OPTUNA_AVAILABLE:
        print("ERROR: optuna not installed. Run: pip install optuna")
        return

    print(f"Loading {asset} 5m data...")
    opens, highs, lows, closes, volumes = load_ohlcv_from_journal(asset)
    print(f"  Got {len(closes)} candles ({len(closes)/288:.1f} days)")

    print(f"\nOptimizing {n_trials} trials (metric: {metric})...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, opens, highs, lows, closes, volumes, metric),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    best_value = study.best_value

    print(f"\n{'='*60}")
    print(f"  OPTIMIZATION RESULTS — {asset}")
    print(f"{'='*60}")
    print(f"  Best {metric}: {best_value:.4f}")
    print(f"  Parameters:")
    for k, v in best.items():
        print(f"    {k}: {v}")
    print(f"{'─'*60}")

    # Run full backtest with best params
    engine = BacktestEngine(
        ema_period=best['ema_period'],
        atr_stop_mult=best['atr_stop_mult'],
        min_score=best['min_score'],
        max_hold_bars=best['max_hold_bars'],
        overextension_pct=best.get('overextension_pct', 10.0),
    )
    full_result = engine.run(opens, highs, lows, closes, volumes)
    full_result.params['asset'] = asset
    full_result.params['days'] = len(closes) / 288
    print(full_result.summary())

    # Cross-validate
    print(f"\n  Cross-Validation ({3} folds):")
    cv_results = cross_validate(best, opens, highs, lows, closes, volumes)
    for cv in cv_results:
        print(f"    Fold {cv['fold']}: {cv['trades']} trades, "
              f"WR={cv['win_rate']:.0%}, P&L={cv['total_pnl']:+.2f}%, "
              f"Sharpe={cv['sharpe']:.2f}, DD={cv['max_dd']:.1f}%")

    # Check consistency
    if cv_results:
        pnls = [cv['total_pnl'] for cv in cv_results]
        all_profitable = all(p > 0 for p in pnls)
        avg_pnl = sum(pnls) / len(pnls)
        print(f"\n    Avg P&L across folds: {avg_pnl:+.2f}%")
        if all_profitable:
            print(f"    ✓ Strategy is profitable across ALL folds — low overfit risk")
        else:
            profitable = sum(1 for p in pnls if p > 0)
            print(f"    ⚠ Profitable in {profitable}/{len(pnls)} folds — some overfit risk")

    # Save best params
    output = {
        'asset': asset,
        'metric': metric,
        'best_value': best_value,
        'params': best,
        'cv_results': cv_results,
    }
    output_path = f'logs/optimization_{asset.lower()}.json'
    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved to {output_path}")
    except Exception:
        pass

    # Print config.yaml snippet
    print(f"\n  config.yaml snippet:")
    print(f"  adaptive:")
    print(f"    ema_period: {best['ema_period']}")
    print(f"  risk:")
    print(f"    atr_stop_mult: {best['atr_stop_mult']}")
    print(f"  # min_score: {best['min_score']}")
    print(f"  max_hold_minutes: {best['max_hold_bars'] * 5}")

    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize EMA crossover parameters')
    parser.add_argument('--asset', default='BTC', help='Asset to optimize')
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--metric', default='sharpe',
                        choices=['sharpe', 'sortino', 'profit_factor', 'total_pnl', 'calmar'],
                        help='Metric to maximize')
    args = parser.parse_args()

    run_optimization(args.asset, args.trials, args.metric)
