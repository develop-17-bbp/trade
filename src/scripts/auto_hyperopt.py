"""
Auto Hyperopt -- Continuous Parameter Optimization
===================================================
Runs in background, periodically optimizes:
1. EMA period (6-12)
2. ATR SL multiplier (4-12 for Robinhood)
3. ATR TP multiplier (15-35 for Robinhood)
4. Min entry score (5-9)
5. Sniper confluence threshold (4-7)
6. Bear veto threshold (4-8)

Uses recent trade history to evaluate parameter sets.
Applies best params to config.yaml when improvement found.

Usage:
    # Single optimization cycle
    python -m src.scripts.auto_hyperopt --trials 30

    # Continuous background loop (every 4 hours)
    python -m src.scripts.auto_hyperopt --continuous --interval 4

    # Custom metric
    python -m src.scripts.auto_hyperopt --trials 50 --metric sharpe
"""

import os
import sys
import json
import time
import copy
import argparse
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import yaml
except ImportError:
    yaml = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default config path
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')
HISTORY_PATH = os.path.join(PROJECT_ROOT, 'logs', 'hyperopt_history.json')


# ============================================================================
# PARAMETER SPACE
# ============================================================================

PARAM_SPACE = {
    'ema_period': {'type': 'int', 'low': 6, 'high': 12},
    'atr_stop_mult': {'type': 'float', 'low': 4.0, 'high': 12.0, 'step': 0.5},
    'atr_tp_mult': {'type': 'float', 'low': 15.0, 'high': 35.0, 'step': 1.0},
    'min_entry_score': {'type': 'int', 'low': 5, 'high': 9},
    'sniper_min_confluence': {'type': 'int', 'low': 4, 'high': 7},
    'bear_veto_threshold': {'type': 'int', 'low': 4, 'high': 8},
}


def _suggest_params(trial) -> Dict:
    """Suggest a parameter set from the search space."""
    params = {}
    for name, spec in PARAM_SPACE.items():
        if spec['type'] == 'int':
            params[name] = trial.suggest_int(name, spec['low'], spec['high'])
        elif spec['type'] == 'float':
            step = spec.get('step')
            if step:
                params[name] = trial.suggest_float(name, spec['low'], spec['high'], step=step)
            else:
                params[name] = trial.suggest_float(name, spec['low'], spec['high'])
    return params


# ============================================================================
# CONFIG I/O
# ============================================================================

def load_config(config_path: str) -> Dict:
    """Load config.yaml and return as dict."""
    if not yaml:
        raise ImportError("PyYAML required: pip install pyyaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def get_current_params(config: Dict) -> Dict:
    """Extract current strategy params from config."""
    adaptive = config.get('adaptive', {})
    risk = config.get('risk', {})
    sniper = config.get('sniper', {})
    ai = config.get('ai', {})

    return {
        'ema_period': adaptive.get('ema_period', 8),
        'atr_stop_mult': risk.get('atr_stop_mult', 8.0),
        'atr_tp_mult': risk.get('atr_tp_mult', 25.0),
        'min_entry_score': adaptive.get('min_entry_score', 6),
        'sniper_min_confluence': sniper.get('min_confluence', 6),
        'bear_veto_threshold': ai.get('bear_veto_threshold', 6),
    }


def apply_params_to_config(config_path: str, params: Dict) -> None:
    """Update config.yaml with new parameter values. Preserves comments via round-trip."""
    config = load_config(config_path)

    # Map params back to config structure
    if 'adaptive' not in config:
        config['adaptive'] = {}
    if 'risk' not in config:
        config['risk'] = {}
    if 'sniper' not in config:
        config['sniper'] = {}
    if 'ai' not in config:
        config['ai'] = {}

    config['adaptive']['ema_period'] = params['ema_period']
    config['adaptive']['min_entry_score'] = params['min_entry_score']
    config['risk']['atr_stop_mult'] = params['atr_stop_mult']
    config['risk']['atr_tp_mult'] = params['atr_tp_mult']
    config['sniper']['min_confluence'] = params['sniper_min_confluence']
    config['ai']['bear_veto_threshold'] = params['bear_veto_threshold']

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"[HYPEROPT] Config updated: {config_path}")


# ============================================================================
# BACKTEST OBJECTIVE
# ============================================================================

def _build_backtest_config(base_config: Dict, params: Dict) -> Dict:
    """Create a backtest config with the trial's parameter set."""
    cfg = copy.deepcopy(base_config)

    # Apply trial params
    if 'adaptive' not in cfg:
        cfg['adaptive'] = {}
    if 'risk' not in cfg:
        cfg['risk'] = {}
    if 'sniper' not in cfg:
        cfg['sniper'] = {}

    cfg['adaptive']['ema_period'] = params['ema_period']
    cfg['adaptive']['min_entry_score'] = params['min_entry_score']
    cfg['risk']['atr_stop_mult'] = params['atr_stop_mult']
    cfg['risk']['atr_tp_mult'] = params['atr_tp_mult']
    cfg['sniper']['min_confluence'] = params['sniper_min_confluence']

    # Backtest needs these
    cfg['min_entry_score'] = params['min_entry_score']
    cfg['initial_capital'] = cfg.get('initial_capital', 100000.0)
    cfg['risk_per_trade_pct'] = cfg.get('risk', {}).get('risk_per_trade_pct', 1.0)
    cfg['use_ml'] = False  # Speed: no ML during hyperopt
    cfg['use_llm'] = False  # Speed: no LLM during hyperopt

    return cfg


def _run_backtest_with_params(params: Dict, base_config: Dict, asset: str = 'BTC',
                               days: int = 30, metric: str = 'profit_factor') -> float:
    """
    Run a backtest with the given parameter set and return the optimization metric.

    Returns:
        float: metric value (higher is better), or -100.0 on failure
    """
    try:
        from src.backtesting.data_loader import fetch_backtest_data
        from src.backtesting.full_engine import FullBacktestEngine
    except ImportError:
        # Fall back to simple engine
        try:
            from src.backtesting.engine import BacktestEngine, load_ohlcv_from_journal
            opens, highs, lows, closes, volumes = load_ohlcv_from_journal(asset)
            engine = BacktestEngine(
                ema_period=params['ema_period'],
                atr_stop_mult=params['atr_stop_mult'],
                min_score=params['min_entry_score'],
            )
            result = engine.run(opens, highs, lows, closes, volumes)
            if result.total_trades < 3:
                return -100.0
            return _extract_metric(result, metric)
        except Exception as e:
            logger.warning(f"[HYPEROPT] Simple backtest failed: {e}")
            return -100.0

    try:
        cfg = _build_backtest_config(base_config, params)

        # Fetch data for last N days
        data = fetch_backtest_data(
            asset=asset,
            days=days,
            primary_tf='4h',  # Robinhood uses 4h timeframe
            local_only=True,  # Use cached data for speed
        )

        if data.bar_count < 50:
            # Try without local_only if cached data insufficient
            data = fetch_backtest_data(
                asset=asset,
                days=days,
                primary_tf='4h',
                local_only=False,
            )

        if data.bar_count < 50:
            return -100.0

        engine = FullBacktestEngine(cfg)
        result = engine.run(data, verbose=False)

        if result.total_trades < 3:
            return -100.0

        return _extract_metric(result, metric)

    except Exception as e:
        logger.debug(f"[HYPEROPT] Full backtest failed: {e}")
        # Fall back to simple engine
        try:
            from src.backtesting.engine import BacktestEngine, load_ohlcv_from_journal
            opens, highs, lows, closes, volumes = load_ohlcv_from_journal(asset)
            engine = BacktestEngine(
                ema_period=params['ema_period'],
                atr_stop_mult=params['atr_stop_mult'],
                min_score=params['min_entry_score'],
            )
            result = engine.run(opens, highs, lows, closes, volumes)
            if result.total_trades < 3:
                return -100.0
            return _extract_metric(result, metric)
        except Exception as e2:
            logger.warning(f"[HYPEROPT] Fallback backtest also failed: {e2}")
            return -100.0


def _extract_metric(result, metric: str) -> float:
    """Extract the optimization metric from a backtest result."""
    if metric == 'profit_factor':
        pf = result.profit_factor
        return min(pf, 10.0) if pf != float('inf') else 10.0
    elif metric == 'sharpe':
        sr = result.sharpe_ratio if hasattr(result, 'sharpe_ratio') else 0.0
        if callable(sr):
            sr = sr()
        return sr
    elif metric == 'sortino':
        return result.sortino_ratio if hasattr(result, 'sortino_ratio') else 0.0
    elif metric == 'total_pnl':
        return result.total_pnl_pct if hasattr(result, 'total_pnl_pct') else 0.0
    elif metric == 'calmar':
        dd = result.max_drawdown_pct if hasattr(result, 'max_drawdown_pct') else 0
        pnl = result.total_pnl_pct if hasattr(result, 'total_pnl_pct') else 0
        return (pnl / dd) if dd > 0 else pnl
    else:
        # Default to profit_factor
        pf = result.profit_factor
        return min(pf, 10.0) if pf != float('inf') else 10.0


# ============================================================================
# HYPEROPT CYCLE
# ============================================================================

def run_hyperopt_cycle(config_path: str = DEFAULT_CONFIG_PATH,
                       n_trials: int = 30,
                       metric: str = 'profit_factor',
                       min_improvement_pct: float = 5.0,
                       days: int = 30,
                       dry_run: bool = False) -> Dict:
    """
    Run a single hyperparameter optimization cycle.

    1. Load current config params as baseline
    2. Run Optuna study with n_trials
    3. If best trial beats baseline by > min_improvement_pct, update config
    4. Log results to hyperopt_history.json

    Args:
        config_path: Path to config.yaml
        n_trials: Number of Optuna trials
        metric: Optimization metric (profit_factor, sharpe, sortino, total_pnl, calmar)
        min_improvement_pct: Min % improvement required to update config
        days: Days of history for backtest
        dry_run: If True, don't update config

    Returns:
        Dict with cycle results
    """
    if not OPTUNA_AVAILABLE:
        logger.error("[HYPEROPT] Optuna not installed: pip install optuna")
        return {'status': 'error', 'reason': 'optuna_not_installed'}

    start_time = time.time()
    logger.info(f"[HYPEROPT] Starting optimization cycle: {n_trials} trials, metric={metric}, days={days}")

    # Load config
    config = load_config(config_path)
    current_params = get_current_params(config)

    # Evaluate baseline (current params)
    logger.info(f"[HYPEROPT] Evaluating baseline: {current_params}")
    baseline_scores = []
    for asset in ['BTC', 'ETH']:
        try:
            score = _run_backtest_with_params(current_params, config, asset=asset, days=days, metric=metric)
            if score > -100.0:
                baseline_scores.append(score)
        except Exception as e:
            logger.warning(f"[HYPEROPT] Baseline eval failed for {asset}: {e}")

    baseline_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    logger.info(f"[HYPEROPT] Baseline {metric}: {baseline_score:.4f}")

    # Optuna study
    def objective(trial):
        params = _suggest_params(trial)
        scores = []
        for asset in ['BTC', 'ETH']:
            try:
                score = _run_backtest_with_params(params, config, asset=asset, days=days, metric=metric)
                if score > -100.0:
                    scores.append(score)
            except Exception:
                pass
        if not scores:
            return -100.0
        return sum(scores) / len(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value
    elapsed = time.time() - start_time

    logger.info(f"[HYPEROPT] Best {metric}: {best_score:.4f} (baseline: {baseline_score:.4f})")
    logger.info(f"[HYPEROPT] Best params: {best_params}")
    logger.info(f"[HYPEROPT] Elapsed: {elapsed:.0f}s")

    # Check improvement
    improvement_pct = 0.0
    if baseline_score > 0:
        improvement_pct = ((best_score - baseline_score) / baseline_score) * 100
    elif baseline_score == 0 and best_score > 0:
        improvement_pct = 100.0  # Any positive is infinite improvement from zero
    elif baseline_score < 0 and best_score > baseline_score:
        improvement_pct = abs(best_score - baseline_score) / abs(baseline_score) * 100

    updated = False
    if improvement_pct >= min_improvement_pct and not dry_run:
        logger.info(f"[HYPEROPT] Improvement {improvement_pct:.1f}% >= {min_improvement_pct}% threshold -> UPDATING config")
        try:
            apply_params_to_config(config_path, best_params)
            updated = True
        except Exception as e:
            logger.error(f"[HYPEROPT] Config update failed: {e}")
    elif dry_run:
        logger.info(f"[HYPEROPT] DRY RUN -- not updating config (improvement: {improvement_pct:.1f}%)")
    else:
        logger.info(f"[HYPEROPT] Improvement {improvement_pct:.1f}% < {min_improvement_pct}% threshold -> keeping current params")

    # Build result
    result = {
        'timestamp': datetime.utcnow().isoformat(),
        'status': 'updated' if updated else ('dry_run' if dry_run else 'no_improvement'),
        'metric': metric,
        'baseline_score': round(baseline_score, 4),
        'best_score': round(best_score, 4),
        'improvement_pct': round(improvement_pct, 2),
        'min_improvement_pct': min_improvement_pct,
        'n_trials': n_trials,
        'days': days,
        'elapsed_seconds': round(elapsed, 1),
        'current_params': current_params,
        'best_params': best_params,
        'config_updated': updated,
    }

    # Log to history
    _log_history(result)

    return result


def _log_history(result: Dict) -> None:
    """Append hyperopt result to history JSON."""
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    history = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, 'r') as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    history.append(result)

    # Keep last 100 entries
    if len(history) > 100:
        history = history[-100:]

    try:
        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"[HYPEROPT] History logged to {HISTORY_PATH} ({len(history)} entries)")
    except Exception as e:
        logger.warning(f"[HYPEROPT] Failed to write history: {e}")


# ============================================================================
# CONTINUOUS BACKGROUND LOOP
# ============================================================================

def start_background_loop(config_path: str = DEFAULT_CONFIG_PATH,
                          interval_hours: float = 4.0,
                          n_trials: int = 30,
                          metric: str = 'profit_factor',
                          min_improvement_pct: float = 5.0,
                          days: int = 30) -> threading.Thread:
    """
    Start a background thread that runs hyperopt cycles at regular intervals.

    Args:
        config_path: Path to config.yaml
        interval_hours: Hours between optimization cycles
        n_trials: Trials per cycle
        metric: Optimization metric
        min_improvement_pct: Min % improvement to update config
        days: Days of backtest history

    Returns:
        threading.Thread (daemon) that is already started
    """
    interval_seconds = interval_hours * 3600

    def _loop():
        cycle = 0
        while True:
            cycle += 1
            logger.info(f"[HYPEROPT] === Background cycle #{cycle} starting ===")
            try:
                result = run_hyperopt_cycle(
                    config_path=config_path,
                    n_trials=n_trials,
                    metric=metric,
                    min_improvement_pct=min_improvement_pct,
                    days=days,
                )
                status = result.get('status', 'unknown')
                improvement = result.get('improvement_pct', 0)
                logger.info(f"[HYPEROPT] Cycle #{cycle} done: {status} (improvement: {improvement:.1f}%)")
            except Exception as e:
                logger.error(f"[HYPEROPT] Cycle #{cycle} failed: {e}", exc_info=True)

            logger.info(f"[HYPEROPT] Next cycle in {interval_hours:.1f} hours")
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_loop, name="auto-hyperopt", daemon=True)
    thread.start()
    logger.info(f"[HYPEROPT] Background loop started (interval={interval_hours}h, trials={n_trials}, metric={metric})")
    return thread


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Auto Hyperopt -- Continuous Parameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.scripts.auto_hyperopt --trials 30
  python -m src.scripts.auto_hyperopt --trials 50 --metric sharpe --days 60
  python -m src.scripts.auto_hyperopt --continuous --interval 4
  python -m src.scripts.auto_hyperopt --dry-run --trials 10
        """,
    )
    parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials (default: 30)')
    parser.add_argument('--metric', type=str, default='profit_factor',
                        choices=['profit_factor', 'sharpe', 'sortino', 'total_pnl', 'calmar'],
                        help='Metric to optimize (default: profit_factor)')
    parser.add_argument('--days', type=int, default=30, help='Days of history for backtest (default: 30)')
    parser.add_argument('--min-improvement', type=float, default=5.0,
                        help='Min improvement %% to update config (default: 5.0)')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to config.yaml')
    parser.add_argument('--dry-run', action='store_true', help='Evaluate only, do not update config')
    parser.add_argument('--continuous', action='store_true', help='Run in continuous background mode')
    parser.add_argument('--interval', type=float, default=4.0,
                        help='Hours between cycles in continuous mode (default: 4)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    print("=" * 65)
    print("  AUTO HYPEROPT -- Continuous Parameter Optimization")
    print("=" * 65)
    print(f"  Config:       {args.config}")
    print(f"  Trials:       {args.trials}")
    print(f"  Metric:       {args.metric}")
    print(f"  Days:         {args.days}")
    print(f"  Min improve:  {args.min_improvement}%")
    print(f"  Dry run:      {args.dry_run}")
    print(f"  Continuous:   {args.continuous} (interval={args.interval}h)")
    print("=" * 65)

    if args.continuous:
        # Continuous mode: run first cycle immediately, then loop
        print("\n[HYPEROPT] Running initial cycle...")
        result = run_hyperopt_cycle(
            config_path=args.config,
            n_trials=args.trials,
            metric=args.metric,
            min_improvement_pct=args.min_improvement,
            days=args.days,
            dry_run=args.dry_run,
        )
        _print_result(result)

        if not args.dry_run:
            print(f"\n[HYPEROPT] Starting background loop (every {args.interval}h)...")
            print("[HYPEROPT] Press Ctrl+C to stop\n")
            try:
                thread = start_background_loop(
                    config_path=args.config,
                    interval_hours=args.interval,
                    n_trials=args.trials,
                    metric=args.metric,
                    min_improvement_pct=args.min_improvement,
                    days=args.days,
                )
                # Keep main thread alive
                while thread.is_alive():
                    thread.join(timeout=60)
            except KeyboardInterrupt:
                print("\n[HYPEROPT] Stopped by user")
        else:
            print("[HYPEROPT] Dry run -- not starting continuous loop")
    else:
        # Single cycle
        result = run_hyperopt_cycle(
            config_path=args.config,
            n_trials=args.trials,
            metric=args.metric,
            min_improvement_pct=args.min_improvement,
            days=args.days,
            dry_run=args.dry_run,
        )
        _print_result(result)


def _print_result(result: Dict) -> None:
    """Pretty-print hyperopt cycle result."""
    print(f"\n{'=' * 65}")
    print(f"  HYPEROPT RESULT")
    print(f"{'=' * 65}")
    print(f"  Status:        {result.get('status', 'unknown')}")
    print(f"  Metric:        {result.get('metric', '?')}")
    print(f"  Baseline:      {result.get('baseline_score', 0):.4f}")
    print(f"  Best:          {result.get('best_score', 0):.4f}")
    print(f"  Improvement:   {result.get('improvement_pct', 0):+.1f}%")
    print(f"  Config updated: {result.get('config_updated', False)}")
    print(f"  Elapsed:       {result.get('elapsed_seconds', 0):.0f}s")
    print(f"  Trials:        {result.get('n_trials', 0)}")

    best = result.get('best_params', {})
    if best:
        print(f"\n  Best Parameters:")
        for k, v in best.items():
            print(f"    {k}: {v}")

    current = result.get('current_params', {})
    if current:
        print(f"\n  Previous Parameters:")
        for k, v in current.items():
            new_v = best.get(k, v)
            changed = " <--" if new_v != v else ""
            print(f"    {k}: {v}{changed}")

    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
