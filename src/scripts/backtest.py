"""
Backtest CLI — Run Full-Fidelity Strategy Backtest
====================================================
Usage:
    python -m src.scripts.backtest --asset BTC --days 30
    python -m src.scripts.backtest --asset ETH --days 90 --timeframe 15m
    python -m src.scripts.backtest --asset BTC --days 30 --csv results/btc_30d.csv
    python -m src.scripts.backtest --asset BTC --start 2025-01-01 --end 2025-03-01
    python -m src.scripts.backtest --asset BTC --days 7 --verbose
    python -m src.scripts.backtest --parallel --ml --local --start 2017-08-17 --end 2026-04-01
"""

import os
import sys
import argparse
import yaml
import multiprocessing

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backtesting.data_loader import fetch_backtest_data
from src.backtesting.full_engine import FullBacktestEngine


def main():
    parser = argparse.ArgumentParser(description='EMA(8) Crossover Strategy Backtester')
    parser.add_argument('--asset', type=str, default='BTC', help='Asset to backtest (BTC or ETH)')
    parser.add_argument('--days', type=int, default=30, help='Days of history')
    parser.add_argument('--timeframe', type=str, default='5m', help='Primary timeframe (1m, 5m, 15m, 1h, 4h)')
    parser.add_argument('--start', type=str, default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=None, help='End date YYYY-MM-DD')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital USD')
    parser.add_argument('--min-score', type=int, default=None, help='Min entry score (default: from config)')
    parser.add_argument('--max-score', type=int, default=None, help='Max entry score cap (default: no cap)')
    parser.add_argument('--hard-stop', type=float, default=None, help='Hard stop pct (e.g. -2.5)')
    parser.add_argument('--cooldown', type=int, default=None, help='Post-close cooldown in minutes')
    parser.add_argument('--short-penalty', type=int, default=None, help='Extra score needed for SHORT entries')
    parser.add_argument('--csv', type=str, default=None, help='Export trades to CSV')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print bar-by-bar output')
    parser.add_argument('--ml', action='store_true', help='Enable ML inference (LightGBM, LSTM, Category B risk models)')
    parser.add_argument('--llm', action='store_true', help='Enable LLM gate (2-pass: Mistral scanner + Llama analyst via Ollama)')
    parser.add_argument('--local', action='store_true', help='Local data only — no API fetch (use cached JSON/parquet)')
    parser.add_argument('--parallel', action='store_true', help='Run BTC + ETH backtests in parallel')
    parser.add_argument('--exchange', type=str, default='binance', help='Data source exchange')

    args = parser.parse_args()

    # Load config
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

    # Override from args
    if args.min_score is not None:
        config['min_entry_score'] = args.min_score
    elif 'min_entry_score' not in config:
        config['min_entry_score'] = 4
    if args.max_score is not None:
        config['max_entry_score'] = args.max_score

    # Hard stop override
    if hasattr(args, 'hard_stop') and args.hard_stop is not None:
        config['hard_stop_pct'] = args.hard_stop
    if hasattr(args, 'cooldown') and args.cooldown is not None:
        config['post_close_cooldown_min'] = args.cooldown
    if hasattr(args, 'short_penalty') and args.short_penalty is not None:
        config['short_score_penalty'] = args.short_penalty

    config['initial_capital'] = args.capital
    config['risk_per_trade_pct'] = config.get('risk_per_trade_pct', 2.0)
    config['use_ml'] = args.ml
    config['use_llm'] = args.llm
    config['asset'] = args.asset  # for per-asset ML model loading

    print("=" * 65)
    print("  EMA(8) CROSSOVER STRATEGY BACKTESTER")
    print("=" * 65)
    print(f"  Asset: {args.asset}")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Period: {args.days} days" + (f" ({args.start} to {args.end})" if args.start else ""))
    print(f"  Capital: ${args.capital:,.0f}")
    print(f"  Min Score: {config['min_entry_score']}")
    print(f"  ML Inference: {'ON' if args.ml else 'OFF'}")
    print(f"  LLM Gate: {'ON' if args.llm else 'OFF'}")
    print("=" * 65)

    # 1. Fetch data
    print("\n[STEP 1] Fetching historical data...")
    data = fetch_backtest_data(
        asset=args.asset,
        days=args.days,
        primary_tf=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        exchange_id=args.exchange,
        local_only=args.local,
    )

    if data.bar_count < 100:
        print(f"  ERROR: Only {data.bar_count} bars — need at least 100")
        sys.exit(1)

    print(f"  Primary: {data.bar_count} bars of {args.timeframe}")
    for tf, tf_data in data.timeframes.items():
        if tf != args.timeframe:
            print(f"  Context: {len(tf_data.get('closes', []))} bars of {tf}")

    # 2. Run backtest
    print("\n[STEP 2] Running backtest...")
    engine = FullBacktestEngine(config)
    metrics = engine.run(data, verbose=args.verbose)

    # 3. Print results
    print(metrics.summary())

    # 4. Export CSV
    if args.csv:
        csv_dir = os.path.dirname(args.csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        metrics.to_csv(args.csv)

    # 5. Quick verdict
    print()
    if metrics.total_trades == 0:
        print("  VERDICT: No trades taken — filters too strict or no signals in period")
    elif metrics.win_rate >= 0.60 and metrics.profit_factor >= 1.0:
        print(f"  VERDICT: PROFITABLE | WR={metrics.win_rate:.1%} PF={metrics.profit_factor:.2f} | Strategy is working")
    elif metrics.win_rate >= 0.45 and metrics.profit_factor >= 0.9:
        print(f"  VERDICT: MARGINAL | WR={metrics.win_rate:.1%} PF={metrics.profit_factor:.2f} | Could improve")
    else:
        print(f"  VERDICT: UNPROFITABLE | WR={metrics.win_rate:.1%} PF={metrics.profit_factor:.2f} | Strategy needs work")


def _run_single_asset(asset, args_dict):
    """Worker function for parallel backtest."""
    sys.argv = ['backtest', '--asset', asset]
    for k, v in args_dict.items():
        if v is True:
            sys.argv.append(f'--{k}')
        elif v is not None and v is not False:
            sys.argv.extend([f'--{k}', str(v)])
    main()


if __name__ == '__main__':
    # Quick check for --parallel flag before argparse
    if '--parallel' in sys.argv:
        # Build args dict for forwarding (strip --parallel and --asset)
        _raw = sys.argv[1:]
        _fwd = {}
        _skip_next = False
        for idx, arg in enumerate(_raw):
            if _skip_next:
                _skip_next = False
                continue
            if arg == '--parallel':
                continue
            if arg == '--asset':
                _skip_next = True
                continue
            if arg.startswith('--'):
                key = arg.lstrip('-').replace('-', '_')
                if idx + 1 < len(_raw) and not _raw[idx + 1].startswith('--'):
                    _fwd[arg.lstrip('-')] = _raw[idx + 1]
                    _skip_next = True
                else:
                    _fwd[arg.lstrip('-')] = True
        # Auto-set CSV outputs
        if 'csv' not in _fwd:
            _fwd_btc = {**_fwd, 'csv': 'results/btc_ml.csv'}
            _fwd_eth = {**_fwd, 'csv': 'results/eth_ml.csv'}
        else:
            _fwd_btc = {**_fwd}
            _fwd_eth = {**_fwd, 'csv': _fwd['csv'].replace('btc', 'eth').replace('BTC', 'ETH')}

        print("=" * 65)
        print("  PARALLEL BACKTEST: BTC + ETH")
        print("=" * 65)
        p_btc = multiprocessing.Process(target=_run_single_asset, args=('BTC', _fwd_btc))
        p_eth = multiprocessing.Process(target=_run_single_asset, args=('ETH', _fwd_eth))
        p_btc.start()
        p_eth.start()
        p_btc.join()
        p_eth.join()
        print("\n" + "=" * 65)
        print("  BOTH BACKTESTS COMPLETE")
        print("=" * 65)
    else:
        main()
