"""
Main Entry Point
=================
Loads config and starts EMA(8) + LLM trading executors.
Supports multiple exchanges running independently in parallel.
"""

import os
import sys
import logging
import threading
import yaml
from dotenv import load_dotenv

# Force UTF-8 output on Windows (prevents UnicodeEncodeError with cp1252)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.trading.executor import TradingExecutor


def run_exchange(config: dict, exchange_name: str, assets: list):
    """Run a single exchange executor in its own thread."""
    # Override config for this exchange
    ex_config = dict(config)
    ex_config['exchange'] = {'name': exchange_name}
    ex_config['assets'] = assets

    try:
        executor = TradingExecutor(ex_config)
        executor.run()
    except Exception as e:
        print(f"  [{exchange_name.upper()}] FATAL: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Load environment variables
    env_path = os.path.join(PROJECT_ROOT, '.env')
    load_dotenv(env_path, override=True)
    print(f"  [ENV] Loaded {env_path}")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(PROJECT_ROOT, 'logs', 'system_output.log'),
                mode='a',
                encoding='utf-8',
            ),
        ],
    )

    # Load config
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check for multi-exchange config
    exchanges = config.get('exchanges', [])

    if len(exchanges) >= 2:
        # Multi-exchange mode — run each in its own thread
        print("=" * 60)
        print("  MULTI-EXCHANGE MODE")
        print(f"  Exchanges: {[e['name'] for e in exchanges]}")
        print("=" * 60)

        threads = []
        for ex_cfg in exchanges:
            name = ex_cfg['name']
            assets = ex_cfg.get('assets', config.get('assets', ['BTC', 'ETH']))

            t = threading.Thread(
                target=run_exchange,
                args=(config, name, assets),
                name=f"executor-{name}",
                daemon=True,
            )
            threads.append(t)
            t.start()
            print(f"  Started {name.upper()} executor: {assets}")

        # Wait for all threads (or Ctrl+C)
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\n  [SHUTDOWN] Stopping all exchanges...")

    elif len(exchanges) == 1:
        # Single exchange from new config format
        ex_cfg = exchanges[0]
        config['exchange'] = {'name': ex_cfg['name']}
        config['assets'] = ex_cfg.get('assets', config.get('assets', ['BTC', 'ETH']))
        executor = TradingExecutor(config)
        executor.run()

    else:
        # Legacy single exchange config
        executor = TradingExecutor(config)
        executor.run()


if __name__ == '__main__':
    main()
