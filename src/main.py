"""
Main Entry Point
=================
Loads config and starts the EMA(8) + LLM trading executor.
"""

import os
import sys
import logging
import yaml
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.trading.executor import TradingExecutor


def main():
    # Load environment variables (override=True ensures .env wins over system env)
    env_path = os.path.join(PROJECT_ROOT, '.env')
    load_dotenv(env_path, override=True)
    print(f"  [ENV] Loaded {env_path}")
    print(f"  [ENV] BYBIT_KEY set: {bool(os.environ.get('BYBIT_TESTNET_KEY'))}")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(PROJECT_ROOT, 'logs', 'system_output.log'),
                mode='a',
            ),
        ],
    )

    # Load config
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Start executor
    executor = TradingExecutor(config)
    executor.run()


if __name__ == '__main__':
    main()
