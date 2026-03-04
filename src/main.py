"""
AI-Driven Crypto Trading System — Entry Point
===============================================
Three-Layer Hybrid Signal Architecture:
  L1 (50%) — Quantitative Engine
  L2 (30%) — Sentiment Layer
  L3 (20% + VETO) — Risk Engine

Usage:
  python -m src.main             # Paper mode (live Binance CCXT real-time data)
"""

import os
import sys
import yaml
from src.trading.executor import TradingExecutor


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file (real-time data only)."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            cfg.pop('demo', None)  # Remove demo flag if present
            return cfg

    # Fall back to example config
    example_path = "config.yaml.example"
    if os.path.exists(example_path):
        with open(example_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            cfg.pop('demo', None)  # Remove demo flag if present
            return cfg

    # Default config (real-time data enforced)
    return {
        'mode': 'testnet',
        'assets': ['BTC', 'ETH'],
        'initial_capital': 100000.0,
        # 'min_return_pct': None,  # Set to 1.0 to enforce 1% daily target override in reports
        'risk': {
            'max_position_size_pct': 2.0,
            'daily_loss_limit_pct': 3.0,
            'max_drawdown_pct': 10.0,
            'risk_per_trade_pct': 1.0,
            'atr_stop_mult': 2.0,
            'atr_tp_mult': 3.0,
        },
        'ai': {
            'use_transformer': False,
            'device': 'cpu',
        },
        'fee_pct': 0.0,
        'slippage_pct': 0.375,
    }


def main():
    """Main entry point -- enforces real-time data only."""
    config = load_config()

    # CLI overrides (transformer is optional)
    if '--transformer' in sys.argv:
        config.setdefault('ai', {})['use_transformer'] = True

    # Reject demo mode CLI flag if user tries it
    if '--demo' in sys.argv or '--live' in sys.argv:
        print("[Warning] Demo mode is no longer supported. Real-time Binance data is enforced.")

    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    executor = TradingExecutor(config)
    executor.run()


if __name__ == "__main__":
    main()
