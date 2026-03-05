"""
AI-Driven Crypto Trading System — Entry Point
===============================================
Three-Layer Hybrid Signal Architecture:
  L1 (50%) — Quantitative Engine
  L2 (30%) — Sentiment Layer
  L3 (20% + VETO) — Risk Engine

Usage:
  python -m src.main             # Paper mode (live Binance CCXT real-time data)
  python -m src.main --retrain   # Run auto-retrain loop for all assets
  python -m src.main --dashboard # Launch real-time monitoring dashboard
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


def run_auto_retrain(config: dict):
    """Run auto-retrain loop for all configured assets."""
    from src.models.auto_retrain import main as retrain_main
    import subprocess

    assets = config.get('assets', ['BTC', 'ETH', 'AAVE'])
    for asset in assets:
        symbol = f"{asset}/USDT"
        model_out = f"models/lgbm_{asset.lower()}_optimized.txt"
        print(f"Running auto-retrain for {symbol}...")
        try:
            # Run the retrain script as subprocess
            result = subprocess.run([
                sys.executable, '-m', 'src.models.auto_retrain',
                '--symbol', symbol,
                '--model-out', model_out,
                '--n-trials', '50'
            ], capture_output=True, text=True, timeout=600)  # 10 min timeout

            if result.returncode == 0:
                print(f"Auto-retrain completed for {symbol}")
            else:
                print(f"Auto-retrain failed for {symbol}: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"Auto-retrain timed out for {symbol}")
        except Exception as e:
            print(f"Error running auto-retrain for {symbol}: {e}")


def run_dashboard():
    """Launch the real-time monitoring dashboard."""
    try:
        from src.dashboard_server import app
        print("🚀 Starting Autonomous Trading Desk Dashboard...")
        print("📊 Dashboard available at: http://localhost:5000")
        print("💡 Press Ctrl+C to stop the dashboard server")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"❌ Dashboard dependencies not installed: {e}")
        print("💡 Install with: pip install flask")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")


def main():
    """Main entry point -- enforces real-time data only."""
    config = load_config()

    # CLI overrides (transformer is optional)
    if '--transformer' in sys.argv:
        config.setdefault('ai', {})['use_transformer'] = True

    # CLI mode override
    if '--mode' in sys.argv:
        mode_idx = sys.argv.index('--mode')
        if mode_idx + 1 < len(sys.argv):
            config['mode'] = sys.argv[mode_idx + 1]

    # Auto-retrain mode
    if '--retrain' in sys.argv:
        print("Starting auto-retrain loop...")
        run_auto_retrain(config)
        return

    # Dashboard mode (Launcher)
    if '--dashboard' in sys.argv:
        print("[DASHBOARD] Launching Strategist Hub Dashboard...")
        import subprocess
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "src/api/dashboard_app.py", "--server.port", "8501"])
        print("Dashboard will be available at http://localhost:8501")

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
