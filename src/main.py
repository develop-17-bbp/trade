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

# Cross-platform: avoid OpenMP/BLAS threading issues (multiple runtimes on Mac; stability on Windows/Linux).
# Set before any native lib (numpy, scipy, lightgbm, torch) loads.
if os.environ.get("OMP_NUM_THREADS") is None:
    os.environ["OMP_NUM_THREADS"] = "1"
if os.environ.get("OPENBLAS_NUM_THREADS") is None:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
if os.environ.get("MKL_NUM_THREADS") is None:
    os.environ["MKL_NUM_THREADS"] = "1"
import signal
import yaml
import logging

try:
    from src.core.paths import ensure_dirs
    from src.core.logging_config import configure_logging
    ensure_dirs()
    configure_logging(level="INFO")
except Exception as _log_cfg_err:
    # Logging config is non-critical — fall back to basicConfig
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger(__name__).warning(f"Logging config failed, using defaults: {_log_cfg_err}")

# Force UTF-8 stdout/stderr on Windows (prevents UnicodeEncodeError from emojis in logs)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    # Also fix the root logging handler so logging output is UTF-8
    for handler in logging.root.handlers:
        if hasattr(handler, 'stream'):
            handler.stream = sys.stderr

from src.trading.executor import TradingExecutor

logger = logging.getLogger(__name__)


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
        host = os.environ.get('DASHBOARD_HOST', '127.0.0.1')
        app.run(debug=False, host=host, port=5000)
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

    # Production API server (--api flag starts REST API on port 11000 in background thread)
    # Usage: python -m src.main --api
    #        python -m src.main --api --api-port 11000
    # Future: Port 11001 reserved for WebSocket streaming
    if '--api' in sys.argv:
        _api_port = 11000
        if '--api-port' in sys.argv:
            _port_idx = sys.argv.index('--api-port')
            if _port_idx + 1 < len(sys.argv):
                try:
                    _api_port = int(sys.argv[_port_idx + 1])
                except ValueError:
                    pass
        try:
            import threading
            from src.api.production_server import run_production_server
            api_thread = threading.Thread(
                target=run_production_server,
                kwargs={"host": "0.0.0.0", "port": _api_port},
                daemon=True, name="production-api"
            )
            api_thread.start()
            logger.info(f"Production API started on port {_api_port}")
        except Exception as _api_err:
            logger.warning(f"[API] Failed to start production API: {_api_err}")

    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    executor = TradingExecutor(config)

    # ── Graceful Shutdown Handler (cross-platform: Windows has SIGINT but not SIGTERM) ──
    def _shutdown_handler(signum, frame):
        try:
            sig_name = signal.Signals(signum).name
        except (ValueError, AttributeError):
            sig_name = str(signum)
        logger.warning(f"[SHUTDOWN] Received {sig_name}. Initiating graceful shutdown...")
        try:
            # Persist state before exit
            from src.persistence.state_store import StateStore
            store = StateStore()

            # Save open positions
            for asset, record in executor.risk_manager.open_positions.items():
                store.save_position(
                    asset=record.asset, direction=record.direction,
                    entry_price=record.entry_price, size=record.size,
                    stop_loss=record.stop_loss, take_profit=record.take_profit,
                    order_id=getattr(record, 'order_id', '')
                )

            # Save risk state
            store.save_risk_state(
                daily_pnl=executor.risk_manager.daily_pnl,
                weekly_pnl=executor.risk_manager.weekly_pnl,
                monthly_pnl=executor.risk_manager.monthly_pnl,
                peak_equity=executor.risk_manager.peak_capital,
                current_equity=executor.risk_manager.current_capital,
                is_shutdown=executor.risk_manager.is_shutdown,
            )

            # Save circuit breakers
            for name, cb in executor.risk_manager.circuit_breakers.items():
                store.save_circuit_breaker(
                    name=name, is_triggered=cb.is_triggered,
                    current_value=cb.current_value, threshold=cb.threshold,
                    last_triggered=cb.last_triggered,
                )

            store.save_system_meta(mode=executor.mode, last_run_time=None)
            store.close()
            logger.info("[SHUTDOWN] State persisted to SQLite. Safe to restart.")
        except Exception as e:
            logger.error(f"[SHUTDOWN] Failed to persist state: {e}")

        try:
            from src.monitoring.alerting import alert_critical
            alert_critical("System Shutdown", f"Trading system stopped via {sig_name}")
        except Exception:
            pass

        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown_handler)

    # ── Model Integrity Check on Startup ──
    try:
        from src.security.model_integrity import verify_all_models
        passed, failed = verify_all_models()
        if failed:
            logger.error(f"[SECURITY] Model integrity check FAILED for: {failed}")
            logger.error("[SECURITY] Run 'python -m src.security.model_integrity --generate' to update checksums")
        else:
            logger.info(f"[SECURITY] All {len(passed)} model files passed integrity check")
    except Exception as e:
        logger.warning(f"[SECURITY] Model integrity check skipped: {e}")

    # ── Restore State from SQLite (Position Reconciliation) ──
    try:
        from src.persistence.state_store import StateStore
        store = StateStore()

        # Restore risk state
        risk_state = store.get_risk_state()
        if risk_state:
            executor.risk_manager.daily_pnl = risk_state.get('daily_pnl', 0.0)
            executor.risk_manager.weekly_pnl = risk_state.get('weekly_pnl', 0.0)
            executor.risk_manager.monthly_pnl = risk_state.get('monthly_pnl', 0.0)
            executor.risk_manager.peak_capital = risk_state.get('peak_equity', executor.initial_capital)
            executor.risk_manager.current_capital = risk_state.get('current_equity', executor.initial_capital)
            if risk_state.get('is_shutdown'):
                executor.risk_manager.is_shutdown = True
                logger.warning("[RECOVERY] System was in SHUTDOWN state. Manual reset required.")

        # Restore circuit breakers
        saved_breakers = store.get_circuit_breakers()
        for cb_data in saved_breakers:
            name = cb_data['name']
            if name in executor.risk_manager.circuit_breakers:
                cb = executor.risk_manager.circuit_breakers[name]
                cb.is_triggered = cb_data['is_triggered']
                cb.current_value = cb_data['current_value']
                cb.last_triggered = cb_data.get('last_triggered')

        # Reconcile positions with exchange
        if executor.price_source.is_authenticated:
            try:
                exchange_balance = executor.price_source.get_balance()
                if 'error' not in exchange_balance:
                    total = exchange_balance.get('total', {})
                    for asset in executor.assets:
                        held = total.get(asset, 0.0)
                        local_has = asset in executor.risk_manager.open_positions
                        if held > 0.0001 and not local_has:
                            logger.warning(f"[RECONCILE] Exchange has {held} {asset} but no local position. Adding to tracking.")
                            price = executor.price_source.fetch_latest_price(f"{asset}/USDT") or 0
                            if price:
                                size_pct = (held * price) / executor.initial_capital
                                executor.risk_manager.register_trade_open(asset, 1, price, size_pct)
                        elif held < 0.0001 and local_has:
                            logger.warning(f"[RECONCILE] Local position for {asset} but exchange shows 0. Removing stale position.")
                            executor.risk_manager.open_positions.pop(asset, None)
            except Exception as e:
                logger.warning(f"[RECONCILE] Exchange reconciliation failed: {e}")

        store.close()
        logger.info("[RECOVERY] State restoration complete")
    except Exception as e:
        logger.warning(f"[RECOVERY] State restoration skipped: {e}")

    # ── Daily Reset Scheduler ──
    try:
        import threading
        from datetime import datetime, timedelta, timezone

        def _schedule_daily_reset():
            """Reset daily P&L counters at 00:00 UTC."""
            while True:
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                sleep_secs = (tomorrow - now).total_seconds()
                import time
                time.sleep(sleep_secs)
                try:
                    executor.risk_manager.reset_daily_pnl()
                    # Also reset the strategy's internal RiskManager so daily_trades counter
                    # and _halted flag don't accumulate across days
                    if hasattr(executor, 'strategy') and hasattr(executor.strategy, 'risk_manager'):
                        executor.strategy.risk_manager.reset_daily()
                    logger.info("[SCHEDULER] Daily P&L reset at 00:00 UTC")
                    # Weekly reset on Mondays
                    if datetime.now(timezone.utc).weekday() == 0:
                        executor.risk_manager.reset_weekly_pnl()
                        logger.info("[SCHEDULER] Weekly P&L reset (Monday)")
                    # Monthly reset on 1st
                    if datetime.now(timezone.utc).day == 1:
                        executor.risk_manager.reset_monthly_pnl()
                        logger.info("[SCHEDULER] Monthly P&L reset (1st)")
                except Exception as e:
                    logger.error(f"[SCHEDULER] Reset failed: {e}")

        reset_thread = threading.Thread(target=_schedule_daily_reset, daemon=True, name="daily-reset")
        reset_thread.start()
        logger.info("[SCHEDULER] Daily reset scheduler started (00:00 UTC)")
    except Exception as e:
        logger.warning(f"[SCHEDULER] Failed to start daily reset: {e}")

    executor.run()


if __name__ == "__main__":
    main()
