from src.risk.dynamic_manager import DynamicRiskManager
from src.trading.backtest import BacktestConfig, run_backtest


def test_dynamic_risk_manager_triggers_shutdown_at_exact_drawdown_limit():
    manager = DynamicRiskManager(initial_capital=100_000.0)
    manager.current_capital = 90_000.0
    manager.peak_capital = 100_000.0

    allowed, reason = manager.check_trade_allowed(
        asset="BTC",
        proposed_size_pct=0.005,
        current_portfolio_heat=0.0,
    )

    assert allowed is False
    assert manager.is_shutdown is True
    assert "Drawdown 10.00%" in reason


def test_backtest_flash_crash_triggers_kill_switch_and_closes_position():
    prices = [100.0] * 60
    highs = [100.5] * 60
    lows = [99.5] * 60
    signals = [0] * 60
    signals[55] = 1

    result = run_backtest(
        prices=prices,
        signals=signals,
        highs=highs,
        lows=lows,
        config=BacktestConfig(
            initial_capital=100_000.0,
            risk_per_trade_pct=10.0,
            max_position_pct=100.0,
            slippage_pct=0.0,
            fee_pct=0.0,
            use_stops=False,
            compound=True,
            use_profit_gate=False,
            flash_crash_bar=56,
            flash_crash_drawdown_pct=20.0,
            kill_switch_drawdown_pct=10.0,
        ),
    )

    assert result.kill_switch_triggered is True
    assert result.kill_switch_bar == 56
    assert result.trades
    assert result.trades[-1].exit_reason == "kill_switch"
    assert "10.00%" in result.kill_switch_reason or "20.00%" in result.kill_switch_reason
