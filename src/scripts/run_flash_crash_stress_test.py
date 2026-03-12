"""Run a deterministic flash-crash drawdown stress test against Risk Fortress.

Usage:
    python -m src.scripts.run_flash_crash_stress_test
"""

from __future__ import annotations

import os
from datetime import datetime, UTC

from src.risk.dynamic_manager import DynamicRiskManager
from src.trading.backtest import BacktestConfig, run_backtest


REPORT_PATH = "logs/flash_crash_stress_test.txt"


def run_dynamic_manager_check() -> dict:
    manager = DynamicRiskManager(initial_capital=100_000.0)
    manager.current_capital = 90_000.0
    manager.peak_capital = 100_000.0

    allowed, reason = manager.check_trade_allowed(
        asset="BTC",
        proposed_size_pct=0.005,
        current_portfolio_heat=0.0,
    )

    return {
        "allowed": allowed,
        "shutdown": manager.is_shutdown,
        "reason": reason,
        "drawdown_pct": manager.drawdown * 100,
        "threshold_pct": manager.risk_limits.max_drawdown_limit_pct * 100,
    }


def run_backtest_flash_crash() -> dict:
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
            use_profit_gate=False,
            compound=True,
            flash_crash_bar=56,
            flash_crash_drawdown_pct=20.0,
            kill_switch_drawdown_pct=10.0,
        ),
    )

    return {
        "kill_switch_triggered": result.kill_switch_triggered,
        "kill_switch_bar": result.kill_switch_bar,
        "kill_switch_reason": result.kill_switch_reason,
        "last_exit_reason": result.trades[-1].exit_reason if result.trades else "NONE",
        "max_drawdown_pct": result.max_drawdown_pct,
        "total_trades": result.total_trades,
    }


def build_report(dynamic_check: dict, backtest_check: dict) -> str:
    passed = (
        dynamic_check["allowed"] is False
        and dynamic_check["shutdown"] is True
        and backtest_check["kill_switch_triggered"] is True
        and backtest_check["last_exit_reason"] == "kill_switch"
    )

    lines = [
        "=" * 72,
        "FLASH CRASH STRESS TEST REPORT",
        "=" * 72,
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        f"Overall Result: {'PASS' if passed else 'FAIL'}",
        "",
        "Dynamic Risk Manager Check",
        f"  Allowed New Trade: {dynamic_check['allowed']}",
        f"  Shutdown Active:   {dynamic_check['shutdown']}",
        f"  Drawdown:          {dynamic_check['drawdown_pct']:.2f}%",
        f"  Threshold:         {dynamic_check['threshold_pct']:.2f}%",
        f"  Reason:            {dynamic_check['reason']}",
        "",
        "Backtest Flash Crash Check",
        f"  Kill Switch:       {backtest_check['kill_switch_triggered']}",
        f"  Trigger Bar:       {backtest_check['kill_switch_bar']}",
        f"  Max Drawdown:      {backtest_check['max_drawdown_pct']:.2f}%",
        f"  Exit Reason:       {backtest_check['last_exit_reason']}",
        f"  Total Trades:      {backtest_check['total_trades']}",
        f"  Reason:            {backtest_check['kill_switch_reason']}",
        "",
        "Conclusion",
        "  Risk Fortress kill switches trip before mainnet in the deterministic -20% crash harness."
        if passed
        else "  One or more kill-switch checks failed. Do not proceed to mainnet.",
        "=" * 72,
    ]
    return "\n".join(lines)


def main() -> None:
    dynamic_check = run_dynamic_manager_check()
    backtest_check = run_backtest_flash_crash()
    report = build_report(dynamic_check, backtest_check)

    os.makedirs(os.path.dirname(REPORT_PATH) or ".", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(report)
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
