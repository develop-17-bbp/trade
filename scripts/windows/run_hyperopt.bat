@echo off
REM ACT's Trading System — Continuous Hyperopt
REM Runs parameter optimization every 4 hours

cd /d C:\Users\convo\trade
echo ============================================================
echo   ACT's Auto Hyperopt — Every 4 Hours
echo   Press Ctrl+C to stop
echo ============================================================

python -m src.scripts.auto_hyperopt --continuous --interval 4
