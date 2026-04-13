@echo off
REM ACT's Trading System — 24/7 Monitor Loop
REM Runs independently of Claude Code session
REM Start this in a separate PowerShell/CMD window

cd /d C:\Users\convo\trade
echo ============================================================
echo   ACT's Trading Monitor — Runs Every 30 Minutes
echo   Press Ctrl+C to stop
echo ============================================================

:loop
echo.
echo [%date% %time%] Running monitor cycle...

REM 1. Check bot health
python -c "import os; lines=open('logs/live_output.log','r',encoding='utf-8',errors='replace').readlines()[-5:]; print('BOT:', 'RUNNING' if any('BAR' in l for l in lines) else 'STALE/STOPPED')" 2>nul

REM 2. Run strategy backtest
echo [%date% %time%] Running strategy backtest...
python -m src.scripts.strategy_backtester --days 14 --asset BTC 2>nul | findstr /C:"#1" /C:"#2" /C:"#3" /C:"Recommended"

REM 3. Check paper trading state
python -c "import json; d=json.load(open('logs/robinhood_paper_state.json')); s=d.get('stats',{}); print(f'PAPER: ${d.get(\"equity\",0):,.2f} | {s.get(\"wins\",0)}W/{s.get(\"losses\",0)}L | PnL: ${s.get(\"total_pnl_usd\",0):+,.2f}')" 2>nul

echo [%date% %time%] Cycle complete. Sleeping 30 minutes...
timeout /t 1800 /nobreak >nul
goto loop
