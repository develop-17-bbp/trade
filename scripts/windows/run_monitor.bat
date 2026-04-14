@echo off
REM ACT's Trading System — 24/7 Monitor Loop
REM Runs independently of Claude Code session
REM Start this in a separate PowerShell/CMD window

cd /d C:\Users\convo\trade
echo ============================================================
echo   ACT's Trading Monitor — Runs Every 10 Minutes
echo   Press Ctrl+C to stop
echo ============================================================

:loop
echo.
echo [%date% %time%] Running monitor cycle...

REM 1. Check bot health via paper state freshness (updated every bar)
python -c "import os, time; f='logs/robinhood_paper_state.json'; age=time.time()-os.path.getmtime(f) if os.path.exists(f) else 9999; print(f'BOT: {\"RUNNING (last update {age:.0f}s ago)\" if age < 120 else \"STALE/STOPPED (no update in {age:.0f}s)\"}')" 2>nul

REM 2. Check API server health
python -c "import urllib.request; r=urllib.request.urlopen('http://localhost:11007/health',timeout=5); print(f'API: OK ({r.read().decode()[:50]})')" 2>nul || echo API: OFFLINE

REM 3. Check dashboard
python -c "import urllib.request; r=urllib.request.urlopen('http://localhost:5173',timeout=5); print('DASHBOARD: OK')" 2>nul || echo DASHBOARD: OFFLINE

REM 4. Check paper trading state
python -c "import json; d=json.load(open('logs/robinhood_paper_state.json')); s=d.get('stats',{}); print(f'PAPER: ${d.get(\"equity\",0):,.2f} | {s.get(\"wins\",0)}W/{s.get(\"losses\",0)}L | PnL: ${s.get(\"total_pnl_usd\",0):+,.2f}')" 2>nul

REM 5. Run strategy backtest (every other cycle — heavy operation)
set /a "cycle_count+=1"
set /a "run_backtest=cycle_count %% 2"
if %run_backtest%==0 (
    echo [%date% %time%] Running strategy backtest...
    python -m src.scripts.strategy_backtester --days 14 --asset BTC 2>nul | findstr /C:"#1" /C:"#2" /C:"#3" /C:"Recommended"
)

REM 6. Quick health dashboard (every 3rd cycle)
set /a "run_health=cycle_count %% 3"
if %run_health%==0 (
    echo [%date% %time%] Running health dashboard...
    python -m src.scripts.health_dashboard --aggregate 2>nul | findstr /C:"OVERALL" /C:"RED" /C:"YELLOW" /C:"GREEN"
)

echo [%date% %time%] Cycle %cycle_count% complete. Sleeping 10 minutes...
timeout /t 600 /nobreak >nul
goto loop
