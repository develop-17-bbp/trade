@echo off
REM ============================================================
REM   ACT's TRADING SYSTEM — MASTER STARTUP
REM   Launches ALL components in separate windows
REM   MUST be run from C:\Users\convo\trade (main repo, NOT worktree)
REM ============================================================

echo.
echo   ACT's AI TRADING SYSTEM — Starting All Components
echo   ==================================================
echo.

REM Force correct directory (main repo, NOT worktree)
cd /d C:\Users\convo\trade
echo   Working directory: %CD%
echo.

REM Set environment variables BEFORE launching anything
set TRADE_API_DEV_MODE=1
set PYTHONUNBUFFERED=1
set PATH=C:\Program Files\nodejs;%PATH%

echo [1/5] Starting Trading Bot...
start "ACTs - Trading Bot" cmd /k "cd /d C:\Users\convo\trade && set PYTHONUNBUFFERED=1 && python -m src.main"
timeout /t 8 /nobreak >nul

echo [2/5] Starting API Server (port 11007)...
start "ACTs - API Server" cmd /k "cd /d C:\Users\convo\trade && set TRADE_API_DEV_MODE=1 && python -m uvicorn src.api.production_server:app --host 0.0.0.0 --port 11007"
timeout /t 4 /nobreak >nul

echo [3/5] Starting Dashboard (port 5173)...
start "ACTs - Dashboard" cmd /k "cd /d C:\Users\convo\trade\frontend && set PATH=C:\Program Files\nodejs;%%PATH%% && node node_modules\vite\bin\vite.js --host"
timeout /t 4 /nobreak >nul

echo [4/5] Starting ML Auto-Retrain (every 4h)...
start "ACTs - FreqAI Retrain" cmd /k "cd /d C:\Users\convo\trade && python -m src.scripts.freqai_retrain --continuous"
timeout /t 2 /nobreak >nul

echo [5/5] Starting Monitor (every 30min)...
start "ACTs - Monitor" cmd /k "cd /d C:\Users\convo\trade && scripts\windows\run_monitor.bat"

echo.
echo   ==================================================
echo   ALL 5 COMPONENTS STARTED
echo   ==================================================
echo.
echo   1. Trading Bot:   New window "ACTs - Trading Bot"
echo   2. API Server:    http://localhost:11007 (dev mode ON)
echo   3. Dashboard:     http://localhost:5173
echo   4. ML Retrain:    Every 4 hours (continuous)
echo   5. Monitor:       Every 30 minutes (backtest + health)
echo.
echo   Open http://localhost:5173 in your browser
echo.
echo   Press any key to close this launcher window...
pause >nul
