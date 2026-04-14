@echo off
REM ============================================================
REM   ACT's TRADING SYSTEM - BACKGROUND STARTUP
REM   Runs headless — no pause, no animation
REM   Designed for Windows Task Scheduler (runs when locked)
REM ============================================================

REM Force correct directory
cd /d C:\Users\convo\trade

REM Set environment variables
set TRADE_API_DEV_MODE=1
set PYTHONUNBUFFERED=1
set PATH=C:\Program Files\nodejs;%PATH%

REM Log startup
echo [%date% %time%] ACT Background Startup >> logs\scheduler.log

REM Kill any existing instances first
taskkill /FI "WINDOWTITLE eq ACTs - Trading Bot*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq ACTs - API Server*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq ACTs - Dashboard*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq ACTs - Continuous Adapt*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq ACTs - Monitor*" /F >nul 2>&1
timeout /t 2 /nobreak >nul

REM Start all components
echo [%date% %time%] Starting Trading Bot... >> logs\scheduler.log
start "ACTs - Trading Bot" /MIN cmd /k "cd /d C:\Users\convo\trade && set PYTHONUNBUFFERED=1 && python -m src.main"
timeout /t 8 /nobreak >nul

echo [%date% %time%] Starting API Server... >> logs\scheduler.log
start "ACTs - API Server" /MIN cmd /k "cd /d C:\Users\convo\trade && set TRADE_API_DEV_MODE=1 && python -m uvicorn src.api.production_server:app --host 0.0.0.0 --port 11007"
timeout /t 4 /nobreak >nul

echo [%date% %time%] Starting Dashboard... >> logs\scheduler.log
start "ACTs - Dashboard" /MIN cmd /k "cd /d C:\Users\convo\trade\frontend && set PATH=C:\Program Files\nodejs;%%PATH%% && node node_modules\vite\bin\vite.js --host"
timeout /t 4 /nobreak >nul

echo [%date% %time%] Starting Continuous Adapt... >> logs\scheduler.log
start "ACTs - Continuous Adapt" /MIN cmd /k "cd /d C:\Users\convo\trade && python -m src.scripts.continuous_adapt --continuous --interval 0.5"
timeout /t 2 /nobreak >nul

echo [%date% %time%] Starting Monitor... >> logs\scheduler.log
start "ACTs - Monitor" /MIN cmd /k "cd /d C:\Users\convo\trade && scripts\windows\run_monitor.bat"
timeout /t 2 /nobreak >nul

echo [%date% %time%] Starting Autonomous Loop... >> logs\scheduler.log
start "ACTs - Autonomous Loop" /MIN cmd /k "cd /d C:\Users\convo\trade && python -m src.scripts.autonomous_loop --interval 2"
timeout /t 2 /nobreak >nul

echo [%date% %time%] Starting Daily Ops... >> logs\scheduler.log
start "ACTs - Daily Ops" /MIN cmd /k "cd /d C:\Users\convo\trade && python -m src.scripts.daily_ops --continuous"
timeout /t 2 /nobreak >nul

echo [%date% %time%] All 7 components launched (minimized) >> logs\scheduler.log
