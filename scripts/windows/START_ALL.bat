@echo off
REM ============================================================
REM   ACT's TRADING SYSTEM — MASTER STARTUP
REM   Launches ALL components in separate windows
REM ============================================================

echo.
echo   ___   _____ _   _
echo  / _ \ / ____^| ^| ^|  ^
echo ^| ^(_^) ^| ^|    ^| ^|_^| ^|___
echo  ^> _ ^<^| ^|    ^| __^|  / __^|
echo ^| ^(_^) ^| ^|____^| ^|_^| \__ \
echo  \___/ \_____^|\__^|_^|___/
echo.
echo   AI TRADING SYSTEM — Starting All Components
echo   ============================================
echo.

cd /d C:\Users\convo\trade

echo [1/5] Starting Trading Bot...
start "ACTs - Trading Bot" cmd /k "cd /d C:\Users\convo\trade && python -m src.main"
timeout /t 5 /nobreak >nul

echo [2/5] Starting API Server...
start "ACTs - API Server" cmd /k "cd /d C:\Users\convo\trade && set TRADE_API_DEV_MODE=1 && python -m uvicorn src.api.production_server:app --host 0.0.0.0 --port 11007"
timeout /t 3 /nobreak >nul

echo [3/5] Starting Dashboard...
start "ACTs - Dashboard" cmd /k "cd /d C:\Users\convo\trade\frontend && set PATH=C:\Program Files\nodejs;%%PATH%% && node node_modules/vite/bin/vite.js --host"
timeout /t 3 /nobreak >nul

echo [4/5] Starting ML Auto-Retrain...
start "ACTs - FreqAI Retrain" cmd /k "cd /d C:\Users\convo\trade && python -m src.scripts.freqai_retrain --continuous"
timeout /t 2 /nobreak >nul

echo [5/5] Starting Monitor...
start "ACTs - Monitor" cmd /k "cd /d C:\Users\convo\trade && scripts\windows\run_monitor.bat"

echo.
echo   ============================================
echo   ALL COMPONENTS STARTED
echo   ============================================
echo.
echo   Trading Bot:  New CMD window "ACTs - Trading Bot"
echo   API Server:   http://localhost:11007
echo   Dashboard:    http://localhost:5173
echo   ML Retrain:   Running every 4 hours
echo   Monitor:      Running every 30 minutes
echo.
echo   Press any key to close this launcher...
pause >nul
