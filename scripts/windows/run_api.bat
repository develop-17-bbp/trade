@echo off
REM ACT's Trading System — API Server for Dashboard
REM Serves the React frontend at port 11007

cd /d C:\Users\convo\trade
set TRADE_API_DEV_MODE=1
echo ============================================================
echo   ACT's API Server — Port 11007
echo   Dashboard at http://localhost:5173
echo ============================================================

python -m uvicorn src.api.production_server:app --host 0.0.0.0 --port 11007
