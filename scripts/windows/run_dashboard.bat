@echo off
REM ACT's Trading System — React Dashboard
REM Starts the frontend dev server

cd /d C:\Users\convo\trade\frontend
echo ============================================================
echo   ACT's Dashboard — http://localhost:5173
echo ============================================================

set PATH=C:\Program Files\nodejs;%PATH%
node node_modules/vite/bin/vite.js --host
