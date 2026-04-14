@echo off
echo.
echo [SHUTDOWN] Stopping all ACT processes...
echo.
taskkill /FI "WINDOWTITLE eq ACT - Trading Bot*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq ACT - Adaptation Loop*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq ACT - Autonomous Loop*" /F >nul 2>&1
echo [OK] All ACT processes stopped.
echo.
pause
