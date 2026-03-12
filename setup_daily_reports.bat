@echo off
REM Setup Daily Trading Report Automation in Windows Task Scheduler
REM Run this script as Administrator to set up automatic daily reports

echo.
echo ======================================================
echo    SETUP: AUTOMATED DAILY TRADING REPORTS
echo ======================================================
echo.

REM Check if running as Administrator
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: This script must be run as Administrator!
    echo.
    echo Please:
    echo   1. Right-click this file
    echo   2. Select "Run as Administrator"
    echo.
    pause
    exit /b 1
)

REM Get the current directory and python executable
set PROJECT_DIR=%~dp0
set PYTHON_EXE=python

echo Setting up automatic daily report generation...
echo.
echo Project Directory: %PROJECT_DIR%
echo.

REM Create the scheduled task
echo Creating Windows Task Scheduler task...
echo.

REM Delete existing task if it exists
schtasks /delete "Trading System - Daily Report" /f 2>nul

REM Create new task - runs daily at 21:00 UTC (9:00 PM)
schtasks /create ^
  /tn "Trading System - Daily Report" ^
  /tr "cd /d %PROJECT_DIR% && %PYTHON_EXE% src\reporting\daily_report_scheduler.py manual" ^
  /sc daily ^
  /st 21:00 ^
  /ru SYSTEM

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================
    echo          SETUP SUCCESSFUL!
    echo ======================================================
    echo.
    echo Daily Report Generation is now scheduled:
    echo   • Task Name: Trading System - Daily Report
    echo   • Schedule: Every day at 21:00 UTC (9:00 PM)
    echo   • Reports Location: logs/daily_reports/
    echo.
    echo Report Files Generated Daily:
    echo   1. report_YYYY-MM-DD.md   (Detailed markdown analysis)
    echo   2. report_YYYY-MM-DD.csv  (Raw data export)
    echo   3. report_YYYY-MM-DD.txt  (Quick summary)
    echo.
    echo To manually generate a report anytime, run:
    echo   generate_daily_report.bat
    echo.
    echo To stop automatic reports, run:
    echo   schtasks /delete "Trading System - Daily Report" /f
    echo.
) else (
    echo.
    echo ERROR: Failed to create task!
    echo Please try again or set up manually.
    echo.
)

echo ======================================================
echo.
pause
