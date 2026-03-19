@echo off
REM Daily Trading Report Generator - Run Manual Report
REM This script generates today's trading reports immediately

echo.
echo ======================================================
echo          DAILY TRADING REPORT GENERATOR
echo ======================================================
echo.

REM Change to project directory
cd /d %~dp0..

REM Run the report generator in manual mode
echo Running manual report generation...
python src\reporting\daily_report_scheduler.py manual

echo.
echo ======================================================
if %ERRORLEVEL% EQU 0 (
    echo         REPORTS GENERATED SUCCESSFULLY
) else (
    echo         ERROR GENERATING REPORTS
)
echo ======================================================
echo.

pause
