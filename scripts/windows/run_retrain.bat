@echo off
REM ACT's Trading System — Continuous ML Retraining
REM Runs FreqAI retrain every 4 hours

cd /d C:\Users\convo\trade
echo ============================================================
echo   ACT's ML Auto-Retrain — Every 4 Hours
echo   Press Ctrl+C to stop
echo ============================================================

python -m src.scripts.freqai_retrain --continuous
