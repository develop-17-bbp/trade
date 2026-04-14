@echo off
REM ═══════════════════════════════════════════════════════════
REM  ACT Trading System — Full Self-Evolving Startup
REM  Launches ALL components needed for autonomous operation:
REM    1. Trading Bot (executor + LLM brain)
REM    2. Continuous Adaptation Loop (backtest + retrain + fine-tune)
REM    3. Autonomous Improvement Loop (self-healing + monitoring)
REM
REM  Prerequisites on GPU system:
REM    - Ollama running locally with CUDA (ollama serve)
REM    - Python environment with all deps installed
REM    - .env file with Robinhood + API credentials
REM ═════════��═══════════════════════════��═════════════════════

echo.
echo =====================================================
echo   ACT SELF-EVOLVING TRADING SYSTEM — FULL STARTUP
echo =====================================================
echo.

REM ── Step 0: Verify Ollama is running locally ──
echo [CHECK] Verifying Ollama is running on localhost:11434...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Ollama not detected! Starting Ollama...
    start "" /MIN ollama serve
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Ollama failed to start. Install from https://ollama.com
        echo [ERROR] Make sure CUDA drivers are installed for GPU acceleration.
        pause
        exit /b 1
    )
)
echo [OK] Ollama is running.

REM ── Step 0b: Verify required models exist ──
echo [CHECK] Checking Ollama models...
for %%m in (mistral:latest) do (
    ollama list 2>nul | findstr /i "%%m" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [PULL] Downloading %%m ...
        ollama pull %%m
    )
)
echo [OK] Base models ready.

cd /d %~dp0
set PYTHONUNBUFFERED=1
set PYTHONPATH=%~dp0

REM ── Kill any existing bot processes ──
echo.
echo [CLEANUP] Stopping any existing ACT processes...
taskkill /FI "WINDOWTITLE eq ACT*" /F >nul 2>&1
timeout /t 2 /nobreak >nul

REM ── Step 1: Start Trading Bot (main executor) ─���
echo.
echo [1/3] Starting Trading Bot (executor + LLM brain)...
start "ACT - Trading Bot" /MIN cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.main 2>&1 | tee logs/main_output.log"
timeout /t 3 /nobreak >nul
echo [OK] Trading Bot started.

REM ��─ Step 2: Start Continuous Adaptation Loop ──
REM  Runs every 4h: refresh data → backtest → evolve → retrain LGBM → fine-tune LLM → deploy
echo.
echo [2/3] Starting Continuous Adaptation Loop (every 4h)...
start "ACT - Adaptation Loop" /MIN cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.scripts.continuous_adapt --continuous --interval 4 2>&1 | tee logs/adapt_output.log"
timeout /t 2 /nobreak >nul
echo [OK] Adaptation Loop started.

REM ─��� Step 3: Start Autonomous Improvement Loop ���─
REM  Runs every 2h: health check → performance audit → auto-fix → strategy discovery
echo.
echo [3/3] Starting Autonomous Improvement Loop (every 2h)...
start "ACT - Autonomous Loop" /MIN cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.scripts.autonomous_loop --interval 2 2>&1 | tee logs/autonomous_loop.log"
timeout /t 2 /nobreak >nul
echo [OK] Autonomous Loop started.

echo.
echo =====================================================
echo   ALL SYSTEMS RUNNING
echo =====================================================
echo.
echo   Trading Bot:      Trading on Robinhood (BTC, ETH)
echo   Adaptation Loop:  Every 4h: backtest + retrain + fine-tune LLMs
echo   Autonomous Loop:  Every 2h: self-heal + monitor + improve
echo.
echo   Ollama:           localhost:11434 (CUDA GPU)
echo   Models:           act-scanner (Mistral) + act-analyst (Llama)
echo                     Falls back to mistral:latest if not fine-tuned yet
echo.
echo   Logs:
echo     logs/main_output.log       — Trading decisions
echo     logs/adapt_output.log      — Adaptation cycles
echo     logs/autonomous_loop.log   — Self-improvement cycles
echo     logs/finetune_history.jsonl — Fine-tuning results
echo.
echo   Self-Evolution Flow:
echo     Trade → Collect data → Label outcomes → Fine-tune LLMs
echo     → Deploy improved models → Trade better → Repeat
echo.
echo   Press Ctrl+C in any window to stop that component.
echo   Close this window to keep all processes running.
echo =====================================================
pause
