@echo off
REM ===============================================================
REM  ACT Trading System -- Full Self-Evolving Startup (RTX 5090)
REM  7 Windows: API + Bot + Adapt + Autonomous + Genetic + Frontend + Tunnel
REM ===============================================================
chcp 65001 >nul

echo.
echo ============================================================
echo.
echo    AAA   CCCCC  TTTTT
echo   A   A C       T
echo   AAAAA C       T      ACT TRADING SYSTEM
echo   A   A C       T      RTX 5090 -- FULL STARTUP
echo   A   A  CCCCC  T
echo.
echo ============================================================
echo.

REM -- Step 0: Verify Ollama --
echo [CHECK] Verifying Ollama on localhost:11434...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Ollama not detected! Starting Ollama...
    start "" /MIN ollama serve
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Ollama failed to start.
        pause
        exit /b 1
    )
)
echo [OK] Ollama running.

REM -- Step 0b: Verify models --
echo [CHECK] Checking Ollama models...
for %%m in (mistral:latest llama3.2:latest) do (
    ollama list 2>nul | findstr /i "%%m" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [PULL] Downloading %%m ...
        ollama pull %%m
    )
)
echo [OK] Models ready: mistral + llama3.2

REM -- Step 0c: Load .env --
echo [CHECK] Loading .env credentials...
if not exist "%~dp0.env" (
    echo [ERROR] .env file not found at %~dp0.env
    echo [ERROR] Create it with ROBINHOOD_API_KEY and ROBINHOOD_PRIVATE_KEY
    pause
    exit /b 1
)
echo [OK] .env found.

cd /d %~dp0
set PYTHONUNBUFFERED=1
set PYTHONPATH=%~dp0

REM -- Load .env into environment --
for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env") do (
    if not "%%A"=="" if not "%%A:~0,1%"=="#" set "%%A=%%B"
)

REM -- Kill existing ACT processes --
echo.
echo [CLEANUP] Stopping existing ACT processes...
taskkill /FI "WINDOWTITLE eq ACT*" /F >nul 2>&1
timeout /t 2 /nobreak >nul
echo [OK] Cleanup done.

REM ============================================================
REM  START ALL 7 WINDOWS
REM ============================================================

REM -- Window 1: API Server --
echo.
echo [1/7] Starting API Server (port 11007)...
start "ACT - API Server" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.api.production_server 2>&1 | tee logs/api_output.log"
timeout /t 3 /nobreak >nul
echo [OK] API Server started on :11007

REM -- Window 2: Trading Bot --
echo [2/7] Starting Trading Bot...
start "ACT - Trading Bot" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.main 2>&1 | tee logs/main_output.log"
timeout /t 3 /nobreak >nul
echo [OK] Trading Bot started.

REM -- Window 3: Adaptation Loop --
echo [3/7] Starting Adaptation Loop (every 1h)...
start "ACT - Adaptation Loop" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.scripts.continuous_adapt --continuous --interval 1 2>&1 | tee logs/adapt_output.log"
timeout /t 2 /nobreak >nul
echo [OK] Adaptation Loop started.

REM -- Window 4: Autonomous Loop --
echo [4/7] Starting Autonomous Loop (every 30min)...
start "ACT - Autonomous Loop" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.scripts.autonomous_loop --interval 0.5 2>&1 | tee logs/autonomous_loop.log"
timeout /t 2 /nobreak >nul
echo [OK] Autonomous Loop started.

REM -- Window 5: Genetic Evolution --
echo [5/7] Starting Genetic Loop (pop=100, every 2h)...
start "ACT - Genetic Loop" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.scripts.genetic_loop --population_size 100 --interval 2 2>&1 | tee logs/genetic_loop.log"
timeout /t 2 /nobreak >nul
echo [OK] Genetic Loop started.

REM -- Window 6: Frontend --
echo [6/7] Starting Frontend (port 5173)...
start "ACT - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
timeout /t 4 /nobreak >nul
echo [OK] Frontend started on :5173

REM -- Window 7: Cloudflare Tunnel --
echo [7/7] Starting Cloudflare Tunnel...
start "ACT - Tunnel" cmd /k "cloudflared tunnel --url http://localhost:5173"
timeout /t 3 /nobreak >nul
echo [OK] Tunnel started (check ACT - Tunnel window for public URL).

REM ============================================================
echo.
echo ============================================================
echo.
echo    ALL 7 SYSTEMS RUNNING  (RTX 5090 + CUDA)
echo.
echo    1  API Server    http://localhost:11007
echo    2  Trading Bot   Robinhood Paper (BTC/ETH)
echo    3  Adapt Loop    Every 1h  -- retrain + fine-tune
echo    4  Auto Loop     Every 30m -- self-heal + monitor
echo    5  Genetic Loop  Every 2h  -- evolve 100 DNA strategies
echo    6  Frontend      http://localhost:5173
echo    7  CF Tunnel     Check ACT-Tunnel window for public URL
echo.
echo    Ollama: localhost:11434  CUDA: RTX 5090
echo    Models: mistral + llama3.2 + neural-chat
echo.
echo    Logs:
echo      logs/main_output.log    -- Trading decisions
echo      logs/adapt_output.log   -- Adaptation cycles
echo      logs/autonomous_loop.log -- Self-improvement
echo      logs/genetic_loop.log   -- Genetic evolution
echo      logs/api_output.log     -- API server
echo.
echo    Check ACT-Tunnel window for remote access URL
echo    Open http://localhost:5173 to view dashboard
echo.
echo ============================================================
pause
