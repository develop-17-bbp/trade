@echo off
REM ===============================================================
REM  ACT Trading System -- Full Self-Evolving Startup (RTX 5090)
REM  7 Windows: API + Bot + Adapt + Autonomous + Genetic + Frontend + Tunnel
REM ===============================================================
chcp 65001 >nul
setlocal enabledelayedexpansion

REM -- Colors --
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "CYAN=[96m"
set "WHITE=[97m"
set "RESET=[0m"

cls

REM -- Animated ASCII Art --
echo.
echo %RED%     AAA   %RESET%%WHITE%CCCCC%RESET%%GREEN%  TTTTT%RESET%
timeout /t 0 /nobreak >nul
echo %RED%   A   A %RESET%%WHITE%C      %RESET%%GREEN%    T  %RESET%
timeout /t 0 /nobreak >nul
echo %RED%   AAAAA %RESET%%WHITE%C      %RESET%%GREEN%    T  %RESET%     %CYAN%ACT TRADING SYSTEM%RESET%
timeout /t 0 /nobreak >nul
echo %RED%   A   A %RESET%%WHITE%C      %RESET%%GREEN%    T  %RESET%     %YELLOW%RTX 5090 -- FULL STARTUP%RESET%
timeout /t 0 /nobreak >nul
echo %RED%   A   A %RESET%%WHITE%CCCCC%RESET%%GREEN%  TTTTT%RESET%
echo.
echo %CYAN%============================================================%RESET%
echo.

REM -- Step 0: Verify Ollama --
echo %YELLOW%[CHECK]%RESET% Verifying Ollama on localhost:11434...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %YELLOW%[WARN]%RESET% Ollama not detected! Starting Ollama...
    start "" /MIN ollama serve
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo %RED%[ERROR]%RESET% Ollama failed to start.
        pause
        exit /b 1
    )
)
echo %GREEN%[OK]%RESET% Ollama running.

REM -- Step 0b: Verify models --
echo %YELLOW%[CHECK]%RESET% Checking Ollama models...
for %%m in (mistral:latest llama3.2:latest) do (
    ollama list 2>nul | findstr /i "%%m" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo %YELLOW%[PULL]%RESET% Downloading %%m ...
        ollama pull %%m
    )
)
echo %GREEN%[OK]%RESET% Models ready: mistral + llama3.2

REM -- Step 0c: Load .env --
echo %YELLOW%[CHECK]%RESET% Loading .env credentials...
if not exist "%~dp0.env" (
    echo %RED%[ERROR]%RESET% .env file not found at %~dp0.env
    pause
    exit /b 1
)
REM Parse .env file safely
for /f "usebackq eol=# tokens=1,2 delims==" %%A in ("%~dp0.env") do (
    set "%%A=%%B"
)
echo %GREEN%[OK]%RESET% .env loaded.

cd /d %~dp0
set PYTHONUNBUFFERED=1
set PYTHONPATH=%~dp0

REM -- Kill existing ACT processes --
echo.
echo %YELLOW%[CLEANUP]%RESET% Stopping existing ACT processes...
taskkill /FI "WINDOWTITLE eq ACT*" /F >nul 2>&1
timeout /t 2 /nobreak >nul
echo %GREEN%[OK]%RESET% Cleanup done.

echo.
echo %CYAN%============================================================%RESET%
echo %CYAN%  LAUNCHING ALL 7 SYSTEMS...%RESET%
echo %CYAN%============================================================%RESET%
echo.

REM -- Window 1: API Server --
echo %YELLOW%[1/7]%RESET% Starting %CYAN%API Server%RESET% (port 11007)...
start "ACT - API Server" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.api.production_server 2>&1 | tee logs/api_output.log"
timeout /t 3 /nobreak >nul
echo %GREEN%[OK]%RESET% API Server started on :11007

REM -- Window 2: Trading Bot --
echo %YELLOW%[2/7]%RESET% Starting %RED%Trading Bot%RESET%...
start "ACT - Trading Bot" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.main 2>&1 | tee logs/main_output.log"
timeout /t 3 /nobreak >nul
echo %GREEN%[OK]%RESET% Trading Bot started.

REM -- Window 3: Adaptation Loop --
echo %YELLOW%[3/7]%RESET% Starting %CYAN%Adaptation Loop%RESET% (every 1h)...
start "ACT - Adaptation Loop" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.scripts.continuous_adapt --continuous --interval 1 2>&1 | tee logs/adapt_output.log"
timeout /t 2 /nobreak >nul
echo %GREEN%[OK]%RESET% Adaptation Loop started.

REM -- Window 4: Autonomous Loop --
echo %YELLOW%[4/7]%RESET% Starting %CYAN%Autonomous Loop%RESET% (every 30min)...
start "ACT - Autonomous Loop" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.scripts.autonomous_loop --interval 0.5 2>&1 | tee logs/autonomous_loop.log"
timeout /t 2 /nobreak >nul
echo %GREEN%[OK]%RESET% Autonomous Loop started.

REM -- Window 5: Genetic Evolution --
echo %YELLOW%[5/7]%RESET% Starting %GREEN%Genetic Loop%RESET% (pop=100, every 2h)...
start "ACT - Genetic Loop" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.scripts.genetic_loop --population_size 100 --interval 2 2>&1 | tee logs/genetic_loop.log"
timeout /t 2 /nobreak >nul
echo %GREEN%[OK]%RESET% Genetic Loop started.

REM -- Window 6: Frontend --
echo %YELLOW%[6/7]%RESET% Starting %CYAN%Frontend%RESET% (port 5173)...
start "ACT - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
timeout /t 4 /nobreak >nul
echo %GREEN%[OK]%RESET% Frontend started on :5173

REM -- Window 7: Cloudflare Tunnel --
echo %YELLOW%[7/7]%RESET% Starting %CYAN%Cloudflare Tunnel%RESET%...
start "ACT - Tunnel" cmd /k "cloudflared tunnel --url http://localhost:5173"
timeout /t 3 /nobreak >nul
echo %GREEN%[OK]%RESET% Tunnel started.

REM ============================================================
echo.
echo %CYAN%============================================================%RESET%
echo.
echo %GREEN%   ALL 7 SYSTEMS RUNNING  ^(RTX 5090 + CUDA^)%RESET%
echo.
echo %WHITE%   1%RESET%  %CYAN%API Server%RESET%    http://localhost:11007
echo %WHITE%   2%RESET%  %RED%Trading Bot%RESET%   Robinhood Paper ^(BTC/ETH^)
echo %WHITE%   3%RESET%  %CYAN%Adapt Loop%RESET%    Every 1h  -- retrain + fine-tune
echo %WHITE%   4%RESET%  %CYAN%Auto Loop%RESET%     Every 30m -- self-heal + monitor
echo %WHITE%   5%RESET%  %GREEN%Genetic Loop%RESET%  Every 2h  -- evolve 100 DNA strategies
echo %WHITE%   6%RESET%  %CYAN%Frontend%RESET%      http://localhost:5173
echo %WHITE%   7%RESET%  %CYAN%CF Tunnel%RESET%     Check ACT-Tunnel window for public URL
echo.
echo %YELLOW%   Ollama: localhost:11434  CUDA: RTX 5090%RESET%
echo %YELLOW%   Models: mistral + llama3.2 + neural-chat%RESET%
echo.
echo %GREEN%   Open http://localhost:5173 to view dashboard%RESET%
echo %GREEN%   Check ACT-Tunnel window for remote access URL%RESET%
echo.
echo %CYAN%============================================================%RESET%
pause
