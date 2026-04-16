@echo off
REM ===============================================================
REM  ACT Trading System -- Full Self-Evolving Startup (RTX 5090)
REM  7 Windows: API + Bot + Adapt + Autonomous + Genetic + Frontend + Tunnel
REM ===============================================================
chcp 65001 >nul
setlocal enabledelayedexpansion

REM -- Enable ANSI colors in Windows Terminal --
reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul 2>&1

cls
echo.
echo [91m     AAA   [0m[97mCCCCC[0m[92m  TTTTT[0m
echo [91m   A   A [0m[97mC      [0m[92m    T  [0m
echo [91m   AAAAA [0m[97mC      [0m[92m    T  [0m     [96mACT TRADING SYSTEM[0m
echo [91m   A   A [0m[97mC      [0m[92m    T  [0m     [93mRTX 5090 -- FULL STARTUP[0m
echo [91m   A   A [0m[97mCCCCC[0m[92m  TTTTT[0m
echo.
echo [96m============================================================[0m
echo.

REM -- Step 0: Verify Ollama --
echo [93m[CHECK][0m Verifying Ollama on localhost:11434...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [93m[WARN][0m Ollama not detected! Starting Ollama...
    start "" /MIN ollama serve
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [91m[ERROR][0m Ollama failed to start.
        pause
        exit /b 1
    )
)
echo [92m[OK][0m Ollama running.

REM -- Step 0b: Verify models --
echo [93m[CHECK][0m Checking Ollama models...
for %%m in (mistral:latest llama3.2:latest) do (
    ollama list 2>nul | findstr /i "%%m" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [93m[PULL][0m Downloading %%m ...
        ollama pull %%m
    )
)
echo [92m[OK][0m Models ready: mistral + llama3.2

REM -- Step 0c: Load .env --
echo [93m[CHECK][0m Loading .env credentials...
if not exist "%~dp0.env" (
    echo [91m[ERROR][0m .env file not found at %~dp0.env
    pause
    exit /b 1
)
for /f "usebackq eol=# tokens=1,2 delims==" %%A in ("%~dp0.env") do (
    set "%%A=%%B"
)
echo [92m[OK][0m .env loaded.

REM -- Install cloudflared if missing --
where cloudflared >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [93m[INSTALL][0m cloudflared not found. Downloading...
    curl -L -o "%~dp0cloudflared.exe" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
    copy "%~dp0cloudflared.exe" "C:\Windows\System32\cloudflared.exe" >nul 2>&1
    echo [92m[OK][0m cloudflared installed.
)

cd /d %~dp0
set PYTHONUNBUFFERED=1
set PYTHONPATH=%~dp0

REM -- Kill existing ACT processes --
echo.
echo [93m[CLEANUP][0m Stopping existing ACT processes...
taskkill /FI "WINDOWTITLE eq ACT*" /F >nul 2>&1
timeout /t 2 /nobreak >nul
echo [92m[OK][0m Cleanup done.

echo.
echo [96m============================================================[0m
echo [96m  LAUNCHING ALL 7 SYSTEMS...[0m
echo [96m============================================================[0m
echo.

REM -- Window 1: API Server --
echo [93m[1/7][0m Starting [96mAPI Server[0m (port 11007)...
start "ACT - API Server" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.api.production_server > logs\api_output.log 2>&1"
timeout /t 3 /nobreak >nul
echo [92m[OK][0m API Server started on :11007

REM -- Window 2: Trading Bot --
echo [93m[2/7][0m Starting [91mTrading Bot[0m...
start "ACT - Trading Bot" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.main > logs\main_output.log 2>&1"
timeout /t 3 /nobreak >nul
echo [92m[OK][0m Trading Bot started.

REM -- Window 3: Adaptation Loop --
echo [93m[3/7][0m Starting [96mAdaptation Loop[0m (every 1h)...
start "ACT - Adaptation Loop" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.scripts.continuous_adapt --continuous --interval 1 > logs\adapt_output.log 2>&1"
timeout /t 2 /nobreak >nul
echo [92m[OK][0m Adaptation Loop started.

REM -- Window 4: Autonomous Loop --
echo [93m[4/7][0m Starting [96mAutonomous Loop[0m (every 30min)...
start "ACT - Autonomous Loop" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.scripts.autonomous_loop --interval 0.5 > logs\autonomous_loop.log 2>&1"
timeout /t 2 /nobreak >nul
echo [92m[OK][0m Autonomous Loop started.

REM -- Window 5: Genetic Evolution --
echo [93m[5/7][0m Starting [92mGenetic Loop[0m (pop=100, every 2h)...
start "ACT - Genetic Loop" cmd /k "cd /d %~dp0 && set PYTHONUNBUFFERED=1 && python -m src.scripts.genetic_loop --population_size 100 --interval 2 > logs\genetic_loop.log 2>&1"
timeout /t 2 /nobreak >nul
echo [92m[OK][0m Genetic Loop started.

REM -- Window 6: Frontend --
echo [93m[6/7][0m Starting [96mFrontend[0m (port 5173)...
start "ACT - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
timeout /t 4 /nobreak >nul
echo [92m[OK][0m Frontend started on :5173

REM -- Window 7: Cloudflare Tunnel --
echo [93m[7/7][0m Starting [96mCloudflare Tunnel[0m...
start "ACT - Tunnel" cmd /k "cloudflared tunnel --url http://localhost:5173"
timeout /t 3 /nobreak >nul
echo [92m[OK][0m Tunnel started.

echo.
echo [96m============================================================[0m
echo.
echo [92m   ALL 7 SYSTEMS RUNNING  (RTX 5090 + CUDA)[0m
echo.
echo    [97m1[0m  [96mAPI Server[0m     http://localhost:11007
echo    [97m2[0m  [91mTrading Bot[0m    Robinhood Paper (BTC/ETH)
echo    [97m3[0m  [96mAdapt Loop[0m     Every 1h  -- retrain + fine-tune
echo    [97m4[0m  [96mAuto Loop[0m      Every 30m -- self-heal + monitor
echo    [97m5[0m  [92mGenetic Loop[0m   Every 2h  -- evolve 100 DNA strategies
echo    [97m6[0m  [96mFrontend[0m       http://localhost:5173
echo    [97m7[0m  [96mCF Tunnel[0m      Check ACT-Tunnel window for URL
echo.
echo    [93mOllama: localhost:11434  CUDA: RTX 5090[0m
echo    [93mModels: mistral + llama3.2 + neural-chat[0m
echo.
echo    [92mOpen http://localhost:5173 to view dashboard[0m
echo    [92mCheck ACT-Tunnel window for remote access URL[0m
echo.
echo [96m============================================================[0m
pause
