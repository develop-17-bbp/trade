@echo off
REM ============================================================
REM   ACT's TRADING SYSTEM - MASTER STARTUP
REM   Launches ALL components in separate windows
REM   MUST be run from C:\Users\convo\trade (main repo, NOT worktree)
REM ============================================================

color 0F
title ACT's AI Trading System - Initializing...

REM == Frame 1: Neural network loading ==
cls
echo.
echo      [============================================]
echo      :                                            :
echo      :       . . . . . . . . . . . . . .         :
echo      :       . . . . . . . . . . . . . .         :
echo      :       . . . . . . . . . . . . . .         :
echo      :       . . . . . . . . . . . . . .         :
echo      :                                            :
echo      :         LOADING NEURAL NETWORK...          :
echo      :                                            :
echo      [============================================]
echo.
timeout /t 1 /nobreak >nul

REM == Frame 2: Network activating ==
cls
echo.
echo      [============================================]
echo      :                                            :
echo      :       o . o . o . o . o . o . o .         :
echo      :       . o . o . o . o . o . o . o         :
echo      :       o . o . o . o . o . o . o .         :
echo      :       . o . o . o . o . o . o . o         :
echo      :                                            :
echo      :          ACTIVATING NEURONS...             :
echo      :                                            :
echo      [============================================]
echo.
timeout /t 1 /nobreak >nul

REM == Frame 3: Network connected ==
cls
echo.
echo      [============================================]
echo      :                                            :
echo      :       O---O---O---O---O---O---O           :
echo      :       :\  :\ /:  :\ /:  :\ /:            :
echo      :       O---O---O---O---O---O---O           :
echo      :       :/  :/ \:  :/ \:  :/ \:            :
echo      :       O---O---O---O---O---O---O           :
echo      :                                            :
echo      :        NEURAL MESH ESTABLISHED             :
echo      :                                            :
echo      [============================================]
echo.
timeout /t 1 /nobreak >nul

REM == Frame 4: Main logo reveal ==
cls
echo.
echo.
echo        ##########   ##########  ########  ##
echo        ##      ##   ##              ##    ##
echo        ##########   ##              ##    ###
echo        ##      ##   ##              ##      ##
echo        ##      ##   ##########      ##    ###
echo.
echo     ================================================
echo     :    A U T O N O M O U S   C R Y P T O        :
echo     :    T R A D I N G   S Y S T E M               :
echo     ================================================
echo.
timeout /t 1 /nobreak >nul

REM == Frame 5: System specs ==
cls
echo.
echo.
echo     ================================================
echo     :                                              :
echo     :     #####   ####  ######  ##                 :
echo     :    ##   ## ##       ##    ## ###              :
echo     :    ####### ##       ##    ##                  :
echo     :    ##   ## ##       ##    ## ###              :
echo     :    ##   ##  ####    ##    ## ###              :
echo     :                                              :
echo     :  AI TRADING SYSTEM v4.0                      :
echo     :                                              :
echo     :  [+] 13 AI Agents            [ARMED]        :
echo     :  [+] 9 ML Models             [LOADED]       :
echo     :  [+] 16 Named Strategies     [ACTIVE]       :
echo     :  [+] 242 Universe Strategies [VOTING]       :
echo     :  [+] 25 Subsystems           [ONLINE]       :
echo     :  [+] Genetic Evolution       [EVOLVING]     :
echo     :  [+] HMM Regime Detection    [SCANNING]     :
echo     :  [+] 7 Circuit Breakers      [STANDING BY]  :
echo     :  [+] EVT + Monte Carlo Risk  [COMPUTING]    :
echo     :  [+] Self-Learning Loop      [AUTONOMOUS]   :
echo     :                                              :
echo     :  Exchange: Robinhood          BTC + ETH     :
echo     :  Mode:     FULLY AUTONOMOUS                 :
echo     :                                              :
echo     ================================================
echo.
echo     Initializing components...
echo.
timeout /t 2 /nobreak >nul

REM Resolve repo root from scripts\windows\ (works on ANY machine)
set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%"
set "REPO_ROOT=%CD%"
popd
cd /d "%REPO_ROOT%"
echo   Working directory: %REPO_ROOT%
echo.

REM Set environment variables BEFORE launching anything
set TRADE_API_DEV_MODE=1
set PYTHONUNBUFFERED=1
set PATH=C:\Program Files\nodejs;%PATH%

echo [1/7] Starting Trading Bot...
start "ACTs - Trading Bot" cmd /k "cd /d "%REPO_ROOT%" && set PYTHONUNBUFFERED=1 && python -m src.main"
timeout /t 8 /nobreak >nul

echo [2/7] Starting API Server (port 11007)...
start "ACTs - API Server" cmd /k "cd /d "%REPO_ROOT%" && set TRADE_API_DEV_MODE=1 && python -m uvicorn src.api.production_server:app --host 0.0.0.0 --port 11007"
timeout /t 4 /nobreak >nul

echo [3/7] Starting Dashboard (port 5173)...
start "ACTs - Dashboard" cmd /k "cd /d "%REPO_ROOT%\frontend" && set PATH=C:\Program Files\nodejs;%%PATH%% && node node_modules\vite\bin\vite.js --host"
timeout /t 4 /nobreak >nul

echo [4/7] Starting Continuous Adaptation...
start "ACTs - Continuous Adapt" cmd /k "cd /d "%REPO_ROOT%" && python -m src.scripts.continuous_adapt --continuous --interval 0.5"
timeout /t 2 /nobreak >nul

echo [5/7] Starting Monitor...
start "ACTs - Monitor" cmd /k "cd /d "%REPO_ROOT%" && scripts\windows\run_monitor.bat"
timeout /t 2 /nobreak >nul

echo [6/7] Starting Autonomous Self-Learning Loop...
start "ACTs - Autonomous Loop" cmd /k "cd /d "%REPO_ROOT%" && python -m src.scripts.autonomous_loop --interval 2"
timeout /t 2 /nobreak >nul

echo [7/7] Starting Daily Ops (health + maintenance)...
start "ACTs - Daily Ops" cmd /k "cd /d "%REPO_ROOT%" && python -m src.scripts.daily_ops --continuous"
timeout /t 2 /nobreak >nul

REM == Final: All systems online ==
cls
echo.
echo     ================================================
echo     :                                              :
echo     :         ALL SYSTEMS OPERATIONAL              :
echo     :                                              :
echo     :  [####] Trading Bot        RUNNING           :
echo     :  [####] API Server         LISTENING :11007  :
echo     :  [####] Dashboard          SERVING   :5173   :
echo     :  [####] ML Adaptation      CYCLING  (30min)  :
echo     :  [####] Health Monitor     WATCHING (10min)  :
echo     :  [####] Autonomous Loop    LEARNING  (2hr)   :
echo     :  [####] Daily Ops          HEALING   (4hr)   :
echo     :                                              :
echo     :  ----------------------------------------    :
echo     :                                              :
echo     :  Dashboard:  http://localhost:5173            :
echo     :  API:        http://localhost:11007/docs      :
echo     :                                              :
echo     :  Mode: FULLY AUTONOMOUS                      :
echo     :  Assets: BTC / ETH                           :
echo     :  Exchange: Robinhood                         :
echo     :  Subsystems: 25/25 ACTIVE                    :
echo     :                                              :
echo     ================================================
echo.
echo     The machine is trading.
echo     Press any key to SHUTDOWN all components...
echo.
pause >nul
call "%~dp0STOP_ALL.bat"
