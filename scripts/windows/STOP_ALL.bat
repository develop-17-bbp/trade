@echo off
REM ============================================================
REM   ACT's TRADING SYSTEM - MASTER SHUTDOWN
REM   Gracefully terminates ALL components
REM ============================================================

color 0F
title ACT's AI Trading System - Shutting Down...

REM == Frame 1: Shutdown initiated ==
cls
echo.
echo     ================================================
echo     :                                              :
echo     :        SHUTDOWN SEQUENCE INITIATED           :
echo     :                                              :
echo     :        Saving state and closing all          :
echo     :        trading components safely...          :
echo     :                                              :
echo     ================================================
echo.
timeout /t 1 /nobreak >nul

REM == Frame 2: Stopping components ==
cls
echo.
echo     ================================================
echo     :                                              :
echo     :  [....] Trading Bot        STOPPING...       :
echo     :  [....] API Server         WAITING...        :
echo     :  [....] Dashboard          WAITING...        :
echo     :  [....] ML Adaptation      WAITING...        :
echo     :  [....] Health Monitor     WAITING...        :
echo     :                                              :
echo     ================================================
echo.

echo   [1/5] Stopping Trading Bot...
taskkill /FI "WINDOWTITLE eq ACTs - Trading Bot*" /F >nul 2>&1
timeout /t 1 /nobreak >nul

cls
echo.
echo     ================================================
echo     :                                              :
echo     :  [DONE] Trading Bot        STOPPED           :
echo     :  [....] API Server         STOPPING...       :
echo     :  [....] Dashboard          WAITING...        :
echo     :  [....] ML Adaptation      WAITING...        :
echo     :  [....] Health Monitor     WAITING...        :
echo     :                                              :
echo     ================================================
echo.

echo   [2/5] Stopping API Server...
taskkill /FI "WINDOWTITLE eq ACTs - API Server*" /F >nul 2>&1
timeout /t 1 /nobreak >nul

cls
echo.
echo     ================================================
echo     :                                              :
echo     :  [DONE] Trading Bot        STOPPED           :
echo     :  [DONE] API Server         STOPPED           :
echo     :  [....] Dashboard          STOPPING...       :
echo     :  [....] ML Adaptation      WAITING...        :
echo     :  [....] Health Monitor     WAITING...        :
echo     :                                              :
echo     ================================================
echo.

echo   [3/5] Stopping Dashboard...
taskkill /FI "WINDOWTITLE eq ACTs - Dashboard*" /F >nul 2>&1
timeout /t 1 /nobreak >nul

cls
echo.
echo     ================================================
echo     :                                              :
echo     :  [DONE] Trading Bot        STOPPED           :
echo     :  [DONE] API Server         STOPPED           :
echo     :  [DONE] Dashboard          STOPPED           :
echo     :  [....] ML Adaptation      STOPPING...       :
echo     :  [....] Health Monitor     WAITING...        :
echo     :                                              :
echo     ================================================
echo.

echo   [4/5] Stopping ML Adaptation...
taskkill /FI "WINDOWTITLE eq ACTs - Continuous Adapt*" /F >nul 2>&1
timeout /t 1 /nobreak >nul

cls
echo.
echo     ================================================
echo     :                                              :
echo     :  [DONE] Trading Bot        STOPPED           :
echo     :  [DONE] API Server         STOPPED           :
echo     :  [DONE] Dashboard          STOPPED           :
echo     :  [DONE] ML Adaptation      STOPPED           :
echo     :  [....] Health Monitor     STOPPING...       :
echo     :                                              :
echo     ================================================
echo.

echo   [5/7] Stopping Health Monitor...
taskkill /FI "WINDOWTITLE eq ACTs - Monitor*" /F >nul 2>&1
timeout /t 1 /nobreak >nul

echo   [6/7] Stopping Autonomous Loop...
taskkill /FI "WINDOWTITLE eq ACTs - Autonomous Loop*" /F >nul 2>&1
timeout /t 1 /nobreak >nul

echo   [7/7] Stopping Daily Ops...
taskkill /FI "WINDOWTITLE eq ACTs - Daily Ops*" /F >nul 2>&1
timeout /t 2 /nobreak >nul

REM == Frame 3: Neural network powering down ==
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
echo      :       DISCONNECTING NEURAL MESH...         :
echo      :                                            :
echo      [============================================]
echo.
timeout /t 1 /nobreak >nul

REM == Frame 4: Nodes going offline ==
cls
echo.
echo      [============================================]
echo      :                                            :
echo      :       o . o . o . o . o . o . o .         :
echo      :       . o . o . o . o . o . o . o         :
echo      :       o . o . o . o . o . o . o .         :
echo      :       . o . o . o . o . o . o . o         :
echo      :                                            :
echo      :        NEURONS POWERING DOWN...            :
echo      :                                            :
echo      [============================================]
echo.
timeout /t 1 /nobreak >nul

REM == Frame 5: Dark ==
cls
echo.
echo      [============================================]
echo      :                                            :
echo      :       . . . . . . . . . . . . . .         :
echo      :       . . . . . . . . . . . . . .         :
echo      :       . . . . . . . . . . . . . .         :
echo      :       . . . . . . . . . . . . . .         :
echo      :                                            :
echo      :          NETWORK OFFLINE                   :
echo      :                                            :
echo      [============================================]
echo.
timeout /t 1 /nobreak >nul

REM == Final: All systems offline ==
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
echo     :  ALL SYSTEMS OFFLINE                         :
echo     :                                              :
echo     :  [DONE] Trading Bot        OFFLINE           :
echo     :  [DONE] API Server         OFFLINE           :
echo     :  [DONE] Dashboard          OFFLINE           :
echo     :  [DONE] ML Adaptation      OFFLINE           :
echo     :  [DONE] Health Monitor     OFFLINE           :
echo     :                                              :
echo     :  ----------------------------------------    :
echo     :                                              :
echo     :  All positions preserved.                    :
echo     :  Trade journal saved.                        :
echo     :  State persisted to disk.                    :
echo     :                                              :
echo     :  Run START_ALL.bat to restart.               :
echo     :                                              :
echo     ================================================
echo.
echo     Shutdown complete. Window closing in 5 seconds...
timeout /t 5 /nobreak >nul
exit
