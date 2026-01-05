@echo off
REM ============================================================================
REM Full Pipeline - Sync + Backtest (Windows)
REM ============================================================================
REM 
REM Runs the complete pipeline:
REM   1. Sync Sharadar SEP data (incremental)
REM   2. Run deterministic backtest
REM
REM Schedule this monthly with Windows Task Scheduler.
REM
REM USAGE:
REM   run_full_pipeline.bat
REM   run_full_pipeline.bat --full-refresh    (for initial setup)
REM
REM ============================================================================

REM Configuration
set PYTHON_PATH=python
set REPO_ROOT=%~dp0..
set LOG_DIR=%REPO_ROOT%\logs
set LOG_FILE=%LOG_DIR%\pipeline_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log

REM Create log directory
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Change to repo directory
cd /d "%REPO_ROOT%"

REM Start logging
echo ====================================================================== >> "%LOG_FILE%"
echo Pipeline started at %date% %time% >> "%LOG_FILE%"
echo ====================================================================== >> "%LOG_FILE%"

echo.
echo ======================================================================
echo WAKE ROBIN BIOTECH SCREENER - FULL PIPELINE
echo ======================================================================
echo Started: %date% %time%
echo Log: %LOG_FILE%
echo.

REM ============================================================================
REM STEP 1: Sync Data
REM ============================================================================
echo.
echo [STEP 1/2] Syncing Sharadar SEP data...
echo.

call scripts\run_sharadar_sync.bat %* >> "%LOG_FILE%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ SYNC FAILED - Check log: %LOG_FILE%
    echo Pipeline aborted at %date% %time% >> "%LOG_FILE%"
    exit /b 1
)

echo ✓ Sync completed

REM ============================================================================
REM STEP 2: Run Backtest
REM ============================================================================
echo.
echo [STEP 2/2] Running backtest...
echo.

call scripts\run_backtest.bat >> "%LOG_FILE%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ BACKTEST FAILED - Check log: %LOG_FILE%
    echo Pipeline aborted at %date% %time% >> "%LOG_FILE%"
    exit /b 1
)

echo ✓ Backtest completed

REM ============================================================================
REM COMPLETE
REM ============================================================================
echo.
echo ======================================================================
echo ✅ PIPELINE COMPLETE
echo ======================================================================
echo Finished: %date% %time%
echo.
echo Outputs:
echo   data\sharadar_sep.csv          (synced prices)
echo   output\runs\sharadar_*\        (backtest results)
echo   logs\                          (run logs)
echo.

echo Pipeline completed at %date% %time% >> "%LOG_FILE%"

exit /b 0
