@echo off
REM ============================================================================
REM Sharadar Backtest Runner - Windows Batch
REM ============================================================================
REM 
REM Runs the deterministic Sharadar backtest after data sync.
REM
REM USAGE:
REM   run_sharadar_backtest.bat
REM   run_sharadar_backtest.bat --start-year 2022 --end-year 2024
REM
REM ============================================================================

REM Configuration - UPDATE THESE PATHS
set PYTHON_PATH=python
set REPO_ROOT=%~dp0..
set PRICES_FILE=%REPO_ROOT%\data\sharadar_sep.csv

REM Change to repo directory
cd /d "%REPO_ROOT%"

REM Check for data file
if not exist "%PRICES_FILE%" (
    echo ERROR: Price data file not found: %PRICES_FILE%
    echo.
    echo Run sync first:
    echo   scripts\run_sharadar_sync.bat
    exit /b 1
)

REM Run backtest
echo.
echo ======================================================================
echo Starting Sharadar Backtest at %date% %time%
echo ======================================================================
echo.

%PYTHON_PATH% scripts/run_sharadar_backtest.py --prices "%PRICES_FILE%" %*

REM Check result
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ======================================================================
    echo BACKTEST FAILED with error code %ERRORLEVEL%
    echo ======================================================================
    exit /b %ERRORLEVEL%
)

echo.
echo ======================================================================
echo Backtest completed successfully at %date% %time%
echo ======================================================================

exit /b 0
