@echo off
REM ============================================================================
REM Sharadar SEP Sync - Windows Batch Runner
REM ============================================================================
REM 
REM This batch file runs the Sharadar SEP sync script.
REM Schedule this with Windows Task Scheduler for daily automated syncs.
REM
REM SETUP:
REM   1. Set your API key as a system environment variable:
REM      - Open System Properties > Advanced > Environment Variables
REM      - Add: NASDAQ_DATA_LINK_API_KEY = your_api_key_here
REM
REM   2. Update PYTHON_PATH and REPO_ROOT below if needed
REM
REM USAGE:
REM   run_sharadar_sync.bat                  (incremental sync)
REM   run_sharadar_sync.bat --full-refresh   (full history)
REM
REM ============================================================================

REM Configuration - UPDATE THESE PATHS
set PYTHON_PATH=python
set REPO_ROOT=%~dp0..

REM Change to repo directory
cd /d "%REPO_ROOT%"

REM Check for API key
if "%NASDAQ_DATA_LINK_API_KEY%"=="" (
    echo ERROR: NASDAQ_DATA_LINK_API_KEY environment variable not set
    echo.
    echo Set it via:
    echo   1. System Properties ^> Advanced ^> Environment Variables
    echo   2. Or run: set NASDAQ_DATA_LINK_API_KEY=your_key_here
    exit /b 1
)

REM Run sync script
echo.
echo ======================================================================
echo Starting Sharadar SEP Sync at %date% %time%
echo ======================================================================
echo.

%PYTHON_PATH% scripts/sync_sharadar_sep.py %*

REM Check result
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ======================================================================
    echo SYNC FAILED with error code %ERRORLEVEL%
    echo ======================================================================
    exit /b %ERRORLEVEL%
)

echo.
echo ======================================================================
echo Sync completed successfully at %date% %time%
echo ======================================================================

exit /b 0
