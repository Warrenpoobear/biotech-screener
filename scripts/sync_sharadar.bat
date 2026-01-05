@echo off
REM ============================================================================
REM Sharadar SEP Daily Sync - Windows Task Scheduler Batch File
REM ============================================================================
REM 
REM SETUP:
REM   1. Edit BIOTECH_SCREENER_PATH below to your actual install path
REM   2. Set your API key (choose one method):
REM      - Environment variable: setx NASDAQ_DATA_LINK_API_KEY "your_key"
REM      - Or create file: data\config\nasdaq_api_key.txt
REM   3. Schedule in Task Scheduler (see SCHEDULING section below)
REM
REM SCHEDULING (Windows Task Scheduler):
REM   1. Open Task Scheduler (taskschd.msc)
REM   2. Create Basic Task > Name: "Sharadar SEP Sync"
REM   3. Trigger: Daily at 6:00 AM
REM   4. Action: Start a program
REM      - Program: C:\path\to\biotech_screener\scripts\sync_sharadar.bat
REM      - Start in: C:\path\to\biotech_screener
REM   5. Finish
REM
REM ============================================================================

REM === CONFIGURATION (EDIT THIS) ===
set BIOTECH_SCREENER_PATH=C:\biotech_screener

REM === DO NOT EDIT BELOW THIS LINE ===

echo ============================================================
echo Sharadar SEP Sync - %date% %time%
echo ============================================================

cd /d "%BIOTECH_SCREENER_PATH%"
if errorlevel 1 (
    echo ERROR: Could not change to directory %BIOTECH_SCREENER_PATH%
    exit /b 1
)

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    exit /b 1
)

REM Run incremental sync
echo Running incremental sync...
python scripts/sync_sharadar_sep.py

if errorlevel 1 (
    echo ERROR: Sync failed with exit code %errorlevel%
    exit /b %errorlevel%
)

echo.
echo Sync completed successfully.
echo Log location: %BIOTECH_SCREENER_PATH%\data\curated\sharadar\

exit /b 0
