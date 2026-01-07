@echo off
REM ============================================================================
REM Wake Robin Screener - Silent/Scheduled Mode (Windows 11 Compatible)
REM For use with Windows Task Scheduler
REM No pauses or prompts - logs everything to file
REM ============================================================================

setlocal EnableDelayedExpansion

REM Change to project directory
cd /d C:\Projects\biotech_screener\biotech-screener

REM Get timestamp using PowerShell (Windows 11 compatible)
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -Format 'yyyy-MM-dd'"') do set TODAY=%%i
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"') do set TIMESTAMP=%%i

REM Setup paths
set OUTPUT_FILE=production_data\results_%TODAY%.json
set LOG_FILE=logs\screener_%TIMESTAMP%.log
set ERROR_FILE=logs\error_%TIMESTAMP%.log

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Log start time
powershell -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'" >> %LOG_FILE%
echo Starting Wake Robin Screener >> %LOG_FILE%
echo Date: %TODAY% >> %LOG_FILE%
echo Output: %OUTPUT_FILE% >> %LOG_FILE%
echo ============================================================================ >> %LOG_FILE%

REM Run screener - capture both stdout and stderr
python run_screen.py --as-of-date %TODAY% --data-dir production_data --output %OUTPUT_FILE% >> %LOG_FILE% 2>> %ERROR_FILE%

REM Check result
if %ERRORLEVEL% EQU 0 (
    powershell -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'" >> %LOG_FILE%
    echo SUCCESS - Screening completed >> %LOG_FILE%
    exit /b 0
) else (
    powershell -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'" >> %LOG_FILE%
    echo ERROR - Screening failed with code %ERRORLEVEL% >> %LOG_FILE%
    echo Check error log: %ERROR_FILE% >> %LOG_FILE%
    exit /b %ERRORLEVEL%
)
