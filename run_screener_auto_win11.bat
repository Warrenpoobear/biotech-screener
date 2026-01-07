@echo off
REM ============================================================================
REM Wake Robin Biotech Screener - Production Runner (Windows 11 Compatible)
REM Automatically uses today's date and logs all output
REM ============================================================================

setlocal EnableDelayedExpansion

REM Change to project directory
cd /d C:\Projects\biotech_screener\biotech-screener

REM Get today's date using PowerShell (Windows 11 compatible)
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -Format 'yyyy-MM-dd'"') do set TODAY=%%i
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"') do set TIMESTAMP=%%i

REM Create output filename with timestamp
set OUTPUT_FILE=results_%TODAY%.json
set LOG_FILE=logs\screener_%TODAY%.log

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

echo ============================================================================
echo Wake Robin Biotech Screener
echo ============================================================================
echo Date: %TODAY%
echo Output: %OUTPUT_FILE%
echo Log: %LOG_FILE%
echo ============================================================================
echo.

REM Run the screener with logging (tee not available on Windows, so use PowerShell)
python run_screen.py --as-of-date %TODAY% --data-dir production_data --output %OUTPUT_FILE% 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%LOG_FILE%'"

REM Check if successful
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================================
    echo SUCCESS! Screening completed
    echo Results: %OUTPUT_FILE%
    echo ============================================================================
) else (
    echo.
    echo ============================================================================
    echo ERROR! Screening failed with error code %ERRORLEVEL%
    echo Check log file: %LOG_FILE%
    echo ============================================================================
)

echo.
pause
