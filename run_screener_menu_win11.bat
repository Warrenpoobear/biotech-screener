@echo off
REM ============================================================================
REM Wake Robin Screener - Interactive Menu (Windows 11 Compatible)
REM ============================================================================

:MENU
cls
echo ============================================================================
echo           WAKE ROBIN BIOTECH SCREENER - INTERACTIVE MENU
echo ============================================================================
echo.
echo 1. Run screening with TODAY's date (recommended)
echo 2. Run screening with CUSTOM date
echo 3. Run tests (module_3_scoring tests)
echo 4. View latest results
echo 5. View logs directory
echo 6. Exit
echo.
echo ============================================================================

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto RUN_TODAY
if "%choice%"=="2" goto RUN_CUSTOM
if "%choice%"=="3" goto RUN_TESTS
if "%choice%"=="4" goto VIEW_RESULTS
if "%choice%"=="5" goto VIEW_LOGS
if "%choice%"=="6" goto EXIT

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto MENU

:RUN_TODAY
cls
echo ============================================================================
echo Running screener with TODAY's date...
echo ============================================================================
cd /d C:\Projects\biotech_screener\biotech-screener

REM Get today's date using PowerShell (Windows 11 compatible)
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -Format 'yyyy-MM-dd'"') do set TODAY=%%i

echo Date: %TODAY%
echo.
python run_screen.py --as-of-date %TODAY% --data-dir production_data --output results_%TODAY%.json

echo.
echo ============================================================================
pause
goto MENU

:RUN_CUSTOM
cls
echo ============================================================================
echo Run screener with CUSTOM date
echo ============================================================================
echo.
set /p custom_date="Enter date (YYYY-MM-DD format): "

echo.
echo Running screener for %custom_date%...
echo.

cd /d C:\Projects\biotech_screener\biotech-screener
python run_screen.py --as-of-date %custom_date% --data-dir production_data --output results_%custom_date%.json

echo.
echo ============================================================================
pause
goto MENU

:RUN_TESTS
cls
echo ============================================================================
echo Running Module 3 Tests...
echo ============================================================================
echo.
cd /d C:\Projects\biotech_screener\biotech-screener
python test_module_3_scoring.py

echo.
echo ============================================================================
pause
goto MENU

:VIEW_RESULTS
cls
echo ============================================================================
echo Latest Result Files
echo ============================================================================
echo.
cd /d C:\Projects\biotech_screener\biotech-screener
dir /o-d *.json | findstr /i "results"
echo.
echo ============================================================================
set /p open="Enter filename to open (or press Enter to return): "
if not "%open%"=="" start notepad %open%
goto MENU

:VIEW_LOGS
cls
echo ============================================================================
echo Log Files
echo ============================================================================
echo.
cd /d C:\Projects\biotech_screener\biotech-screener
if exist logs (
    dir /o-d logs\*.log
) else (
    echo No logs directory found.
)
echo.
echo ============================================================================
pause
goto MENU

:EXIT
echo.
echo Exiting...
exit /b 0
