@echo off
REM run_production.bat - Run biotech screener and save to production_data
REM
REM Usage:
REM   run_production.bat 2024-12-15
REM   run_production.bat 2024-12-15 --archive
REM   run_production.bat 2024-12-15 --archive --enable-coinvest

setlocal

if "%1"=="" (
    echo Usage: run_production.bat ^<YYYY-MM-DD^> [options]
    echo.
    echo Examples:
    echo   run_production.bat 2024-12-15
    echo   run_production.bat 2024-12-15 --archive
    echo   run_production.bat 2024-12-15 --archive -- --enable-coinvest
    exit /b 1
)

set AS_OF_DATE=%1
shift

REM Collect remaining arguments
set EXTRA_ARGS=
:loop
if "%1"=="" goto done
set EXTRA_ARGS=%EXTRA_ARGS% %1
shift
goto loop
:done

echo Running production screen for %AS_OF_DATE%...
python run_production_screen.py --as-of-date %AS_OF_DATE% %EXTRA_ARGS%

endlocal
