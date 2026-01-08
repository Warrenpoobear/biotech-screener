@echo off
REM =============================================================================
REM WAKE ROBIN - REPORT GENERATOR
REM Simple wrapper for generate_all_reports.py
REM =============================================================================

echo.
echo ================================================================================
echo WAKE ROBIN - REPORT GENERATOR
echo ================================================================================
echo.

REM Check if results file provided
if "%~1"=="" (
    echo Usage: generate_reports.bat [results_file.json]
    echo.
    echo Example: generate_reports.bat results_ZERO_BUG_FIXED.json
    echo.
    echo Or drag-and-drop a JSON file onto this batch file!
    echo.
    pause
    exit /b 1
)

REM Check if file exists
if not exist "%~1" (
    echo ERROR: File not found: %~1
    echo.
    pause
    exit /b 1
)

REM Run the Python script
python generate_all_reports.py "%~1"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo SUCCESS - All reports generated!
    echo ================================================================================
    echo.
    echo Open outputs folder? (Y/N)
    set /p OPEN=
    if /i "%OPEN%"=="Y" explorer outputs
) else (
    echo.
    echo ================================================================================
    echo ERROR - Report generation failed
    echo ================================================================================
    echo.
    echo Check the error messages above for details.
)

echo.
pause
