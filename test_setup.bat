@echo off
REM Test script to verify report generator setup

echo.
echo ================================================================================
echo WAKE ROBIN - REPORT GENERATOR TEST
echo ================================================================================
echo.

echo Testing Python availability...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.7+ and add to PATH
    pause
    exit /b 1
)
echo OK - Python found
echo.

echo Testing generate_all_reports.py...
if not exist "generate_all_reports.py" (
    echo ERROR: generate_all_reports.py not found!
    echo Make sure both .bat and .py files are in the same directory
    pause
    exit /b 1
)
echo OK - Script found
echo.

echo Testing for results files...
dir /b *.json 2>nul | find "result" >nul
if %ERRORLEVEL% EQU 0 (
    echo Found result files:
    dir /b *result*.json
) else (
    echo WARNING: No result files found
    echo Run the screening pipeline first to generate results
)
echo.

echo ================================================================================
echo SETUP CHECK COMPLETE
echo ================================================================================
echo.
echo To generate reports, run:
echo   generate_reports.bat [your_results_file.json]
echo.
echo Or drag-and-drop a JSON file onto generate_reports.bat
echo.
pause
