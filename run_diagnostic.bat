@echo off
REM ============================================================================
REM Wake Robin - Score Quality Diagnostic
REM ============================================================================

echo.
echo ============================================================================
echo WAKE ROBIN - SCORE QUALITY DIAGNOSTIC
echo ============================================================================
echo.
echo This tool analyzes your screening results for:
echo   - Score bucketing / quantization issues
echo   - Universe contamination (invalid tickers)
echo   - Tie mass (tickers with identical scores)
echo   - Overall score quality grade (A-F)
echo.

REM Check if diagnostic script exists
if not exist "score_quality_diagnostic.py" (
    echo ERROR: score_quality_diagnostic.py not found!
    echo Please make sure the Python script is in the current directory.
    pause
    exit /b 1
)

echo Running diagnostic...
echo.

python score_quality_diagnostic.py

if errorlevel 1 (
    echo.
    echo ERROR: Diagnostic failed!
    echo.
    echo If no results file was found, try:
    echo   python score_quality_diagnostic.py --file your_results.json
    pause
    exit /b 1
)

echo.
echo Diagnostic complete!
echo.
pause
