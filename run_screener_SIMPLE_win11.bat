@echo off
REM ============================================================================
REM Wake Robin Screener - Ultra Simple Version (Windows 11 Compatible)
REM ============================================================================

cd /d C:\Projects\biotech_screener\biotech-screener

echo ============================================================================
echo Wake Robin Biotech Screener
echo ============================================================================
echo.

REM Get today's date using PowerShell
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -Format 'yyyy-MM-dd'"') do set TODAY=%%i

echo Running screener for: %TODAY%
echo.

python run_screen.py --as-of-date %TODAY% --data-dir production_data --output results_%TODAY%.json --enable-enhancements --enable-coinvest --enable-short-interest

echo.
echo ============================================================================
echo Screening complete!
echo Results saved to: results_%TODAY%.json
echo ============================================================================
echo.
pause
