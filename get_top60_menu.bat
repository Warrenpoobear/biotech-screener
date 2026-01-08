@echo off
REM ============================================================================
REM Wake Robin - Top 60 Report Generator (Using Python Script)
REM ============================================================================

echo.
echo ============================================================================
echo WAKE ROBIN - TOP 60 SECURITIES REPORT
echo ============================================================================
echo.

REM Check if Python script exists
if not exist "extract_top60.py" (
    echo ERROR: extract_top60.py not found!
    echo Please make sure the Python script is in the current directory.
    pause
    exit /b 1
)

echo What would you like to do?
echo.
echo   1. Display top 60 (console only)
echo   2. Save to text file
echo   3. Save to CSV file (for Excel)
echo   4. Save to both text and CSV
echo.
choice /C 1234 /N /M "Choose option (1/2/3/4): "

if errorlevel 4 goto BOTH
if errorlevel 3 goto CSV
if errorlevel 2 goto TEXT
if errorlevel 1 goto CONSOLE

:CONSOLE
echo.
python extract_top60.py
goto END

:TEXT
echo.
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -Format 'yyyy-MM-dd'"') do set TODAY=%%i
set OUTPUT=top60_%TODAY%.txt
python extract_top60.py --output %OUTPUT% --format text
if errorlevel 0 notepad %OUTPUT%
goto END

:CSV
echo.
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -Format 'yyyy-MM-dd'"') do set TODAY=%%i
set OUTPUT=top60_%TODAY%.csv
python extract_top60.py --output %OUTPUT% --format csv --quiet
if errorlevel 0 start "" "%OUTPUT%"
goto END

:BOTH
echo.
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -Format 'yyyy-MM-dd'"') do set TODAY=%%i
set OUTPUT=top60_%TODAY%
python extract_top60.py --output %OUTPUT% --format both
if errorlevel 0 (
    echo.
    echo Files generated:
    echo   - %OUTPUT%.txt
    echo   - %OUTPUT%.csv
    echo.
    echo Opening text file...
    notepad %OUTPUT%.txt
)
goto END

:END
echo.
pause
