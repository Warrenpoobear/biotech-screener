@echo off
REM ============================================================================
REM Wake Robin - Top 60 to CSV Export
REM ============================================================================

echo.
echo Exporting Top 60 to CSV...
echo.

REM Get today's date
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -Format 'yyyy-MM-dd'"') do set TODAY=%%i

REM Find results file
set RESULTS_FILE=results_%TODAY%.json
if not exist "%RESULTS_FILE%" set RESULTS_FILE=test_no_overlay.json
if not exist "%RESULTS_FILE%" (
    for /f "delims=" %%f in ('dir /b /o-d results_*.json 2^>nul') do (
        set RESULTS_FILE=%%f
        goto FOUND
    )
)

:FOUND
echo Source: %RESULTS_FILE%

REM Set output filename
set OUTPUT_CSV=top60_%TODAY%.csv

echo Output: %OUTPUT_CSV%
echo.

REM Generate CSV
python -c "import json, csv; data=json.load(open('%RESULTS_FILE%')); ranked=sorted(data['module_5_composite']['ranked_securities'], key=lambda x: int(x.get('composite_rank', 999)))[:60]; f=open('%OUTPUT_CSV%', 'w', newline=''); writer=csv.writer(f); writer.writerow(['Rank', 'Ticker', 'Composite_Score', 'Weight_Percent', 'Financial_Score', 'Clinical_Score', 'Catalyst_Score', 'Stage']); [writer.writerow([s['composite_rank'], s['ticker'], f\"{float(s['composite_score']):.2f}\", f\"{float(s['position_weight'])*100:.2f}\", f\"{float(s.get('financial_normalized', 0)):.2f}\", f\"{float(s.get('clinical_dev_normalized', 0)):.2f}\", f\"{float(s.get('catalyst_normalized', 50)):.2f}\", s.get('stage_bucket', 'unknown')]) for s in ranked]; f.close(); print('CSV exported successfully!'); print(f'Total positions: 60'); print(f'Total weight: {sum(float(s[\"position_weight\"]) for s in ranked)*100:.2f}%%')"

echo.
echo Opening CSV in Excel...
start "" "%OUTPUT_CSV%"

echo.
pause
