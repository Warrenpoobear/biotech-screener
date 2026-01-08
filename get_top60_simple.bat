@echo off
REM ============================================================================
REM Wake Robin - Quick Top 60 Display (SIMPLE VERSION)
REM ============================================================================

echo.
echo Generating Top 60 Report...
echo.

REM Get today's date
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -Format 'yyyy-MM-dd'"') do set TODAY=%%i

REM Default to today's results or latest file
set RESULTS_FILE=results_%TODAY%.json
if not exist "%RESULTS_FILE%" set RESULTS_FILE=test_no_overlay.json
if not exist "%RESULTS_FILE%" (
    for /f "delims=" %%f in ('dir /b /o-d results_*.json 2^>nul') do (
        set RESULTS_FILE=%%f
        goto FOUND
    )
)

:FOUND
echo Using: %RESULTS_FILE%
echo.

REM Generate report
python -c "import json; data=json.load(open('%RESULTS_FILE%')); ranked=sorted(data['module_5_composite']['ranked_securities'], key=lambda x: int(x.get('composite_rank', 999)))[:60]; print('='*80); print('TOP 60 SECURITIES - ' + '%RESULTS_FILE%'); print('='*80); print(f\"\n{'Rank':<6}{'Ticker':<8}{'Score':<10}{'Weight':<10}{'Financial':<12}{'Clinical':<10}\"); print('-'*70); [print(f\"{s['composite_rank']:<6}{s['ticker']:<8}{float(s['composite_score']):<10.2f}{float(s['position_weight'])*100:>8.2f}%%  {float(s.get('financial_normalized', 0)):>10.2f}  {float(s.get('clinical_dev_normalized', 0)):>9.2f}\") for s in ranked]; print('\n' + '='*80); print(f\"Total: {sum(float(s['position_weight']) for s in ranked)*100:.2f}%% | Avg Score: {sum(float(s['composite_score']) for s in ranked)/60:.2f}\"); print('='*80)"

echo.
pause
