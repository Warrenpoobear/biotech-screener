# run_historical_batch.ps1
# Simple batch runner for historical backtesting

param(
    [string]$DataDir = "production_data",
    [string]$OutputDir = "backtest_results"
)

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Define test dates (weekly for last 3 months)
$TestDates = @(
    "2025-10-07",
    "2025-10-14",
    "2025-10-21",
    "2025-10-28",
    "2025-11-04",
    "2025-11-11",
    "2025-11-18",
    "2025-11-25",
    "2025-12-02",
    "2025-12-09",
    "2025-12-16",
    "2025-12-23",
    "2025-12-30",
    "2026-01-06"
)

Write-Host "============================================================"
Write-Host "HISTORICAL BACKTEST BATCH RUNNER"
Write-Host "============================================================"
Write-Host ""
Write-Host "Data directory: $DataDir"
Write-Host "Output directory: $OutputDir"
Write-Host "Test dates: $($TestDates.Count)"
Write-Host ""

foreach ($date in $TestDates) {
    Write-Host "[$date]"
    Write-Host ""
}

$confirm = Read-Host "Press Enter to start batch run (Ctrl+C to cancel)"

$results = @()
$successCount = 0
$failCount = 0

foreach ($date in $TestDates) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running: $date"
    Write-Host "============================================================"
    
    $outputFile = Join-Path $OutputDir "ranked_$date.json"
    
    try {
        # Run pipeline
        python run_screen.py `
            --as-of-date $date `
            --data-dir $DataDir `
            --output $outputFile `
            2>&1 | Tee-Object -Variable output
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ SUCCESS" -ForegroundColor Green
            $successCount++
            
            $results += [PSCustomObject]@{
                Date = $date
                Status = "Success"
                OutputFile = $outputFile
            }
        } else {
            Write-Host "✗ FAILED" -ForegroundColor Red
            $failCount++
            
            $results += [PSCustomObject]@{
                Date = $date
                Status = "Failed"
                Error = "Exit code: $LASTEXITCODE"
            }
        }
    }
    catch {
        Write-Host "✗ ERROR: $_" -ForegroundColor Red
        $failCount++
        
        $results += [PSCustomObject]@{
            Date = $date
            Status = "Error"
            Error = $_.Exception.Message
        }
    }
}

# Summary
Write-Host ""
Write-Host "============================================================"
Write-Host "BATCH RUN COMPLETE"
Write-Host "============================================================"
Write-Host ""
Write-Host "Total runs:     $($TestDates.Count)"
Write-Host "Successful:     $successCount" -ForegroundColor Green
Write-Host "Failed:         $failCount" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Green" })
Write-Host ""

# Save summary
$summaryFile = Join-Path $OutputDir "batch_summary.csv"
$results | Export-Csv -Path $summaryFile -NoTypeInformation
Write-Host "Summary saved to: $summaryFile"

# Show results
Write-Host ""
Write-Host "Results:"
$results | Format-Table -AutoSize

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Review outputs in: $OutputDir"
Write-Host "  2. Analyze results:"
Write-Host "     python analyze_defensive_impact.py --output $OutputDir/ranked_2026-01-06.json"
Write-Host ""
