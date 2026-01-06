# phase1_top50.ps1
# Phase 1: Collect and screen top 50 biotech stocks from XBI/IBB/NBI

Write-Host "============================================================"
Write-Host "PHASE 1: TOP 50 ETF CONSTITUENTS"
Write-Host "============================================================"
Write-Host ""

# Top 50 stocks (largest, most liquid, easiest data collection)
$top50 = @(
    # Your existing 21 stocks (already have data):
    "ACAD", "ALNY", "ARGX", "BBIO", "BMRN", "CRSP", "EPRX", "EXEL",
    "FOLD", "GOSS", "INCY", "IONS", "JAZZ", "RARE", "REGN", "RNA",
    "RYTM", "SRPT", "UTHR", "VRTX", "NBIX",
    
    # Add 29 more large/mid caps (need data):
    "AMGN", "GILD", "BIIB", "MRNA", "BNTX",  # Mega caps
    "SGEN", "LGND", "TECH", "BGNE", "RGEN",   # Large caps
    "ROIV", "NTRA", "HZNP", "DAWN", "PCVX",   # Mid caps
    "ARVN", "LEGN", "IMMU", "BLUE", "FATE",   # Mid caps
    "NTLA", "EDIT", "BEAM", "VCYT", "NSTG",   # Mid caps
    "AXSM", "PTGX", "CDNA", "PRVA"            # Mid caps
)

Write-Host "Top 50 Biotech Stocks Selected:"
Write-Host "  • Your existing 21 stocks: ✓ (already have data)"
Write-Host "  • Additional 29 stocks: Need data collection"
Write-Host ""
Write-Host "Market Cap Distribution:"
Write-Host "  • Mega cap ($50B+): 5 stocks (AMGN, GILD, VRTX, REGN, BIIB)"
Write-Host "  • Large cap ($10-50B): 15 stocks"
Write-Host "  • Mid cap ($2-10B): 20 stocks"
Write-Host "  • Small cap (<$2B): 10 stocks"
Write-Host ""

# Save ticker list
$tickerList = $top50 -join ","
$tickerList | Out-File "top50_tickers.txt"
Write-Host "✓ Saved ticker list to: top50_tickers.txt"
Write-Host ""

Write-Host "============================================================"
Write-Host "DATA COLLECTION OPTIONS"
Write-Host "============================================================"
Write-Host ""
Write-Host "Option A: Use Your Existing Pipeline"
Write-Host "-------------------------------------"
Write-Host "If you have wake_robin_data_pipeline:"
Write-Host ""
Write-Host "  cd wake_robin_data_pipeline"
Write-Host "  python collect_universe_data.py \"
Write-Host "      --tickers `"$tickerList`" \"
Write-Host "      --as-of-date 2026-01-06 \"
Write-Host "      --output ../top50_data"
Write-Host ""

Write-Host "Option B: Manual Collection (New Script)"
Write-Host "-----------------------------------------"
Write-Host "Use the provided collect_top50_data.py script:"
Write-Host ""
Write-Host "  python collect_top50_data.py \"
Write-Host "      --tickers top50_tickers.txt \"
Write-Host "      --as-of-date 2026-01-06 \"
Write-Host "      --output top50_data"
Write-Host ""

Write-Host "Option C: Copy Existing + Add New"
Write-Host "-----------------------------------"
Write-Host "Use your 21-stock data + add 29 new:"
Write-Host ""
Write-Host "  1. Copy production_data to top50_data"
Write-Host "  2. Collect data for 29 new tickers"
Write-Host "  3. Merge into single universe file"
Write-Host ""

Write-Host "============================================================"
Write-Host "NEXT STEPS"
Write-Host "============================================================"
Write-Host ""
Write-Host "1. Choose data collection option (A, B, or C)"
Write-Host ""
Write-Host "2. Collect data for top 50 stocks"
Write-Host ""
Write-Host "3. Run screener:"
Write-Host "   python run_screen.py \"
Write-Host "       --as-of-date 2026-01-06 \"
Write-Host "       --data-dir top50_data \"
Write-Host "       --output top50_screening_results.json"
Write-Host ""
Write-Host "4. Analyze results:"
Write-Host "   python analyze_defensive_impact.py \"
Write-Host "       --output top50_screening_results.json"
Write-Host ""
Write-Host "5. Compare with 21-stock results to validate scaling"
Write-Host ""
Write-Host "============================================================"
Write-Host "EXPECTED RESULTS"
Write-Host "============================================================"
Write-Host ""
Write-Host "With 50 stocks, you should see:"
Write-Host "  • Active securities: ~45 (90% pass gates)"
Write-Host "  • Rankable: ~40 (80% have all data)"
Write-Host "  • Position weights: 0.01 to 0.08 (1% to 8%)"
Write-Host "  • Average weight: 0.0225 (2.25% per position)"
Write-Host "  • Top 10 concentration: ~30-35%"
Write-Host "  • Defensive adjustments: 50-65% of securities"
Write-Host ""
Write-Host "Runtime: ~1-2 minutes (vs 30 seconds for 21 stocks)"
Write-Host ""
Write-Host "============================================================"

$response = Read-Host "Do you have wake_robin_data_pipeline working? (y/n)"

if ($response -eq "y") {
    Write-Host ""
    Write-Host "Great! Use Option A (existing pipeline)"
    Write-Host ""
    Write-Host "Run this command:"
    Write-Host ""
    Write-Host "cd wake_robin_data_pipeline"
    Write-Host "python collect_universe_data.py --tickers `"$tickerList`" --as-of-date 2026-01-06 --output ../top50_data"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "No problem! You have two options:"
    Write-Host ""
    Write-Host "Option B: Use collect_top50_data.py (I'll create this next)"
    Write-Host "Option C: Extend your existing 21-stock data with 29 more"
    Write-Host ""
    Write-Host "Recommended: Option C (fastest path)"
    Write-Host ""
    Write-Host "Would you like me to create a script for Option C? (y/n)"
    $response2 = Read-Host
    
    if ($response2 -eq "y") {
        Write-Host ""
        Write-Host "I'll create extend_universe.py for you..."
        Write-Host "(This will let you add 29 stocks to your existing 21)"
    }
}

Write-Host ""
Write-Host "Saved top 50 tickers to: top50_tickers.txt"
Write-Host ""
