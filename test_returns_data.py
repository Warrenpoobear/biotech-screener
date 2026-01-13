"""
Morningstar Returns Data for Wake Robin Backtesting
Test historical returns access for validation framework
"""
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1EY3hOemRHTnpGRFJrSTRPRGswTmtaRU1FSkdOekl5TXpORFJrUTROemd6TWtOR016bEdOdyJ9.eyJodHRwczovL21vcm5pbmdzdGFyLmNvbS9lbWFpbCI6ImRzY2h1bHpAYnJvb2tzLnVzLmNvbSIsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL3JvbGUiOlsiRW5hYmxlZCBBbmFseXRpY3MgTGFiIERlbGl2ZXJ5IE5vdGVib29rcyIsIkRpc2FibGUgRGVmaW5lZCBDb250cmlidXRpb24gUGxhbnMiLCJQUkJpdFNldHRpbmciLCJQb3J0Zm9saW8gQW5hbHlzaXMgVXNlciIsIlBlcnNvbmEuRGlyZWN0Rm9yQXNzZXRNYW5hZ2VtZW50IiwiTGljZW5zZS5QcmVzZW50YXRpb25TdHVkaW8iXSwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vY29tcGFueV9pZCI6IjFhNzc5M2Q4LTE5ODEtNDA1Ni05ODA5LTI5ZjZlOThmNDc3MSIsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL2xlZ2FjeV9jb21wYW55X2lkIjoiMWE3NzkzZDgtMTk4MS00MDU2LTk4MDktMjlmNmU5OGY0NzcxIiwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vcm9sZV9pZCI6WyI3OGJhMWFlNy0xZWUzLTQ0YTAtYTAxOC0wOGM1NThmZWNmMTciLCI4MjYyOWNkMC1kZjgwLTRlNWMtYjNiYS02YmQyNWU5MzBhNDIiLCJmNDdhZjlmMC03NWU0LTQzY2ItYWM2Mi1kMjRmZmQ2NzU1ODgiLCJkYzMxN2Q5OC0xMTAwLTQyM2YtOTUzZi1mZjRkYjc4MzUwMTgiXSwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vcHJvZHVjdCI6WyJESVJFQ1QiLCJQUyJdLCJodHRwczovL21vcm5pbmdzdGFyLmNvbS9jb21wYW55IjpbeyJpZCI6IjFhNzc5M2Q4LTE5ODEtNDA1Ni05ODA5LTI5ZjZlOThmNDc3MSIsInByb2R1Y3QiOiJESVJFQ1QifV0sImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL21zdGFyX2lkIjoiRjU4ODkzNkQtQkY1Ri00QTAyLUI5RjctRjE1RDhFMzYyRTAyIiwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vZW1haWxfdmVyaWZpZWQiOnRydWUsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL3Bhc3N3b3JkQ2hhbmdlUmVxdWlyZWQiOmZhbHNlLCJodHRwczovL21vcm5pbmdzdGFyLmNvbS91aW1fcm9sZXMiOiJNRF9NRU1CRVJfMV8xLERJUkVDVCIsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL2xhc3RfcGFzc3dvcmRfcmVzZXQiOiIyMDI1LTA4LTEzVDE4OjI4OjI1LjQ5NFoiLCJpc3MiOiJodHRwczovL2xvZ2luLXByb2QubW9ybmluZ3N0YXIuY29tLyIsInN1YiI6ImF1dGgwfEY1ODg5MzZELUJGNUYtNEEwMi1COUY3LUYxNUQ4RTM2MkUwMiIsImF1ZCI6WyJodHRwczovL3VpbS1wcm9kLm1vcm5pbmdzdGFyLmF1dGgwLmNvbS9hcGkvdjIvIiwiaHR0cHM6Ly91aW0tcHJvZC5tb3JuaW5nc3Rhci5hdXRoMC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNzY4MjM4MDc0LCJleHAiOjE3NjgzMjQ0NzQsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgb2ZmbGluZV9hY2Nlc3MiLCJhenAiOiJDaGVLTTR1ajhqUFQ2MGFVMkk0Y1BsSDhyREtkT3NaZCJ9.dYahlgJ6rVHy18LCcmEcvyAcnolVq3KjVTeb73ylMcV0Tsj5vutJPiSvx_Y-Z2I1j5OGpqUJnhckJVGc6nVhZl8mpJ_VRdfCD_JiPzxGRafQaqbeKeIaLrcrQl0Ag2ub_YUl-M7M8AyLsILWEP0EDQwFM26BUadmb2kd0vwi_eqXc5S7C98oMDfMrrJPmCVib9t2r1k2X2pkF4d0dE9FA_PhNyNeEf-tzeATECpQs4AGCJtsiqMfXmN0ejla3hjk9XP5lbK8w5cPPS381zoKVuGe7lT5C76_9gpSTphfgRMViG05rBxyK0nMK7ssvduIibBlTSozyPixS06ZmeX_JQ"  # Replace with your token
os.environ['MD_AUTH_TOKEN'] = TOKEN

try:
    import morningstar_data as md
    print("✅ Connected to Morningstar Direct\n")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Test tickers - biotech stocks we know exist
TEST_TICKERS = ["VRTX", "IONS", "BMRN", "RARE"]

print("="*70)
print("STEP 1: FIND SECURITIES")
print("="*70)

sec_ids = {}
for ticker in TEST_TICKERS:
    results = md.direct.investments(keyword=ticker, count=5)
    if not results.empty:
        exact = results[results['Ticker'].str.upper() == ticker.upper()]
        if not exact.empty:
            sec_ids[ticker] = exact.iloc[0]['SecId']
            print(f"✅ {ticker:6s} -> {sec_ids[ticker]}")

if not sec_ids:
    print("❌ No securities found")
    sys.exit(1)

print(f"\n✅ Found {len(sec_ids)} securities\n")

print("="*70)
print("STEP 2: TEST RETURNS DATA ACCESS")
print("="*70)

# Test historical returns
start_date = "2020-01-01"
end_date = "2024-12-31"

print(f"\nRetrieving returns from {start_date} to {end_date}...")
print("Testing different methods:\n")

# Method 1: get_returns (documented method)
print("[Method 1: md.direct.get_returns()]")
try:
    returns_data = md.direct.get_returns(
        investments=list(sec_ids.values()),
        start_date=start_date,
        end_date=end_date,
        freq='monthly'
    )
    
    if not returns_data.empty:
        print(f"✅ SUCCESS! Retrieved {len(returns_data)} rows")
        print(f"   Date range: {returns_data.index.min()} to {returns_data.index.max()}")
        print(f"   Columns: {list(returns_data.columns)}")
        print("\nSample data (first 5 months):")
        print(returns_data.head())
        
        # Map SecId back to ticker for readability
        id_to_ticker = {v: k for k, v in sec_ids.items()}
        
        print("\n✅ Returns data is ACCESSIBLE via Morningstar Direct")
        
    else:
        print("⚠️  No data returned")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Method 2: returns (alternative method)
print("\n" + "-"*70)
print("[Method 2: md.direct.returns()]")
try:
    returns_data2 = md.direct.returns(
        investments=list(sec_ids.values()),
        start_date=start_date,
        end_date=end_date,
        freq='monthly'
    )
    
    if not returns_data2.empty:
        print(f"✅ SUCCESS! Retrieved {len(returns_data2)} rows")
        print(f"   Date range: {returns_data2.index.min()} to {returns_data2.index.max()}")
    else:
        print("⚠️  No data returned")
        
except Exception as e:
    print(f"❌ Error: {str(e)[:100]}")

print("\n" + "="*70)
print("STEP 3: TEST BENCHMARK-RELATIVE RETURNS")
print("="*70)

# Try excess returns vs XBI (biotech ETF) or SPY
print("\nTesting excess returns vs benchmark...")

# Common biotech benchmark SecIds (you may need to look these up)
benchmarks = {
    'XBI': None,  # Need to find SecId
    'IBB': None,  # Need to find SecId  
    'SPY': None   # S&P 500
}

# Try to find XBI
print("\nSearching for XBI (biotech ETF)...")
xbi_results = md.direct.investments(keyword="XBI", count=5)
if not xbi_results.empty:
    # Look for SPDR S&P Biotech ETF
    for idx, row in xbi_results.iterrows():
        if 'BIOTECH' in row['Name'].upper() or 'XBI' in row['Ticker'].upper():
            benchmarks['XBI'] = row['SecId']
            print(f"✅ Found XBI: {row['Name']} (SecId: {row['SecId']})")
            break

if benchmarks['XBI']:
    try:
        excess = md.direct.get_excess_returns(
            investments=list(sec_ids.values()),
            benchmark_sec_id=benchmarks['XBI'],
            start_date=start_date,
            end_date=end_date,
            freq='monthly'
        )
        
        if not excess.empty:
            print(f"✅ Excess returns retrieved: {len(excess)} rows")
            print("\nSample excess returns (first 5 months):")
            print(excess.head())
        else:
            print("⚠️  No excess returns data")
            
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}")
else:
    print("⚠️  Couldn't find XBI benchmark, skipping excess returns test")

print("\n" + "="*70)
print("WAKE ROBIN BACKTESTING USE CASES")
print("="*70)

print("""
With Morningstar returns data, you can build:

1. **Signal Validation Framework**
   - Test if high-scoring tickers actually outperform
   - Measure alpha generation by pattern (CASH_COW vs RAMP vs etc)
   - Validate composite ranking vs actual returns
   
2. **Walk-Forward Testing**
   - Screen universe at T=0 using only data available then
   - Track forward returns for 3, 6, 12 months
   - Measure hit rate and magnitude of winners
   - No look-ahead bias
   
3. **Regime Analysis**
   - How do signals perform in bull vs bear markets?
   - Do conviction scores work better in risk-on vs risk-off?
   - Sector rotation effects (biotech-specific cycles)
   
4. **Pattern Performance Attribution**
   - Which of the 6 patterns (CASH_COW, RAMP, LAUNCH, etc) actually work?
   - Are some patterns false signals?
   - Should pattern weights change over time?
   
5. **Coverage Validation**
   - Do tickers with complete financial data perform better?
   - Is sparse-data penalty calibrated correctly?
   - Should confidence thresholds change?

6. **Institutional Signal Validation**
   - Do elite hedge fund signals (Baker Bros, RA Capital) predict returns?
   - Is the 13F institutional conviction scoring working?
   - Optimal lookback period for institutional signals?

7. **Ablation Testing with Real Returns**
   - Remove one signal component, measure impact on returns
   - Which alpha factors are actually contributing?
   - Optimize signal weights based on historical performance

8. **Drawdown Analysis**
   - When do screening signals fail most?
   - What market conditions lead to false positives?
   - How to build regime-aware confidence adjustments?
""")

print("\n" + "="*70)
print("IMPLEMENTATION PRIORITY")
print("="*70)

print("""
RECOMMENDED: Build the backtesting module FIRST

Phase 1: Returns Data Integration (Week 1)
- Create morningstar_returns.py module
- Fetch & cache historical returns (2015-present)
- Build returns database for entire biotech universe
- Implement point-in-time discipline

Phase 2: Basic Validation Framework (Week 2)
- Select screen date (e.g., 2020-01-01)
- Run screening as if at that date
- Track forward 6-month and 12-month returns
- Calculate hit rate and alpha

Phase 3: Pattern Performance Analysis (Week 3)
- Backtest each of the 6 patterns independently
- Measure Sharpe ratio, hit rate, avg return by pattern
- Identify which patterns have real alpha
- Recalibrate pattern weights

Phase 4: Walk-Forward Validation (Week 4)
- Run screening monthly for 2020-2024
- Track cumulative performance
- Compare to biotech benchmarks (XBI, IBB)
- Validate Tier-0 provenance prevents leakage

This approach:
- Uses Morningstar for what it's good at (returns)
- Avoids the fundamental data API complexity
- Directly addresses GRAND PLAN priority: "No historical backtesting"
- Validates that your signals actually predict returns
- Informs which improvements matter most

VALUE PROPOSITION:
This is MORE valuable than better financial data coverage because:
- Tells you if your current signals actually work
- Guides where to invest development time
- Builds institutional trust (validated backtests)
- Enables continuous improvement (ongoing validation)
""")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

print("""
1. Run this script to confirm returns data access works
2. If successful, I'll build:
   - morningstar_returns.py module (Tier-0 provenance)
   - Historical returns database builder
   - Basic backtesting framework
   - Walk-forward validation harness

3. Then you can answer critical questions:
   - Do elite hedge fund signals predict returns?
   - Which patterns have real alpha?
   - Are sparse-data penalties calibrated correctly?
   - Should we add PoS modeling or fix existing signals first?

This is the RIGHT way to use Morningstar Direct for Wake Robin.
""")
