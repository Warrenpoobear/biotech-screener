"""
Biotech Financial Data Coverage Test
Compare Morningstar Direct vs Wake Robin's current data sources

This tests coverage for the critical financial metrics that Wake Robin needs:
- Cash and equivalents (currently 60% missing)
- Total debt (currently 80% missing)
- Market cap
- Revenue
"""
import os
import sys
import pandas as pd
from decimal import Decimal

# Set your token
TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1EY3hOemRHTnpGRFJrSTRPRGswTmtaRU1FSkdOekl5TXpORFJrUTROemd6TWtOR016bEdOdyJ9.eyJodHRwczovL21vcm5pbmdzdGFyLmNvbS9lbWFpbCI6ImRzY2h1bHpAYnJvb2tzLnVzLmNvbSIsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL3JvbGUiOlsiRW5hYmxlZCBBbmFseXRpY3MgTGFiIERlbGl2ZXJ5IE5vdGVib29rcyIsIkRpc2FibGUgRGVmaW5lZCBDb250cmlidXRpb24gUGxhbnMiLCJQUkJpdFNldHRpbmciLCJQb3J0Zm9saW8gQW5hbHlzaXMgVXNlciIsIlBlcnNvbmEuRGlyZWN0Rm9yQXNzZXRNYW5hZ2VtZW50IiwiTGljZW5zZS5QcmVzZW50YXRpb25TdHVkaW8iXSwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vY29tcGFueV9pZCI6IjFhNzc5M2Q4LTE5ODEtNDA1Ni05ODA5LTI5ZjZlOThmNDc3MSIsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL2xlZ2FjeV9jb21wYW55X2lkIjoiMWE3NzkzZDgtMTk4MS00MDU2LTk4MDktMjlmNmU5OGY0NzcxIiwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vcm9sZV9pZCI6WyI3OGJhMWFlNy0xZWUzLTQ0YTAtYTAxOC0wOGM1NThmZWNmMTciLCI4MjYyOWNkMC1kZjgwLTRlNWMtYjNiYS02YmQyNWU5MzBhNDIiLCJmNDdhZjlmMC03NWU0LTQzY2ItYWM2Mi1kMjRmZmQ2NzU1ODgiLCJkYzMxN2Q5OC0xMTAwLTQyM2YtOTUzZi1mZjRkYjc4MzUwMTgiXSwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vcHJvZHVjdCI6WyJESVJFQ1QiLCJQUyJdLCJodHRwczovL21vcm5pbmdzdGFyLmNvbS9jb21wYW55IjpbeyJpZCI6IjFhNzc5M2Q4LTE5ODEtNDA1Ni05ODA5LTI5ZjZlOThmNDc3MSIsInByb2R1Y3QiOiJESVJFQ1QifV0sImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL21zdGFyX2lkIjoiRjU4ODkzNkQtQkY1Ri00QTAyLUI5RjctRjE1RDhFMzYyRTAyIiwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vZW1haWxfdmVyaWZpZWQiOnRydWUsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL3Bhc3N3b3JkQ2hhbmdlUmVxdWlyZWQiOmZhbHNlLCJodHRwczovL21vcm5pbmdzdGFyLmNvbS91aW1fcm9sZXMiOiJNRF9NRU1CRVJfMV8xLERJUkVDVCIsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL2xhc3RfcGFzc3dvcmRfcmVzZXQiOiIyMDI1LTA4LTEzVDE4OjI4OjI1LjQ5NFoiLCJpc3MiOiJodHRwczovL2xvZ2luLXByb2QubW9ybmluZ3N0YXIuY29tLyIsInN1YiI6ImF1dGgwfEY1ODg5MzZELUJGNUYtNEEwMi1COUY3LUYxNUQ4RTM2MkUwMiIsImF1ZCI6WyJodHRwczovL3VpbS1wcm9kLm1vcm5pbmdzdGFyLmF1dGgwLmNvbS9hcGkvdjIvIiwiaHR0cHM6Ly91aW0tcHJvZC5tb3JuaW5nc3Rhci5hdXRoMC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNzY4MjM4MDc0LCJleHAiOjE3NjgzMjQ0NzQsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgb2ZmbGluZV9hY2Nlc3MiLCJhenAiOiJDaGVLTTR1ajhqUFQ2MGFVMkk0Y1BsSDhyREtkT3NaZCJ9.dYahlgJ6rVHy18LCcmEcvyAcnolVq3KjVTeb73ylMcV0Tsj5vutJPiSvx_Y-Z2I1j5OGpqUJnhckJVGc6nVhZl8mpJ_VRdfCD_JiPzxGRafQaqbeKeIaLrcrQl0Ag2ub_YUl-M7M8AyLsILWEP0EDQwFM26BUadmb2kd0vwi_eqXc5S7C98oMDfMrrJPmCVib9t2r1k2X2pkF4d0dE9FA_PhNyNeEf-tzeATECpQs4AGCJtsiqMfXmN0ejla3hjk9XP5lbK8w5cPPS381zoKVuGe7lT5C76_9gpSTphfgRMViG05rBxyK0nMK7ssvduIibBlTSozyPixS06ZmeX_JQ"
os.environ['MD_AUTH_TOKEN'] = TOKEN

try:
    import morningstar_data as md
    print("✅ Connected to Morningstar Direct\n")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Expanded test universe - mix of biotech companies at different stages
BIOTECH_TICKERS = [
    # Large cap commercial
    "VRTX", "REGN", "BIIB", "AMGN", "GILD",
    # Mid cap commercial
    "IONS", "BMRN", "ALNY", "SRPT", "EXEL",
    # Small/mid cap clinical
    "KRTX", "RARE", "BLUE", "NTLA", "CRSP",
    # Smaller biotech
    "BEAM", "FOLD", "RGNX", "EDIT", "VRTX"
]

print("="*70)
print("BIOTECH UNIVERSE COVERAGE TEST")
print("="*70)
print(f"\nTesting {len(BIOTECH_TICKERS)} biotech tickers")
print("Goal: Identify data coverage rate for key financial metrics\n")

# Step 1: Find all securities
print("Step 1: Searching for securities...")
sec_ids = {}
failed_searches = []

for ticker in BIOTECH_TICKERS:
    try:
        results = md.direct.investments(keyword=ticker, count=5)
        if not results.empty:
            exact = results[results['Ticker'].str.upper() == ticker.upper()]
            if not exact.empty:
                sec_ids[ticker] = exact.iloc[0]['SecId']
                print(f"  ✓ {ticker}")
            else:
                failed_searches.append(ticker)
                print(f"  ✗ {ticker} - no exact match")
        else:
            failed_searches.append(ticker)
            print(f"  ✗ {ticker} - not found")
    except Exception as e:
        failed_searches.append(ticker)
        print(f"  ✗ {ticker} - error: {e}")

print(f"\n✅ Found {len(sec_ids)}/{len(BIOTECH_TICKERS)} tickers")
if failed_searches:
    print(f"❌ Failed: {', '.join(failed_searches)}")

if not sec_ids:
    print("\n❌ No securities found. Exiting.")
    sys.exit(1)

# Step 2: Test different data point names for critical metrics
print("\n" + "="*70)
print("Step 2: Testing data point availability")
print("="*70)

# These are potential data point names - we'll test which ones work
data_point_tests = {
    'Cash': [
        'CashAndEquivalents',
        'Cash',
        'CashAndCashEquivalents', 
        'TotalCash',
        'CashAndShortTermInvestments'
    ],
    'Debt': [
        'TotalDebt',
        'LongTermDebt',
        'TotalLiabilities',
        'DebtToEquity',
        'NetDebt'
    ],
    'Revenue': [
        'Revenue',
        'TotalRevenue',
        'Sales',
        'TTMRevenue',
        'RevenueGrowth'
    ],
    'Market Cap': [
        'MarketCap',
        'MarketCapitalization',
        'EquityMarketCap'
    ],
    'Other Key Metrics': [
        'TotalAssets',
        'NetIncome',
        'OperatingIncome',
        'SharesOutstanding',
        'EnterpriseValue'
    ]
}

# Test with one ticker
test_ticker = list(sec_ids.keys())[0]
test_sec_id = sec_ids[test_ticker]
print(f"\nTesting with {test_ticker}...\n")

working_data_points = []

for category, point_names in data_point_tests.items():
    print(f"{category}:")
    for point_name in point_names:
        try:
            data = md.direct.get_investment_data(
                investments=[test_sec_id],
                data_points=[point_name]
            )
            if not data.empty and point_name in data.columns:
                value = data[point_name].iloc[0]
                if pd.notna(value):
                    print(f"  ✅ {point_name:35s} = {value}")
                    working_data_points.append(point_name)
                else:
                    print(f"  ⚠️  {point_name:35s} = NULL")
            else:
                print(f"  ❌ {point_name:35s} - not available")
        except Exception as e:
            print(f"  ❌ {point_name:35s} - error: {str(e)[:50]}")
    print()

if not working_data_points:
    print("❌ No working data points found!")
    print("\nTry checking Morningstar Direct documentation for correct field names:")
    print("Help > Morningstar Data Python Reference > Available Data Points")
    sys.exit(1)

# Step 3: Test coverage across all tickers
print("="*70)
print("Step 3: Coverage test across all tickers")
print("="*70)

print(f"\nTesting {len(working_data_points)} working data points on {len(sec_ids)} tickers...")

all_sec_ids = list(sec_ids.values())

try:
    data = md.direct.get_investment_data(
        investments=all_sec_ids,
        data_points=working_data_points
    )
    
    print(f"\n✅ Retrieved data for {len(data)} securities\n")
    
    # Calculate coverage statistics
    print("Coverage Statistics:")
    print("-" * 70)
    
    for col in working_data_points:
        if col in data.columns:
            total = len(data)
            non_null = data[col].notna().sum()
            coverage_pct = (non_null / total * 100) if total > 0 else 0
            
            print(f"  {col:35s} {non_null:3d}/{total:3d} ({coverage_pct:5.1f}%)")
    
    # Show the actual data
    print("\n" + "="*70)
    print("Sample Data (first 10 tickers):")
    print("="*70)
    
    # Map sec_id back to ticker for display
    id_to_ticker = {v: k for k, v in sec_ids.items()}
    data['Ticker'] = data['SecId'].map(id_to_ticker)
    
    display_cols = ['Ticker'] + working_data_points[:5]  # Show first 5 metrics
    print(data[display_cols].head(10).to_string())
    
    # Export results
    output_file = "morningstar_coverage_test.csv"
    data.to_csv(output_file, index=False)
    print(f"\n✅ Full results saved to: {output_file}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Coverage summary and recommendations
print("\n" + "="*70)
print("WAKE ROBIN INTEGRATION RECOMMENDATION")
print("="*70)

print("""
Compare these results with your current Yahoo Finance coverage:

Current Wake Robin gaps (from project docs):
- Cash coverage: ~40% (60% missing)
- Debt coverage: ~20% (80% missing)

Questions to answer:
1. Does Morningstar provide >80% coverage for cash/debt?
2. Is the data fresher than Yahoo Finance?
3. Are the data points reliable and consistent?

If Morningstar coverage is significantly better (>20% improvement),
then integration makes sense and will reduce false positives.

If coverage is similar or worse, stick with current free sources.

Next step: Review the coverage statistics above and decide on integration.
""")
