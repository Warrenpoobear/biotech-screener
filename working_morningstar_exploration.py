"""
Explore Morningstar Direct Financial Data - Using REAL API Methods
Based on actual introspection results
"""
import os
import sys
import pandas as pd

# Set your token here
TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1EY3hOemRHTnpGRFJrSTRPRGswTmtaRU1FSkdOekl5TXpORFJrUTROemd6TWtOR016bEdOdyJ9.eyJodHRwczovL21vcm5pbmdzdGFyLmNvbS9lbWFpbCI6ImRzY2h1bHpAYnJvb2tzLnVzLmNvbSIsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL3JvbGUiOlsiRW5hYmxlZCBBbmFseXRpY3MgTGFiIERlbGl2ZXJ5IE5vdGVib29rcyIsIkRpc2FibGUgRGVmaW5lZCBDb250cmlidXRpb24gUGxhbnMiLCJQUkJpdFNldHRpbmciLCJQb3J0Zm9saW8gQW5hbHlzaXMgVXNlciIsIlBlcnNvbmEuRGlyZWN0Rm9yQXNzZXRNYW5hZ2VtZW50IiwiTGljZW5zZS5QcmVzZW50YXRpb25TdHVkaW8iXSwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vY29tcGFueV9pZCI6IjFhNzc5M2Q4LTE5ODEtNDA1Ni05ODA5LTI5ZjZlOThmNDc3MSIsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL2xlZ2FjeV9jb21wYW55X2lkIjoiMWE3NzkzZDgtMTk4MS00MDU2LTk4MDktMjlmNmU5OGY0NzcxIiwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vcm9sZV9pZCI6WyI3OGJhMWFlNy0xZWUzLTQ0YTAtYTAxOC0wOGM1NThmZWNmMTciLCI4MjYyOWNkMC1kZjgwLTRlNWMtYjNiYS02YmQyNWU5MzBhNDIiLCJmNDdhZjlmMC03NWU0LTQzY2ItYWM2Mi1kMjRmZmQ2NzU1ODgiLCJkYzMxN2Q5OC0xMTAwLTQyM2YtOTUzZi1mZjRkYjc4MzUwMTgiXSwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vcHJvZHVjdCI6WyJESVJFQ1QiLCJQUyJdLCJodHRwczovL21vcm5pbmdzdGFyLmNvbS9jb21wYW55IjpbeyJpZCI6IjFhNzc5M2Q4LTE5ODEtNDA1Ni05ODA5LTI5ZjZlOThmNDc3MSIsInByb2R1Y3QiOiJESVJFQ1QifV0sImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL21zdGFyX2lkIjoiRjU4ODkzNkQtQkY1Ri00QTAyLUI5RjctRjE1RDhFMzYyRTAyIiwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vZW1haWxfdmVyaWZpZWQiOnRydWUsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL3Bhc3N3b3JkQ2hhbmdlUmVxdWlyZWQiOmZhbHNlLCJodHRwczovL21vcm5pbmdzdGFyLmNvbS91aW1fcm9sZXMiOiJNRF9NRU1CRVJfMV8xLERJUkVDVCIsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL2xhc3RfcGFzc3dvcmRfcmVzZXQiOiIyMDI1LTA4LTEzVDE4OjI4OjI1LjQ5NFoiLCJpc3MiOiJodHRwczovL2xvZ2luLXByb2QubW9ybmluZ3N0YXIuY29tLyIsInN1YiI6ImF1dGgwfEY1ODg5MzZELUJGNUYtNEEwMi1COUY3LUYxNUQ4RTM2MkUwMiIsImF1ZCI6WyJodHRwczovL3VpbS1wcm9kLm1vcm5pbmdzdGFyLmF1dGgwLmNvbS9hcGkvdjIvIiwiaHR0cHM6Ly91aW0tcHJvZC5tb3JuaW5nc3Rhci5hdXRoMC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNzY4MjM4MDc0LCJleHAiOjE3NjgzMjQ0NzQsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgb2ZmbGluZV9hY2Nlc3MiLCJhenAiOiJDaGVLTTR1ajhqUFQ2MGFVMkk0Y1BsSDhyREtkT3NaZCJ9.dYahlgJ6rVHy18LCcmEcvyAcnolVq3KjVTeb73ylMcV0Tsj5vutJPiSvx_Y-Z2I1j5OGpqUJnhckJVGc6nVhZl8mpJ_VRdfCD_JiPzxGRafQaqbeKeIaLrcrQl0Ag2ub_YUl-M7M8AyLsILWEP0EDQwFM26BUadmb2kd0vwi_eqXc5S7C98oMDfMrrJPmCVib9t2r1k2X2pkF4d0dE9FA_PhNyNeEf-tzeATECpQs4AGCJtsiqMfXmN0ejla3hjk9XP5lbK8w5cPPS381zoKVuGe7lT5C76_9gpSTphfgRMViG05rBxyK0nMK7ssvduIibBlTSozyPixS06ZmeX_JQ"  # Replace with your current token
os.environ['MD_AUTH_TOKEN'] = TOKEN

try:
    import morningstar_data as md
    print("✅ Connected to Morningstar Direct\n")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    sys.exit(1)

# Test tickers - biotech stocks from your project
TEST_TICKERS = ["VRTX", "KRTX", "IONS", "BMRN", "RARE"]

print("="*70)
print("STEP 1: SEARCH FOR BIOTECH SECURITIES")
print("="*70)

sec_ids = {}
for ticker in TEST_TICKERS:
    print(f"\nSearching for {ticker}...")
    try:
        # Use the REAL API method we discovered
        results = md.direct.investments(keyword=ticker, count=5)
        
        if not results.empty:
            # Filter for exact ticker match
            exact_matches = results[results['Ticker'].str.upper() == ticker.upper()]
            
            if not exact_matches.empty:
                sec_id = exact_matches.iloc[0]['SecId']
                name = exact_matches.iloc[0]['Name']
                sec_ids[ticker] = sec_id
                print(f"  ✅ Found: {name}")
                print(f"     SecId: {sec_id}")
                print(f"     Ticker: {exact_matches.iloc[0]['Ticker']}")
            else:
                print(f"  ⚠️  Found results but no exact match")
                print(f"     Top result: {results.iloc[0]['Name']}")
        else:
            print(f"  ❌ No results found")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

print(f"\n✅ Successfully found {len(sec_ids)} out of {len(TEST_TICKERS)} tickers")

if not sec_ids:
    print("\n❌ No securities found. Cannot continue.")
    sys.exit(1)

print("\n" + "="*70)
print("STEP 2: LIST AVAILABLE DATA SETS")
print("="*70)

try:
    datasets = md.direct.get_morningstar_data_sets()
    print(f"\nTotal datasets: {len(datasets)}")
    
    # Look for financially relevant datasets
    financial_keywords = ['financial', 'balance', 'income', 'cash', 'fundamental', 
                         'key', 'statistic', 'ratio', 'health']
    
    print("\nFinancially relevant datasets:")
    for idx, row in datasets.iterrows():
        dataset_id = row['datasetId']
        dataset_name = row['name']
        
        # Check if any financial keyword is in the name
        if any(keyword in dataset_name.lower() for keyword in financial_keywords):
            print(f"  • {dataset_id:40s} - {dataset_name}")
            
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*70)
print("STEP 3: RETRIEVE FINANCIAL DATA FOR ONE TICKER")
print("="*70)

# Pick the first found ticker
test_ticker = list(sec_ids.keys())[0]
test_sec_id = sec_ids[test_ticker]

print(f"\nTesting data retrieval for {test_ticker} (SecId: {test_sec_id})")

# Try different data point combinations
data_point_attempts = [
    # Attempt 1: Simple market data
    {
        'name': 'Market Data',
        'data_points': ['Name', 'Ticker', 'ClosePrice', 'MarketCap']
    },
    # Attempt 2: Financial health metrics
    {
        'name': 'Financial Health',
        'data_points': ['TotalAssets', 'TotalDebt', 'Cash', 'Revenue']
    },
    # Attempt 3: Key statistics
    {
        'name': 'Key Stats',
        'data_points': ['NetIncome', 'OperatingIncome', 'GrossMargin', 'DebtToEquity']
    },
    # Attempt 4: Use a dataset ID
    {
        'name': 'Via Dataset',
        'data_points': 'KeyStatistics'  # Try using dataset name directly
    }
]

for attempt in data_point_attempts:
    print(f"\n[{attempt['name']}]")
    try:
        data = md.direct.get_investment_data(
            investments=[test_sec_id],
            data_points=attempt['data_points']
        )
        
        if not data.empty:
            print(f"  ✅ Retrieved {len(data.columns)} data points")
            print("\nAvailable columns:")
            for col in data.columns:
                value = data[col].iloc[0] if len(data) > 0 else None
                print(f"    • {col:30s} = {value}")
        else:
            print("  ⚠️  No data returned")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*70)
print("STEP 4: TRY BULK RETRIEVAL FOR ALL TICKERS")
print("="*70)

print(f"\nAttempting to get data for all {len(sec_ids)} tickers at once...")

try:
    all_sec_ids = list(sec_ids.values())
    
    # Try with simple data points first
    data = md.direct.get_investment_data(
        investments=all_sec_ids,
        data_points=['Name', 'Ticker', 'ClosePrice', 'MarketCap']
    )
    
    print(f"✅ Retrieved data for {len(data)} securities")
    print("\nSummary:")
    print(data[['Name', 'Ticker', 'ClosePrice', 'MarketCap']].to_string())
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*70)
print("NEXT STEPS FOR WAKE ROBIN INTEGRATION")
print("="*70)
print("""
Based on this exploration, we now know:

1. ✅ How to search for securities: md.direct.investments(keyword=ticker)
2. ✅ How to get data: md.direct.get_investment_data(investments, data_points)
3. ✅ What datasets are available: md.direct.get_morningstar_data_sets()

To complete the integration, we need to:

A. Identify the EXACT data points for:
   - Cash and cash equivalents
   - Total debt
   - Market capitalization
   - Revenue (TTM)
   - Operating expenses
   - Shares outstanding

B. Test coverage rate:
   - Try this on 20-30 biotech tickers
   - Calculate % with complete data
   - Compare to Yahoo Finance coverage

C. Check data freshness:
   - How stale is the data?
   - When was it last updated?

D. Build the Wake Robin wrapper with:
   - Tier-0 provenance tracking
   - Coverage waterfall (Morningstar → Yahoo → SEC)
   - Point-in-time discipline
   - Atomic file operations

Run this script and share the results. We'll use them to build the production code.
""")
