"""
Test Morningstar Direct API connection
Run this at work with VPN/network access to Morningstar Direct
"""
import os
import sys

# Set authentication token from environment or hardcode for testing
# Replace with your actual token copied from Analytics Lab
TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1EY3hOemRHTnpGRFJrSTRPRGswTmtaRU1FSkdOekl5TXpORFJrUTROemd6TWtOR016bEdOdyJ9.eyJodHRwczovL21vcm5pbmdzdGFyLmNvbS9lbWFpbCI6ImRzY2h1bHpAYnJvb2tzLnVzLmNvbSIsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL3JvbGUiOlsiRW5hYmxlZCBBbmFseXRpY3MgTGFiIERlbGl2ZXJ5IE5vdGVib29rcyIsIkRpc2FibGUgRGVmaW5lZCBDb250cmlidXRpb24gUGxhbnMiLCJQUkJpdFNldHRpbmciLCJQb3J0Zm9saW8gQW5hbHlzaXMgVXNlciIsIlBlcnNvbmEuRGlyZWN0Rm9yQXNzZXRNYW5hZ2VtZW50IiwiTGljZW5zZS5QcmVzZW50YXRpb25TdHVkaW8iXSwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vY29tcGFueV9pZCI6IjFhNzc5M2Q4LTE5ODEtNDA1Ni05ODA5LTI5ZjZlOThmNDc3MSIsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL2xlZ2FjeV9jb21wYW55X2lkIjoiMWE3NzkzZDgtMTk4MS00MDU2LTk4MDktMjlmNmU5OGY0NzcxIiwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vcm9sZV9pZCI6WyI3OGJhMWFlNy0xZWUzLTQ0YTAtYTAxOC0wOGM1NThmZWNmMTciLCI4MjYyOWNkMC1kZjgwLTRlNWMtYjNiYS02YmQyNWU5MzBhNDIiLCJmNDdhZjlmMC03NWU0LTQzY2ItYWM2Mi1kMjRmZmQ2NzU1ODgiLCJkYzMxN2Q5OC0xMTAwLTQyM2YtOTUzZi1mZjRkYjc4MzUwMTgiXSwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vcHJvZHVjdCI6WyJESVJFQ1QiLCJQUyJdLCJodHRwczovL21vcm5pbmdzdGFyLmNvbS9jb21wYW55IjpbeyJpZCI6IjFhNzc5M2Q4LTE5ODEtNDA1Ni05ODA5LTI5ZjZlOThmNDc3MSIsInByb2R1Y3QiOiJESVJFQ1QifV0sImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL21zdGFyX2lkIjoiRjU4ODkzNkQtQkY1Ri00QTAyLUI5RjctRjE1RDhFMzYyRTAyIiwiaHR0cHM6Ly9tb3JuaW5nc3Rhci5jb20vZW1haWxfdmVyaWZpZWQiOnRydWUsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL3Bhc3N3b3JkQ2hhbmdlUmVxdWlyZWQiOmZhbHNlLCJodHRwczovL21vcm5pbmdzdGFyLmNvbS91aW1fcm9sZXMiOiJNRF9NRU1CRVJfMV8xLERJUkVDVCIsImh0dHBzOi8vbW9ybmluZ3N0YXIuY29tL2xhc3RfcGFzc3dvcmRfcmVzZXQiOiIyMDI1LTA4LTEzVDE4OjI4OjI1LjQ5NFoiLCJpc3MiOiJodHRwczovL2xvZ2luLXByb2QubW9ybmluZ3N0YXIuY29tLyIsInN1YiI6ImF1dGgwfEY1ODg5MzZELUJGNUYtNEEwMi1COUY3LUYxNUQ4RTM2MkUwMiIsImF1ZCI6WyJodHRwczovL3VpbS1wcm9kLm1vcm5pbmdzdGFyLmF1dGgwLmNvbS9hcGkvdjIvIiwiaHR0cHM6Ly91aW0tcHJvZC5tb3JuaW5nc3Rhci5hdXRoMC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNzY4MjM4MDc0LCJleHAiOjE3NjgzMjQ0NzQsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgb2ZmbGluZV9hY2Nlc3MiLCJhenAiOiJDaGVLTTR1ajhqUFQ2MGFVMkk0Y1BsSDhyREtkT3NaZCJ9.dYahlgJ6rVHy18LCcmEcvyAcnolVq3KjVTeb73ylMcV0Tsj5vutJPiSvx_Y-Z2I1j5OGpqUJnhckJVGc6nVhZl8mpJ_VRdfCD_JiPzxGRafQaqbeKeIaLrcrQl0Ag2ub_YUl-M7M8AyLsILWEP0EDQwFM26BUadmb2kd0vwi_eqXc5S7C98oMDfMrrJPmCVib9t2r1k2X2pkF4d0dE9FA_PhNyNeEf-tzeATECpQs4AGCJtsiqMfXmN0ejla3hjk9XP5lbK8w5cPPS381zoKVuGe7lT5C76_9gpSTphfgRMViG05rBxyK0nMK7ssvduIibBlTSozyPixS06ZmeX_JQ"

os.environ['MD_AUTH_TOKEN'] = TOKEN

# Check if morningstar_data is installed
try:
    import morningstar_data as md
    print("✅ morningstar_data package imported successfully")
except ImportError as e:
    print(f"❌ Failed to import morningstar_data: {e}")
    print("\nPlease install with: pip install morningstar-data --break-system-packages")
    sys.exit(1)

# Test connection and list available datasets
print("\nTesting connection to Morningstar Direct...")
try:
    datasets = md.direct.get_morningstar_data_sets()
    print(f"✅ Successfully connected to Morningstar Direct")
    print(f"Available datasets: {len(datasets)}")
    print("\nFirst 5 datasets:")
    for i, ds in enumerate(datasets[:5]):
        print(f"  {i+1}. {ds}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nPossible issues:")
    print("  - Token expired (valid 24 hours)")
    print("  - Not connected to corporate network/VPN")
    print("  - Invalid Morningstar Direct license")
    sys.exit(1)

# Test pulling data for a specific ticker
print("\n" + "="*60)
print("Testing data retrieval for a sample ticker (AAPL)...")
print("="*60)

try:
    # Try to search for a security
    from morningstar_data import InvestmentIdentifier
    
    # Search for AAPL
    ticker = "AAPL"
    results = md.direct.search_investments(
        term=ticker,
        investment_type="CommonStock",
        limit=1
    )
    
    if results:
        print(f"✅ Found {ticker}")
        print(f"   Investment ID: {results[0].get('SecId', 'N/A')}")
        print(f"   Name: {results[0].get('Name', 'N/A')}")
        
        # Try to get some basic data
        sec_id = results[0].get('SecId')
        if sec_id:
            print(f"\n   Attempting to retrieve financial data...")
            # This will depend on what data sets are available in your license
            # We'll just confirm the search works for now
    else:
        print(f"⚠️  No results found for {ticker}")
        
except Exception as e:
    print(f"⚠️  Data retrieval test encountered an issue: {e}")
    print("   This may be normal if certain data sets aren't in your license")

print("\n" + "="*60)
print("Basic connectivity test complete!")
print("="*60)
