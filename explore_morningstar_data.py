"""
Explore Morningstar Direct data availability for biotech stocks
This script will help determine what financial data we can use for Wake Robin
"""
import os
import sys
import json
from pprint import pprint

# Set your token here (update with fresh token daily)
TOKEN = "YOUR_TOKEN_HERE"  # Replace with your actual token
os.environ['MD_AUTH_TOKEN'] = TOKEN

try:
    import morningstar_data as md
    print("✅ Connected to Morningstar Direct\n")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    sys.exit(1)

# Test tickers - mix of biotech stocks
TEST_TICKERS = [
    "VRTX",  # Vertex - large cap commercial
    "KRTX",  # Karuna (mentioned in your docs)
    "IONS",  # Ionis - established biotech
    "BMRN",  # BioMarin - commercial stage
    "RARE",  # Ultragenyx - smaller biotech
]

print("="*70)
print("EXPLORING AVAILABLE DATASETS")
print("="*70)

try:
    datasets = md.direct.get_morningstar_data_sets()
    print(f"\nTotal datasets available: {len(datasets)}")
    print("\nDataset names (first 20):")
    for i, ds in enumerate(datasets[:20], 1):
        # Try to get the actual name from the dataset object
        if isinstance(ds, dict):
            name = ds.get('name', ds.get('datasetId', str(ds)))
        else:
            name = str(ds)
        print(f"  {i:2d}. {name}")
    
    if len(datasets) > 20:
        print(f"  ... and {len(datasets) - 20} more")
except Exception as e:
    print(f"Error listing datasets: {e}")

print("\n" + "="*70)
print("TESTING DATA RETRIEVAL FOR BIOTECH TICKERS")
print("="*70)

for ticker in TEST_TICKERS:
    print(f"\n{'='*70}")
    print(f"Ticker: {ticker}")
    print(f"{'='*70}")
    
    try:
        # Method 1: Try direct search
        print(f"\n[1] Searching for {ticker}...")
        search_results = md.direct.search_investments(
            term=ticker,
            investment_type="Stock"
        )
        
        if search_results:
            result = search_results[0] if isinstance(search_results, list) else search_results
            print(f"✅ Found: {result.get('Name', 'Unknown')}")
            sec_id = result.get('SecId') or result.get('secId')
            print(f"   SecId: {sec_id}")
            
            if sec_id:
                # Try to get fundamental data
                print(f"\n[2] Attempting to retrieve fundamental data...")
                
                try:
                    # Try getting key stats/financials
                    # Note: exact method names may vary based on API version
                    financials = md.direct.get_investment_data(
                        identifiers=[sec_id],
                        data_points=[
                            "Name",
                            "MarketCap",
                            "TotalAssets", 
                            "TotalDebt",
                            "Cash",
                            "Revenue",
                            "NetIncome"
                        ]
                    )
                    
                    print("✅ Retrieved financial data:")
                    if isinstance(financials, dict):
                        for key, value in financials.items():
                            print(f"   {key}: {value}")
                    else:
                        print(f"   {financials}")
                        
                except AttributeError as e:
                    print(f"⚠️  Method 'get_investment_data' not available: {e}")
                    print("   Trying alternative methods...")
                    
                    # Try alternative data access methods
                    try:
                        # Check if there's a get_data method
                        data = md.direct.get_data(
                            security_id=sec_id,
                            data_set="KeyStatistics"
                        )
                        print(f"✅ Retrieved via get_data: {data}")
                    except Exception as e2:
                        print(f"⚠️  Alternative method failed: {e2}")
                        
                except Exception as e:
                    print(f"⚠️  Could not retrieve financial data: {e}")
        else:
            print(f"❌ No results found for {ticker}")
            
    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("API EXPLORATION COMPLETE")
print("="*70)
print("\nNext steps:")
print("1. Review which data points are available above")
print("2. Check Morningstar Direct documentation for your license")
print("3. Identify the correct API methods for financial data")
print("4. We'll build a custom wrapper for Wake Robin integration")
