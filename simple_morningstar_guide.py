"""
Simple Morningstar Direct Data Access - Using Known Working Methods
Based on Morningstar Data Python Package v1.13.0
"""
import os
import sys

# Set your token here
TOKEN = "YOUR_TOKEN_HERE"  # Replace with your current token
os.environ['MD_AUTH_TOKEN'] = TOKEN

try:
    import morningstar_data as md
    print("✅ Package imported successfully\n")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

print("="*70)
print("METHOD 1: List Available Data Sets (WE KNOW THIS WORKS)")
print("="*70)

try:
    datasets = md.direct.get_morningstar_data_sets()
    print(f"\n✅ Found {len(datasets)} data sets")
    
    print("\nAll available data sets:")
    for i, ds in enumerate(datasets, 1):
        print(f"  {i:3d}. {ds}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*70)
print("METHOD 2: Check Official Documentation")
print("="*70)
print("""
To find the correct API methods for YOUR Morningstar Direct license:

1. In Morningstar Direct, go to the top menu
2. Click: Help > Morningstar Data Python Reference
3. This opens the documentation specific to your installation

Key sections to review:
- "Investment Data" - methods to get securities
- "Data Sets" - available financial data
- "Lookup" - searching for securities
- "Holdings Data" - portfolio holdings

Common patterns in Morningstar API:
- md.direct.get_investment_data() - get data for specific securities
- md.InvestmentIdentifier() - create security identifiers
- md.direct.search_securities() - might be the search method
- md.direct.get_data() - generic data retrieval
""")

print("\n" + "="*70)
print("METHOD 3: Example from Morningstar Docs")
print("="*70)
print("""
According to PyPI page, typical usage looks like:

    from morningstar_data import InvestmentList
    
    # Create a list of investments
    investment_list = InvestmentList(
        name="MyBiotechList",
        identifiers=["VRTX", "IONS", "BMRN"]
    )
    
    # Get data for the list
    data = md.direct.get_investment_list_data(
        investment_list=investment_list,
        data_set="FinancialHealth"  # or other dataset name
    )

But you need to check the docs for exact method names and parameters.
""")

print("\n" + "="*70)
print("RECOMMENDED NEXT STEPS")
print("="*70)
print("""
1. Run the introspect script:
   python introspect_morningstar.py
   
   This will show ALL available methods in your installation
   
2. Check the official documentation:
   Help > Morningstar Data Python Reference
   
3. Look for these specific capabilities:
   - How to search for a security by ticker
   - How to get financial statement data (balance sheet, income statement)
   - How to retrieve key statistics (market cap, cash, debt)
   - What data sets contain biotech-relevant data
   
4. Once you find the right methods, we'll build the Wake Robin wrapper

The key is finding the correct method names that YOUR version/license supports.
""")
