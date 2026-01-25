# Add this function to your extractor (or run standalone)
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_manager_pipeline(manager_name, cik, quarter_end, cusip_map, universe):
    """Show where holdings drop off in the pipeline"""
    from edgar_13f_extractor_CORRECTED import (
        find_13f_filing,
        fetch_information_table_xml,
        parse_information_table
    )

    print(f"\n{'='*70}")
    print(f"Pipeline Analysis: {manager_name} (Q{quarter_end})")
    print(f"{'='*70}")

    # Step 1: Find filing
    filing = find_13f_filing(cik, quarter_end)
    if not filing:
        print("? No 13F filing found")
        return
    print(f"? Filing found: {filing['accession']}")

    # Step 2: Fetch XML
    xml_content, _ = fetch_information_table_xml(cik, filing['accession'])
    if not xml_content:
        print("? XML not found")
        return
    print(f"? XML fetched")

    # Step 3: Parse holdings
    holdings = parse_information_table(xml_content)
    print(f"? Parsed {len(holdings)} holdings")

    # Step 4: CUSIP mapping
    cusips_in_map = sum(1 for h in holdings if h.get('cusip') in cusip_map)
    print(f"  ? {cusips_in_map}/{len(holdings)} CUSIPs in map ({cusips_in_map/len(holdings)*100:.1f}%)")

    # Step 5: Universe matching
    universe_tickers = {s['ticker'] for s in universe if s.get('ticker')}
    matched = 0
    for h in holdings:
        cusip = h.get('cusip')
        if cusip in cusip_map:
            ticker = cusip_map[cusip]
            if ticker in universe_tickers:
                matched += 1

    print(f"  ? {matched}/{len(holdings)} matched to universe ({matched/len(holdings)*100:.1f}%)")

    # Show top unmapped CUSIPs
    unmapped = [h['cusip'] for h in holdings if h.get('cusip') not in cusip_map][:10]
    if unmapped:
        print(f"\nTop 10 unmapped CUSIPs: {', '.join(unmapped)}")

# Run for all managers
with open('production_data/manager_registry.json') as f:
    registry = json.load(f)

with open('production_data/cusip_static_map.json') as f:
    cusip_map = json.load(f)

with open('production_data/universe.json') as f:
    universe = json.load(f)

for mgr in registry['elite_core']:
    try:
        analyze_manager_pipeline(mgr['name'], mgr['cik'], '2024-09-30', cusip_map, universe)
    except Exception as e:
        logger.error(f"Error analyzing {mgr['name']}: {e}", exc_info=True)
