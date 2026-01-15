import json

# Load universe
universe = json.load(open('production_data/universe.json'))

# Build CUSIP map from universe
cusip_map = {}
for stock in universe:
    ticker = stock.get('ticker', '')
    cusip = stock.get('cusip', '')
    if ticker and cusip and ticker != '_XBI_BENCHMARK_':
        # Normalize CUSIP (9 chars)
        cusip_clean = cusip.strip()[:9]
        cusip_map[cusip_clean] = {
            'ticker': ticker,
            'cusip': cusip_clean,
            'name': stock.get('name', ''),
            'source': 'universe',
            'mapped_at': '2026-01-09T16:00:00'
        }

# Merge with existing manual entries
try:
    existing = json.load(open('production_data/cusip_static_map.json'))
    for cusip, data in existing.items():
        if cusip not in cusip_map:
            cusip_map[cusip] = data
except:
    pass

# Save
with open('production_data/cusip_static_map.json', 'w') as f:
    json.dump(cusip_map, f, indent=2)

print(f'Created CUSIP map with {len(cusip_map)} entries')
print(f'  From universe: {sum(1 for v in cusip_map.values() if v.get("source") == "universe")}')
print(f'  Manual: {sum(1 for v in cusip_map.values() if v.get("source") != "universe")}')
