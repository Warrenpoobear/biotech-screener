import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load universe
with open('production_data/universe.json') as f:
    universe = json.load(f)

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
    with open('production_data/cusip_static_map.json') as f:
        existing = json.load(f)
    for cusip, data in existing.items():
        if cusip not in cusip_map:
            cusip_map[cusip] = data
except FileNotFoundError:
    logger.info("No existing cusip_static_map.json found, creating new one")
except json.JSONDecodeError as e:
    logger.warning(f"Failed to parse existing cusip_static_map.json: {e}")

# Save
with open('production_data/cusip_static_map.json', 'w') as f:
    json.dump(cusip_map, f, indent=2)

print(f'Created CUSIP map with {len(cusip_map)} entries')
print(f'  From universe: {sum(1 for v in cusip_map.values() if v.get("source") == "universe")}')
print(f'  Manual: {sum(1 for v in cusip_map.values() if v.get("source") != "universe")}')
