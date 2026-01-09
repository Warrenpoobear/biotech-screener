import json
import sys
from pathlib import Path

# Load the mapper's query function
sys.path.insert(0, '.')
from cusip_mapper import CUSIPMapper

mapper = CUSIPMapper('production_data')

# Read CUSIPs
cusips = Path('baker_bros_cusips.txt').read_text().splitlines()

mapped = 0
for cusip in cusips:
    result = mapper.query(cusip)
    if result:
        mapped += 1
        print(f'{cusip} -> {result.get("ticker", "?")}')

print(f'\nMapped {mapped}/{len(cusips)}')
print(f'Cache now has {len(mapper.cache)} entries')
