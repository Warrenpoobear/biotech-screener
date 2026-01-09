import json
import urllib.request
import time

# Load the CUSIPs we extracted from Baker Bros
cusips = open('baker_bros_cusips.txt').read().splitlines()
print(f'Mapping {len(cusips)} CUSIPs from Baker Bros via OpenFIGI...')

# Build batch requests (5 CUSIPs per request)
cusip_map = {}
batch_size = 5

for i in range(0, len(cusips), batch_size):
    batch = cusips[i:i+batch_size]
    
    # Build OpenFIGI request (CUSIP ? ticker direction)
    jobs = [{"idType": "ID_CUSIP", "idValue": cusip} for cusip in batch]
    payload = json.dumps(jobs).encode()
    
    req = urllib.request.Request(
        'https://api.openfigi.com/v3/mapping',
        data=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            results = json.loads(response.read())
            
            for j, result in enumerate(results):
                cusip = batch[j]
                if 'data' in result:
                    for item in result['data']:
                        if item.get('securityType') == 'Common Stock':
                            ticker = item.get('ticker')
                            if ticker:
                                cusip_map[cusip] = ticker
                                print(f'{len(cusip_map)}/{len(cusips)}: {cusip} -> {ticker}')
                                break
    except Exception as e:
        print(f'Batch {i//batch_size + 1} FAILED: {e}')
    
    time.sleep(1.5)  # Rate limit
    
# Save as simple string map
with open('production_data/cusip_static_map.json', 'w') as f:
    json.dump(cusip_map, f, indent=2)

print(f'\nMapped {len(cusip_map)}/{len(cusips)} CUSIPs')
