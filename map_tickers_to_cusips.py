import json
import logging
import urllib.request
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('production_data/universe.json') as f:
    universe = json.load(f)
tickers = [s['ticker'] for s in universe if s.get('ticker') and s['ticker'] != '_XBI_BENCHMARK_']

print(f'Mapping {len(tickers)} tickers to CUSIPs via OpenFIGI...')

cusip_map = {}
for i, ticker in enumerate(tickers):
    # OpenFIGI ticker lookup
    payload = json.dumps([{"idType": "TICKER", "idValue": ticker, "exchCode": "US"}]).encode()
    req = urllib.request.Request(
        'https://api.openfigi.com/v3/mapping',
        data=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read())
            if result and 'data' in result[0]:
                for item in result[0]['data']:
                    if item.get('securityType') == 'Common Stock':
                        cusip = item.get('compositeFIGI', '')[-9:]  # Last 9 chars
                        if cusip:
                            cusip_map[cusip] = {
                                'ticker': ticker,
                                'cusip': cusip,
                                'name': item.get('name', ''),
                                'source': 'openfigi',
                                'mapped_at': '2026-01-09T16:30:00'
                            }
                            print(f'{i+1}/{len(tickers)}: {ticker} -> {cusip}')
                            break
    except Exception as e:
        print(f'{i+1}/{len(tickers)}: {ticker} FAILED: {e}')
    
    time.sleep(0.25)  # Rate limit
    
    if (i+1) % 50 == 0:
        print(f'Progress: {i+1}/{len(tickers)} mapped')

# Save
with open('production_data/cusip_static_map.json', 'w') as f:
    json.dump(cusip_map, f, indent=2)

print(f'\nFinal: {len(cusip_map)} CUSIPs mapped from {len(tickers)} tickers')
