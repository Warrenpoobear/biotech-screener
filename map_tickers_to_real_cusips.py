import json
import urllib.request
import time

universe = json.load(open('production_data/universe.json'))
tickers = [s['ticker'] for s in universe if s.get('ticker') and s['ticker'] != '_XBI_BENCHMARK_']

print(f'Mapping {len(tickers)} tickers to REAL CUSIPs...')

cusip_map = {}
for i, ticker in enumerate(tickers):
    payload = json.dumps([{"idType": "TICKER", "idValue": ticker, "exchCode": "US"}]).encode()
    req = urllib.request.Request(
        'https://api.openfigi.com/v3/mapping',
        data=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read())
                if result and 'data' in result[0]:
                    for item in result[0]['data']:
                        # Look for the actual CUSIP field
                        cusip = item.get('cusip')  # Real CUSIP field
                        if cusip and item.get('securityType') == 'Common Stock':
                            cusip_map[cusip] = {
                                'ticker': ticker,
                                'cusip': cusip,
                                'name': item.get('name', ''),
                                'source': 'openfigi',
                                'mapped_at': '2026-01-09T17:00:00'
                            }
                            print(f'{i+1}/{len(tickers)}: {ticker} -> {cusip}')
                            break
                break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = (attempt + 1) * 5
                print(f'{i+1}/{len(tickers)}: {ticker} rate limited, waiting {wait}s...')
                time.sleep(wait)
            else:
                print(f'{i+1}/{len(tickers)}: {ticker} FAILED: {e}')
                break
        except Exception as e:
            print(f'{i+1}/{len(tickers)}: {ticker} FAILED: {e}')
            break
    
    time.sleep(1.0)
    
    if (i+1) % 50 == 0:
        print(f'Progress: {i+1}/{len(tickers)}, {len(cusip_map)} mapped')

with open('production_data/cusip_static_map.json', 'w') as f:
    json.dump(cusip_map, f, indent=2)

print(f'\nFinal: {len(cusip_map)} CUSIPs mapped')
