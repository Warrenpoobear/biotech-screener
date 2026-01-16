#!/usr/bin/env python3
"""Test Morningstar provider against full universe."""
import json
from wake_robin_data_pipeline.morningstar_data_provider import MorningstarDataProvider
from datetime import date
import statistics
import time

# Load universe
with open('production_data/universe.json', 'r') as f:
    universe = json.load(f)

tickers = [s['ticker'] for s in universe if s.get('ticker') and not s['ticker'].startswith('_')]
print(f'Testing {len(tickers)} tickers from universe...\n')

provider = MorningstarDataProvider()
results = {'success': [], 'failed': []}
start = time.time()

# Test first 50 tickers (change to tickers for full test)
test_tickers = tickers[:50]

for i, ticker in enumerate(test_tickers):
    try:
        returns = provider.get_daily_returns(ticker, date.today(), 60)
        if len(returns) >= 30:
            vol = statistics.stdev(returns) * (252 ** 0.5)
            results['success'].append((ticker, len(returns), vol))
            print(f'{i+1:3}. {ticker}: {len(returns)} returns, vol={vol:.1%}')
        else:
            results['failed'].append((ticker, f'Only {len(returns)} returns'))
            print(f'{i+1:3}. {ticker}: Only {len(returns)} returns')
    except Exception as e:
        results['failed'].append((ticker, str(e)[:50]))
        print(f'{i+1:3}. {ticker}: ERROR - {str(e)[:40]}')

elapsed = time.time() - start
total = len(test_tickers)

print(f'\n{"="*50}')
print(f'RESULTS')
print(f'{"="*50}')
print(f'Success: {len(results["success"])}/{total} ({100*len(results["success"])/total:.1f}%)')
print(f'Failed:  {len(results["failed"])}/{total}')
print(f'Time:    {elapsed:.1f}s ({total/elapsed:.1f} tickers/sec)')

if results['success']:
    vols = [v[2] for v in results['success']]
    print(f'\nVolatility Stats:')
    print(f'  Min: {min(vols):.1%}')
    print(f'  Max: {max(vols):.1%}')
    print(f'  Avg: {statistics.mean(vols):.1%}')
