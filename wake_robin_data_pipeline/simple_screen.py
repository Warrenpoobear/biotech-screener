import json

# Load latest snapshot
with open('outputs/universe_snapshot_latest.json') as f:
    companies = json.load(f)

print('\nWake Robin Universe - 20 Ticker Pilot')
print('='*80)
print(f'{"Ticker":<8} {"Price":>8} {"MCap":>8} {"Cash":>8} {"Coverage":>8} {"Stage":<12} {"Status"}')
print('-'*80)

passed = []
filtered = []

# Apply your financial coverage gate
for company in companies:
    ticker = company['ticker']
    price = company['market_data']['price']
    mcap = company['market_data']['market_cap'] / 1e9
    cash = company['financials'].get('cash')
    coverage = company['data_quality']['financial_coverage']
    lead_stage = company['clinical']['lead_stage']
    
    # Format cash
    cash_str = f'${cash/1e6:.0f}M' if cash else 'N/A'
    
    # Simple screening criteria
    if coverage >= 75 and cash and cash > 100e6:
        status = '✓ PASS'
        passed.append(ticker)
        print(f'{ticker:<8} ${price:7.2f} ${mcap:6.2f}B {cash_str:>8} {coverage:6.1f}% {lead_stage:<12} {status}')
    else:
        reason = 'Low coverage' if coverage < 75 else 'Low cash' if not cash or cash <= 100e6 else 'Other'
        status = f'✗ FAIL ({reason})'
        filtered.append((ticker, reason))
        print(f'{ticker:<8} ${price:7.2f} ${mcap:6.2f}B {cash_str:>8} {coverage:6.1f}% {lead_stage:<12} {status}')

print('='*80)
print(f'\nSummary: {len(passed)} passed, {len(filtered)} filtered out')
print(f'Pass rate: {len(passed)/len(companies)*100:.1f}%')

if filtered:
    print('\nFiltered out:')
    for ticker, reason in filtered:
        print(f'  • {ticker}: {reason}')

print('\nPassed screening:')
for ticker in passed:
    print(f'  • {ticker}')
