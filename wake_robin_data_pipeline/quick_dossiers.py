import json
from pathlib import Path

Path('dossiers/top5').mkdir(parents=True, exist_ok=True)
companies = json.load(open('outputs/universe_snapshot_latest.json'))

for i, c in enumerate(companies[:5], 1):
    cash = c['financials'].get('cash')
    cash_str = f"${cash/1e6:.0f}M" if cash else "N/A"
    
    content = f"""# {c['ticker']} - Rank #{i}

Price: ${c['market_data']['price']:.2f}
Market Cap: ${c['market_data']['market_cap']/1e9:.2f}B
Cash: {cash_str}

## Clinical
Stage: {c['clinical']['lead_stage']}
Active Trials: {c['clinical']['active_trials']}/{c['clinical']['total_trials']}

## Liquidity
Volume: {c['market_data'].get('volume_avg_30d', 0):,} shares/day

## Data Quality
Overall: {c['data_quality']['overall_coverage']:.0f}%
Financial: {c['data_quality']['financial_coverage']:.0f}%
"""
    
    Path(f'dossiers/top5/{i}_{c["ticker"]}.md').write_text(content)
    print(f'{i}. {c["ticker"]} - {cash_str}')

print('\nDossiers created in dossiers/top5/')
