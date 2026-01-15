import json
from datetime import datetime

# Load latest snapshot
with open('outputs/universe_snapshot_latest.json') as f:
    companies = json.load(f)

# Scoring weights (adjust based on your thesis)
STAGE_SCORES = {
    'commercial': 10,
    'phase_3': 8,
    'phase_2': 5,
    'phase_1': 2,
    'unknown': 0
}

# Financial strength score (0-10)
def financial_score(company):
    cash = company['financials'].get('cash', 0)
    mcap = company['market_data']['market_cap']
    
    if not cash or mcap == 0:
        return 0
    
    # Cash as % of market cap
    cash_pct = (cash / mcap) * 100
    
    # Score: higher cash % = higher score
    if cash_pct > 20:
        return 10
    elif cash_pct > 10:
        return 8
    elif cash_pct > 5:
        return 6
    elif cash_pct > 2:
        return 4
    else:
        return 2

# Pipeline activity score (0-10)
def pipeline_score(company):
    active = company['clinical']['active_trials']
    total = company['clinical']['total_trials']
    
    if total == 0:
        return 0
    
    # Active ratio + absolute count
    ratio_score = (active / total) * 5
    volume_score = min(active / 5, 5)  # Max 5 points for 25+ active
    
    return min(ratio_score + volume_score, 10)

# Composite score
def calculate_score(company):
    stage = company['clinical']['lead_stage']
    
    stage_pts = STAGE_SCORES.get(stage, 0)
    financial_pts = financial_score(company)
    pipeline_pts = pipeline_score(company)
    
    # Weighted composite (adjust weights as needed)
    composite = (stage_pts * 0.4) + (financial_pts * 0.3) + (pipeline_pts * 0.3)
    
    return {
        'ticker': company['ticker'],
        'composite_score': round(composite, 2),
        'stage_score': stage_pts,
        'financial_score': round(financial_pts, 2),
        'pipeline_score': round(pipeline_pts, 2),
        'price': company['market_data']['price'],
        'market_cap': company['market_data']['market_cap'] / 1e9,
        'cash': company['financials'].get('cash', 0) / 1e6 if company['financials'].get('cash') else 0,
        'lead_stage': stage,
        'active_trials': company['clinical']['active_trials']
    }

# Score all companies
scored = [calculate_score(c) for c in companies]

# Sort by composite score (ASCENDING: lower score = better = rank 1)
# Validation showed inverted ranking: high scores predicted underperformance
ranked = sorted(scored, key=lambda x: x['composite_score'], reverse=False)

# Display ranked list
print('\n' + '='*100)
print('WAKE ROBIN BIOTECH SCREENER - RANKED LIST')
print('='*100)
print(f'{"Rank":<6} {"Ticker":<8} {"Score":<8} {"Stage":<8} {"Fin":<6} {"Pipe":<6} {"Price":<10} {"MCap":<10} {"Cash":<10}')
print('-'*100)

for i, company in enumerate(ranked, 1):
    print(f'{i:<6} {company["ticker"]:<8} {company["composite_score"]:<8.2f} '
          f'{company["stage_score"]:<8} {company["financial_score"]:<6.2f} '
          f'{company["pipeline_score"]:<6.2f} ${company["price"]:<9.2f} '
          f'${company["market_cap"]:<9.2f}B ${company["cash"]:<9.0f}M')

print('='*100)

# Top 5 recommendations
print('\n🎯 TOP 5 INVESTMENT CANDIDATES:\n')
for i, company in enumerate(ranked[:5], 1):
    print(f'{i}. {company["ticker"]} - Score: {company["composite_score"]:.2f}')
    print(f'   Price: ${company["price"]:.2f} | MCap: ${company["market_cap"]:.2f}B | Cash: ${company["cash"]:.0f}M')
    print(f'   Stage: {company["lead_stage"]} | Active Trials: {company["active_trials"]}')
    print()

# Save ranked list
output = {
    'timestamp': datetime.now().isoformat(),
    'universe_size': len(ranked),
    'ranked_list': ranked
}

with open('outputs/ranked_list_latest.json', 'w') as f:
    json.dump(output, f, indent=2)

print('✓ Ranked list saved to: outputs/ranked_list_latest.json')
