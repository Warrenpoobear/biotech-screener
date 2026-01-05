import json
from pathlib import Path
import math

companies = json.load(open('outputs/universe_snapshot_latest.json'))

def score_company(c):
    cash = c['financials'].get('cash') or 0
    mcap = c['market_data']['market_cap'] or 1
    
    # Calculate base cash score
    if cash > 0 and mcap > 0:
        ratio = min((cash / mcap) * 100, 30.0)
        cash_score = math.log(1 + ratio) * 2.5
    else:
        cash_score = 0
    
    # FRESHNESS PENALTY: degrade financial score if stale
    sec_stale = c['data_quality'].get('sec_stale', True)
    if sec_stale:
        cash_score *= 0.5  # 50% penalty for stale data
    
    # Stage score
    stage_scores = {'commercial': 10, 'phase_3': 8, 'phase_2': 5, 'phase_1': 2, 'unknown': 0}
    stage_score = stage_scores.get(c['clinical']['lead_stage'], 0)
    
    # Runway calculation with sanity checks
    runway_months = None
    runway_label = "N/A"
    if cash > 0:
        revenue = c['financials'].get('revenue_ttm') or 0
        monthly_burn = (revenue * 0.25) / 12 if revenue > 0 else 10e6 / 12
        
        if monthly_burn <= 0:
            runway_months = None
            runway_label = "self-funding/positive OCF"
            runway_score = 10  # Self-funding is good
        else:
            raw_runway = (cash / monthly_burn)
            runway_months = min(60, raw_runway)  # CAP AT 60 MONTHS for scoring
            runway_label = f"~{int(runway_months)}mo" if runway_months < 60 else "60+ mo (capped)"
            
            if runway_months > 24: runway_score = 10
            elif runway_months > 18: runway_score = 8
            elif runway_months > 12: runway_score = 6
            else: runway_score = 3
    else:
        runway_score = 5
    
    # Pipeline score
    active = c['clinical']['active_trials']
    total = c['clinical']['total_trials']
    pipeline_score = min((active / total) * 10 + min(active / 3, 5), 10) if total > 0 else 0
    
    # Composite (adjusted weights)
    composite = stage_score * 0.30 + cash_score * 0.25 + runway_score * 0.25 + pipeline_score * 0.20
    
    return {
        'composite': round(composite, 2),
        'stage_score': stage_score,
        'cash_score': round(cash_score, 2),
        'cash_score_raw': round(cash_score / 0.5, 2) if sec_stale else round(cash_score, 2),
        'runway_score': runway_score,
        'pipeline_score': round(pipeline_score, 2),
        'runway_months': runway_months,
        'runway_label': runway_label,
        'stale_penalty_applied': sec_stale
    }

# Score and rank
ranked = sorted([(c, score_company(c)) for c in companies], 
                key=lambda x: x[1]['composite'], reverse=True)

print('\nTOP 5 RANKINGS (PIT-Aware Scoring):\n')
print(f"{'Rank':<6} {'Ticker':<8} {'Score':<8} {'Cash':<10} {'Runway':<15} {'Stale?':<8} {'Stage':<12}")
print('-' * 85)

for i, (c, scores) in enumerate(ranked[:5], 1):
    cash = c['financials'].get('cash')
    cash_str = f"${cash/1e6:.0f}M" if cash else "N/A"
    stale_flag = "YES" if scores['stale_penalty_applied'] else "NO"
    
    print(f"{i:<6} {c['ticker']:<8} {scores['composite']:<8.2f} {cash_str:<10} {scores['runway_label']:<15} {stale_flag:<8} {c['clinical']['lead_stage']:<12}")

print('\n' + '='*85)
print('Note: Stale SEC data receives 50% penalty on cash component')
print('Note: Runway capped at 60 months to prevent burn artifacts')
print('='*85)
