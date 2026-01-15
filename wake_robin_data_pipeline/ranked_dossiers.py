import json
from pathlib import Path
import math

Path('dossiers/top5').mkdir(parents=True, exist_ok=True)
companies = json.load(open('outputs/universe_snapshot_latest.json'))

def score_company(c):
    cash = c['financials'].get('cash') or 0
    mcap = c['market_data']['market_cap'] or 1
    
    if cash > 0 and mcap > 0:
        ratio = min((cash / mcap) * 100, 30.0)
        cash_score = math.log(1 + ratio) * 2.5
    else:
        cash_score = 0
    
    stage_scores = {'commercial': 10, 'phase_3': 8, 'phase_2': 5, 'phase_1': 2, 'unknown': 0}
    stage_score = stage_scores.get(c['clinical']['lead_stage'], 0)
    
    runway_months = None
    if cash > 0:
        revenue = c['financials'].get('revenue_ttm') or 0
        burn = (revenue * 0.25 / 4) if revenue > 0 else 10e6
        runway_months = (cash / burn) * 3
        runway_score = 10 if runway_months > 24 else 8 if runway_months > 18 else 6 if runway_months > 12 else 3
    else:
        runway_score = 5
    
    active = c['clinical']['active_trials']
    total = c['clinical']['total_trials']
    pipeline_score = min((active / total) * 10 + min(active / 3, 5), 10) if total > 0 else 0
    
    composite = stage_score * 0.30 + cash_score * 0.25 + runway_score * 0.25 + pipeline_score * 0.20
    
    return {
        'composite': round(composite, 2),
        'stage_score': stage_score,
        'cash_score': round(cash_score, 2),
        'runway_score': runway_score,
        'pipeline_score': round(pipeline_score, 2),
        'runway_months': runway_months
    }

ranked = sorted([(c, score_company(c)) for c in companies], 
                key=lambda x: x[1]['composite'], reverse=True)

print('\nTOP 5 RANKINGS (Improved Scoring):\n')
print(f"{'Rank':<6} {'Ticker':<8} {'Score':<8} {'Cash':<10} {'Runway':<12} {'Stage':<12}")
print('-' * 70)

for i, (c, scores) in enumerate(ranked[:5], 1):
    cash = c['financials'].get('cash')
    cash_str = f"${cash/1e6:.0f}M" if cash else "N/A"
    runway_str = f"~{scores['runway_months']:.0f}mo" if scores['runway_months'] else "N/A"
    
    print(f"{i:<6} {c['ticker']:<8} {scores['composite']:<8.2f} {cash_str:<10} {runway_str:<12} {c['clinical']['lead_stage']:<12}")
    
    content = f"""# Investment Dossier: {c['ticker']}
**Rank: #{i} | Composite Score: {scores['composite']:.2f}/10**

## Executive Summary
- Company: {c['market_data'].get('company_name', c['ticker'])}
- Price: ${c['market_data']['price']:.2f}
- Market Cap: ${c['market_data']['market_cap']/1e9:.2f}B
- Cash: {cash_str}
- Clinical Stage: {c['clinical']['lead_stage']}
- Active Trials: {c['clinical']['active_trials']}/{c['clinical']['total_trials']}

## Score Breakdown

| Component | Score | Weight | Contribution |
|-----------|-------|--------|--------------|
| Stage | {scores['stage_score']:.1f}/10 | 30% | {scores['stage_score'] * 0.30:.2f} |
| Cash (capped) | {scores['cash_score']:.1f}/10 | 25% | {scores['cash_score'] * 0.25:.2f} |
| Runway | {scores['runway_score']:.1f}/10 | 25% | {scores['runway_score'] * 0.25:.2f} |
| Pipeline | {scores['pipeline_score']:.1f}/10 | 20% | {scores['pipeline_score'] * 0.20:.2f} |
| **Composite** | **{scores['composite']:.2f}/10** | 100% | **{scores['composite']:.2f}** |

## Financial Reality Check
- Cash: {cash_str}
- Estimated Runway: {runway_str}
- Market Cap: ${c['market_data']['market_cap']/1e9:.2f}B
- Daily Volume: {c['market_data'].get('volume_avg_30d', 0):,} shares

## Data Quality
- Overall Coverage: {c['data_quality']['overall_coverage']:.0f}%
- Financial Coverage: {c['data_quality']['financial_coverage']:.0f}%

## Risk Warnings
"""
    
    warnings = []
    if c['data_quality']['financial_coverage'] < 50:
        warnings.append(f"[CRITICAL] Only {c['data_quality']['financial_coverage']:.0f}% financial data coverage")
    
    if scores['runway_months'] and scores['runway_months'] < 12:
        warnings.append(f"[HIGH RISK] Dilution risk - Runway < 12 months")
    elif scores['runway_months'] and scores['runway_months'] < 18:
        warnings.append(f"[WATCH] Runway < 18 months - monitor for financing")
    
    if c['market_data']['market_cap'] < 1e9:
        warnings.append(f"[LIQUIDITY] Market cap < $1B - position sizing limited")
    
    if c['clinical']['lead_stage'] in ['phase_1', 'phase_2']:
        warnings.append(f"[CLINICAL RISK] Lead asset in {c['clinical']['lead_stage']}")
    
    if not cash:
        warnings.append(f"[DATA GAP] No cash data available - verify SEC filings")
    
    if warnings:
        content += '\n'.join(f"- {w}" for w in warnings)
    else:
        content += "- [PASS] No major risk flags identified\n"
    
    content += "\n\n---\n*Algorithmically generated. Verify all data before investment decisions.*\n"
    
    with open(f'dossiers/top5/{i}_{c["ticker"]}.md', 'w', encoding='utf-8') as f:
        f.write(content)

print('\n✅ Ranked dossiers created in dossiers/top5/')
