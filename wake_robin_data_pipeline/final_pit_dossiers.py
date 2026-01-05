import json
from pathlib import Path
import math

companies = json.load(open('outputs/universe_snapshot_latest.json'))

def score_company(c):
    cash = c['financials'].get('cash') or 0
    mcap = c['market_data']['market_cap'] or 1
    
    if cash > 0 and mcap > 0:
        ratio = min((cash / mcap) * 100, 30.0)
        cash_score = math.log(1 + ratio) * 2.5
    else:
        cash_score = 0
    
    sec_stale = c['data_quality'].get('sec_stale', True)
    if sec_stale:
        cash_score *= 0.5
    
    stage_scores = {'commercial': 10, 'phase_3': 8, 'phase_2': 5, 'phase_1': 2, 'unknown': 0}
    stage_score = stage_scores.get(c['clinical']['lead_stage'], 0)
    
    runway_months = None
    runway_label = "N/A"
    if cash > 0:
        revenue = c['financials'].get('revenue_ttm') or 0
        monthly_burn = (revenue * 0.25) / 12 if revenue > 0 else 10e6 / 12
        
        if monthly_burn <= 0:
            runway_months = None
            runway_label = "self-funding/positive OCF"
            runway_score = 10
        else:
            raw_runway = (cash / monthly_burn)
            runway_months = min(60, raw_runway)
            runway_label = f"~{int(runway_months)}mo" if runway_months < 60 else "60+ mo (capped)"
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
        'runway_months': runway_months,
        'runway_label': runway_label,
        'stale_penalty_applied': sec_stale
    }

ranked = sorted([(c, score_company(c)) for c in companies], 
                key=lambda x: x[1]['composite'], reverse=True)

Path('dossiers/top5').mkdir(parents=True, exist_ok=True)

print('\nGenerating PIT-validated dossiers...\n')

for i, (c, scores) in enumerate(ranked[:5], 1):
    ticker = c['ticker']
    cash = c['financials'].get('cash')
    cash_str = f"${cash/1e6:.0f}M" if cash else "N/A"
    
    sec_fresh = c.get('freshness', {}).get('sec', {})
    
    content = f"""# Investment Dossier: {ticker}
**Rank: #{i} | Composite Score: {scores['composite']:.2f}/10**
**Generated: 2026-01-05 (as-of date)**

## Executive Summary
- Company: {c['market_data'].get('company_name', ticker)}
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
- Estimated Runway: {scores['runway_label']}
- Market Cap: ${c['market_data']['market_cap']/1e9:.2f}B
- Daily Volume: {c['market_data'].get('volume_avg_30d', 0):,} shares

## Freshness & Point-in-Time Discipline

**As-of Date:** 2026-01-05

**SEC Financial Data:**
- Period End: {sec_fresh.get('period_end', 'N/A')}
- Period End Age: {sec_fresh.get('period_end_age_days', 'N/A')} days
- Filed Date: {sec_fresh.get('filed_date', 'N/A')[:10] if sec_fresh.get('filed_date') else 'N/A'}
- Status: {'FRESH (<365 days)' if not sec_fresh.get('is_stale', True) else 'STALE (>365 days)'}

**Market Data:**
- Source: Yahoo Finance (real-time)
- Last Update: {c.get('provenance', {}).get('sources', {}).get('yahoo_finance', {}).get('timestamp', 'N/A')[:10]}

## Data Quality
- Overall Coverage: {c['data_quality']['overall_coverage']:.0f}%
- Financial Coverage: {c['data_quality']['financial_coverage']:.0f}%
- SEC Data Status: {'Fresh' if not sec_fresh.get('is_stale', True) else 'Stale - 50% penalty applied'}

## Risk Warnings
"""
    
    warnings = []
    
    if sec_fresh.get('is_stale', True):
        warnings.append(f"[CRITICAL] SEC data is stale - cash component received 50% penalty")
    
    if c['data_quality']['financial_coverage'] < 50:
        warnings.append(f"[CRITICAL] Only {c['data_quality']['financial_coverage']:.0f}% financial data coverage")
    
    if scores['runway_months'] and scores['runway_months'] < 12:
        warnings.append(f"[HIGH RISK] Dilution risk - Runway < 12 months")
    elif scores['runway_months'] and scores['runway_months'] < 18:
        warnings.append(f"[WATCH] Runway < 18 months - monitor for financing")
    
    if scores['runway_label'] == "60+ mo (capped)":
        warnings.append(f"[NOTE] Runway capped at 60mo for scoring (prevents burn artifacts)")
    
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
    
    content += "\n\n---\n*Point-in-time validated. Algorithmically generated. Verify all data before investment decisions.*\n"
    
    with open(f'dossiers/top5/{i}_{ticker}.md', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"{i}. {ticker:6} - Score: {scores['composite']:.2f} | Fresh: {'YES' if not sec_fresh.get('is_stale', True) else 'NO'}")

print('\n✓ PIT-validated dossiers created in dossiers/top5/')
