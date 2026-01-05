import json
from datetime import datetime, date
from pathlib import Path

def parse_iso_date(d):
    if not d:
        return None
    try:
        return datetime.fromisoformat(d[:10]).date()
    except:
        return None

def sec_freshness(asof_date, period_end, filed_date):
    """Calculate freshness metrics for SEC data"""
    end_dt = parse_iso_date(period_end)
    filed_dt = parse_iso_date(filed_date)
    
    end_age = (asof_date - end_dt).days if end_dt else None
    filed_age = (asof_date - filed_dt).days if filed_dt else None
    
    # Conservative: mark stale if period end > 365 days old
    is_stale = True
    if end_age is not None:
        is_stale = end_age > 365
    
    return {
        'period_end': period_end,
        'filed_date': filed_date,
        'period_end_age_days': end_age,
        'filed_age_days': filed_age,
        'is_stale': is_stale
    }

# Load current snapshot
snapshot = json.load(open('outputs/universe_snapshot_latest.json'))
asof = date(2026, 1, 5)

# Add freshness to each company
for company in snapshot:
    # Get SEC provenance data
    sec = company.get('provenance', {}).get('sources', {}).get('sec_edgar', {})
    
    # For now, use the timestamp as filed_date (we collected it, so it was filed before that)
    # In reality, you would extract this from the SEC filing XML
    timestamp = sec.get('timestamp', '')
    
    # TEMPORARY: Estimate period_end as ~45 days before filed
    # In production, extract this from dei:DocumentPeriodEndDate in the XBRL
    if timestamp:
        filed_dt = parse_iso_date(timestamp)
        if filed_dt:
            # Rough estimate: Q3 filing in Nov = period end in Sept (45 days prior)
            period_end_estimate = date(filed_dt.year, filed_dt.month - 2, 1) if filed_dt.month > 2 else date(filed_dt.year-1, 10, 1)
            period_end_str = period_end_estimate.isoformat()
        else:
            period_end_str = None
    else:
        period_end_str = None
    
    # Calculate freshness
    freshness = sec_freshness(asof, period_end_str, timestamp)
    
    # Add to company record
    if 'freshness' not in company:
        company['freshness'] = {}
    company['freshness']['sec'] = freshness
    
    # Update data quality flags
    company['data_quality']['sec_stale'] = freshness['is_stale']

# Save updated snapshot
with open('outputs/universe_snapshot_latest.json', 'w') as f:
    json.dump(snapshot, f, indent=2)

print("Added freshness metadata to snapshot")
print("\nSample check - JAZZ:")
jazz = [x for x in snapshot if x['ticker']=='JAZZ'][0]
print(json.dumps(jazz['freshness'], indent=2))
print(f"Is stale: {jazz['data_quality']['sec_stale']}")
