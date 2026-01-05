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

# Load current snapshot
snapshot = json.load(open('outputs/universe_snapshot_latest.json'))
asof = date(2026, 1, 5)

print("Extracting REAL period_end dates from SEC XBRL...\n")

for company in snapshot:
    ticker = company['ticker']
    
    # Get the most recent cash fact from SEC data
    # The SEC Company Facts API includes the "end" date for each fact
    cash_facts = company.get('financials', {}).get('_raw_sec_cash_facts', [])
    
    if cash_facts:
        # Get the most recent fact (they're usually sorted by end date)
        most_recent = cash_facts[-1] if isinstance(cash_facts, list) else cash_facts
        period_end = most_recent.get('end') if isinstance(most_recent, dict) else None
        filed = most_recent.get('filed') if isinstance(most_recent, dict) else None
    else:
        # Fallback to current estimate
        sec = company.get('provenance', {}).get('sources', {}).get('sec_edgar', {})
        period_end = company.get('freshness', {}).get('sec', {}).get('period_end')
        filed = sec.get('timestamp', '')
    
    # Calculate freshness with real dates
    end_dt = parse_iso_date(period_end)
    filed_dt = parse_iso_date(filed)
    
    end_age = (asof - end_dt).days if end_dt else None
    filed_age = (asof - filed_dt).days if filed_dt else None
    
    is_stale = True
    if end_age is not None:
        is_stale = end_age > 365
    
    # Update freshness metadata
    company['freshness']['sec'] = {
        'period_end': period_end,
        'filed_date': filed,
        'period_end_age_days': end_age,
        'filed_age_days': filed_age,
        'is_stale': is_stale
    }
    
    company['data_quality']['sec_stale'] = is_stale
    
    if ticker in ['JAZZ', 'INCY', 'SRPT', 'BMRN', 'EXEL']:
        print(f"{ticker:6} | Period End: {period_end or 'N/A':12} | Age: {end_age or 'N/A':>4} days | Stale: {is_stale}")

# Save updated snapshot
with open('outputs/universe_snapshot_latest.json', 'w') as f:
    json.dump(snapshot, f, indent=2)

print("\n✓ Real period_end dates extracted and saved")
