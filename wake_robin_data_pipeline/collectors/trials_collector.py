"""
trials_collector.py - Collect clinical trial data from ClinicalTrials.gov
Free, no API key required. Rate limit: reasonable (1 req/sec safe)
"""
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import hashlib

def get_cache_path(company_name: str) -> Path:
    """Get cache file path for company."""
    cache_dir = Path(__file__).parent.parent / "cache" / "trials"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Use sanitized company name as filename
    safe_name = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).strip()
    return cache_dir / f"{safe_name}.json"

def is_cache_valid(cache_path: Path, max_age_hours: int = 24) -> bool:
    """Check if cache is fresh enough."""
    if not cache_path.exists():
        return False
    
    age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
    return age < timedelta(hours=max_age_hours)

def search_company_trials(company_name: str, max_results: int = 50) -> List[dict]:
    """
    Search ClinicalTrials.gov for trials by company/sponsor name.
    
    Args:
        company_name: Company or sponsor name to search
        max_results: Maximum number of trials to return
        
    Returns:
        List of trial summary dicts
    """
    try:
        # ClinicalTrials.gov v2 API search endpoint
        url = "https://clinicaltrials.gov/api/v2/studies"
        
        params = {
            'query.spons': company_name,  # Search sponsor field
            'pageSize': min(max_results, 100),  # API max is 100
            'format': 'json',
            'fields': 'NCTId,BriefTitle,OverallStatus,Phase,Condition,StartDate,CompletionDate,EnrollmentCount,LeadSponsorName'
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        studies = data.get('studies', [])
        
        trials = []
        for study in studies:
            protocol = study.get('protocolSection', {})
            
            # Extract key fields
            id_module = protocol.get('identificationModule', {})
            status_module = protocol.get('statusModule', {})
            design_module = protocol.get('designModule', {})
            conditions_module = protocol.get('conditionsModule', {})
            sponsor_module = protocol.get('sponsorCollaboratorsModule', {})
            
            trial = {
                'nct_id': id_module.get('nctId', ''),
                'title': id_module.get('briefTitle', ''),
                'status': status_module.get('overallStatus', ''),
                'phase': ', '.join(design_module.get('phases', [])),
                'condition': ', '.join(conditions_module.get('conditions', [])),
                'start_date': status_module.get('startDateStruct', {}).get('date', ''),
                'completion_date': status_module.get('completionDateStruct', {}).get('date', ''),
                'enrollment': status_module.get('enrollmentInfo', {}).get('count', 0),
                'sponsor': sponsor_module.get('leadSponsor', {}).get('name', '')
            }
            
            trials.append(trial)
        
        return trials
        
    except Exception as e:
        print(f"  Warning: ClinicalTrials.gov search failed: {e}")
        return []

def aggregate_trial_stats(trials: List[dict]) -> dict:
    """
    Aggregate statistics from trials list.
    
    Returns:
        dict with counts by phase, status, etc.
    """
    if not trials:
        return {
            'total_trials': 0,
            'by_phase': {},
            'by_status': {},
            'active_trials': 0,
            'completed_trials': 0
        }
    
    stats = {
        'total_trials': len(trials),
        'by_phase': {},
        'by_status': {},
        'active_trials': 0,
        'completed_trials': 0,
        'conditions': set()
    }
    
    for trial in trials:
        # Count by phase
        phase = trial.get('phase', 'UNKNOWN')
        stats['by_phase'][phase] = stats['by_phase'].get(phase, 0) + 1
        
        # Count by status
        status = trial.get('status', 'UNKNOWN')
        stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
        
        # Active vs completed
        if status in ['RECRUITING', 'ACTIVE_NOT_RECRUITING', 'ENROLLING_BY_INVITATION']:
            stats['active_trials'] += 1
        elif status in ['COMPLETED']:
            stats['completed_trials'] += 1
        
        # Collect conditions
        condition = trial.get('condition', '')
        if condition:
            stats['conditions'].add(condition)
    
    # Convert set to sorted list
    stats['conditions'] = sorted(list(stats['conditions']))
    
    return stats

def fetch_trials_data(ticker: str, company_name: str) -> dict:
    """
    Fetch clinical trial data for a company.
    
    Args:
        ticker: Stock ticker (for reference)
        company_name: Full company name to search
        
    Returns:
        dict with trial data and provenance
    """
    try:
        # Search for company trials
        trials = search_company_trials(company_name)
        
        # Aggregate statistics
        stats = aggregate_trial_stats(trials)
        
        # Determine lead stage from active trials
        lead_stage = "unknown"
        if trials:
            phases = stats['by_phase']
            if 'PHASE4' in phases or any('POST' in p for p in phases):
                lead_stage = "commercial"
            elif 'PHASE3' in phases:
                lead_stage = "phase_3"
            elif 'PHASE2' in phases:
                lead_stage = "phase_2"
            elif 'PHASE1' in phases:
                lead_stage = "phase_1"
            elif 'EARLY_PHASE1' in phases or 'NA' in phases:
                lead_stage = "preclinical"
        
        data = {
            "ticker": ticker,
            "company_name": company_name,
            "success": True,
            "trials": trials[:10],  # Keep top 10 for detail
            "summary": {
                **stats,
                "lead_stage": lead_stage
            },
            "provenance": {
                "source": "ClinicalTrials.gov API v2",
                "timestamp": datetime.now().isoformat(),
                "search_query": company_name,
                "url": f"https://clinicaltrials.gov/search?term={company_name.replace(' ', '+')}"
            }
        }
        
        return data
        
    except Exception as e:
        return {
            "ticker": ticker,
            "company_name": company_name,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def collect_trials_data(ticker: str, company_name: str, force_refresh: bool = False) -> dict:
    """
    Main entry point: collect clinical trial data with caching.
    """
    # Handle None or empty company names (e.g., benchmark tickers)
    if not company_name:
        return {
            "ticker": ticker,
            "success": True,
            "summary": {
                "total_trials": 0,
                "active_trials": 0,
                "completed_trials": 0,
                "lead_stage": "not_applicable",
                "by_phase": {},
                "conditions": [],
                "top_trials": []
            },
            "from_cache": False,
            "skipped": True,
            "reason": "No company name (benchmark or index ticker)"
        }

    cache_path = get_cache_path(company_name)
    
    # Check cache first
    if not force_refresh and is_cache_valid(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
            cached['from_cache'] = True
            return cached
    
    # Fetch fresh data
    data = fetch_trials_data(ticker, company_name)
    
    # Cache successful results
    if data.get('success'):
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    data['from_cache'] = False
    return data

def collect_batch(ticker_company_map: dict, delay_seconds: float = 1.0) -> dict:
    """
    Collect trial data for multiple companies with rate limiting.
    
    Args:
        ticker_company_map: dict mapping ticker -> company_name
        delay_seconds: Delay between requests
        
    Returns:
        dict mapping ticker to trial data
    """
    results = {}
    total = len(ticker_company_map)
    
    print(f"\n🧬 Collecting ClinicalTrials.gov data for {total} companies...")
    
    for i, (ticker, company_name) in enumerate(ticker_company_map.items(), 1):
        print(f"[{i}/{total}] Fetching {ticker} ({company_name})...", end=" ")
        
        data = collect_trials_data(ticker, company_name)
        results[ticker] = data
        
        if data.get('success'):
            summary = data['summary']
            total_trials = summary['total_trials']
            active = summary['active_trials']
            lead_stage = summary['lead_stage']
            cached = " (cached)" if data.get('from_cache') else ""
            print(f"✓ {total_trials} trials, {active} active, Lead: {lead_stage}{cached}")
        else:
            print(f"✗ {data.get('error', 'Unknown error')}")
        
        # Rate limiting
        if i < total and not data.get('from_cache'):
            time.sleep(delay_seconds)
    
    successful = sum(1 for d in results.values() if d.get('success'))
    print(f"\n✓ Successfully collected data for {successful}/{total} companies")
    
    return results

if __name__ == "__main__":
    # Test with a single company
    test_ticker = "VRTX"
    test_company = "Vertex Pharmaceuticals"
    
    print(f"Testing ClinicalTrials.gov collector with {test_company}...")
    
    data = collect_trials_data(test_ticker, test_company, force_refresh=True)
    print(json.dumps(data, indent=2))
