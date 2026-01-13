#!/usr/bin/env python3
"""
ClinicalTrials.gov Historical Data Fetcher

Fetches clinical trial data to reconstruct point-in-time clinical stage snapshots.

The ClinicalTrials.gov API provides trial data including:
- Phase (1, 2, 3, 4)
- Status (Recruiting, Active, Completed, etc.)
- Start/completion dates
- Sponsor information

Usage:
    python clinicaltrials_gov.py --ticker VRTX --as-of 2023-01-15
    python clinicaltrials_gov.py --company "Vertex Pharmaceuticals" --as-of 2023-01-15
"""

import argparse
import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional
import urllib.request
import urllib.error
import urllib.parse

# ClinicalTrials.gov API base URL (v2)
CT_BASE_URL = "https://clinicaltrials.gov/api/v2"

# Cache directory
CACHE_DIR = Path("data/cache/clinicaltrials")

# Mapping of tickers to company names for searching
TICKER_TO_COMPANY = {
    'VRTX': 'Vertex Pharmaceuticals',
    'REGN': 'Regeneron Pharmaceuticals',
    'ALNY': 'Alnylam Pharmaceuticals',
    'BMRN': 'BioMarin Pharmaceutical',
    'SGEN': 'Seagen',
    'BIIB': 'Biogen',
    'GILD': 'Gilead Sciences',
    'AMGN': 'Amgen',
    'CRSP': 'CRISPR Therapeutics',
    'EDIT': 'Editas Medicine',
    'NTLA': 'Intellia Therapeutics',
    'BEAM': 'Beam Therapeutics',
    'MRNA': 'Moderna',
    'BNTX': 'BioNTech',
    'NVAX': 'Novavax',
    # Add more mappings as needed
}

# Phase to stage bucket mapping
PHASE_TO_STAGE = {
    'PHASE1': 'early',
    'PHASE2': 'mid',
    'PHASE3': 'late',
    'PHASE4': 'commercial',
    'NA': 'preclinical',
    'EARLY_PHASE1': 'early',
}


def get_company_name(ticker: str) -> Optional[str]:
    """Get company name for a ticker."""
    # Check hardcoded mapping first
    if ticker.upper() in TICKER_TO_COMPANY:
        return TICKER_TO_COMPANY[ticker.upper()]

    # Try to load from cache
    cache_file = CACHE_DIR / "ticker_to_company.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cache = json.load(f)
            if ticker.upper() in cache:
                return cache[ticker.upper()]

    return None


def search_trials(sponsor: str, as_of_date: str,
                  max_results: int = 100) -> List[Dict]:
    """
    Search for clinical trials by sponsor as of a specific date.

    Uses ClinicalTrials.gov API v2.
    """
    target_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()

    # Build query - search for trials that existed before as_of_date
    # We look for trials where start date <= as_of_date
    query = {
        'query.spons': sponsor,
        'filter.overallStatus': 'RECRUITING,ACTIVE_NOT_RECRUITING,ENROLLING_BY_INVITATION,NOT_YET_RECRUITING,COMPLETED,SUSPENDED,TERMINATED,WITHDRAWN',
        'pageSize': max_results,
        'format': 'json'
    }

    url = f"{CT_BASE_URL}/studies?" + urllib.parse.urlencode(query)

    try:
        req = urllib.request.Request(url)
        req.add_header('Accept', 'application/json')

        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode())
            studies = data.get('studies', [])

            # Filter to trials that existed as of target_date
            valid_trials = []
            for study in studies:
                protocol = study.get('protocolSection', {})
                status_module = protocol.get('statusModule', {})

                # Get study start date
                start_info = status_module.get('startDateStruct', {})
                start_date_str = start_info.get('date', '')

                if start_date_str:
                    try:
                        # Parse date (format: YYYY-MM-DD or YYYY-MM or YYYY)
                        if len(start_date_str) == 10:
                            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
                        elif len(start_date_str) == 7:
                            start_date = datetime.strptime(start_date_str + "-01", "%Y-%m-%d").date()
                        else:
                            start_date = datetime.strptime(start_date_str + "-01-01", "%Y-%m-%d").date()

                        if start_date <= target_date:
                            valid_trials.append(study)
                    except ValueError:
                        # Include if we can't parse date
                        valid_trials.append(study)
                else:
                    # Include if no start date
                    valid_trials.append(study)

            return valid_trials

    except urllib.error.HTTPError as e:
        print(f"  HTTP Error {e.code}: {e.reason}")
        return []
    except Exception as e:
        print(f"  Error searching trials: {e}")
        return []


def extract_trial_info(study: Dict) -> Dict:
    """Extract relevant information from a study."""
    protocol = study.get('protocolSection', {})

    # Identification
    id_module = protocol.get('identificationModule', {})
    nct_id = id_module.get('nctId', '')
    title = id_module.get('briefTitle', '')

    # Status
    status_module = protocol.get('statusModule', {})
    overall_status = status_module.get('overallStatus', '')

    # Design
    design_module = protocol.get('designModule', {})
    phases = design_module.get('phases', [])
    phase = phases[0] if phases else 'NA'

    # Conditions
    conditions_module = protocol.get('conditionsModule', {})
    conditions = conditions_module.get('conditions', [])

    return {
        'nct_id': nct_id,
        'title': title,
        'phase': phase,
        'status': overall_status,
        'conditions': conditions
    }


def determine_lead_stage(trials: List[Dict]) -> Dict:
    """
    Determine the lead clinical stage from a list of trials.

    Returns the most advanced stage (Phase 3 > Phase 2 > Phase 1).
    """
    phase_counts = {'PHASE1': 0, 'PHASE2': 0, 'PHASE3': 0, 'PHASE4': 0}
    active_phases = set()

    for trial in trials:
        info = extract_trial_info(trial)
        phase = info['phase'].upper().replace(' ', '')

        # Only count active/recruiting trials for lead stage
        if info['status'] in ['RECRUITING', 'ACTIVE_NOT_RECRUITING',
                              'ENROLLING_BY_INVITATION', 'NOT_YET_RECRUITING']:
            active_phases.add(phase)

        # Count all phases
        if 'PHASE3' in phase or phase == 'PHASE3':
            phase_counts['PHASE3'] += 1
        elif 'PHASE2' in phase or phase == 'PHASE2':
            phase_counts['PHASE2'] += 1
        elif 'PHASE1' in phase or phase == 'PHASE1':
            phase_counts['PHASE1'] += 1
        elif 'PHASE4' in phase or phase == 'PHASE4':
            phase_counts['PHASE4'] += 1

    # Determine lead stage (most advanced active phase)
    if 'PHASE4' in active_phases or phase_counts['PHASE4'] > 0:
        lead_stage = 'COMMERCIAL'
        stage_bucket = 'commercial'
    elif 'PHASE3' in active_phases or phase_counts['PHASE3'] > 0:
        lead_stage = 'PHASE3'
        stage_bucket = 'late'
    elif 'PHASE2' in active_phases or phase_counts['PHASE2'] > 0:
        lead_stage = 'PHASE2'
        stage_bucket = 'mid'
    elif 'PHASE1' in active_phases or phase_counts['PHASE1'] > 0:
        lead_stage = 'PHASE1'
        stage_bucket = 'early'
    else:
        lead_stage = 'PRECLINICAL'
        stage_bucket = 'preclinical'

    return {
        'lead_stage': lead_stage,
        'stage_bucket': stage_bucket,
        'phase_counts': phase_counts,
        'total_trials': len(trials),
        'active_phases': list(active_phases)
    }


def get_historical_clinical(ticker: str, as_of_date: str) -> Optional[Dict]:
    """
    Get historical clinical stage data for a ticker as of a specific date.
    """
    company_name = get_company_name(ticker)
    if not company_name:
        # Try ticker as sponsor name
        company_name = ticker

    print(f"    Searching for sponsor: {company_name}")

    trials = search_trials(company_name, as_of_date)

    if not trials:
        return {
            'ticker': ticker,
            'as_of_date': as_of_date,
            'company_name': company_name,
            'lead_stage': 'UNKNOWN',
            'stage_bucket': 'unknown',
            'trial_count': 0,
            'error': 'No trials found'
        }

    stage_info = determine_lead_stage(trials)

    # Extract individual trial details
    trial_details = [extract_trial_info(t) for t in trials[:20]]  # Limit for size

    return {
        'ticker': ticker,
        'as_of_date': as_of_date,
        'company_name': company_name,
        'lead_stage': stage_info['lead_stage'],
        'stage_bucket': stage_info['stage_bucket'],
        'trial_count': stage_info['total_trials'],
        'phase_counts': stage_info['phase_counts'],
        'active_phases': stage_info['active_phases'],
        'sample_trials': trial_details[:5],
        'data_source': 'ClinicalTrials.gov'
    }


def fetch_batch(tickers: List[str], as_of_date: str,
                delay: float = 0.5) -> Dict[str, Dict]:
    """
    Fetch historical clinical data for multiple tickers.
    """
    results = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{total}] Fetching {ticker}...")
        try:
            data = get_historical_clinical(ticker, as_of_date)
            if data:
                results[ticker] = data
                print(f"    Stage: {data.get('lead_stage', 'N/A')}, "
                      f"Trials: {data.get('trial_count', 0)}")
            else:
                print("    No data")
        except Exception as e:
            print(f"    Error: {e}")
            results[ticker] = {'ticker': ticker, 'error': str(e)}

        # Rate limiting
        time.sleep(delay)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical clinical trial data from ClinicalTrials.gov"
    )
    parser.add_argument('--ticker', type=str, help='Single ticker to fetch')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers')
    parser.add_argument('--company', type=str, help='Company name to search')
    parser.add_argument('--as-of', type=str, required=True,
                        help='Point-in-time date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between requests (default: 0.5s)')

    args = parser.parse_args()

    # Collect tickers
    tickers = []
    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    elif args.company:
        # Direct company search
        print(f"Searching for {args.company} as of {args.as_of}")
        trials = search_trials(args.company, args.as_of)
        stage_info = determine_lead_stage(trials)
        print(f"  Found {len(trials)} trials")
        print(f"  Lead stage: {stage_info['lead_stage']}")
        print(f"  Phase counts: {stage_info['phase_counts']}")
        return
    else:
        parser.error("Specify --ticker, --tickers, or --company")

    print(f"Fetching historical clinical data for {len(tickers)} tickers as of {args.as_of}")

    results = fetch_batch(tickers, args.as_of, delay=args.delay)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'as_of_date': args.as_of,
                'fetched_at': datetime.now().isoformat(),
                'count': len(results),
                'tickers': results
            }, f, indent=2)
        print(f"\nSaved to {args.output}")
    else:
        print(json.dumps(results, indent=2))

    # Summary
    by_stage = {}
    for r in results.values():
        stage = r.get('stage_bucket', 'unknown')
        by_stage[stage] = by_stage.get(stage, 0) + 1
    print(f"\nBy stage: {by_stage}")


if __name__ == "__main__":
    main()
