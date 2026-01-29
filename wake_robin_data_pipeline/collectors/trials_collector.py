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

# Sponsor alias mapping: ticker -> list of CT.gov sponsor name variations
# Many companies sponsor trials under subsidiary names or different legal entities
SPONSOR_ALIASES = {
    # Companies using different legal entities
    'MRNA': ['ModernaTX, Inc.', 'Moderna TX', 'Moderna'],
    'QURE': ['UniQure Biopharma B.V.', 'uniQure', 'Uniqure'],
    'PRTA': ['Prothena Biosciences Ltd.', 'Prothena Biosciences', 'Prothena'],
    'AUTL': ['Autolus Limited', 'Autolus Therapeutics', 'Autolus'],
    'MGTX': ['MeiraGTx UK II Ltd', 'MeiraGTx', 'Meiragtx'],
    'IMCR': ['Immunocore Ltd', 'Immunocore', 'Immunocore Holdings'],
    'BCYC': ['BicycleTx Limited', 'BicycleTx', 'Bicycle Therapeutics'],
    'PHVS': ['Pharvaris Netherlands B.V.', 'Pharvaris', 'Pharvaris N.V.'],
    'CNTA': ['Centessa Pharmaceuticals (UK) Limited', 'Centessa Pharmaceuticals', 'Centessa'],
    'SLN': ['Silence Therapeutics plc', 'Silence Therapeutics', 'Silence'],
    'ADCT': ['ADC Therapeutics S.A.', 'ADC Therapeutics', 'ADCT'],
    'ALKS': ['Alkermes, Inc.', 'Alkermes plc', 'Alkermes'],
    'INDV': ['Indivior Inc.', 'Indivior PLC', 'Indivior'],
    'JAZZ': ['Jazz Pharmaceuticals', 'Jazz Pharmaceuticals plc', 'Jazz'],
    'NVCR': ['NovoCure Ltd.', 'NovoCure Limited', 'Novocure', 'NovoCure'],
    'XERS': ['Xeris Pharmaceuticals', 'Xeris Biopharma', 'Xeris'],
    'ZLAB': ['Zai Lab (Shanghai) Co., Ltd.', 'Zai Lab Limited', 'Zai Lab'],
    'TERN': ['Terns, Inc.', 'Terns Pharmaceuticals', 'Terns'],
    'SRRK': ['Scholar Rock, Inc.', 'Scholar Rock Holding', 'Scholar Rock'],
    'GHRS': ['GH Research Ireland Limited', 'GH Research PLC', 'GH Research'],
    'CMPS': ['COMPASS Pathways', 'COMPASS Pathways Plc', 'COMPASS'],
    'ATAI': ['atai Therapeutics, Inc.', 'ATAI Life Sciences', 'atai'],
    'MNMD': ['Definium Therapeutics', 'Mind Medicine', 'MindMed'],  # MindMed uses Definium subsidiary
    'REPL': ['Replimune Inc.', 'Replimune Group', 'Replimune'],
    'GMAB': ['Genmab', 'Genmab A/S'],
    'CYTK': ['Cytokinetics', 'Cytokinetics, Incorporated'],
    'AZN': ['AstraZeneca', 'AstraZeneca PLC', 'Astrazeneca'],
    # Roivant subsidiaries (holding company structure)
    'ROIV': ['Roivant Sciences', 'Immunovant Sciences GmbH', 'Immunovant',
             'Myovant Sciences GmbH', 'Myovant', 'Kinevant Sciences GmbH',
             'Dermavant Sciences', 'Aruvant Sciences'],
    # More companies with known variations
    'CVAC': ['CureVac', 'CureVac N.V.', 'CureVac AG'],
    'RPRX': ['Royalty Pharma', 'Royalty Pharma plc'],  # May not sponsor trials directly
    'TWST': ['Twist Bioscience', 'Twist Bioscience Corporation'],
    # Additional aliases discovered through search
    'HRMY': ['Harmony Biosciences Management, Inc.', 'Harmony Biosciences', 'Harmony'],
    'SGMT': ['Sagimet Biosciences Inc.', 'Sagimet Biosciences', 'Sagimet'],
    'NAMS': ['NewAmsterdam Pharma', 'NewAmsterdam Pharma Company', 'NewAmsterdam'],
    'RANI': ['RANI Therapeutics', 'Rani Therapeutics Holdings', 'Rani'],
    'EOLS': ['Evolus, Inc.', 'Evolus Inc', 'Evolus'],
    'KALA': ['Kala Pharmaceuticals, Inc.', 'Kala Pharmaceuticals', 'KALA BIO', 'Kala'],
    'ADPT': ['Adaptive Biotechnologies', 'Adaptive Biotechnologies Corporation', 'Adaptive'],
    # More companies with different sponsor names
    'MTSR': ['Metsera', 'Metsera Inc', 'Metsera, Inc.'],
    'LGND': ['Ligand Pharmaceuticals', 'Ligand Pharmaceuticals Incorporated', 'Ligand'],
    'OBIO': ['Orchestra BioMed, Inc', 'Orchestra BioMed', 'Orchestra'],
}


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

def _clean_company_name(company_name: str) -> str:
    """Clean company name by removing common suffixes."""
    clean_name = company_name
    for suffix in [' SE', ' BV', ' N.V.', ' Inc', ' Inc.', ' Corporation', ' Corp', ' Corp.',
                   ' Pharmaceuticals', ' Therapeutics', ' Biosciences', ' Pharmaceutical',
                   ' plc', ' PLC', ' Ltd', ' Ltd.', ' Limited', ' S.A.', ' SA', ' A/S',
                   ' Holdings', ' Group', ' - American']:
        if clean_name.endswith(suffix):
            clean_name = clean_name[:-len(suffix)].strip()
    return clean_name


def _search_single_sponsor(sponsor_name: str, max_results: int = 50) -> List[dict]:
    """
    Search ClinicalTrials.gov for a single sponsor name.

    Returns:
        List of trial summary dicts
    """
    url = "https://clinicaltrials.gov/api/v2/studies"

    # Use query.spons for broader matching (includes partial matches)
    params = {
        'query.spons': sponsor_name,
        'pageSize': min(max_results, 100),
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


def search_company_trials(company_name: str, ticker: str = None, max_results: int = 50) -> List[dict]:
    """
    Search ClinicalTrials.gov for trials by company/sponsor name.

    Uses a multi-step search strategy:
    1. Check SPONSOR_ALIASES for known alternative names (by ticker)
    2. Try the cleaned company name
    3. Deduplicate results by NCT ID

    Args:
        company_name: Company or sponsor name to search
        ticker: Stock ticker (for alias lookup)
        max_results: Maximum number of trials to return

    Returns:
        List of trial summary dicts
    """
    try:
        all_trials = {}  # Use dict to dedupe by NCT ID
        search_terms_used = []

        # Step 1: Check for known aliases by ticker
        if ticker and ticker in SPONSOR_ALIASES:
            aliases = SPONSOR_ALIASES[ticker]
            for alias in aliases:
                try:
                    trials = _search_single_sponsor(alias, max_results)
                    for trial in trials:
                        nct_id = trial.get('nct_id')
                        if nct_id and nct_id not in all_trials:
                            all_trials[nct_id] = trial
                    if trials:
                        search_terms_used.append(alias)
                except Exception:
                    continue

        # Step 2: Try cleaned company name if we haven't found enough trials
        if len(all_trials) < max_results:
            clean_name = _clean_company_name(company_name)
            try:
                trials = _search_single_sponsor(clean_name, max_results)
                for trial in trials:
                    nct_id = trial.get('nct_id')
                    if nct_id and nct_id not in all_trials:
                        all_trials[nct_id] = trial
                if trials:
                    search_terms_used.append(clean_name)
            except Exception:
                pass

        # Step 3: Try original company name if still nothing
        if not all_trials:
            try:
                trials = _search_single_sponsor(company_name, max_results)
                for trial in trials:
                    nct_id = trial.get('nct_id')
                    if nct_id and nct_id not in all_trials:
                        all_trials[nct_id] = trial
                if trials:
                    search_terms_used.append(company_name)
            except Exception:
                pass

        return list(all_trials.values())[:max_results]

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
        ticker: Stock ticker (for alias lookup and reference)
        company_name: Full company name to search

    Returns:
        dict with trial data and provenance
    """
    try:
        # Search for company trials (pass ticker for alias lookup)
        trials = search_company_trials(company_name, ticker=ticker)
        
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
