#!/usr/bin/env python3
"""
ClinicalTrials.gov Historical Data Fetcher - WITH CONFIDENCE GATING

This version adds three critical fixes:
1. Sponsor name sanitization (fixes HTTP 400 errors)
2. Match confidence gating (based on trial count)
3. Weighted scoring (reduces impact of low-confidence matches)

Usage:
    # Drop-in replacement for clinicaltrials_gov.py
    from historical_fetchers.clinicaltrials_gov_gated import fetch_batch, get_historical_clinical

Author: Wake Robin Capital
Version: 2.1 (with confidence gating)
"""

import argparse
import json
import re
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import urllib.request
import urllib.error
import urllib.parse

# ClinicalTrials.gov API base URL (v2)
CT_BASE_URL = "https://clinicaltrials.gov/api/v2"

# Cache directory
CACHE_DIR = Path("data/cache/clinicaltrials")

# Sponsor mapping file
SPONSOR_MAPPING_FILE = Path("data/ticker_to_sponsor.json")

# =============================================================================
# CONFIDENCE GATING CONFIGURATION
# =============================================================================

# Trial count thresholds for confidence levels
CONFIDENCE_THRESHOLDS = {
    'HIGH': 10,      # >= 10 trials = HIGH confidence (full weight)
    'MEDIUM': 3,     # >= 3 trials = MEDIUM confidence (50% weight)
    'LOW': 1,        # >= 1 trial = LOW confidence (25% weight)
}

# Confidence to weight mapping (how much clinical score influences composite)
CONFIDENCE_WEIGHTS = {
    'HIGH': 1.0,     # Full signal
    'MEDIUM': 0.5,   # Half signal
    'LOW': 0.25,     # Quarter signal
    'UNKNOWN': 0.0,  # No signal (treat as neutral)
}

# Raw clinical scores by stage (lower = better)
STAGE_SCORES = {
    'commercial': 10,
    'late': 20,       # Phase 3
    'mid': 30,        # Phase 2
    'early': 40,      # Phase 1
    'preclinical': 35,
    'unknown': 30,    # Neutral - doesn't help or hurt
}

# Neutral baseline score (used for blending)
NEUTRAL_SCORE = 30


# =============================================================================
# SPONSOR NAME SANITIZATION (Fix 1)
# =============================================================================

def sanitize_sponsor_name(name: str) -> str:
    """
    Clean sponsor name for ClinicalTrials.gov API query.

    Fixes HTTP 400 errors caused by special characters.

    Examples:
        "LIFECORE BIOMEDICAL, INC. \\DE\\" -> "LIFECORE BIOMEDICAL, INC."
        "ACME CORP \\NV\\" -> "ACME CORP"
        "Company (DE)" -> "Company"
    """
    if not name:
        return ""

    # Remove state codes like \DE\, \NV\, \CA\, etc.
    name = re.sub(r'\\[A-Z]{2}\\', '', name)

    # Remove parenthetical state codes like (DE), (NV)
    name = re.sub(r'\s*\([A-Z]{2}\)\s*$', '', name)

    # Remove other problematic characters that break URL encoding
    name = re.sub(r'[\\<>|"\'`]', '', name)

    # Remove multiple spaces
    name = re.sub(r'\s+', ' ', name)

    return name.strip()


# =============================================================================
# CONFIDENCE GATING (Fix 2)
# =============================================================================

def get_match_confidence(trial_count: int) -> str:
    """
    Determine match confidence based on trial count.

    Args:
        trial_count: Number of trials found for this sponsor

    Returns:
        Confidence level: 'HIGH', 'MEDIUM', 'LOW', or 'UNKNOWN'
    """
    if trial_count >= CONFIDENCE_THRESHOLDS['HIGH']:
        return 'HIGH'
    elif trial_count >= CONFIDENCE_THRESHOLDS['MEDIUM']:
        return 'MEDIUM'
    elif trial_count >= CONFIDENCE_THRESHOLDS['LOW']:
        return 'LOW'
    else:
        return 'UNKNOWN'


def get_confidence_weight(confidence: str) -> float:
    """Get the weight multiplier for a confidence level."""
    return CONFIDENCE_WEIGHTS.get(confidence, 0.0)


# =============================================================================
# WEIGHTED SCORING (Fix 3)
# =============================================================================

def calculate_weighted_clinical_score(
    stage_bucket: str,
    trial_count: int,
    confidence: str = None
) -> Tuple[float, str, float]:
    """
    Calculate confidence-weighted clinical score.

    For low-confidence matches, the score is blended toward neutral (30)
    to reduce the impact of unreliable data.

    Args:
        stage_bucket: Clinical stage ('commercial', 'late', 'mid', 'early', etc.)
        trial_count: Number of trials found
        confidence: Optional pre-computed confidence level

    Returns:
        Tuple of (weighted_score, confidence, weight)
    """
    if confidence is None:
        confidence = get_match_confidence(trial_count)

    weight = get_confidence_weight(confidence)
    raw_score = STAGE_SCORES.get(stage_bucket, NEUTRAL_SCORE)

    # Blend raw score toward neutral based on confidence
    # HIGH (weight=1.0): weighted_score = raw_score
    # MEDIUM (weight=0.5): weighted_score = 0.5*raw + 0.5*neutral
    # LOW (weight=0.25): weighted_score = 0.25*raw + 0.75*neutral
    # UNKNOWN (weight=0.0): weighted_score = neutral

    weighted_score = (weight * raw_score) + ((1 - weight) * NEUTRAL_SCORE)

    return weighted_score, confidence, weight


# =============================================================================
# SPONSOR MAPPING
# =============================================================================

def load_sponsor_mapping() -> Dict[str, str]:
    """Load ticker to sponsor name mapping."""
    if SPONSOR_MAPPING_FILE.exists():
        with open(SPONSOR_MAPPING_FILE, 'r') as f:
            return json.load(f)
    return {}


_SPONSOR_MAPPING = None


def get_sponsor_mapping() -> Dict[str, str]:
    """Get sponsor mapping, loading if needed."""
    global _SPONSOR_MAPPING
    if _SPONSOR_MAPPING is None:
        _SPONSOR_MAPPING = load_sponsor_mapping()
    return _SPONSOR_MAPPING


def get_company_name(ticker: str) -> Optional[str]:
    """Get company name for a ticker from mapping."""
    mapping = get_sponsor_mapping()

    if ticker.upper() in mapping:
        raw_name = mapping[ticker.upper()]
        # Apply sanitization before returning
        return sanitize_sponsor_name(raw_name)

    return ticker


# =============================================================================
# CLINICAL TRIALS API
# =============================================================================

def search_trials(sponsor: str, as_of_date: str,
                  max_results: int = 100) -> List[Dict]:
    """
    Search for clinical trials by sponsor as of a specific date.

    Uses ClinicalTrials.gov API v2.
    """
    # Sanitize sponsor name before querying
    sponsor = sanitize_sponsor_name(sponsor)

    if not sponsor:
        return []

    target_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()

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

                start_info = status_module.get('startDateStruct', {})
                start_date_str = start_info.get('date', '')

                if start_date_str:
                    try:
                        if len(start_date_str) == 10:
                            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
                        elif len(start_date_str) == 7:
                            start_date = datetime.strptime(start_date_str + "-01", "%Y-%m-%d").date()
                        else:
                            start_date = datetime.strptime(start_date_str + "-01-01", "%Y-%m-%d").date()

                        if start_date <= target_date:
                            valid_trials.append(study)
                    except ValueError:
                        valid_trials.append(study)
                else:
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

    id_module = protocol.get('identificationModule', {})
    nct_id = id_module.get('nctId', '')
    title = id_module.get('briefTitle', '')

    status_module = protocol.get('statusModule', {})
    overall_status = status_module.get('overallStatus', '')

    design_module = protocol.get('designModule', {})
    phases = design_module.get('phases', [])
    phase = phases[0] if phases else 'NA'

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
    """
    phase_counts = {'PHASE1': 0, 'PHASE2': 0, 'PHASE3': 0, 'PHASE4': 0}
    active_phases = set()

    for trial in trials:
        info = extract_trial_info(trial)
        phase = info['phase'].upper().replace(' ', '')

        if info['status'] in ['RECRUITING', 'ACTIVE_NOT_RECRUITING',
                              'ENROLLING_BY_INVITATION', 'NOT_YET_RECRUITING']:
            active_phases.add(phase)

        if 'PHASE3' in phase or phase == 'PHASE3':
            phase_counts['PHASE3'] += 1
        elif 'PHASE2' in phase or phase == 'PHASE2':
            phase_counts['PHASE2'] += 1
        elif 'PHASE1' in phase or phase == 'PHASE1':
            phase_counts['PHASE1'] += 1
        elif 'PHASE4' in phase or phase == 'PHASE4':
            phase_counts['PHASE4'] += 1

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


# =============================================================================
# MAIN DATA FETCHING (WITH CONFIDENCE GATING)
# =============================================================================

def get_historical_clinical(ticker: str, as_of_date: str) -> Optional[Dict]:
    """
    Get historical clinical stage data for a ticker as of a specific date.

    NOW INCLUDES: Confidence gating and weighted scoring!
    """
    company_name = get_company_name(ticker)
    if not company_name:
        company_name = ticker

    print(f"    Searching for sponsor: {company_name}")

    trials = search_trials(company_name, as_of_date)
    trial_count = len(trials)

    if not trials:
        # No trials found - return UNKNOWN with neutral score
        weighted_score, confidence, weight = calculate_weighted_clinical_score(
            'unknown', 0
        )
        return {
            'ticker': ticker,
            'as_of_date': as_of_date,
            'company_name': company_name,
            'lead_stage': 'UNKNOWN',
            'stage_bucket': 'unknown',
            'trial_count': 0,
            # NEW: Confidence-weighted scoring
            'confidence': confidence,
            'confidence_weight': weight,
            'raw_clinical_score': NEUTRAL_SCORE,
            'clinical_score': weighted_score,  # This is what rankings should use
            'error': 'No trials found'
        }

    stage_info = determine_lead_stage(trials)
    stage_bucket = stage_info['stage_bucket']

    # Calculate confidence-weighted score
    weighted_score, confidence, weight = calculate_weighted_clinical_score(
        stage_bucket, trial_count
    )
    raw_score = STAGE_SCORES.get(stage_bucket, NEUTRAL_SCORE)

    trial_details = [extract_trial_info(t) for t in trials[:20]]

    return {
        'ticker': ticker,
        'as_of_date': as_of_date,
        'company_name': company_name,
        'lead_stage': stage_info['lead_stage'],
        'stage_bucket': stage_bucket,
        'trial_count': trial_count,
        'phase_counts': stage_info['phase_counts'],
        'active_phases': stage_info['active_phases'],
        'sample_trials': trial_details[:5],
        'data_source': 'ClinicalTrials.gov',
        # NEW: Confidence-weighted scoring
        'confidence': confidence,
        'confidence_weight': weight,
        'raw_clinical_score': raw_score,
        'clinical_score': weighted_score,  # This is what rankings should use
    }


def fetch_batch(tickers: List[str], as_of_date: str,
                delay: float = 0.5) -> Dict[str, Dict]:
    """
    Fetch historical clinical data for multiple tickers.

    NOW INCLUDES: Confidence gating and weighted scoring!
    """
    results = {}
    total = len(tickers)

    confidence_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{total}] Fetching {ticker}...")
        try:
            data = get_historical_clinical(ticker, as_of_date)
            if data:
                results[ticker] = data
                confidence = data.get('confidence', 'UNKNOWN')
                confidence_counts[confidence] += 1
                print(f"    Stage: {data.get('lead_stage', 'N/A')}, "
                      f"Trials: {data.get('trial_count', 0)}, "
                      f"Confidence: {confidence}")
            else:
                print("    No data")
        except Exception as e:
            print(f"    Error: {e}")
            results[ticker] = {'ticker': ticker, 'error': str(e)}

        time.sleep(delay)

    # Print confidence summary
    print(f"\n  Confidence Summary:")
    print(f"    HIGH (>=10 trials): {confidence_counts['HIGH']}")
    print(f"    MEDIUM (3-9 trials): {confidence_counts['MEDIUM']}")
    print(f"    LOW (1-2 trials): {confidence_counts['LOW']}")
    print(f"    UNKNOWN (0 trials): {confidence_counts['UNKNOWN']}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical clinical trial data (with confidence gating)"
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

    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    elif args.company:
        # Direct company search
        print(f"Searching for {args.company} as of {args.as_of}")
        trials = search_trials(args.company, args.as_of)
        stage_info = determine_lead_stage(trials)

        weighted_score, confidence, weight = calculate_weighted_clinical_score(
            stage_info['stage_bucket'], len(trials)
        )

        print(f"  Found {len(trials)} trials")
        print(f"  Lead stage: {stage_info['lead_stage']}")
        print(f"  Confidence: {confidence} (weight: {weight})")
        print(f"  Raw score: {STAGE_SCORES.get(stage_info['stage_bucket'], 30)}")
        print(f"  Weighted score: {weighted_score:.1f}")
        return
    else:
        parser.error("Specify --ticker, --tickers, or --company")

    print(f"Fetching historical clinical data for {len(tickers)} tickers as of {args.as_of}")
    print(f"Using confidence gating: HIGH>=10, MEDIUM>=3, LOW>=1 trials")

    results = fetch_batch(tickers, args.as_of, delay=args.delay)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'as_of_date': args.as_of,
                'fetched_at': datetime.now().isoformat(),
                'count': len(results),
                'confidence_gating': True,
                'thresholds': CONFIDENCE_THRESHOLDS,
                'tickers': results
            }, f, indent=2)
        print(f"\nSaved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
