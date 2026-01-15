"""
Refactored ClinicalTrials.gov integration with confidence-weighted scoring.

Key improvements:
1. Sponsor name sanitization (fixes HTTP 400s)
2. Match confidence gating based on trial count
3. Clinical weight shrinkage for low-confidence matches
4. Proper handling of big pharma vs focused biotech

Usage:
    from clinicaltrials_gov_refactored import get_clinical_data_with_confidence
    
    data = get_clinical_data_with_confidence('VRTX')
    # Returns: {
    #     'stage': 'PHASE3',
    #     'trial_count': 45,
    #     'confidence': 'HIGH',
    #     'clinical_weight': 1.0,
    #     'raw_score': 20,
    #     'weighted_score': 20
    # }
"""

import re
import requests
import urllib.parse
from typing import Optional, Dict, Tuple
from pathlib import Path
import json


CT_BASE_URL = "https://clinicaltrials.gov/api/v2"


def sanitize_sponsor_name(name: str) -> str:
    """
    Remove special characters that break ClinicalTrials.gov API.
    
    Fixes HTTP 400 errors from names like:
    - "LIFECORE BIOMEDICAL, INC. \\DE\\"
    - "Company Name \\NV\\"
    """
    # Remove state incorporation codes (e.g., \DE\, \NV\)
    name = re.sub(r'\\[A-Z]{2}\\', '', name)
    
    # Remove other problematic characters
    name = re.sub(r'[\\<>|]', '', name)
    
    # Normalize whitespace
    name = ' '.join(name.split())
    
    return name.strip()


def get_sponsor_mapping() -> Dict[str, str]:
    """Load ticker → sponsor name mapping."""
    mapping_path = Path('data/ticker_to_sponsor.json')
    
    if not mapping_path.exists():
        return {}
    
    with open(mapping_path, 'r') as f:
        return json.load(f)


def get_company_name(ticker: str) -> Optional[str]:
    """
    Get company name for ticker, with sanitization.
    
    Returns sanitized sponsor name suitable for ClinicalTrials.gov API.
    """
    mapping = get_sponsor_mapping()
    
    if ticker.upper() in mapping:
        raw_name = mapping[ticker.upper()]
        return sanitize_sponsor_name(raw_name)
    
    # Fallback to ticker (likely to fail, but sanitize anyway)
    return sanitize_sponsor_name(ticker)


def search_clinical_trials(sponsor: str, max_results: int = 100) -> Dict:
    """
    Search ClinicalTrials.gov for sponsor's trials.
    
    Returns raw API response with trial data.
    """
    # Sanitize sponsor name before sending to API
    sponsor_clean = sanitize_sponsor_name(sponsor)
    
    params = {
        'query.spons': sponsor_clean,
        'filter.overallStatus': 'RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED,ENROLLING_BY_INVITATION',
        'pageSize': max_results,
        'format': 'json'
    }
    
    url = f"{CT_BASE_URL}/studies"
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️  API Error for {sponsor}: {e}")
        return {'studies': [], 'totalCount': 0}


def get_lead_stage_from_trials(trials: list) -> str:
    """
    Determine most advanced clinical stage from trial list.
    
    Priority: PHASE4 > PHASE3 > PHASE2 > PHASE1 > EARLY_PHASE1 > NA
    """
    stage_priority = {
        'PHASE4': 1,
        'PHASE3': 2,
        'PHASE2': 3, 
        'PHASE1': 4,
        'EARLY_PHASE1': 5,
        'NA': 6
    }
    
    if not trials:
        return 'UNKNOWN'
    
    # Find most advanced stage
    lead_stage = 'UNKNOWN'
    best_priority = 999
    
    for trial in trials:
        try:
            phase = trial.get('protocolSection', {}).get('designModule', {}).get('phases', [])
            if phase and len(phase) > 0:
                trial_phase = phase[0]
                priority = stage_priority.get(trial_phase, 999)
                if priority < best_priority:
                    lead_stage = trial_phase
                    best_priority = priority
        except (KeyError, IndexError):
            continue
    
    # Map to simplified stages
    stage_map = {
        'PHASE4': 'COMMERCIAL',
        'PHASE3': 'PHASE3',
        'PHASE2': 'PHASE2',
        'PHASE1': 'PHASE1',
        'EARLY_PHASE1': 'PHASE1',
        'NA': 'UNKNOWN'
    }
    
    return stage_map.get(lead_stage, 'UNKNOWN')


def get_match_confidence(trial_count: int) -> str:
    """
    Assess confidence in clinical data match based on trial count.
    
    HIGH: ≥10 trials (strong signal)
    MEDIUM: 3-9 trials (moderate signal)
    LOW: 1-2 trials (weak signal)
    UNKNOWN: 0 trials (no signal)
    """
    if trial_count >= 10:
        return 'HIGH'
    elif trial_count >= 3:
        return 'MEDIUM'
    elif trial_count >= 1:
        return 'LOW'
    else:
        return 'UNKNOWN'


def get_clinical_weight(confidence: str) -> float:
    """
    Get weight multiplier based on match confidence.
    
    HIGH: 1.0 (full weight)
    MEDIUM: 0.5 (half weight)
    LOW: 0.25 (quarter weight)
    UNKNOWN: 0.0 (no weight - neutral)
    """
    weights = {
        'HIGH': 1.0,
        'MEDIUM': 0.5,
        'LOW': 0.25,
        'UNKNOWN': 0.0
    }
    return weights.get(confidence, 0.0)


def get_raw_stage_score(stage: str) -> int:
    """
    Get raw clinical stage score (before confidence weighting).
    
    Lower score = better (follows penalty-style scoring).
    
    COMMERCIAL: 10 (best - proven product)
    PHASE3: 20 (good - near approval)
    PHASE2: 30 (moderate - mid-stage)
    PHASE1: 40 (early - high risk)
    UNKNOWN: 30 (neutral - no penalty, no bonus)
    """
    scores = {
        'COMMERCIAL': 10,
        'PHASE3': 20,
        'PHASE2': 30,
        'PHASE1': 40,
        'UNKNOWN': 30  # Neutral baseline
    }
    return scores.get(stage, 30)


def get_clinical_data_with_confidence(ticker: str) -> Dict:
    """
    Get clinical trial data with confidence-weighted scoring.
    
    Returns dict with:
        - stage: Clinical stage (COMMERCIAL, PHASE3, PHASE2, PHASE1, UNKNOWN)
        - trial_count: Number of trials found
        - confidence: Match confidence (HIGH, MEDIUM, LOW, UNKNOWN)
        - clinical_weight: Weight multiplier (0.0 to 1.0)
        - raw_score: Unweighted stage score
        - weighted_score: Final confidence-weighted score
        - sponsor: Sponsor name used for search
    """
    # Get sponsor name
    sponsor = get_company_name(ticker)
    if not sponsor:
        return {
            'stage': 'UNKNOWN',
            'trial_count': 0,
            'confidence': 'UNKNOWN',
            'clinical_weight': 0.0,
            'raw_score': 30,
            'weighted_score': 30,
            'sponsor': None
        }
    
    # Search trials
    data = search_clinical_trials(sponsor, max_results=100)
    trials = data.get('studies', [])
    trial_count = len(trials)
    
    # Determine stage
    stage = get_lead_stage_from_trials(trials)
    
    # Assess confidence
    confidence = get_match_confidence(trial_count)
    clinical_weight = get_clinical_weight(confidence)
    
    # Calculate scores
    raw_score = get_raw_stage_score(stage)
    
    # Weighted score calculation:
    # - HIGH confidence: Use raw score as-is
    # - MEDIUM/LOW confidence: Blend toward neutral (30)
    # - UNKNOWN: Use neutral score (30)
    
    if confidence == 'UNKNOWN':
        weighted_score = 30  # Neutral
    else:
        # Blend between raw score and neutral based on weight
        weighted_score = (raw_score * clinical_weight) + (30 * (1 - clinical_weight))
    
    return {
        'stage': stage,
        'trial_count': trial_count,
        'confidence': confidence,
        'clinical_weight': clinical_weight,
        'raw_score': raw_score,
        'weighted_score': weighted_score,
        'sponsor': sponsor
    }


def format_clinical_summary(data: Dict) -> str:
    """Format clinical data for human-readable output."""
    
    confidence_icon = {
        'HIGH': '🟢',
        'MEDIUM': '🟡',
        'LOW': '🟠',
        'UNKNOWN': '⚪'
    }
    
    icon = confidence_icon.get(data['confidence'], '?')
    
    if data['confidence'] == 'UNKNOWN':
        return f"{icon} Stage: {data['stage']}, Trials: {data['trial_count']}, Confidence: {data['confidence']} (neutral)"
    else:
        return (
            f"{icon} Stage: {data['stage']}, Trials: {data['trial_count']}, "
            f"Confidence: {data['confidence']} ({data['clinical_weight']:.0%} weight), "
            f"Score: {data['raw_score']} → {data['weighted_score']:.1f}"
        )


# Example usage and testing
if __name__ == '__main__':
    test_tickers = ['VRTX', 'SNY', 'XERS', 'CRSP']
    
    print("Testing confidence-weighted clinical scoring:\n")
    
    for ticker in test_tickers:
        print(f"{ticker}:")
        data = get_clinical_data_with_confidence(ticker)
        print(f"  {format_clinical_summary(data)}")
        print()
