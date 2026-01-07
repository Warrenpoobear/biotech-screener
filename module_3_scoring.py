"""
Module 3 Catalyst Scoring System
Converts catalyst events into numeric impact scores (0-100 scale)
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Event Severity Classification
# ============================================================================

# Severity mappings: event_type -> severity_level
EVENT_SEVERITY = {
    # Critical positive events
    'PHASE_ADVANCE_P2_TO_P3': 'CRITICAL_POSITIVE',
    'PHASE_ADVANCE_P3_TO_NDA': 'CRITICAL_POSITIVE',
    'FDA_APPROVAL': 'CRITICAL_POSITIVE',
    'BREAKTHROUGH_DESIGNATION': 'CRITICAL_POSITIVE',
    
    # Severe negative events
    'TRIAL_TERMINATION': 'SEVERE_NEGATIVE',
    'TRIAL_SUSPENDED': 'SEVERE_NEGATIVE',
    'FDA_REJECTION': 'SEVERE_NEGATIVE',
    'SAFETY_HOLD': 'SEVERE_NEGATIVE',
    
    # Positive events
    'ENROLLMENT_COMPLETE': 'POSITIVE',
    'ENROLLMENT_STARTED': 'POSITIVE',
    'FAST_TRACK_GRANTED': 'POSITIVE',
    'ORPHAN_DESIGNATION': 'POSITIVE',
    
    # Negative events
    'ENROLLMENT_DELAY': 'NEGATIVE',
    'TRIAL_DELAYED': 'NEGATIVE',
}

# Severity -> Score mapping
SEVERITY_SCORES = {
    'CRITICAL_POSITIVE': 75.0,
    'POSITIVE': 60.0,
    'NEGATIVE': 40.0,
    'SEVERE_NEGATIVE': 20.0,
}

# Default neutral score
DEFAULT_SCORE = 50.0

# ============================================================================
# Ticker-Level Aggregation Logic
# ============================================================================

def calculate_ticker_catalyst_score(ticker: str, summary) -> Dict:
    """
    Calculate aggregate catalyst score for a ticker.
    
    Args:
        ticker: Ticker symbol
        summary: TickerCatalystSummary object (dataclass, NOT dict)
    
    Returns:
        Dict with score and metadata
    """
    # FIXED: Access dataclass attributes directly, not via .get()
    # Check if it's a dict (for backward compatibility) or a dataclass
    if isinstance(summary, dict):
        # Handle dictionary format (test suite)
        events = summary.get('events', [])
    else:
        # Handle dataclass format (production pipeline)
        events = getattr(summary, 'events', [])
    
    # If no events, return neutral score
    if not events:
        return {
            'ticker': ticker,
            'score': DEFAULT_SCORE,
            'reason': 'NO_EVENTS',
            'event_count': 0,
            'severity_breakdown': {}
        }
    
    # Classify events by severity
    severity_counts = {}
    event_details = []
    
    for event in events:
        # Access event fields (handle both dict and dataclass)
        if isinstance(event, dict):
            event_type = event.get('event_type', 'UNKNOWN')
        else:
            event_type = getattr(event, 'event_type', 'UNKNOWN')
        
        severity = EVENT_SEVERITY.get(event_type, 'NEUTRAL')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        event_details.append({
            'event_type': event_type,
            'severity': severity
        })
    
    # Calculate score using hierarchical logic
    score = _calculate_hierarchical_score(severity_counts)
    
    return {
        'ticker': ticker,
        'score': score,
        'reason': _determine_reason(severity_counts),
        'event_count': len(events),
        'severity_breakdown': severity_counts,
        'event_details': event_details
    }


def _calculate_hierarchical_score(severity_counts: Dict[str, int]) -> float:
    """
    Calculate score using hierarchical logic.
    
    Priority rules:
    1. If ANY severe negative -> 20
    2. If ANY critical positive -> 75
    3. Otherwise, blend positive/negative events
    """
    # Rule 1: Severe negatives override everything
    if severity_counts.get('SEVERE_NEGATIVE', 0) > 0:
        return 20.0
    
    # Rule 2: Critical positives (in absence of severe negatives)
    if severity_counts.get('CRITICAL_POSITIVE', 0) > 0:
        return 75.0
    
    # Rule 3: Blend positive and negative events
    positive_count = severity_counts.get('POSITIVE', 0)
    negative_count = severity_counts.get('NEGATIVE', 0)
    
    if positive_count > 0 and negative_count == 0:
        # Pure positive
        return 60.0 + min(10.0, positive_count * 5.0)  # Max 70
    
    if negative_count > 0 and positive_count == 0:
        # Pure negative
        return 40.0 - min(5.0, negative_count * 2.5)  # Min 35
    
    if positive_count > 0 and negative_count > 0:
        # Mixed: slight positive bias
        net_score = 50.0 + (positive_count * 3.0) - (negative_count * 2.0)
        return max(35.0, min(65.0, net_score))
    
    # No classified events
    return DEFAULT_SCORE


def _determine_reason(severity_counts: Dict[str, int]) -> str:
    """Generate human-readable reason for score."""
    if severity_counts.get('SEVERE_NEGATIVE', 0) > 0:
        return 'SEVERE_NEGATIVE_EVENTS'
    if severity_counts.get('CRITICAL_POSITIVE', 0) > 0:
        return 'CRITICAL_POSITIVE_EVENTS'
    if severity_counts.get('POSITIVE', 0) > 0 and severity_counts.get('NEGATIVE', 0) == 0:
        return 'POSITIVE_EVENTS'
    if severity_counts.get('NEGATIVE', 0) > 0 and severity_counts.get('POSITIVE', 0) == 0:
        return 'NEGATIVE_EVENTS'
    if severity_counts.get('POSITIVE', 0) > 0 and severity_counts.get('NEGATIVE', 0) > 0:
        return 'MIXED_EVENTS'
    return 'NO_CLASSIFIED_EVENTS'


# ============================================================================
# Batch Processing Interface
# ============================================================================

def score_catalyst_events(
    catalyst_summaries: Dict,  # ticker -> TickerCatalystSummary
    active_tickers: List[str]
) -> Dict[str, Dict]:
    """
    Score catalyst events for all active tickers.
    
    Args:
        catalyst_summaries: Dict of ticker -> TickerCatalystSummary objects
        active_tickers: List of active tickers to score
    
    Returns:
        Dict of ticker -> score_result
    """
    logger.info(f"Scoring catalyst events for {len(active_tickers)} tickers")
    
    results = {}
    
    for ticker in active_tickers:
        summary = catalyst_summaries.get(ticker)
        
        if summary is None:
            # No events for this ticker
            results[ticker] = {
                'ticker': ticker,
                'score': DEFAULT_SCORE,
                'reason': 'NO_SUMMARY',
                'event_count': 0,
                'severity_breakdown': {}
            }
        else:
            # Calculate score from summary
            score_result = calculate_ticker_catalyst_score(ticker, summary)
            results[ticker] = score_result
    
    logger.info(f"Scoring complete. Generated {len(results)} scores")
    
    return results


# ============================================================================
# Validation & Diagnostics
# ============================================================================

def validate_scores(scores: Dict[str, Dict]) -> Dict:
    """
    Validate score outputs for sanity.
    
    Returns:
        Dict with validation metrics
    """
    if not scores:
        return {
            'valid': False,
            'reason': 'NO_SCORES'
        }
    
    score_values = [s['score'] for s in scores.values() if 'score' in s]
    
    # Check for differentiation
    unique_scores = len(set(score_values))
    
    return {
        'valid': True,
        'total_tickers': len(scores),
        'unique_scores': unique_scores,
        'min_score': min(score_values) if score_values else None,
        'max_score': max(score_values) if score_values else None,
        'mean_score': sum(score_values) / len(score_values) if score_values else None,
    }
