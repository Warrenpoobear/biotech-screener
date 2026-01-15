"""
module_5_composite_FIXED_WITH_TIEBREAKER.py

CRITICAL FIXES:
1. Deterministic tie-breaker for identical composite scores
2. Proper handling of None scores (missingness penalty instead of default 50.0)
3. Maintains all existing functionality

CHANGES FROM ORIGINAL:
- Line ~400: Added store_unrounded_scores() to preserve full precision
- Line ~450: Added deterministic_sort() for tie-breaking
- Line ~100: Added handle_missing_scores() for None values

Tie-break order when composite scores are identical:
  1. Unrounded composite (higher precision)
  2. Clinical score
  3. Financial score
  4. Ticker (alphabetical)
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any


def handle_missing_scores(
    financial_score: Optional[float],
    clinical_score: Optional[float],
    catalyst_score: Optional[float],
) -> tuple:
    """
    Handle missing scores with explicit penalties instead of defaults.
    
    FIX #2: Instead of defaulting to 50.0, apply explicit penalties:
    - Missing financial → 30.0 (mild penalty, some biotechs are pre-revenue)
    - Missing clinical → 20.0 (severe penalty, clinical is core)
    - Missing catalyst → 50.0 (neutral, expected when no events)
    
    Returns:
        (financial, clinical, catalyst, missingness_flags)
    """
    flags = []
    
    # Financial: Mild penalty
    if financial_score is None:
        financial_score = 30.0
        flags.append("missing_financial")
    
    # Clinical: Severe penalty (this is core)
    if clinical_score is None:
        clinical_score = 20.0
        flags.append("missing_clinical")
    
    # Catalyst: Neutral (often no events)
    if catalyst_score is None:
        catalyst_score = 50.0
        # Don't flag as missing if it's just no events
    
    return financial_score, clinical_score, catalyst_score, flags


def store_unrounded_scores(securities: List[Dict]) -> None:
    """
    Store full-precision composite scores for tie-breaking.
    
    FIX #4: Preserve unrounded scores before quantization.
    """
    for sec in securities:
        # Store full precision before any rounding
        if 'composite_score' in sec:
            sec['composite_score_unrounded'] = sec['composite_score']


def deterministic_sort(securities: List[Dict]) -> List[Dict]:
    """
    Sort with deterministic tie-breaking.
    
    FIX #4: When composite scores tie, use this order:
    1. Unrounded composite (higher precision)
    2. Clinical score (higher is better)
    3. Financial score (higher is better)
    4. Ticker (alphabetical, stable)
    
    This ensures identical rankings across runs and prevents
    random portfolio membership changes due to hash-based dict ordering.
    """
    def sort_key(sec):
        # Extract scores with defaults for missing values
        composite_unrounded = float(sec.get('composite_score_unrounded', sec.get('composite_score', 0)))
        clinical = float(sec.get('clinical_dev_normalized', 0))
        financial = float(sec.get('financial_normalized', 0))
        ticker = sec.get('ticker', '')
        
        # Return tuple for sorting
        # Use negative for scores (higher is better)
        # Use positive for ticker (alphabetical)
        return (
            -composite_unrounded,  # Primary: highest composite wins
            -clinical,              # Tie-break 1: highest clinical wins
            -financial,             # Tie-break 2: highest financial wins
            ticker                  # Tie-break 3: alphabetical (stable)
        )
    
    return sorted(securities, key=sort_key)


def compute_composite_scores(
    financial_scores: List[Dict],
    clinical_scores: List[Dict],
    catalyst_scores: List[Dict],
    weights: Dict[str, float] = None
) -> List[Dict]:
    """
    Compute composite scores with proper None handling and tie-breaking.
    
    INTEGRATION POINT: Use this function in your Module 5 scoring.
    """
    if weights is None:
        weights = {
            'financial': 0.35,
            'clinical': 0.40,
            'catalyst': 0.25
        }
    
    # Create lookups
    fin_lookup = {s['ticker']: s for s in financial_scores}
    clin_lookup = {s['ticker']: s for s in clinical_scores}
    cat_lookup = {s['ticker']: s for s in catalyst_scores}
    
    # Get all tickers
    all_tickers = set(fin_lookup.keys()) | set(clin_lookup.keys()) | set(cat_lookup.keys())
    
    results = []
    
    for ticker in all_tickers:
        # Get scores (may be None)
        fin_data = fin_lookup.get(ticker, {})
        clin_data = clin_lookup.get(ticker, {})
        cat_data = cat_lookup.get(ticker, {})
        
        fin_score = fin_data.get('financial_normalized')
        clin_score = clin_data.get('clinical_score')
        cat_score = cat_data.get('score')
        
        # Handle missing scores with penalties
        fin_score, clin_score, cat_score, miss_flags = handle_missing_scores(
            fin_score, clin_score, cat_score
        )
        
        # Compute weighted composite
        composite = (
            fin_score * weights['financial'] +
            clin_score * weights['clinical'] +
            cat_score * weights['catalyst']
        )
        
        results.append({
            'ticker': ticker,
            'composite_score': composite,
            'composite_score_unrounded': composite,  # Store full precision
            'financial_normalized': fin_score,
            'clinical_dev_normalized': clin_score,
            'catalyst_normalized': cat_score,
            'missingness_flags': miss_flags,
            'weights': weights.copy()
        })
    
    # Apply deterministic sort
    results = deterministic_sort(results)
    
    # Add ranks
    for i, sec in enumerate(results, 1):
        sec['composite_rank'] = i
    
    return results


# EXAMPLE INTEGRATION INTO YOUR MODULE 5:
"""
In your module_5_composite.py, replace the scoring section with:

# Import at top
from module_5_composite_FIXED_WITH_TIEBREAKER import (
    compute_composite_scores,
    handle_missing_scores,
    deterministic_sort
)

# In your main function:
ranked_securities = compute_composite_scores(
    financial_scores=module_2_results,
    clinical_scores=module_4_results['scores'],
    catalyst_scores=module_3_results['scored'],
    weights={
        'financial': 0.35,
        'clinical': 0.40,
        'catalyst': 0.25
    }
)

# ranked_securities is now properly sorted with tie-breaking!
"""


def test_tie_breaking():
    """Test deterministic tie-breaking."""
    
    # Create test data with ties
    securities = [
        {'ticker': 'AAA', 'composite_score': 75.0, 'composite_score_unrounded': 75.02, 'clinical_dev_normalized': 80, 'financial_normalized': 70},
        {'ticker': 'BBB', 'composite_score': 75.0, 'composite_score_unrounded': 75.01, 'clinical_dev_normalized': 80, 'financial_normalized': 70},
        {'ticker': 'CCC', 'composite_score': 75.0, 'composite_score_unrounded': 75.03, 'clinical_dev_normalized': 80, 'financial_normalized': 70},
        {'ticker': 'DDD', 'composite_score': 75.0, 'composite_score_unrounded': 75.01, 'clinical_dev_normalized': 85, 'financial_normalized': 65},
    ]
    
    print("BEFORE sorting (all show 75.0):")
    for s in securities:
        print(f"  {s['ticker']}: {s['composite_score']:.2f}")
    
    sorted_secs = deterministic_sort(securities)
    
    print("\nAFTER deterministic sort:")
    for s in sorted_secs:
        print(f"  {s['ticker']}: {s['composite_score']:.2f} (unrounded: {s['composite_score_unrounded']:.4f}, clin: {s['clinical_dev_normalized']}, fin: {s['financial_normalized']})")
    
    print("\nExpected order: CCC (75.03), AAA (75.02), DDD (75.01, higher clinical), BBB (75.01, lower clinical)")
    print(f"Actual order: {', '.join([s['ticker'] for s in sorted_secs])}")


def test_missing_handling():
    """Test missing score penalties."""
    
    print("\nTesting missing score handling:\n")
    
    # Test 1: All scores present
    f, c, cat, flags = handle_missing_scores(80.0, 75.0, 60.0)
    print(f"All present: fin={f}, clin={c}, cat={cat}, flags={flags}")
    
    # Test 2: Missing financial
    f, c, cat, flags = handle_missing_scores(None, 75.0, 60.0)
    print(f"Missing fin: fin={f}, clin={c}, cat={cat}, flags={flags}")
    
    # Test 3: Missing clinical (severe)
    f, c, cat, flags = handle_missing_scores(80.0, None, 60.0)
    print(f"Missing clin: fin={f}, clin={c}, cat={cat}, flags={flags}")
    
    # Test 4: Missing all
    f, c, cat, flags = handle_missing_scores(None, None, None)
    print(f"Missing all: fin={f}, clin={c}, cat={cat}, flags={flags}")


if __name__ == "__main__":
    print("="*80)
    print("MODULE 5 FIXES - TIE-BREAKING & NONE HANDLING TESTS")
    print("="*80)
    
    test_tie_breaking()
    print("\n" + "="*80)
    test_missing_handling()
    print("="*80)

# Compatibility wrapper for run_screen.py
def compute_module_5_composite(universe_result=None, financial_result=None, clinical_result=None, 
                                catalyst_result=None, weights=None, **kwargs):
    """
    Wrapper to handle run_screen.py calling convention.
    
    run_screen.py passes module results (dicts with 'scores' keys)
    compute_composite_scores expects lists of score dicts
    """
    # Extract scores from module results
    financial_scores = []
    if financial_result and isinstance(financial_result, dict):
        financial_scores = financial_result.get('scores', [])
    
    clinical_scores = []
    if clinical_result and isinstance(clinical_result, dict):
        clinical_scores = clinical_result.get('scores', [])
    
    catalyst_scores = []
    if catalyst_result and isinstance(catalyst_result, dict):
        catalyst_scores = catalyst_result.get('scored', [])
    
    # Use default weights if not provided
    if weights is None:
        weights = {'financial': 0.35, 'clinical': 0.40, 'catalyst': 0.25}
    
    # Call the actual function
    result = compute_composite_scores(
        financial_scores=financial_scores,
        clinical_scores=clinical_scores,
        catalyst_scores=catalyst_scores,
        weights=weights
    )
    
    # Wrap in expected format for run_screen.py
    if isinstance(result, list):
        return {
            'ranked_securities': result,
            'diagnostic_counts': {'rankable': len(result), 'excluded': 0}
        }
    return result