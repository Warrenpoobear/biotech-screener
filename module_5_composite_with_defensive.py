"""
module_5_composite_with_defensive.py

Simple wrapper that adds defensive overlays to your existing Module 5.

USAGE:
------
Replace your existing call:
    output = rank_securities(scores_by_ticker, active_tickers, as_of_date, ...)

With:
    from module_5_composite_with_defensive import rank_securities_with_defensive
    output = rank_securities_with_defensive(scores_by_ticker, active_tickers, as_of_date, ...)

That's it! Everything else stays the same.
"""

from module_5_composite import rank_securities
from defensive_overlay_adapter import enrich_with_defensive_overlays, validate_defensive_integration


def rank_securities_with_defensive(
    scores_by_ticker: dict,
    active_tickers: set,
    as_of_date: str,
    normalization: str = "cohort",
    weights: dict = None,
    cohort_mode: str = "stage_only",
    coinvest_signals: dict = None,
    apply_defensive_multiplier: bool = True,
    apply_position_sizing: bool = True,
    validate: bool = False,
) -> dict:
    """
    Rank securities with defensive overlays integrated.
    
    This is a drop-in replacement for module_5_composite.rank_securities()
    that adds defensive overlay functionality.
    
    Args:
        ... (same as rank_securities)
        apply_defensive_multiplier: Apply correlation-based score adjustments
        apply_position_sizing: Calculate inverse-vol position weights
        validate: Print validation diagnostics
    
    Returns:
        Same structure as rank_securities() with added fields:
        - defensive_notes: List of defensive adjustments per security
        - defensive_multiplier: The multiplier applied
        - position_weight: Calculated position size (if sizing enabled)
        - composite_score_before_defensive: Original score before adjustments
    """
    # Call your existing Module 5
    output = rank_securities(
        scores_by_ticker=scores_by_ticker,
        active_tickers=active_tickers,
        as_of_date=as_of_date,
        normalization=normalization,
        weights=weights,
        cohort_mode=cohort_mode,
        coinvest_signals=coinvest_signals,
    )
    
    # Add defensive overlays
    enrich_with_defensive_overlays(
        output,
        scores_by_ticker,
        apply_multiplier=apply_defensive_multiplier,
        apply_position_sizing=apply_position_sizing,
    )
    
    # Optionally validate
    if validate:
        validate_defensive_integration(output)
    
    return output


# Convenience exports
__all__ = [
    "rank_securities_with_defensive",
    "enrich_with_defensive_overlays",
    "validate_defensive_integration",
]


if __name__ == "__main__":
    print("""
    module_5_composite_with_defensive.py
    
    This is a wrapper that adds defensive overlays to your existing Module 5.
    
    INTEGRATION STEPS:
    ------------------
    
    1. Ensure these files are in the same directory:
       - module_5_composite.py (your existing file)
       - defensive_overlay_adapter.py (helper functions)
       - module_5_composite_with_defensive.py (this wrapper)
    
    2. In your runner/pipeline code, change:
       
       FROM:
           from module_5_composite import rank_securities
           output = rank_securities(scores_by_ticker, active_tickers, as_of_date, ...)
       
       TO:
           from module_5_composite_with_defensive import rank_securities_with_defensive
           output = rank_securities_with_defensive(scores_by_ticker, active_tickers, as_of_date, ...)
    
    3. That's it! The output will now include:
       - Defensive score adjustments (correlation-based)
       - Position weights (inverse-vol with caps)
       - Defensive notes explaining adjustments
    
    OPTIONAL: Add validation=True to see diagnostics:
        output = rank_securities_with_defensive(..., validate=True)
    
    TESTING:
    --------
    Run this file directly to test the integration:
        python module_5_composite_with_defensive.py
    """)
