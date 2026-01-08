"""
module_5_composite_with_defensive.py

Simple wrapper that adds defensive overlays to your existing Module 5.
"""

import json
from pathlib import Path
from module_5_composite import compute_module_5_composite
from defensive_overlay_adapter import enrich_with_defensive_overlays, validate_defensive_integration


from module_5_composite import compute_module_5_composite
from defensive_overlay_adapter import enrich_with_defensive_overlays, validate_defensive_integration


def compute_module_5_composite_with_defensive(
    universe_result: dict,
    financial_result: dict,
    catalyst_result: dict,
    clinical_result: dict,
    as_of_date: str,
    weights: dict = None,
    normalization: str = "rank",
    coinvest_signals: dict = None,
    cohort_mode: str = "stage_only",
    apply_defensive_multiplier: bool = False,
    apply_position_sizing: bool = True,
    validate: bool = False,
) -> dict:
    """
    Rank securities with defensive overlays integrated.
    
    Drop-in replacement for compute_module_5_composite() with defensive overlays.
    """
    # Call your existing Module 5
    output = compute_module_5_composite(
        universe_result=universe_result,
        financial_result=financial_result,
        catalyst_result=catalyst_result,
        clinical_result=clinical_result,
        as_of_date=as_of_date,
        weights=weights,
        normalization=normalization,
        coinvest_signals=coinvest_signals,
        cohort_mode=cohort_mode,
    )
    
    # Build defensive_features lookup from RAW universe file
    # Module 1 strips out defensive_features, so we load the raw file
    defensive_by_ticker = {}
    
    # Try to find the raw universe file
    # Check common locations
    universe_paths = [
        Path("production_data/universe.json"),
        Path("wake_robin_data_pipeline/outputs/universe_snapshot_latest.json"),
        Path("wake_robin_data_pipeline/outputs/universe.json"),
        Path("test_data/universe.json"),
    ]
    
    raw_universe = None
    for path in universe_paths:
        if path.exists():
            print(f"\nDEBUG: Loading defensive features from {path}")
            with open(path, 'r') as f:
                raw_universe = json.load(f)
            break
    
    if raw_universe:
        # Handle both dict and array formats
        if isinstance(raw_universe, dict):
            securities = raw_universe.get("active_securities", [])
        else:
            securities = raw_universe  # Direct array
        
        for sec in securities:
            ticker = sec.get("ticker")
            if ticker and "defensive_features" in sec:
                defensive_by_ticker[ticker] = {"defensive_features": sec["defensive_features"]}
        
        print(f"DEBUG: Extracted defensive_features for {len(defensive_by_ticker)} tickers")
    else:
        print("WARNING: Could not find raw universe file for defensive features")

    # Add defensive overlays
    enrich_with_defensive_overlays(
        output,
        defensive_by_ticker,
        apply_multiplier=apply_defensive_multiplier,
        apply_position_sizing=apply_position_sizing,
        top_n=60,  # Enable top-N selection
    )
    
    # Optionally validate
    if validate:
        validate_defensive_integration(output)
    
    return output


# Convenience exports
__all__ = [
    "compute_module_5_composite_with_defensive",
    "enrich_with_defensive_overlays",
    "validate_defensive_integration",
]


if __name__ == "__main__":
    print("module_5_composite_with_defensive.py - Wrapper for defensive overlays")
    print("Run: python test_defensive_integration.py")
