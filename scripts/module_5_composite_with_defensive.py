"""
module_5_composite_with_defensive.py

Simple wrapper that adds defensive overlays to your existing Module 5.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List

from module_5_composite import compute_module_5_composite
from defensive_overlay_adapter import enrich_with_defensive_overlays, validate_defensive_integration

logger = logging.getLogger(__name__)

# Default search paths for universe file with defensive features
DEFAULT_UNIVERSE_PATHS = [
    "production_data/universe.json",
    "wake_robin_data_pipeline/outputs/universe_snapshot_latest.json",
    "wake_robin_data_pipeline/outputs/universe.json",
    "data/universe.json",
]


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
    universe_path: Optional[str] = None,
    universe_search_paths: Optional[List[str]] = None,
    enhancement_result: dict = None,
) -> dict:
    """
    Rank securities with defensive overlays integrated.

    Drop-in replacement for compute_module_5_composite() with defensive overlays.

    Args:
        universe_result: Module 1 output
        financial_result: Module 2 output
        catalyst_result: Module 3 output
        clinical_result: Module 4 output
        as_of_date: Analysis date (YYYY-MM-DD)
        weights: Override default weights
        normalization: "rank" (default) or "zscore"
        coinvest_signals: Optional co-invest overlay data
        cohort_mode: "stage_only" (recommended) or "stage_mcap"
        apply_defensive_multiplier: Apply defensive score multiplier
        apply_position_sizing: Apply position sizing based on volatility
        validate: Run validation checks
        universe_path: Explicit path to universe file with defensive_features
        universe_search_paths: Custom list of paths to search for universe file
        enhancement_result: Optional enhancement module results (PoS, regime, SI)

    Returns:
        Module 5 output enriched with defensive overlay fields
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
        enhancement_result=enhancement_result,
    )

    # Build defensive_features lookup from RAW universe file
    # Module 1 strips out defensive_features, so we load the raw file
    defensive_by_ticker = {}
    raw_universe = None
    loaded_from = None

    # If explicit path provided, use it
    if universe_path:
        path = Path(universe_path)
        if path.exists():
            with open(path, 'r') as f:
                raw_universe = json.load(f)
            loaded_from = path
        else:
            logger.warning(f"Specified universe_path does not exist: {universe_path}")

    # Otherwise search configured paths
    if raw_universe is None:
        search_paths = universe_search_paths or DEFAULT_UNIVERSE_PATHS
        for path_str in search_paths:
            path = Path(path_str)
            if path.exists():
                with open(path, 'r') as f:
                    raw_universe = json.load(f)
                loaded_from = path
                break

    if raw_universe:
        logger.info(f"Loaded defensive features from {loaded_from}")

        # Handle both dict and array formats
        if isinstance(raw_universe, dict):
            securities = raw_universe.get("active_securities", [])
        else:
            securities = raw_universe  # Direct array

        for sec in securities:
            ticker = sec.get("ticker")
            if ticker and "defensive_features" in sec:
                defensive_by_ticker[ticker] = {"defensive_features": sec["defensive_features"]}

        logger.info(f"Extracted defensive_features for {len(defensive_by_ticker)} tickers")
    else:
        logger.warning(
            "Could not find raw universe file for defensive features. "
            f"Searched: {universe_search_paths or DEFAULT_UNIVERSE_PATHS}"
        )

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
