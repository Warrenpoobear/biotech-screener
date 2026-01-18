"""
module_5_composite_with_defensive.py

Production wrapper that adds defensive overlays to Module 5.

v1.1.0 (2026-01-18): Merged v2 features into production:
  - Monotonic caps (risk gates can't be "outvoted")
  - Confidence weighting (module-level confidence affects weights)
  - Hybrid aggregation (weighted-sum + weakest-link blend)
  - Determinism hash (SHA256 for audit parity)
  - Volatility-adjusted weighting

The v2 scorer is now the default. Set use_v2_scoring=False to revert to v1.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from module_5_composite import compute_module_5_composite
from module_5_composite_v2 import compute_module_5_composite_v2
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
    use_v2_scoring: bool = True,
    market_data_by_ticker: Optional[Dict[str, Dict[str, Any]]] = None,
) -> dict:
    """
    Rank securities with defensive overlays integrated.

    Production wrapper for Module 5 composite scoring with v2 enhancements.

    Args:
        universe_result: Module 1 output
        financial_result: Module 2 output
        catalyst_result: Module 3 output
        clinical_result: Module 4 output
        as_of_date: Analysis date (YYYY-MM-DD)
        weights: Override default weights
        normalization: "rank" (default) or "zscore" (v1 only)
        coinvest_signals: Optional co-invest overlay data
        cohort_mode: "stage_only" (recommended) or "stage_mcap" (v1 only)
        apply_defensive_multiplier: Apply defensive score multiplier
        apply_position_sizing: Apply position sizing based on volatility
        validate: Run validation checks
        universe_path: Explicit path to universe file with defensive_features
        universe_search_paths: Custom list of paths to search for universe file
        enhancement_result: Optional enhancement module results (PoS, regime, SI)
        use_v2_scoring: Use v2 scoring with caps/confidence/hybrid (default True)
        market_data_by_ticker: Optional market data for volatility adjustment (v2)

    Returns:
        Module 5 output enriched with defensive overlay fields

    V2 Features (enabled by default):
        - Monotonic caps: Risk gates can't be "outvoted" by strong modules
        - Confidence weighting: Module confidence affects effective weights
        - Hybrid aggregation: alpha*weighted_sum + (1-alpha)*min_critical
        - Determinism hash: SHA256 for audit parity
        - Volatility adjustment: Weight adjustment based on realized vol
    """
    if use_v2_scoring:
        logger.info("Module 5: Using v2 scoring (caps/confidence/hybrid/hash)")
        output = compute_module_5_composite_v2(
            universe_result=universe_result,
            financial_result=financial_result,
            catalyst_result=catalyst_result,
            clinical_result=clinical_result,
            as_of_date=as_of_date,
            weights=weights,
            coinvest_signals=coinvest_signals,
            enhancement_result=enhancement_result,
            market_data_by_ticker=market_data_by_ticker,
        )
    else:
        logger.info("Module 5: Using v1 scoring (legacy)")
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


__version__ = "1.1.0"

# Convenience exports
__all__ = [
    "compute_module_5_composite_with_defensive",
    "enrich_with_defensive_overlays",
    "validate_defensive_integration",
    "compute_module_5_composite_v2",  # Direct access to v2 scorer
]


if __name__ == "__main__":
    print(f"module_5_composite_with_defensive.py v{__version__}")
    print("Production wrapper with v2 features: caps/confidence/hybrid/hash")
    print()
    print("V2 Features (default):")
    print("  - Monotonic caps: Risk gates can't be 'outvoted'")
    print("  - Confidence weighting: Module confidence affects weights")
    print("  - Hybrid aggregation: weighted_sum + weakest_link blend")
    print("  - Determinism hash: SHA256 audit trail")
    print("  - Volatility adjustment: Weight scaling by realized vol")
    print()
    print("Usage:")
    print("  # Default (v2 scoring)")
    print("  result = compute_module_5_composite_with_defensive(...)")
    print()
    print("  # Revert to v1")
    print("  result = compute_module_5_composite_with_defensive(..., use_v2_scoring=False)")
    print()
    print("Test: python -m pytest tests/test_module_5_v2_integration.py -v")
